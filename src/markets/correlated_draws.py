"""Correlated Monte Carlo draws through the scoreline → player-xP graph.

Phase 1: sample Dixon–Coles (λ_h, λ_a, ρ) around a fit; rebuild team pots;
re-score both sides. Teammates (and opponents) share the same match shock.

Phase 2 (optional): Dirichlet draws on goal/assist weights so within-team
shares are negatively correlated and sum to 1.

Store the draw matrix for planning (captain pairs, CVaR) — do not collapse to
independent ``xp_se`` only.

  python -m src.markets.correlated_draws   # small demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets import numagent_xp as nx
from src.markets import scoreline as sl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Fallback if calibration file missing. Prefer
# ``data/processed/correlated_draws_defaults.json`` from
# ``python -m src.markets.calibrate_correlated_draws``.
DEFAULT_SIGMA_LOG_LAM = 0.30  # fallback; prefer correlated_draws_defaults.json
DEFAULT_SIGMA_IDIO = 2.94  # scalar fallback ≈ MID mean sigma_out; prefer per-player
DEFAULT_SIGMA_RHO = 0.05
DEFAULT_DIRICHLET_STRENGTH = 40.0


def load_calibrated_defaults(
    *,
    default_sigma_log_lam: float = DEFAULT_SIGMA_LOG_LAM,
    default_sigma_idio: float = DEFAULT_SIGMA_IDIO,
) -> tuple[float, float]:
    path = PROCESSED / "correlated_draws_defaults.json"
    if not path.exists():
        return float(default_sigma_log_lam), float(default_sigma_idio)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return (
            float(doc.get("sigma_log_lam", default_sigma_log_lam)),
            float(doc.get("sigma_idio", default_sigma_idio)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return float(default_sigma_log_lam), float(default_sigma_idio)


def load_calibrated_sigma_log_lam(default: float = DEFAULT_SIGMA_LOG_LAM) -> float:
    return load_calibrated_defaults(default_sigma_log_lam=default)[0]


@dataclass(frozen=True)
class FixtureDraws:
    """Correlated xP draws for one fixture."""

    player_ids: tuple[str, ...]
    side: tuple[str, ...]  # home/away per player
    xp: np.ndarray  # shape (n_scenarios, n_players)
    xp_goals: np.ndarray
    mu_point: np.ndarray  # deterministic score_fixture mean
    lam_h: np.ndarray
    lam_a: np.ndarray
    meta: dict[str, Any]

    @property
    def n_scenarios(self) -> int:
        return int(self.xp.shape[0])

    @property
    def n_players(self) -> int:
        return int(self.xp.shape[1])

    def mean_xp(self) -> np.ndarray:
        return self.xp.mean(axis=0)

    def corr_xp(self) -> np.ndarray:
        """Player×player correlation of total xP across scenarios."""
        if self.n_scenarios < 3:
            return np.eye(self.n_players)
        c = np.corrcoef(self.xp, rowvar=False)
        return np.nan_to_num(c, nan=0.0)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for j, pid in enumerate(self.player_ids):
            rows.append(
                {
                    "player_id": pid,
                    "side": self.side[j],
                    "mu_point": float(self.mu_point[j]),
                    "mu_draw": float(self.xp[:, j].mean()),
                    "sd_draw": float(self.xp[:, j].std(ddof=1)),
                    "mu_goals_draw": float(self.xp_goals[:, j].mean()),
                }
            )
        return pd.DataFrame(rows)


def sample_lambdas(
    lam_h: float,
    lam_a: float,
    rho: float,
    n: int,
    *,
    sigma_log_lam: float = DEFAULT_SIGMA_LOG_LAM,
    sigma_rho: float = DEFAULT_SIGMA_RHO,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lognormal shocks on λ; Gaussian on ρ (clipped)."""
    rng = rng or np.random.default_rng(0)
    n = int(n)
    z_h = rng.normal(0.0, float(sigma_log_lam), size=n)
    z_a = rng.normal(0.0, float(sigma_log_lam), size=n)
    # Mild positive correlation of total goals shock (match tempo)
    z_tot = rng.normal(0.0, float(sigma_log_lam) * 0.5, size=n)
    lh = np.clip(np.exp(np.log(max(lam_h, 1e-6)) + z_h + z_tot), 0.15, 4.0)
    la = np.clip(np.exp(np.log(max(lam_a, 1e-6)) + z_a + z_tot), 0.15, 4.0)
    rh = np.clip(float(rho) + rng.normal(0.0, float(sigma_rho), size=n), -0.2, 0.2)
    return lh, la, rh


def dirichlet_shares(
    weights: np.ndarray,
    *,
    strength: float = DEFAULT_DIRICHLET_STRENGTH,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Dirichlet draw; zeros stay zero; positive weights get compositional noise."""
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.zeros_like(w)
    alpha = strength * (w / s)
    alpha = np.maximum(alpha, 1e-6)
    # Dirichlet only on positive mass indices
    pos = w > 0
    out = np.zeros_like(w)
    draw = rng.dirichlet(alpha[pos])
    out[pos] = draw
    return out


def _player_ids(df: pd.DataFrame) -> list[str]:
    if "player_id" in df.columns:
        return [str(x) for x in df["player_id"].tolist()]
    if "name" in df.columns:
        return [str(x) for x in df["name"].tolist()]
    return [str(i) for i in range(len(df))]


def _base_weights(players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Goal/assist weights used for Dirichlet (same as score_team_side)."""
    n = len(players)
    m = pd.to_numeric(players.get("m", 0), errors="coerce").fillna(0.0).to_numpy(float)
    sm = pd.to_numeric(players.get("sigma_m", 0), errors="coerce").fillna(0.0).to_numpy(float)
    rG = pd.to_numeric(players.get("rG", 0), errors="coerce").fillna(0.0).to_numpy(float)
    drG = pd.to_numeric(players.get("drG", 0), errors="coerce").fillna(0.0).to_numpy(float)
    rA = pd.to_numeric(players.get("rA", 0), errors="coerce").fillna(0.0).to_numpy(float)
    drA = pd.to_numeric(players.get("drA", 0), errors="coerce").fillna(0.0).to_numpy(float)
    wG = np.zeros(n)
    wA = np.zeros(n)
    for i in range(n):
        wG[i], _ = nx.weight_and_delta(rG[i], drG[i], m[i], sm[i])
        wA[i], _ = nx.weight_and_delta(rA[i], drA[i], m[i], sm[i])
    return wG, wA


def _resolve_sigma_out(
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    n_p: int,
    *,
    sigma_out: np.ndarray | list[float] | None,
    sigma_idio: float | None,
    default_idio: float,
) -> np.ndarray:
    """Per-player idio SD. Prefer ``sigma_out``; else scalar ``sigma_idio``; else default.

    Also reads optional ``sigma_out`` columns on the player frames (home then away).
    """
    if sigma_out is not None:
        arr = np.asarray(sigma_out, dtype=float).reshape(-1)
        if arr.size != n_p:
            raise ValueError(f"sigma_out length {arr.size} != n_players {n_p}")
        return np.maximum(arr, 0.0)
    # Column on frames (draws-only; never xp_se)
    parts = []
    for frame in (home_players, away_players):
        if "sigma_out" in frame.columns:
            parts.append(pd.to_numeric(frame["sigma_out"], errors="coerce").fillna(0.0).to_numpy(float))
        else:
            parts.append(None)
    if parts[0] is not None and parts[1] is not None:
        return np.maximum(np.concatenate(parts), 0.0)
    scale = float(default_idio if sigma_idio is None else sigma_idio)
    return np.full(n_p, max(scale, 0.0), dtype=float)


def draw_fixture(
    lam_h: float,
    lam_a: float,
    rho: float,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    *,
    n_scenarios: int = 200,
    alpha_g: float = 0.96,
    alpha_a: float = 0.90,
    sigma_log_lam: float | None = None,
    sigma_idio: float | None = None,
    sigma_out: np.ndarray | list[float] | None = None,
    dirichlet_shares_flag: bool = False,
    dirichlet_strength: float = DEFAULT_DIRICHLET_STRENGTH,
    seed: int = 0,
) -> FixtureDraws:
    """Return correlated xP draws for one fixture.

    Match shock is shared. Optional Dirichlet makes within-team shares
    negatively correlated. Independent outcome noise uses per-player
    ``sigma_out`` (predictive residual SD) when provided — **not** ``xp_se``.
    Scalar ``sigma_idio`` is the fallback when ``sigma_out`` is absent.
    """
    cal_lam, cal_idio = load_calibrated_defaults()
    if sigma_log_lam is None:
        sigma_log_lam = cal_lam
    rng = np.random.default_rng(int(seed))
    n_scenarios = int(n_scenarios)
    mat0 = sl.build_matrix(lam_h, lam_a, rho)
    point = nx.score_fixture(
        mat0,
        home_players,
        away_players,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        d_alpha_g=0.0,
        d_alpha_a=0.0,
    )
    ids = _player_ids(point)
    sides = tuple(str(s) for s in point["side"].tolist())
    mu_point = point["xp"].to_numpy(float)

    wG_h, wA_h = _base_weights(home_players)
    wG_a, wA_a = _base_weights(away_players)

    lh, la, rh = sample_lambdas(
        lam_h, lam_a, rho, n_scenarios, sigma_log_lam=sigma_log_lam, rng=rng
    )
    n_h = len(home_players)
    n_a = len(away_players)
    n_p = n_h + n_a
    xp = np.zeros((n_scenarios, n_p))
    xp_goals = np.zeros((n_scenarios, n_p))

    for s in range(n_scenarios):
        mat = sl.build_matrix(float(lh[s]), float(la[s]), float(rh[s]))
        mom = sl.team_moments(mat)
        sG_h = sA_h = sG_a = sA_a = None
        if dirichlet_shares_flag:
            sG_h = dirichlet_shares(wG_h, strength=dirichlet_strength, rng=rng)
            sA_h = dirichlet_shares(wA_h, strength=dirichlet_strength, rng=rng)
            sG_a = dirichlet_shares(wG_a, strength=dirichlet_strength, rng=rng)
            sA_a = dirichlet_shares(wA_a, strength=dirichlet_strength, rng=rng)
        home = nx.score_team_side(
            home_players,
            xg=mom["xg_home"],
            xcs=mom["xcs_home"],
            xgc=mom["xgc_home"],
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            sG_override=sG_h,
            sA_override=sA_h,
        )
        away = nx.score_team_side(
            away_players,
            xg=mom["xg_away"],
            xcs=mom["xcs_away"],
            xgc=mom["xgc_away"],
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            sG_override=sG_a,
            sA_override=sA_a,
        )
        xp[s, :n_h] = home["xp"].to_numpy(float)
        xp[s, n_h:] = away["xp"].to_numpy(float)
        xp_goals[s, :n_h] = home["xp_goals"].to_numpy(float)
        xp_goals[s, n_h:] = away["xp_goals"].to_numpy(float)

    idio = _resolve_sigma_out(
        home_players,
        away_players,
        n_p,
        sigma_out=sigma_out,
        sigma_idio=sigma_idio,
        default_idio=cal_idio,
    )
    if np.any(idio > 0):
        xp = xp + rng.normal(0.0, 1.0, size=xp.shape) * idio.reshape(1, -1)

    return FixtureDraws(
        player_ids=tuple(ids),
        side=sides,
        xp=xp,
        xp_goals=xp_goals,
        mu_point=mu_point,
        lam_h=lh,
        lam_a=la,
        meta={
            "n_scenarios": n_scenarios,
            "sigma_log_lam": float(sigma_log_lam),
            "sigma_out_mean": float(idio.mean()),
            "sigma_out": idio.tolist(),
            "dirichlet_shares": dirichlet_shares_flag,
            "dirichlet_strength": dirichlet_strength,
            "lam_h0": float(lam_h),
            "lam_a0": float(lam_a),
            "rho0": float(rho),
            "seed": int(seed),
        },
    )


def _toy_side(prefix: str, clubs: str) -> pd.DataFrame:
    """Minimal 4-player side for demos/tests."""
    return pd.DataFrame(
        {
            "player_id": [f"{prefix}_GK", f"{prefix}_D", f"{prefix}_M1", f"{prefix}_M2"],
            "name": [f"{prefix}_GK", f"{prefix}_D", f"{prefix}_M1", f"{prefix}_M2"],
            "position": ["GKP", "DEF", "MID", "MID"],
            "rG": [0.001, 0.005, 0.025, 0.015],
            "drG": [0.0, 0.002, 0.008, 0.005],
            "rA": [0.001, 0.004, 0.015, 0.012],
            "drA": [0.0, 0.002, 0.005, 0.004],
            "m": [90.0, 90.0, 90.0, 75.0],
            "sigma_m": [0.0, 5.0, 5.0, 10.0],
            "defcon_mu": [0.0, 11.0, 8.0, 7.0],
            "defcon_sig": [1.0, 3.0, 3.0, 3.0],
            "defcon_n": [10, 10, 10, 10],
            "bonus_mu": [0.2, 0.3, 0.5, 0.3],
            "bonus_se": [0.1, 0.1, 0.1, 0.1],
            "card_mu": [-0.05, -0.1, -0.1, -0.1],
            "card_se": [0.05, 0.05, 0.05, 0.05],
            "saves_mu": [1.0, 0.0, 0.0, 0.0],
            "saves_se": [0.2, 0.0, 0.0, 0.0],
        }
    )


def demo_report(*, n: int = 400, seed: int = 0) -> dict[str, Any]:
    home = _toy_side("H", "H")
    away = _toy_side("A", "A")
    draws = draw_fixture(
        1.7,
        1.1,
        -0.05,
        home,
        away,
        n_scenarios=n,
        seed=seed,
        dirichlet_shares_flag=True,
        # Demo: structural + Dirichlet; idio off so coupling is visible
        sigma_idio=0.0,
    )
    corr = draws.corr_xp()
    ids = list(draws.player_ids)
    i_m1 = ids.index("H_M1")
    i_m2 = ids.index("H_M2")
    i_gk_a = ids.index("A_GK")
    # Goal-component corr among home mids (share + team xG)
    gcorr = np.corrcoef(draws.xp_goals[:, i_m1], draws.xp_goals[:, i_m2])[0, 1]
    # Home attack total xP vs away GK (xGC / xCS coupling)
    cross = float(np.corrcoef(draws.xp[:, i_m1], draws.xp[:, i_gk_a])[0, 1])
    mean_err = float(np.max(np.abs(draws.mean_xp() - draws.mu_point)))
    doc = {
        "n_scenarios": n,
        "max_abs_mean_err_vs_point": mean_err,
        "corr_home_mids_total_xp": float(corr[i_m1, i_m2]),
        "corr_home_mids_goal_xp": float(gcorr),
        "corr_home_m1_vs_away_gk": cross,
        "summary": draws.to_frame().to_dict(orient="records"),
        "meta": draws.meta,
    }
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    doc = demo_report(n=args.n, seed=args.seed)
    out_json = PROCESSED / "correlated_draws_demo.json"
    out_md = REPORTS / "correlated_draws.md"
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    lines = [
        "# Correlated draws (scoreline → player xP)",
        "",
        "Monte Carlo through Dixon–Coles λ shocks + optional Dirichlet shares.",
        "Same match factor moves a whole team (and the opposition) together.",
        "",
        f"- Scenarios: **{doc['n_scenarios']}**",
        f"- Max |mean(draw) − point μ|: **{doc['max_abs_mean_err_vs_point']:.4f}**",
        f"- Corr home M1–M2 total xP: **{doc['corr_home_mids_total_xp']:+.3f}**",
        f"- Corr home M1–M2 goal xP: **{doc['corr_home_mids_goal_xp']:+.3f}**",
        f"- Corr home M1 vs away GK: **{doc['corr_home_m1_vs_away_gk']:+.3f}**",
        "",
        "Independent `xp_se` cannot encode these; store the draw matrix for planning.",
        "",
        f"JSON: `{(out_json).relative_to(ROOT)}`",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in doc if k != "summary"}, indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
