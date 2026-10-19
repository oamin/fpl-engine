"""Calibrate correlated-draw σ_λ with heterogeneous σ_out vs realised corr.

1. Estimate σ_log_lam from football-data xG vs goals (overdispersion; often ≈0).
2. Realised pairwise residual correlations from ``score_xp_v2_closed``.
3. Build as-of ``sigma_out`` (``outcome_spread``); use position means as toy
   idio; sweep σ_log_lam. ``xp_se`` is never used in draws.

  python -m src.markets.calibrate_correlated_draws
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS
from src.markets.correlated_draws import DEFAULT_DIRICHLET_STRENGTH, _toy_side, draw_fixture

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
MIN_MINUTES = 60.0


def _stem(season: str) -> str:
    return season.replace("-", "_")


def load_fd_sides() -> pd.DataFrame:
    frames = []
    for season in CLOSED_SEASONS:
        path = PROCESSED / f"scoreline_fd_moments_{_stem(season)}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calibrate_sigma_from_xg(fd: pd.DataFrame) -> dict[str, Any]:
    """σ proxies from side-level xG vs goals."""
    xg = pd.to_numeric(fd["xg"], errors="coerce").to_numpy(float)
    goals = pd.to_numeric(fd["goals"], errors="coerce").to_numpy(float)
    mask = np.isfinite(xg) & np.isfinite(goals) & (xg > 0)
    xg, goals = xg[mask], goals[mask]
    resid = goals - xg
    var_resid = float(np.var(resid, ddof=1))
    mean_xg = float(np.mean(xg))
    mean_xg2 = float(np.mean(xg**2))
    excess = var_resid - mean_xg
    sigma_od = float(np.sqrt(max(0.0, excess) / mean_xg2)) if mean_xg2 > 0 else 0.0
    log_r = np.log(goals + 0.5) - np.log(xg + 0.5)
    sigma_log = float(np.std(log_r, ddof=1))
    return {
        "n_sides": int(mask.sum()),
        "mean_xg": mean_xg,
        "var_goals_minus_xg": var_resid,
        "poisson_mean_xg": mean_xg,
        "excess_var": float(excess),
        "sigma_from_overdispersion": sigma_od,
        "sigma_from_log_resid": sigma_log,
        "log_resid_mean": float(np.mean(log_r)),
        "note": (
            "Overdispersion σ is 0 when Var(G−xG) ≤ E[xG] (common here). "
            "Log-residual SD mixes outcome noise with λ error — upper bound only. "
            "Player co-movement needs σ_idio; match-shock alone is not residual-scale."
        ),
    }


def realised_pair_corrs(v2: pd.DataFrame) -> dict[str, Any]:
    """Pairwise residual corr: same-team attackers; attacker vs opp GK."""
    df = v2.copy()
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["score_xp_v2"] = pd.to_numeric(df["score_xp_v2"], errors="coerce")
    df = df.dropna(subset=["total_points", "score_xp_v2"])
    df = df.drop_duplicates(["season", "fixture_id", "element"], keep="first")
    df["resid"] = df["total_points"] - df["score_xp_v2"]

    atk_x: list[float] = []
    atk_y: list[float] = []
    gk_x: list[float] = []
    gk_y: list[float] = []
    by_season: list[dict[str, Any]] = []

    for season, sdf in df.groupby("season", sort=True):
        ax: list[float] = []
        ay: list[float] = []
        gx: list[float] = []
        gy: list[float] = []
        for _, g in sdf.groupby("fixture_id"):
            atk = g.loc[
                g["position"].isin(["MID", "FWD"]) & (g["minutes"] >= MIN_MINUTES)
            ]
            for _, tg in atk.groupby("team"):
                if len(tg) < 2:
                    continue
                r = tg["resid"].to_numpy(float)
                for i, j in combinations(range(len(r)), 2):
                    ax.append(float(r[i]))
                    ay.append(float(r[j]))
            teams = list(g["team"].astype(str).unique())
            if len(teams) != 2:
                continue
            for atk_team, gk_team in ((teams[0], teams[1]), (teams[1], teams[0])):
                a = g.loc[
                    (g["team"].astype(str) == atk_team)
                    & g["position"].isin(["MID", "FWD"])
                    & (g["minutes"] >= MIN_MINUTES)
                ]
                gk = g.loc[
                    (g["team"].astype(str) == gk_team)
                    & (g["position"] == "GKP")
                    & (g["minutes"] >= MIN_MINUTES)
                ]
                if a.empty or gk.empty:
                    continue
                rg = float(gk["resid"].iloc[0])
                for val in a["resid"].to_numpy(float):
                    gx.append(float(val))
                    gy.append(rg)
        c_atk = float(np.corrcoef(ax, ay)[0, 1]) if len(ax) > 20 else float("nan")
        c_gk = float(np.corrcoef(gx, gy)[0, 1]) if len(gx) > 20 else float("nan")
        by_season.append(
            {
                "season": season,
                "n_atk_pairs": len(ax),
                "corr_atk_atk_resid": c_atk,
                "n_atk_gk_pairs": len(gx),
                "corr_atk_opp_gk_resid": c_gk,
            }
        )
        atk_x.extend(ax)
        atk_y.extend(ay)
        gk_x.extend(gx)
        gk_y.extend(gy)

    resid_sd = float(df.loc[df["minutes"] >= MIN_MINUTES, "resid"].std(ddof=1))
    return {
        "pooled": {
            "n_atk_pairs": len(atk_x),
            "corr_atk_atk_resid": float(np.corrcoef(atk_x, atk_y)[0, 1])
            if len(atk_x) > 20
            else None,
            "n_atk_gk_pairs": len(gk_x),
            "corr_atk_opp_gk_resid": float(np.corrcoef(gk_x, gk_y)[0, 1])
            if len(gk_x) > 20
            else None,
            "resid_sd_minutes60": resid_sd,
        },
        "by_season": by_season,
        "definition": (
            f"resid = total_points − score_xp_v2; minutes≥{int(MIN_MINUTES)}; "
            "unique element×fixture; attackers=MID/FWD"
        ),
    }


def _pair_corrs_from_xp(xp: np.ndarray, ids: list[str]) -> dict[str, float]:
    i1, i2 = ids.index("H_M1"), ids.index("H_M2")
    i_gk = ids.index("A_GK")
    return {
        "corr_home_mids_xp": float(np.corrcoef(xp[:, i1], xp[:, i2])[0, 1]),
        "corr_home_m1_vs_away_gk": float(np.corrcoef(xp[:, i1], xp[:, i_gk])[0, 1]),
    }


def structural_draws(
    sigma_log_lam: float,
    *,
    n_scenarios: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match-shock structural xP (σ_idio=0) + goal component."""
    home = _toy_side("H", "H")
    away = _toy_side("A", "A")
    d = draw_fixture(
        1.7,
        1.1,
        -0.05,
        home,
        away,
        n_scenarios=n_scenarios,
        seed=seed,
        sigma_log_lam=float(sigma_log_lam),
        sigma_idio=0.0,
        dirichlet_shares_flag=False,
        dirichlet_strength=DEFAULT_DIRICHLET_STRENGTH,
    )
    return d.xp, d.xp_goals, list(d.player_ids)


def _toy_sigma_out_vector(pos_sigma: dict[str, float]) -> np.ndarray:
    """Home then away toy order: GK, DEF, MID, MID × 2 sides."""
    order = ["GKP", "DEF", "MID", "MID", "GKP", "DEF", "MID", "MID"]
    return np.array([float(pos_sigma.get(p, 3.0)) for p in order], dtype=float)


def match_hetero_sigma_out(
    target_atk_gk: float,
    target_atk_atk: float,
    pos_sigma: dict[str, float],
    *,
    sigma_grid: np.ndarray | None = None,
    n_scenarios: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Sweep σ_log_lam with fixed per-position σ_out (resid-scale idio)."""
    sigma_grid = (
        sigma_grid
        if sigma_grid is not None
        else np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50])
    )
    sigma_out = _toy_sigma_out_vector(pos_sigma)
    rows: list[dict[str, Any]] = []
    for sig in sigma_grid:
        xp_struct, xp_goals, ids = structural_draws(
            float(sig), n_scenarios=n_scenarios, seed=seed
        )
        rng = np.random.default_rng(seed + 91)
        xp = xp_struct + rng.normal(0.0, 1.0, size=xp_struct.shape) * sigma_out.reshape(
            1, -1
        )
        pair = _pair_corrs_from_xp(xp, ids)
        err_gk = (pair["corr_home_m1_vs_away_gk"] - target_atk_gk) ** 2
        err_atk = (pair["corr_home_mids_xp"] - target_atk_atk) ** 2
        rows.append(
            {
                "sigma_log_lam": float(sig),
                "sigma_out_mid": float(pos_sigma.get("MID", 3.0)),
                "sigma_out_gkp": float(pos_sigma.get("GKP", 2.7)),
                "corr_home_mids_xp": pair["corr_home_mids_xp"],
                "corr_home_m1_vs_away_gk": pair["corr_home_m1_vs_away_gk"],
                "loss": float(2.0 * err_gk + err_atk),
            }
        )
    best = min(rows, key=lambda r: r["loss"])
    return {
        "mode": "hetero_sigma_out",
        "pos_sigma": pos_sigma,
        "grid": rows,
        "best": best,
        "note": (
            "Idio = position mean sigma_out (expanding residual SD). "
            "xp_se is not used. Marginal draw SD ≈ sigma_out; coupling from σ_λ."
        ),
    }


def match_two_param(
    target_atk_gk: float,
    target_atk_atk: float,
    *,
    sigma_grid: np.ndarray | None = None,
    idio_grid: np.ndarray | None = None,
    n_scenarios: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Minimise 2·(Δgk)² + (Δmid)² over (σ_log_lam, σ_idio)."""
    sigma_grid = (
        sigma_grid
        if sigma_grid is not None
        else np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25])
    )
    idio_grid = (
        idio_grid
        if idio_grid is not None
        else np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    )
    rows: list[dict[str, Any]] = []
    for sig in sigma_grid:
        xp_struct, xp_goals, ids = structural_draws(
            float(sig), n_scenarios=n_scenarios, seed=seed
        )
        c_goals = float(
            np.corrcoef(xp_goals[:, ids.index("H_M1")], xp_goals[:, ids.index("H_M2")])[
                0, 1
            ]
        )
        for idio in idio_grid:
            idio = float(idio)
            if idio > 0:
                rng = np.random.default_rng(seed + 17 + int(round(1000 * idio)))
                xp = xp_struct + rng.normal(0.0, idio, size=xp_struct.shape)
            else:
                xp = xp_struct
            pair = _pair_corrs_from_xp(xp, ids)
            err_gk = (pair["corr_home_m1_vs_away_gk"] - target_atk_gk) ** 2
            err_atk = (pair["corr_home_mids_xp"] - target_atk_atk) ** 2
            rows.append(
                {
                    "sigma_log_lam": float(sig),
                    "sigma_idio": idio,
                    "corr_home_mids_xp": pair["corr_home_mids_xp"],
                    "corr_home_mids_goals_struct": c_goals,
                    "corr_home_m1_vs_away_gk": pair["corr_home_m1_vs_away_gk"],
                    "loss": float(2.0 * err_gk + err_atk),
                }
            )
    best = min(rows, key=lambda r: r["loss"])
    # Also report: best at matching opp-GK only; mid-only; and match-shock only
    best_gk = min(rows, key=lambda r: (r["corr_home_m1_vs_away_gk"] - target_atk_gk) ** 2)
    best_mid = min(rows, key=lambda r: (r["corr_home_mids_xp"] - target_atk_atk) ** 2)
    match_only = [r for r in rows if r["sigma_idio"] == 0.0]
    return {
        "grid": rows,
        "best": best,
        "best_opp_gk_only": best_gk,
        "best_mid_mid_only": best_mid,
        "match_shock_only_note": (
            "With σ_idio=0, toy mid–mid and mid–GK |corr| stay near 1 for all "
            f"σ_λ in the grid (example σ={match_only[-1]['sigma_log_lam']:.2f}: "
            f"mid={match_only[-1]['corr_home_mids_xp']:+.3f}, "
            f"gk={match_only[-1]['corr_home_m1_vs_away_gk']:+.3f})."
        ),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    xg = doc["sigma_from_xg"]
    real = doc["realised"]["pooled"]
    best = doc["matched"]["best"]
    pos = doc["matched"].get("pos_sigma", {})
    lines = [
        "# Correlated draws — calibration vs realised co-movement",
        "",
        "## 1. σ_log_lam from football-data xG vs goals",
        "",
        f"- Sides: **{xg['n_sides']}** (four closed seasons)",
        f"- Var(goals − xG): **{xg['var_goals_minus_xg']:.3f}** vs Poisson E[xG]=**{xg['mean_xg']:.3f}**",
        f"- Excess var: **{xg['excess_var']:+.3f}** → overdispersion σ_λ ≈ **{xg['sigma_from_overdispersion']:.3f}**",
        f"- SD of log((G+0.5)/(xG+0.5)): **{xg['sigma_from_log_resid']:.3f}** (upper bound; includes outcome noise)",
        "",
        xg["note"],
        "",
        "## 2. Realised player residual co-movement",
        "",
        doc["realised"]["definition"],
        "",
        f"- Same-team MID/FWD resid corr: **{real['corr_atk_atk_resid']:+.3f}** (n={real['n_atk_pairs']})",
        f"- Attacker vs opposing GKP resid corr: **{real['corr_atk_opp_gk_resid']:+.3f}** (n={real['n_atk_gk_pairs']})",
        f"- Residual SD (minutes≥60): **{real['resid_sd_minutes60']:.2f}** pts",
        "",
        "| season | atk–atk | atk–opp GK |",
        "| --- | ---: | ---: |",
    ]
    for r in doc["realised"]["by_season"]:
        lines.append(
            f"| {r['season']} | {r['corr_atk_atk_resid']:+.3f} | {r['corr_atk_opp_gk_resid']:+.3f} |"
        )
    lines += [
        "",
        "## 3. Heterogeneous σ_out + match σ_λ (toy fixture)",
        "",
        "Idio = position mean **sigma_out** (expanding residual SD; see "
        "`reports/outcome_spread.md`). **Not** `xp_se`. "
        "Structural path = λ shocks → scoreline → player xP; then "
        "N(0, σ_out,j²) per player.",
        "",
        doc["matched"].get("note", ""),
        "",
        f"- Position σ_out priors: "
        + ", ".join(f"{k}={v:.2f}" for k, v in sorted(pos.items())),
        f"- Chosen **σ_log_lam = {best['sigma_log_lam']:.3f}**",
        f"- Draw corr home mids: **{best['corr_home_mids_xp']:+.3f}** "
        f"(target {real['corr_atk_atk_resid']:+.3f})",
        f"- Draw corr home M1 vs away GK: **{best['corr_home_m1_vs_away_gk']:+.3f}** "
        f"(target {real['corr_atk_opp_gk_resid']:+.3f})",
        "",
        "### Sweep",
        "",
        "| σ_log_lam | mid–mid xP | M1 vs away GK | loss |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(doc["matched"]["grid"], key=lambda x: x["sigma_log_lam"]):
        mark = " ←" if r["sigma_log_lam"] == best["sigma_log_lam"] else ""
        lines.append(
            f"| {r['sigma_log_lam']:.3f} | {r['corr_home_mids_xp']:+.3f} | "
            f"{r['corr_home_m1_vs_away_gk']:+.3f} | {r['loss']:.4f}{mark} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        doc["verdict"],
        "",
        f"Defaults: `{(PROCESSED / 'correlated_draws_defaults.json').relative_to(ROOT)}`",
        f"JSON: `{(PROCESSED / 'correlated_draws_calibration.json').relative_to(ROOT)}`",
        f"σ_out table: `{(PROCESSED / 'outcome_spread_closed.csv').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scenarios", type=int, default=400)
    args = ap.parse_args(argv)

    from src.markets.outcome_spread import expand_outcome_spread

    fd = load_fd_sides()
    v2 = pd.read_csv(CLOSED_V2)
    sigma_xg = calibrate_sigma_from_xg(fd)
    realised = realised_pair_corrs(v2)
    target_gk = float(realised["pooled"]["corr_atk_opp_gk_resid"])
    target_atk = float(realised["pooled"]["corr_atk_atk_resid"])

    spread = expand_outcome_spread(v2)
    pos_sigma = (
        spread.loc[spread["n_out"] >= 3]
        .groupby("position")["sigma_out"]
        .mean()
        .to_dict()
    )
    pos_sigma = {str(k): float(v) for k, v in pos_sigma.items()}

    matched = match_hetero_sigma_out(
        target_gk, target_atk, pos_sigma, n_scenarios=args.n_scenarios
    )
    best = matched["best"]
    fallback_idio = float(pos_sigma.get("MID", realised["pooled"]["resid_sd_minutes60"]))

    recommend = {
        "sigma_log_lam": float(best["sigma_log_lam"]),
        "sigma_idio": fallback_idio,
        "sigma_out_by_position": pos_sigma,
        "source": "hetero_sigma_out_plus_matched_sigma_log_lam",
        "dirichlet_shares_default": False,
        "dirichlet_note": (
            "Keep Dirichlet off for match-coupling calibration; "
            "enable later for within-team share risk"
        ),
        "caveat": (
            "Prefer per-player sigma_out from outcome_spread; "
            "sigma_idio is scalar fallback only. xp_se never enters draws."
        ),
    }
    gk_err = abs(best["corr_home_m1_vs_away_gk"] - target_gk)
    atk_err = abs(best["corr_home_mids_xp"] - target_atk)
    verdict = (
        f"Ship **σ_log_lam={best['sigma_log_lam']:.3f}** with per-player **σ_out** "
        f"(position means MID={pos_sigma.get('MID', float('nan')):.2f}, "
        f"GKP={pos_sigma.get('GKP', float('nan')):.2f}). "
        f"Draw mid–mid {best['corr_home_mids_xp']:+.3f} vs realised {target_atk:+.3f} "
        f"(|Δ|={atk_err:.3f}); opp-GK {best['corr_home_m1_vs_away_gk']:+.3f} vs "
        f"{target_gk:+.3f} (|Δ|={gk_err:.3f}). "
        f"Scalar fallback σ_idio={fallback_idio:.2f}. xp_se stays shrink-only."
    )

    doc = {
        "sigma_from_xg": sigma_xg,
        "realised": realised,
        "matched": matched,
        "recommended": recommend,
        "verdict": verdict,
    }
    out_json = PROCESSED / "correlated_draws_calibration.json"
    out_md = REPORTS / "correlated_draws_calibration.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)

    side = PROCESSED / "correlated_draws_defaults.json"
    side.write_text(json.dumps(recommend, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"recommended": recommend, "verdict": verdict}, indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
