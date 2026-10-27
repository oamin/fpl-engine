"""Fixture-specific covariance: M → minutes → volatile shares → xP.

1. Sample a discrete scoreline from Dixon–Coles ``M``.
2. Draw minutes (sparse competition groups + independent noise).
3. Dirichlet goal/assist shares with strength from rate SEs (volatile).
4. Score via ``score_team_side`` on the *realised* scoreline moments
   (goals, CS, GC from the drawn (h,a)), with share overrides.

Cards stay on independent priors (not joint). Saves/DefCon use existing
conditionals given minutes / xGC — not extra GA folklore.

  python -m src.markets.fixture_covariance
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
from src.markets.correlated_draws import dirichlet_shares
from src.markets.minutes_covariance import load_merged
from src.markets.sparse_minutes import (
    CompetitionGroup,
    build_season_groups,
    draw_club_minutes,
    find_club_competition_groups,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Lower strength ⇒ more share volatility. Floor avoids degeneracy.
SHARE_STRENGTH_FLOOR = 4.0
SHARE_STRENGTH_CEIL = 80.0
SHARE_STRENGTH_BASE = 25.0


def sample_scoreline(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Draw (home_goals, away_goals) from M. Tail bin = GRID_MAX."""
    flat = matrix.ravel()
    flat = flat / flat.sum()
    idx = int(rng.choice(flat.size, p=flat))
    n = matrix.shape[0]
    i, j = divmod(idx, n)
    k_max = n - 1
    return int(min(i, k_max)), int(min(j, k_max))


def moments_from_scoreline(goals_h: int, goals_a: int) -> dict[str, float]:
    """Deterministic side moments for a realised scoreline."""
    import math

    return {
        "xg_home": float(goals_h),
        "xg_away": float(goals_a),
        "xcs_home": 1.0 if goals_a == 0 else 0.0,
        "xcs_away": 1.0 if goals_h == 0 else 0.0,
        "xgc_home": float(math.floor(goals_a / 2.0)),
        "xgc_away": float(math.floor(goals_h / 2.0)),
    }


def share_strength(r: np.ndarray, dr: np.ndarray) -> float:
    """Map relative rate SE to Dirichlet strength (higher SE → lower strength)."""
    r = np.asarray(r, dtype=float)
    dr = np.asarray(dr, dtype=float)
    mask = r > 1e-8
    if not mask.any():
        return SHARE_STRENGTH_BASE
    cv = float(np.median(dr[mask] / r[mask]))
    # cv~0.2 → strength~25; cv~0.5 → strength~10
    strength = SHARE_STRENGTH_BASE / max(cv, 0.15) * 0.2
    return float(np.clip(strength, SHARE_STRENGTH_FLOOR, SHARE_STRENGTH_CEIL))


def _weights_for_shares(players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    m = pd.to_numeric(players["m"], errors="coerce").fillna(0.0).to_numpy(float)
    sm = (
        pd.to_numeric(players["sigma_m"], errors="coerce").fillna(0.0).to_numpy(float)
        if "sigma_m" in players.columns
        else np.zeros(len(players))
    )
    rG = pd.to_numeric(players["rG"], errors="coerce").fillna(0.0).to_numpy(float)
    drG = pd.to_numeric(players["drG"], errors="coerce").fillna(0.0).to_numpy(float)
    rA = pd.to_numeric(players["rA"], errors="coerce").fillna(0.0).to_numpy(float)
    drA = pd.to_numeric(players["drA"], errors="coerce").fillna(0.0).to_numpy(float)
    n = len(players)
    wG = np.zeros(n)
    wA = np.zeros(n)
    for i in range(n):
        # Goals: π ∝ r (m gates club membership). Assists: still ∝ r·m/90.
        wG[i], _ = nx.goal_share_weight(rG[i], drG[i])
        wA[i], _ = nx.weight_and_delta(rA[i], drA[i], m[i], sm[i])
    # Zero weights for non-playing
    wG = np.where(m > 0, wG, 0.0)
    wA = np.where(m > 0, wA, 0.0)
    return wG, wA, share_strength(rG, drG), share_strength(rA, drA)


@dataclass
class FixtureCovDraws:
    player_ids: tuple[str, ...]
    side: tuple[str, ...]
    xp: np.ndarray
    goals_h: np.ndarray
    goals_a: np.ndarray
    minutes: np.ndarray
    mu_point: np.ndarray
    meta: dict[str, Any]

    def corr_xp(self) -> np.ndarray:
        if self.xp.shape[0] < 3:
            return np.eye(self.xp.shape[1])
        return np.nan_to_num(np.corrcoef(self.xp, rowvar=False), nan=0.0)


def draw_fixture_covariant(
    lam_h: float,
    lam_a: float,
    rho: float,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    *,
    home_groups: list[CompetitionGroup] | None = None,
    away_groups: list[CompetitionGroup] | None = None,
    n_scenarios: int = 400,
    alpha_g: float = 0.96,
    alpha_a: float = 0.90,
    seed: int = 0,
) -> FixtureCovDraws:
    """Monte Carlo fixture draws with scoreline + sparse minutes + share volatility."""
    rng = np.random.default_rng(int(seed))
    mat = sl.build_matrix(lam_h, lam_a, rho)
    point_mom = sl.team_moments(mat)
    point_h = nx.score_team_side(
        home_players,
        xg=point_mom["xg_home"],
        xcs=point_mom["xcs_home"],
        xgc=point_mom["xgc_home"],
        alpha_g=alpha_g,
        alpha_a=alpha_a,
    )
    point_a = nx.score_team_side(
        away_players,
        xg=point_mom["xg_away"],
        xcs=point_mom["xcs_away"],
        xgc=point_mom["xgc_away"],
        alpha_g=alpha_g,
        alpha_a=alpha_a,
    )
    ids_h = [
        str(x)
        for x in (
            point_h["player_id"] if "player_id" in point_h.columns else point_h["name"]
        ).tolist()
    ]
    ids_a = [
        str(x)
        for x in (
            point_a["player_id"] if "player_id" in point_a.columns else point_a["name"]
        ).tolist()
    ]
    ids = tuple(ids_h + ids_a)
    sides = tuple(["home"] * len(ids_h) + ["away"] * len(ids_a))
    mu_point = np.concatenate(
        [point_h["xp"].to_numpy(float), point_a["xp"].to_numpy(float)]
    )

    n_h, n_a = len(home_players), len(away_players)
    n_p = n_h + n_a
    xp = np.zeros((n_scenarios, n_p))
    minutes = np.zeros((n_scenarios, n_p))
    goals_h = np.zeros(n_scenarios, dtype=int)
    goals_a = np.zeros(n_scenarios, dtype=int)
    home_groups = home_groups or []
    away_groups = away_groups or []

    for s in range(n_scenarios):
        gh, ga = sample_scoreline(mat, rng=rng)
        goals_h[s], goals_a[s] = gh, ga
        mom = moments_from_scoreline(gh, ga)

        h = draw_club_minutes(home_players, home_groups, rng=rng)
        a = draw_club_minutes(away_players, away_groups, rng=rng)
        minutes[s, :n_h] = h["m"].to_numpy(float)
        minutes[s, n_h:] = a["m"].to_numpy(float)

        wG_h, wA_h, stG_h, stA_h = _weights_for_shares(h)
        wG_a, wA_a, stG_a, stA_a = _weights_for_shares(a)
        sG_h = dirichlet_shares(wG_h, strength=stG_h, rng=rng)
        sA_h = dirichlet_shares(wA_h, strength=stA_h, rng=rng)
        sG_a = dirichlet_shares(wG_a, strength=stG_a, rng=rng)
        sA_a = dirichlet_shares(wA_a, strength=stA_a, rng=rng)

        home_sc = nx.score_team_side(
            h,
            xg=mom["xg_home"],
            xcs=mom["xcs_home"],
            xgc=mom["xgc_home"],
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            sG_override=sG_h,
            sA_override=sA_h,
        )
        away_sc = nx.score_team_side(
            a,
            xg=mom["xg_away"],
            xcs=mom["xcs_away"],
            xgc=mom["xgc_away"],
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            sG_override=sG_a,
            sA_override=sA_a,
        )
        xp[s, :n_h] = home_sc["xp"].to_numpy(float)
        xp[s, n_h:] = away_sc["xp"].to_numpy(float)

    return FixtureCovDraws(
        player_ids=ids,
        side=sides,
        xp=xp,
        goals_h=goals_h,
        goals_a=goals_a,
        minutes=minutes,
        mu_point=mu_point,
        meta={
            "n_scenarios": n_scenarios,
            "lam_h": float(lam_h),
            "lam_a": float(lam_a),
            "rho": float(rho),
            "seed": int(seed),
            "n_home_groups": len(home_groups),
            "n_away_groups": len(away_groups),
        },
    )


def _toy_side(prefix: str, *, competing_fb: bool = True) -> pd.DataFrame:
    """Toy side with optional FB competition (Kerkez/Robertson style)."""
    rows = {
        "player_id": [f"{prefix}_GK", f"{prefix}_D1", f"{prefix}_D2", f"{prefix}_M1", f"{prefix}_M2", f"{prefix}_F"],
        "element": [1, 2, 3, 4, 5, 6] if prefix == "H" else [11, 12, 13, 14, 15, 16],
        "name": [f"{prefix}_GK", f"{prefix}_D1", f"{prefix}_D2", f"{prefix}_M1", f"{prefix}_M2", f"{prefix}_F"],
        "position": ["GKP", "DEF", "DEF", "MID", "MID", "FWD"],
        "rG": [0.001, 0.008, 0.004, 0.030, 0.018, 0.040],
        "drG": [0.0, 0.003, 0.002, 0.015, 0.008, 0.018],
        "rA": [0.001, 0.010, 0.006, 0.020, 0.015, 0.012],
        "drA": [0.0, 0.004, 0.003, 0.008, 0.006, 0.005],
        "m": [90.0, 70.0, 35.0, 85.0, 70.0, 80.0],
        "sigma_m": [0.0, 12.0, 18.0, 8.0, 12.0, 10.0],
        "defcon_mu": [0.0, 12.0, 10.0, 6.0, 7.0, 4.0],
        "defcon_sig": [1.0, 3.0, 3.0, 3.0, 3.0, 2.0],
        "defcon_n": [10] * 6,
        "bonus_mu": [0.2, 0.3, 0.2, 0.5, 0.3, 0.4],
        "bonus_se": [0.1] * 6,
        "card_mu": [-0.05, -0.1, -0.1, -0.1, -0.1, -0.08],
        "card_se": [0.05] * 6,
        "saves_mu": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "saves_se": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    if not competing_fb:
        rows["m"][1], rows["m"][2] = 85.0, 85.0
    return pd.DataFrame(rows)


def _toy_fb_group(prefix: str) -> CompetitionGroup:
    el = (2, 3) if prefix == "H" else (12, 13)
    return CompetitionGroup(
        team=prefix,
        elements=el,
        names=(f"{prefix}_D1", f"{prefix}_D2"),
        positions=("DEF", "DEF"),
        mean_minutes=(70.0, 35.0),
        min_pair_corr=-0.95,
    )


def build_club_side(
    merged: pd.DataFrame,
    team: str,
    *,
    min_mean_minutes: float = 15.0,
    max_players: int = 15,
) -> tuple[pd.DataFrame, list[CompetitionGroup]]:
    """Build a scoreable side + competition groups from 2025/26 merged history."""
    club = merged.loc[merged["team"] == team].copy()
    if club.empty:
        raise ValueError(f"No rows for team={team!r}")
    # Mean minutes prior + identity
    mean_m = (
        club.groupby("element", as_index=False)
        .agg(
            name=("name", "first"),
            position=("position", "first"),
            m=("minutes", "mean"),
            sigma_m=("minutes", "std"),
        )
    )
    mean_m["sigma_m"] = mean_m["sigma_m"].fillna(10.0).clip(lower=5.0)
    mean_m = mean_m.loc[mean_m["m"] >= min_mean_minutes].copy()
    mean_m = mean_m.sort_values("m", ascending=False).head(max_players)

    rates = nx.player_rates_from_history(club, id_col="element")
    club_p = club.copy()
    for col, default in (
        ("defensive_contribution", 0.0),
        ("bonus", 0.0),
        ("yellow_cards", 0.0),
        ("red_cards", 0.0),
        ("saves", 0.0),
    ):
        if col not in club_p.columns:
            club_p[col] = default
    priors = nx.history_priors_for_xp(club_p, id_col="element")
    side = mean_m.merge(rates, on="element", how="left").merge(
        priors, on="element", how="left", suffixes=("", "_pr")
    )
    if "position_pr" in side.columns:
        side["position"] = side["position"].fillna(side["position_pr"])
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.2),
        ("bonus_se", 0.1),
        ("card_mu", -0.1),
        ("card_se", 0.05),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", 3.0),
        ("defcon_n", 5),
    ):
        if c not in side.columns:
            side[c] = default
        else:
            side[c] = pd.to_numeric(side[c], errors="coerce").fillna(default)
    # Position: Vaastav uses GK not GKP
    side["position"] = side["position"].replace({"GK": "GKP"})
    side["player_id"] = side["name"].astype(str)
    # Keep one GK if present
    groups, _ = find_club_competition_groups(club, team)
    # Restrict groups to players on this side
    keep = set(side["element"].astype(int))
    groups = [g for g in groups if set(g.elements) <= keep]
    return side.reset_index(drop=True), groups


def _pair_corr_minutes(
    draws: FixtureCovDraws, name_a: str, name_b: str
) -> float | None:
    ids = list(draws.player_ids)
    if name_a not in ids or name_b not in ids:
        return None
    i, j = ids.index(name_a), ids.index(name_b)
    return float(np.corrcoef(draws.minutes[:, i], draws.minutes[:, j])[0, 1])


def _pair_corr_xp(draws: FixtureCovDraws, name_a: str, name_b: str) -> float | None:
    ids = list(draws.player_ids)
    if name_a not in ids or name_b not in ids:
        return None
    i, j = ids.index(name_a), ids.index(name_b)
    return float(np.corrcoef(draws.xp[:, i], draws.xp[:, j])[0, 1])


def demo(*, n: int = 500, seed: int = 0) -> dict[str, Any]:
    home = _toy_side("H", competing_fb=True)
    away = _toy_side("A", competing_fb=True)
    h_groups = [_toy_fb_group("H")]
    a_groups = [_toy_fb_group("A")]
    draws = draw_fixture_covariant(
        1.7,
        1.1,
        -0.05,
        home,
        away,
        home_groups=h_groups,
        away_groups=a_groups,
        n_scenarios=n,
        seed=seed,
    )
    ids = list(draws.player_ids)
    corr = draws.corr_xp()
    i_m1, i_m2 = ids.index("H_M1"), ids.index("H_M2")
    i_d1, i_d2 = ids.index("H_D1"), ids.index("H_D2")
    i_gk_a = ids.index("A_GK")
    i_f = ids.index("H_F")

    c_min_fb = float(np.corrcoef(draws.minutes[:, i_d1], draws.minutes[:, i_d2])[0, 1])
    c_xp_mids = float(corr[i_m1, i_m2])
    c_xp_atk_gk = float(corr[i_m1, i_gk_a])
    c_xp_fb = float(corr[i_d1, i_d2])

    blank = draws.goals_h == 0
    haul = draws.goals_h >= 2
    mu_f_blank = float(draws.xp[blank, i_f].mean()) if blank.any() else float("nan")
    mu_f_haul = float(draws.xp[haul, i_f].mean()) if haul.any() else float("nan")

    return {
        "mode": "toy",
        "n_scenarios": n,
        "corr_home_mids_xp": c_xp_mids,
        "corr_home_m1_vs_away_gk": c_xp_atk_gk,
        "corr_home_fb_minutes": c_min_fb,
        "corr_home_fb_xp": c_xp_fb,
        "mean_xp_fwd_when_home_blank": mu_f_blank,
        "mean_xp_fwd_when_home_2plus": mu_f_haul,
        "mean_abs_err_vs_point": float(
            np.mean(np.abs(draws.xp.mean(axis=0) - draws.mu_point))
        ),
        "scoreline_mean_h": float(draws.goals_h.mean()),
        "scoreline_mean_a": float(draws.goals_a.mean()),
        "meta": draws.meta,
        "competition_pairs": [
            {"side": "home", "a": "H_D1", "b": "H_D2", "minutes_corr": c_min_fb}
        ],
        "summary": [
            {
                "player_id": pid,
                "side": draws.side[j],
                "mu_point": float(draws.mu_point[j]),
                "mu_draw": float(draws.xp[:, j].mean()),
                "sd_draw": float(draws.xp[:, j].std(ddof=1)),
                "mean_minutes": float(draws.minutes[:, j].mean()),
            }
            for j, pid in enumerate(ids)
        ],
    }


def demo_real(
    home_team: str,
    away_team: str,
    *,
    lam_h: float = 1.6,
    lam_a: float = 1.2,
    rho: float = -0.05,
    n: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Real-club fixture using 2025/26 rates + greedy competition pairs."""
    merged = load_merged()
    home, h_groups = build_club_side(merged, home_team)
    away, a_groups = build_club_side(merged, away_team)
    draws = draw_fixture_covariant(
        lam_h,
        lam_a,
        rho,
        home,
        away,
        home_groups=h_groups,
        away_groups=a_groups,
        n_scenarios=n,
        seed=seed,
    )

    # Highlight competition pairs + top attackers vs opp GK
    pair_rows = []
    for side, groups in (("home", h_groups), ("away", a_groups)):
        for g in groups:
            c_m = _pair_corr_minutes(draws, g.names[0], g.names[1])
            c_x = _pair_corr_xp(draws, g.names[0], g.names[1])
            pair_rows.append(
                {
                    "side": side,
                    "a": g.names[0],
                    "b": g.names[1],
                    "positions": "/".join(g.positions),
                    "hist_corr": g.min_pair_corr,
                    "draw_minutes_corr": c_m,
                    "draw_xp_corr": c_x,
                }
            )

    # Attack vs opp GK: highest-μ home mid/fwd vs away GK
    ids = list(draws.player_ids)
    home_atk = home.loc[home["position"].isin(["MID", "FWD"])].sort_values(
        "m", ascending=False
    )
    away_gk = away.loc[away["position"] == "GKP"]
    c_atk_gk = None
    atk_name = gk_name = None
    if len(home_atk) and len(away_gk):
        atk_name = str(home_atk.iloc[0]["name"])
        gk_name = str(away_gk.iloc[0]["name"])
        c_atk_gk = _pair_corr_xp(draws, atk_name, gk_name)

    # Two highest-m home mids
    home_mids = home.loc[home["position"] == "MID"].sort_values("m", ascending=False)
    c_mids = None
    mid_names: list[str] = []
    if len(home_mids) >= 2:
        mid_names = [str(home_mids.iloc[0]["name"]), str(home_mids.iloc[1]["name"])]
        c_mids = _pair_corr_xp(draws, mid_names[0], mid_names[1])

    top = (
        pd.DataFrame(
            [
                {
                    "player_id": pid,
                    "side": draws.side[j],
                    "mu_point": float(draws.mu_point[j]),
                    "mu_draw": float(draws.xp[:, j].mean()),
                    "sd_draw": float(draws.xp[:, j].std(ddof=1)),
                    "mean_minutes": float(draws.minutes[:, j].mean()),
                }
                for j, pid in enumerate(ids)
            ]
        )
        .sort_values("mu_draw", ascending=False)
        .head(12)
    )

    return {
        "mode": "real",
        "home_team": home_team,
        "away_team": away_team,
        "n_scenarios": n,
        "n_home_players": len(home),
        "n_away_players": len(away),
        "n_home_groups": len(h_groups),
        "n_away_groups": len(a_groups),
        "competition_pairs": pair_rows,
        "corr_home_mids_xp": c_mids,
        "home_mid_names": mid_names,
        "corr_home_attack_vs_away_gk": c_atk_gk,
        "attack_name": atk_name,
        "away_gk_name": gk_name,
        "mean_abs_err_vs_point": float(
            np.mean(np.abs(draws.xp.mean(axis=0) - draws.mu_point))
        ),
        "scoreline_mean_h": float(draws.goals_h.mean()),
        "scoreline_mean_a": float(draws.goals_a.mean()),
        "meta": draws.meta,
        "summary": top.to_dict(orient="records"),
        "home_groups": [
            {"names": list(g.names), "corr": g.min_pair_corr} for g in h_groups
        ],
        "away_groups": [
            {"names": list(g.names), "corr": g.min_pair_corr} for g in a_groups
        ],
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    mode = doc.get("mode", "toy")
    if mode == "real":
        title = (
            f"# Fixture covariance — {doc['home_team']} vs {doc['away_team']} "
            "(2025/26 rates)"
        )
        intro = (
            "Real club sides from merged GW history; greedy competition pairs "
            "(corr≤−0.7); scoreline from M; volatile shares."
        )
    else:
        title = "# Fixture covariance demo (M → minutes → shares → xP)"
        intro = (
            "Toy home favourites with a competing FB pair (sparse minutes) and "
            "volatile goal/assist Dirichlet shares. Scoreline sampled from Dixon–Coles M."
        )
    lines = [
        title,
        "",
        intro,
        "",
        f"- Scenarios: **{doc['n_scenarios']}**",
        f"- Mean |draw μ − point μ|: **{doc['mean_abs_err_vs_point']:.3f}**",
        f"- Scoreline mean: **{doc['scoreline_mean_h']:.2f}–{doc['scoreline_mean_a']:.2f}**",
        "",
    ]
    if mode == "real":
        lines += [
            f"- Home players / groups: **{doc['n_home_players']}** / **{doc['n_home_groups']}**",
            f"- Away players / groups: **{doc['n_away_players']}** / **{doc['n_away_groups']}**",
            "",
            "## Competition pairs (hist → draw)",
            "",
            "| side | A | B | pos | hist corr | draw min corr | draw xP corr |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for r in doc.get("competition_pairs", []):
            dm = r["draw_minutes_corr"]
            dx = r["draw_xp_corr"]
            dm_s = f"{dm:+.3f}" if dm is not None else "n/a"
            dx_s = f"{dx:+.3f}" if dx is not None else "n/a"
            lines.append(
                f"| {r['side']} | {r['a']} | {r['b']} | {r['positions']} | "
                f"{r['hist_corr']:+.3f} | {dm_s} | {dx_s} |"
            )
        mid_s = (
            f"{doc['corr_home_mids_xp']:+.3f}"
            if doc.get("corr_home_mids_xp") is not None
            else "n/a"
        )
        gk_s = (
            f"{doc['corr_home_attack_vs_away_gk']:+.3f}"
            if doc.get("corr_home_attack_vs_away_gk") is not None
            else "n/a"
        )
        lines += [
            "",
            "## Scoreline couplings",
            "",
            f"- Home mids ({', '.join(doc.get('home_mid_names') or [])}) xP corr: **{mid_s}**",
            f"- {doc.get('attack_name')} vs {doc.get('away_gk_name')} xP corr: **{gk_s}**",
            "",
        ]
    else:
        lines += [
            "## Induced dependence",
            "",
            f"- Corr home M1–M2 xP: **{doc['corr_home_mids_xp']:+.3f}**",
            f"- Corr home M1 vs away GK xP: **{doc['corr_home_m1_vs_away_gk']:+.3f}**",
            f"- Corr competing FB minutes: **{doc['corr_home_fb_minutes']:+.3f}**",
            f"- Corr competing FB xP: **{doc['corr_home_fb_xp']:+.3f}**",
            f"- Home FWD xP | blank: **{doc['mean_xp_fwd_when_home_blank']:.2f}**; "
            f"| ≥2 goals: **{doc['mean_xp_fwd_when_home_2plus']:.2f}**",
            "",
        ]
    lines += [
        "## Player summary (top by draw μ)",
        "",
        "| id | side | point μ | draw μ | draw SD | mean m |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["summary"]:
        lines.append(
            f"| {r['player_id']} | {r['side']} | {r['mu_point']:.2f} | "
            f"{r['mu_draw']:.2f} | {r['sd_draw']:.2f} | {r['mean_minutes']:.1f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        doc["verdict"],
        "",
        f"JSON: `{(PROCESSED / 'fixture_covariance_demo.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--home", type=str, default="")
    ap.add_argument("--away", type=str, default="")
    ap.add_argument("--lam-h", type=float, default=1.6)
    ap.add_argument("--lam-a", type=float, default=1.2)
    ap.add_argument("--rebuild-groups", action="store_true")
    args = ap.parse_args(argv)

    if args.rebuild_groups or (args.home and args.away):
        # Refresh greedy groups JSON when running real clubs
        gdoc = build_season_groups()
        (PROCESSED / "sparse_minutes_groups_2025_26.json").write_text(
            json.dumps(gdoc, indent=2), encoding="utf-8"
        )
        from src.markets.sparse_minutes import write_report as write_groups

        write_groups(gdoc, REPORTS / "sparse_minutes_groups_2025_26.md")

    if args.home and args.away:
        doc = demo_real(
            args.home,
            args.away,
            lam_h=args.lam_h,
            lam_a=args.lam_a,
            n=args.n,
            seed=args.seed,
        )
        pair_ok = [
            r
            for r in doc["competition_pairs"]
            if r["draw_minutes_corr"] is not None and r["draw_minutes_corr"] < -0.2
        ]
        gk = doc.get("corr_home_attack_vs_away_gk")
        mid = doc.get("corr_home_mids_xp")
        gk_s = f"{gk:+.2f}" if gk is not None else "n/a"
        mid_s = f"{mid:+.2f}" if mid is not None else "n/a"
        doc["verdict"] = (
            f"{doc['home_team']} vs {doc['away_team']}: "
            f"{len(doc['competition_pairs'])} competition pairs "
            f"({len(pair_ok)} with draw minutes corr < −0.2). "
            f"Home mids xP {mid_s}; attack–opp GK {gk_s}. "
            "Real groups + M path produce fixture-specific minutes and scoreline coupling."
        )
        out_name = "fixture_covariance_real.json"
        md_name = "fixture_covariance_real.md"
    else:
        doc = demo(n=args.n, seed=args.seed)
        ok_fb = doc["corr_home_fb_minutes"] < -0.3
        ok_gk = doc["corr_home_m1_vs_away_gk"] < 0.0
        ok_mids = doc["corr_home_mids_xp"] > 0.05
        doc["verdict"] = (
            f"Sparse FB minutes corr {doc['corr_home_fb_minutes']:+.2f} "
            f"({'ok' if ok_fb else 'weak'}). "
            f"Attack–opp GK {doc['corr_home_m1_vs_away_gk']:+.2f} "
            f"({'ok' if ok_gk else 'weak'}). "
            f"Home mids {doc['corr_home_mids_xp']:+.2f} "
            f"({'ok' if ok_mids else 'weak'}). "
            "Fixture path induces the right qualitative couplings without a dense Σ."
        )
        out_name = "fixture_covariance_demo.json"
        md_name = "fixture_covariance_demo.md"

    out_json = PROCESSED / out_name
    out_md = REPORTS / md_name
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    # Point write_report at the right relative JSON name
    write_report(doc, out_md)
    # Fix JSON path line in report
    text = out_md.read_text(encoding="utf-8")
    text = text.replace(
        "data/processed/fixture_covariance_demo.json",
        f"data/processed/{out_name}",
    )
    out_md.write_text(text, encoding="utf-8")
    print(json.dumps({k: doc[k] for k in doc if k not in ("summary",)}, indent=2, default=str))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
