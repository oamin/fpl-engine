"""Walk-forward as-of audit of fixture covariance (minutes pairs + scoreline coupling).

Not a forecast-gate / climb screen. Checks:

1. Competition pairs discovered on GW < t only (as-of).
2. Out-of-sample realised minutes corr for those pairs on later fixtures.
3. On a subsample of fixtures: MC draw minutes corr for as-of pairs, and
   attack–opp-GK / teammate-attack xP corr under the M→shares path.
4. Realised residual co-movement from ``score_xp_v2_closed`` (reference).

  python -m src.markets.fixture_covariance_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS
from src.markets.fixture_covariance import (
    build_club_side,
    draw_fixture_covariant,
)
from src.markets.minutes_covariance import load_merged, merged_path
from src.markets.sparse_minutes import CORR_MAX, find_club_competition_groups

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

MIN_HIST_GW = 8  # start discovering pairs after enough history
MIN_PAIR_OOS_OBS = 5
N_DRAW_SCENARIOS = 120
N_FIXTURES_PER_SEASON = 24  # MC subsample
CORR_MAX_DEFAULT = CORR_MAX


def asof_groups_for_team(
    season_df: pd.DataFrame,
    team: str,
    gw_t: int,
    *,
    corr_max: float = CORR_MAX_DEFAULT,
) -> list:
    """Competition groups using only fixtures with GW < gw_t."""
    club = season_df.loc[
        (season_df["team"] == team) & (season_df["GW"] < int(gw_t))
    ].copy()
    if club.empty:
        return []
    groups, _ = find_club_competition_groups(club, team, corr_max=corr_max)
    return groups


def assert_asof_pairs_ignore_future(season_df: pd.DataFrame, team: str, gw_t: int) -> bool:
    """Poison GW >= t minutes; as-of groups at t must be unchanged."""
    base = asof_groups_for_team(season_df, team, gw_t)
    poisoned = season_df.copy()
    mask = (poisoned["team"] == team) & (poisoned["GW"] >= int(gw_t))
    poisoned.loc[mask, "minutes"] = 999.0
    other = asof_groups_for_team(poisoned, team, gw_t)
    base_keys = {(g.elements, round(g.min_pair_corr, 6)) for g in base}
    other_keys = {(g.elements, round(g.min_pair_corr, 6)) for g in other}
    return base_keys == other_keys


def walk_minutes_oos(
    season_df: pd.DataFrame,
    *,
    min_hist_gw: int = MIN_HIST_GW,
    corr_max: float = CORR_MAX_DEFAULT,
    min_oos_obs: int = MIN_PAIR_OOS_OBS,
) -> dict[str, Any]:
    """Discover pairs as-of each GW; accumulate OOS (m_a, m_b) on that GW's fixtures."""
    # obs[(team, ea, eb)] -> lists
    buckets: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
    n_pair_discoveries = 0
    gws = sorted(int(g) for g in season_df["GW"].unique() if int(g) >= min_hist_gw)

    for gw in gws:
        for team, club_all in season_df.groupby("team"):
            groups = asof_groups_for_team(season_df, str(team), gw, corr_max=corr_max)
            if not groups:
                continue
            n_pair_discoveries += len(groups)
            # Fixtures this club plays in gw t
            now = club_all.loc[club_all["GW"] == gw]
            if now.empty:
                continue
            for fixture, fdf in now.groupby("fixture"):
                mins = {
                    int(r.element): float(r.minutes)
                    for r in fdf.itertuples(index=False)
                }
                for g in groups:
                    a, b = int(g.elements[0]), int(g.elements[1])
                    if a not in mins or b not in mins:
                        continue
                    key = (str(team), min(a, b), max(a, b))
                    buckets.setdefault(key, []).append((mins[a], mins[b]))

    pair_rows = []
    for (team, ea, eb), pts in buckets.items():
        if len(pts) < min_oos_obs:
            continue
        xa = np.array([p[0] for p in pts], dtype=float)
        xb = np.array([p[1] for p in pts], dtype=float)
        if float(xa.std()) < 1e-6 or float(xb.std()) < 1e-6:
            continue
        c = float(np.corrcoef(xa, xb)[0, 1])
        if not np.isfinite(c):
            continue
        pair_rows.append(
            {
                "team": team,
                "element_a": ea,
                "element_b": eb,
                "n_oos": len(pts),
                "oos_minutes_corr": c,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return {
            "n_pair_discoveries": n_pair_discoveries,
            "n_pairs_evaluated": 0,
            "frac_oos_neg": None,
            "median_oos_corr": None,
            "mean_oos_corr": None,
            "p10_oos_corr": None,
            "pairs": [],
        }
    c = pairs["oos_minutes_corr"].to_numpy(float)
    return {
        "n_pair_discoveries": n_pair_discoveries,
        "n_pairs_evaluated": int(len(pairs)),
        "frac_oos_neg": float(np.mean(c < 0)),
        "median_oos_corr": float(np.median(c)),
        "mean_oos_corr": float(np.mean(c)),
        "p10_oos_corr": float(np.quantile(c, 0.10)),
        "pairs": pairs.to_dict(orient="records"),
    }


def realised_residual_coupling(v2: pd.DataFrame, season: str) -> dict[str, Any]:
    """Same-team MID/FWD resid corr and attack vs opp GK (minutes≥60)."""
    from itertools import combinations

    df = v2.loc[v2["season"] == season].copy()
    if df.empty:
        return {"n_atk": 0, "corr_atk_atk": None, "corr_atk_gk": None}
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["score_xp_v2"] = pd.to_numeric(df["score_xp_v2"], errors="coerce")
    df = df.dropna(subset=["total_points", "score_xp_v2"])
    df = df.drop_duplicates(["fixture_id", "element"], keep="first")
    df["resid"] = df["total_points"] - df["score_xp_v2"]
    ax, ay, gx, gy = [], [], [], []
    for _, g in df.groupby("fixture_id"):
        atk = g.loc[g["position"].isin(["MID", "FWD"]) & (g["minutes"] >= 60)]
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
        for at, gt in ((teams[0], teams[1]), (teams[1], teams[0])):
            a = g.loc[
                (g["team"].astype(str) == at)
                & g["position"].isin(["MID", "FWD"])
                & (g["minutes"] >= 60)
            ]
            gk = g.loc[
                (g["team"].astype(str) == gt)
                & (g["position"] == "GKP")
                & (g["minutes"] >= 60)
            ]
            if a.empty or gk.empty:
                continue
            rg = float(gk["resid"].iloc[0])
            for v in a["resid"].to_numpy(float):
                gx.append(float(v))
                gy.append(rg)
    out: dict[str, Any] = {
        "n_atk_pairs": len(ax),
        "n_atk_gk_pairs": len(gx),
        "corr_atk_atk": float(np.corrcoef(ax, ay)[0, 1]) if len(ax) > 20 else None,
        "corr_atk_gk": float(np.corrcoef(gx, gy)[0, 1]) if len(gx) > 20 else None,
    }
    return out


def _fixture_schedule(season_df: pd.DataFrame) -> pd.DataFrame:
    """Unique fixtures with home/away team labels from was_home if present."""
    cols = ["fixture", "GW", "team"]
    if "was_home" in season_df.columns:
        cols.append("was_home")
    sub = season_df[cols].drop_duplicates()
    if "was_home" in sub.columns:
        home = sub.loc[sub["was_home"] == True][["fixture", "GW", "team"]].rename(  # noqa: E712
            columns={"team": "home_team"}
        )
        away = sub.loc[sub["was_home"] == False][["fixture", "team"]].rename(  # noqa: E712
            columns={"team": "away_team"}
        )
        sched = home.merge(away, on="fixture", how="inner")
    else:
        # Fallback: two teams per fixture, order lexicographic as home/away
        rows = []
        for (fx, gw), g in sub.groupby(["fixture", "GW"]):
            teams = sorted(g["team"].astype(str).unique())
            if len(teams) != 2:
                continue
            rows.append(
                {"fixture": fx, "GW": gw, "home_team": teams[0], "away_team": teams[1]}
            )
        sched = pd.DataFrame(rows)
    return sched.sort_values(["GW", "fixture"]).reset_index(drop=True)


def sample_mc_coupling(
    season: str,
    season_df: pd.DataFrame,
    *,
    n_fixtures: int = N_FIXTURES_PER_SEASON,
    n_scenarios: int = N_DRAW_SCENARIOS,
    seed: int = 0,
    min_hist_gw: int = MIN_HIST_GW,
) -> dict[str, Any]:
    """MC subsample: as-of pairs + draw minutes/xP couplings."""
    rng = np.random.default_rng(seed)
    sched = _fixture_schedule(season_df)
    sched = sched.loc[sched["GW"] >= min_hist_gw]
    if sched.empty:
        return {"n_fixtures": 0}
    # Stratify pick across GW bands
    take = min(n_fixtures, len(sched))
    idx = np.linspace(0, len(sched) - 1, take).astype(int)
    picked = sched.iloc[idx]

    pair_min_corrs: list[float] = []
    atk_gk_corrs: list[float] = []
    mid_mid_corrs: list[float] = []
    n_pairs_tested = 0
    n_ok_pairs = 0

    for row in picked.itertuples(index=False):
        gw = int(row.GW)
        home_team, away_team = str(row.home_team), str(row.away_team)
        # As-of history for rates/groups
        hist = season_df.loc[season_df["GW"] < gw]
        if hist.empty:
            continue
        try:
            home, h_groups = build_club_side(hist, home_team)
            away, a_groups = build_club_side(hist, away_team)
        except (ValueError, KeyError):
            continue
        # Prefer as-of groups over full-history groups from build_club_side
        h_groups = asof_groups_for_team(season_df, home_team, gw)
        a_groups = asof_groups_for_team(season_df, away_team, gw)
        # Restrict to players on side
        h_keep = set(home["element"].astype(int))
        a_keep = set(away["element"].astype(int))
        h_groups = [g for g in h_groups if set(g.elements) <= h_keep]
        a_groups = [g for g in a_groups if set(g.elements) <= a_keep]
        if home.empty or away.empty:
            continue

        draws = draw_fixture_covariant(
            1.5,
            1.2,
            -0.05,
            home,
            away,
            home_groups=h_groups,
            away_groups=a_groups,
            n_scenarios=n_scenarios,
            seed=int(rng.integers(0, 1_000_000)),
        )
        ids = list(draws.player_ids)
        name_to_i = {pid: i for i, pid in enumerate(ids)}

        for g in h_groups + a_groups:
            n_pairs_tested += 1
            na, nb = g.names[0], g.names[1]
            if na not in name_to_i or nb not in name_to_i:
                continue
            ia, ib = name_to_i[na], name_to_i[nb]
            c = float(np.corrcoef(draws.minutes[:, ia], draws.minutes[:, ib])[0, 1])
            if np.isfinite(c):
                pair_min_corrs.append(c)
                if c < -0.2:
                    n_ok_pairs += 1

        # Attack vs opp GK
        home_atk = home.loc[home["position"].isin(["MID", "FWD"])].sort_values(
            "m", ascending=False
        )
        away_gk = away.loc[away["position"] == "GKP"]
        if len(home_atk) and len(away_gk):
            an = str(home_atk.iloc[0]["name"])
            gn = str(away_gk.iloc[0]["name"])
            if an in name_to_i and gn in name_to_i:
                c = float(
                    np.corrcoef(
                        draws.xp[:, name_to_i[an]], draws.xp[:, name_to_i[gn]]
                    )[0, 1]
                )
                if np.isfinite(c):
                    atk_gk_corrs.append(c)

        home_mids = home.loc[home["position"] == "MID"].sort_values("m", ascending=False)
        if len(home_mids) >= 2:
            m1, m2 = str(home_mids.iloc[0]["name"]), str(home_mids.iloc[1]["name"])
            if m1 in name_to_i and m2 in name_to_i:
                c = float(
                    np.corrcoef(
                        draws.xp[:, name_to_i[m1]], draws.xp[:, name_to_i[m2]]
                    )[0, 1]
                )
                if np.isfinite(c):
                    mid_mid_corrs.append(c)

    def _summ(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"n": 0, "median": None, "mean": None, "frac_neg": None}
        a = np.asarray(xs, dtype=float)
        return {
            "n": int(len(a)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "frac_neg": float(np.mean(a < 0)),
        }

    return {
        "n_fixtures": int(len(picked)),
        "n_scenarios": n_scenarios,
        "n_pairs_tested": n_pairs_tested,
        "n_pairs_draw_min_lt_-0.2": n_ok_pairs,
        "frac_pairs_ok": float(n_ok_pairs / n_pairs_tested) if n_pairs_tested else None,
        "draw_pair_minutes": _summ(pair_min_corrs),
        "draw_atk_opp_gk": _summ(atk_gk_corrs),
        "draw_home_mids": _summ(mid_mid_corrs),
    }


def audit_season(
    season: str,
    *,
    v2: pd.DataFrame | None,
    n_fixtures: int,
    n_scenarios: int,
    seed: int,
) -> dict[str, Any]:
    path = merged_path(season)
    if not path.exists():
        return {"season": season, "skipped": True, "reason": "missing_merged"}
    season_df = load_merged(path)
    asof_ok = assert_asof_pairs_ignore_future(
        season_df, str(season_df["team"].iloc[0]), MIN_HIST_GW + 2
    )
    # Stronger as-of: test a few clubs
    teams = sorted(season_df["team"].astype(str).unique())[:5]
    asof_checks = [
        assert_asof_pairs_ignore_future(season_df, t, 15) for t in teams
    ]
    minutes_oos = walk_minutes_oos(season_df)
    mc = sample_mc_coupling(
        season,
        season_df,
        n_fixtures=n_fixtures,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    resid = (
        realised_residual_coupling(v2, season)
        if v2 is not None
        else {"corr_atk_atk": None, "corr_atk_gk": None}
    )
    return {
        "season": season,
        "skipped": False,
        "asof_passed": bool(asof_ok and all(asof_checks)),
        "asof_n_club_checks": len(asof_checks),
        "minutes_oos": {
            k: minutes_oos[k]
            for k in minutes_oos
            if k != "pairs"
        },
        "mc_draws": mc,
        "realised_resid": resid,
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fixture covariance — walk-forward as-of audit",
        "",
        "Diagnostic (not a climb / likelihood gate). Competition pairs from **GW < t** only.",
        "",
        f"- Min history GW before discovery: **{doc['defs']['min_hist_gw']}**",
        f"- Pair corr threshold: **≤ {doc['defs']['corr_max']}**",
        f"- MC subsample: **{doc['defs']['n_fixtures_per_season']}** fixtures × "
        f"**{doc['defs']['n_scenarios']}** scenarios / season",
        "",
        f"- As-of poison checks: **{'PASS' if doc['asof_all_passed'] else 'FAIL'}**",
        "",
        "## By season",
        "",
        "| season | as-of | OOS pairs | frac OOS min corr<0 | median OOS | "
        "MC pairs ok | med draw min | med atk–GK | med mids | resid atk–atk | resid atk–GK |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in doc["seasons"]:
        if s.get("skipped"):
            lines.append(f"| {s['season']} | — | — | — | — | — | — | — | — | — | — |")
            continue
        mo = s["minutes_oos"]
        mc = s["mc_draws"]
        rr = s["realised_resid"]
        dpm = mc.get("draw_pair_minutes") or {}
        dag = mc.get("draw_atk_opp_gk") or {}
        dmm = mc.get("draw_home_mids") or {}

        def fmt(x: Any, nd: int = 2) -> str:
            if x is None:
                return "n/a"
            return f"{float(x):+.{nd}f}" if nd else f"{float(x):.0%}"

        lines.append(
            f"| {s['season']} | {'PASS' if s['asof_passed'] else 'FAIL'} | "
            f"{mo.get('n_pairs_evaluated', 0)} | "
            f"{fmt(mo.get('frac_oos_neg'), 0)} | {fmt(mo.get('median_oos_corr'))} | "
            f"{fmt(mc.get('frac_pairs_ok'), 0)} | {fmt(dpm.get('median'))} | "
            f"{fmt(dag.get('median'))} | {fmt(dmm.get('median'))} | "
            f"{fmt(rr.get('corr_atk_atk'))} | {fmt(rr.get('corr_atk_gk'))} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- **OOS minutes:** as-of pairs should stay negatively correlated on later fixtures.",
        "- **MC pairs ok:** fraction of as-of competition pairs with draw minutes corr < −0.2.",
        "- **Draw atk–GK / mids:** sign checks for scoreline coupling (compare to resid columns).",
        "",
        "## Verdict",
        "",
        doc["verdict"],
        "",
        f"JSON: `{(PROCESSED / 'fixture_covariance_audit.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fixtures", type=int, default=N_FIXTURES_PER_SEASON)
    ap.add_argument("--n-scenarios", type=int, default=N_DRAW_SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seasons",
        default=",".join(CLOSED_SEASONS),
        help="Comma-separated closed seasons",
    )
    args = ap.parse_args(argv)
    seasons = tuple(s.strip() for s in args.seasons.split(",") if s.strip())

    v2 = pd.read_csv(CLOSED_V2) if CLOSED_V2.exists() else None
    rows = []
    for i, season in enumerate(seasons):
        print(f"auditing {season}...", flush=True)
        rows.append(
            audit_season(
                season,
                v2=v2,
                n_fixtures=args.n_fixtures,
                n_scenarios=args.n_scenarios,
                seed=args.seed + 17 * i,
            )
        )

    asof_all = all(r.get("asof_passed") for r in rows if not r.get("skipped"))
    # Aggregate OOS
    oos_medians = [
        r["minutes_oos"]["median_oos_corr"]
        for r in rows
        if not r.get("skipped") and r["minutes_oos"].get("median_oos_corr") is not None
    ]
    oos_fracs = [
        r["minutes_oos"]["frac_oos_neg"]
        for r in rows
        if not r.get("skipped") and r["minutes_oos"].get("frac_oos_neg") is not None
    ]
    mc_ok = [
        r["mc_draws"]["frac_pairs_ok"]
        for r in rows
        if not r.get("skipped") and r["mc_draws"].get("frac_pairs_ok") is not None
    ]
    gk_med = [
        r["mc_draws"]["draw_atk_opp_gk"]["median"]
        for r in rows
        if not r.get("skipped")
        and (r["mc_draws"].get("draw_atk_opp_gk") or {}).get("median") is not None
    ]

    verdict_bits = [
        f"As-of checks {'PASS' if asof_all else 'FAIL'} on {len(rows)} seasons.",
    ]
    if oos_fracs:
        verdict_bits.append(
            f"OOS minutes: median frac neg {float(np.mean(oos_fracs)):.0%}, "
            f"median corr {float(np.median(oos_medians)):+.2f}."
        )
    if mc_ok:
        verdict_bits.append(
            f"MC competition pairs with draw min corr < −0.2: "
            f"{float(np.mean(mc_ok)):.0%} (mean across seasons)."
        )
    if gk_med:
        verdict_bits.append(
            f"MC attack–opp GK median corr {float(np.median(gk_med)):+.2f} "
            f"(want negative)."
        )
    # Pass heuristic for "mechanism ok"
    mech_ok = (
        asof_all
        and oos_fracs
        and float(np.mean(oos_fracs)) >= 0.7
        and mc_ok
        and float(np.mean(mc_ok)) >= 0.7
        and gk_med
        and float(np.median(gk_med)) < 0.0
    )
    verdict_bits.append(
        "Mechanism audit: **PASS** — proceed toward planning glue."
        if mech_ok
        else "Mechanism audit: **HOLD** — inspect weak season rows before wiring planning."
    )

    doc = {
        "defs": {
            "min_hist_gw": MIN_HIST_GW,
            "corr_max": CORR_MAX_DEFAULT,
            "n_fixtures_per_season": args.n_fixtures,
            "n_scenarios": args.n_scenarios,
            "note": "Diagnostic only; not write_gated_report / not a μ screen",
        },
        "asof_all_passed": asof_all,
        "mechanism_pass": mech_ok,
        "seasons": rows,
        "verdict": " ".join(verdict_bits),
    }
    out_json = PROCESSED / "fixture_covariance_audit.json"
    out_md = REPORTS / "fixture_covariance_audit.md"
    # Drop bulky pair lists already stripped
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(json.dumps({"verdict": doc["verdict"], "mechanism_pass": mech_ok}, indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
