"""Gated screen: factored M-event likelihood vs Gaussian-on-total.

Pre-registered: ``protocol.m_discrete_event_likelihood_screen``.

Baseline: Gaussian log score of total_points under locked FD centreline
``score_xp_v2`` μ (σ=3).
Treatment: sum of component log scores from the **same** locked board —
Gaussian G/A/rest, Bernoulli CS via P(CS) from M, categorical ⌊GA/2⌋ from
M marginal for GKP/DEF.

Mean layer untouched. No mode/median μ. No fitted β. FD only.

  PYTHONPATH=. python3 -m src.eval.m_discrete_event_likelihood_screen
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.markets.scoreline import build_matrix
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
)
from src.rules.fpl_2026 import (
    ASSIST_POINTS,
    CS_POINTS,
    clean_sheet_points,
    goal_points,
    goals_conceded_points,
    normalize_position,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "m_discrete_event_likelihood_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/m_discrete_event_likelihood_screen.json")
PROCESSED = ROOT / "data" / "processed"

TREATMENT = "l3_factored_m_events"
BASELINE = "l3_gaussian_total"

GW_START = 8
GW_END = 38
MIN_MINUTES = 60.0
MIN_HISTORY = 3
SIGMA = 3.0
EPS = 1e-12


def _now() -> float:
    return time.time()


def _gc_deduction_pmf(matrix: np.ndarray, *, side: str) -> np.ndarray:
    """P(⌊GA/2⌋ = k) for ``side`` under Dixon–Coles ``matrix``."""
    n = int(matrix.shape[0])
    k_max = n - 1
    max_k = k_max // 2
    p = np.zeros(max_k + 1, dtype=float)
    if side == "home":
        marg = matrix.sum(axis=0)  # away goals
    else:
        marg = matrix.sum(axis=1)  # home goals
    for j in range(n):
        gj = j if j < k_max else k_max
        p[int(gj // 2)] += float(marg[j])
    s = float(p.sum())
    if s <= 0.0:
        p[0] = 1.0
        return p
    return p / s


def _log_categorical(k: int, pmf: np.ndarray) -> float:
    kk = int(k)
    if kk < 0:
        return math.log(EPS)
    if kk >= len(pmf):
        # Tail mass beyond grid — fold into last bin.
        return math.log(max(float(pmf[-1]), EPS))
    return math.log(max(float(pmf[kk]), EPS))


def _log_bernoulli(event: bool, p: float) -> float:
    pp = float(min(1.0 - EPS, max(EPS, p)))
    return math.log(pp) if event else math.log(1.0 - pp)


def _fixture_event_pack(fit: dict[str, Any]) -> dict[str, Any]:
    lh = float(fit["lam_h"])
    la = float(fit["lam_a"])
    rho = float(fit.get("rho") or 0.0)
    M = build_matrix(lh, la, rho)
    return {
        "p_cs_home": float(M[:, 0].sum()),
        "p_cs_away": float(M[0, :].sum()),
        "gc_pmf_home": _gc_deduction_pmf(M, side="home"),
        "gc_pmf_away": _gc_deduction_pmf(M, side="away"),
        "xg_home": float(fit["xg_home"]),
        "xcs_home": float(fit["xcs_home"]),
        "xgc_home": float(fit["xgc_home"]),
        "xg_away": float(fit["xg_away"]),
        "xcs_away": float(fit["xcs_away"]),
        "xgc_away": float(fit["xgc_away"]),
        "home": str(fit["home"]),
        "away": str(fit["away"]),
        "home_n": str(fit["home_n"]),
        "away_n": str(fit["away_n"]),
        "date_iso": str(fit.get("date_iso") or ""),
    }


def _score_gw_board(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    """Per player-fixture centreline components + M event packs (DGW-safe rows)."""
    if side.empty or not fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fits:
        pack = _fixture_event_pack(fit)
        for team, team_n, prefix in (
            (pack["home"], pack["home_n"], "home"),
            (pack["away"], pack["away_n"], "away"),
        ):
            club = side.loc[side["team"].astype(str) == team].copy()
            if club.empty and "team_n" in side.columns:
                club = side.loc[side["team_n"] == team_n].copy()
            if club.empty:
                continue
            scored = numagent_xp.score_team_side(
                club,
                xg=float(pack[f"xg_{prefix}"]),
                xcs=float(pack[f"xcs_{prefix}"]),
                xgc=float(pack[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            ).copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["team_n"] = team_n
            scored["side"] = prefix
            scored["p_cs_m"] = float(pack[f"p_cs_{prefix}"])
            scored["gc_pmf"] = [pack[f"gc_pmf_{prefix}"]] * len(scored)
            scored["fixture_key"] = (
                f"{pack['date_iso']}:{pack['home_n']}:{pack['away_n']}"
            )
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _actual_fixture_components(week: pd.DataFrame) -> pd.DataFrame:
    """Per element-fixture actual P(G)/P(A)/P(CS)/P(GC) + total_points."""
    rows: list[dict[str, Any]] = []
    for r in week.itertuples(index=False):
        eid = int(r.element) if pd.notna(r.element) else None
        if eid is None:
            continue
        mins = int(float(r.minutes or 0))
        pos = normalize_position(str(r.position))
        goals = int(float(getattr(r, "goals_scored", 0) or 0))
        assists = int(float(getattr(r, "assists", 0) or 0))
        if bool(r.was_home):
            gc = int(float(r.team_a_score or 0))
            side = "home"
            team_n = norm_team(str(r.team))
        else:
            gc = int(float(r.team_h_score or 0))
            side = "away"
            team_n = norm_team(str(r.team))
        cs = mins >= 60 and gc == 0
        date_iso = ""
        if hasattr(r, "kickoff_time") and pd.notna(r.kickoff_time):
            date_iso = str(r.kickoff_time)[:10]
        rows.append(
            {
                "element": eid,
                "position": pos,
                "minutes": mins,
                "team_n": team_n,
                "side": side,
                "goals": goals,
                "assists": assists,
                "goals_conceded": gc,
                "P_G": float(goal_points(pos, goals)),
                "P_A": float(assists * ASSIST_POINTS),
                "P_CS": float(clean_sheet_points(pos, mins, cs)),
                "P_GC": float(goals_conceded_points(pos, mins, gc)),
                "total_points": float(getattr(r, "total_points", 0) or 0),
                "date_iso": date_iso,
                "fixture": int(r.fixture) if hasattr(r, "fixture") and pd.notna(r.fixture) else -1,
            }
        )
    return pd.DataFrame(rows)


def run_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    parts: list[pd.DataFrame] = []

    for gw in range(GW_START, GW_END + 1):
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw), season=season)
        week = merged.loc[merged["GW"] == int(gw)].copy()
        if week.empty or hist.empty:
            continue
        alpha_g, alpha_a = _asof_alphas(hist)
        side, alpha_g, alpha_a = build_frozen_side_table(
            hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            season=season,
            historical_rate_baseline=True,
        )
        if side.empty:
            continue
        fx = fixtures_for_gw(enriched, int(gw))
        fits = centreline_fixture_fits(fx, fd)
        if len(fits) < 5:
            continue
        board = _score_gw_board(side, fits, alpha_g=alpha_g, alpha_a=alpha_a)
        if board.empty:
            continue
        act = _actual_fixture_components(week)
        if act.empty:
            continue

        # Join board ↔ actuals on element + team + side + date (DGW-safe).
        board["element"] = pd.to_numeric(board["element"], errors="coerce")
        board["team_n"] = board["team_n"].map(norm_team)
        board["date_iso"] = board["fixture_key"].astype(str).str.split(":").str[0]
        act["team_n"] = act["team_n"].map(norm_team)
        j = board.merge(
            act,
            on=["element", "team_n", "side", "date_iso"],
            how="inner",
            suffixes=("", "_act"),
        )
        if j.empty:
            # Fallback without date if kickoff missing on either side.
            j = board.merge(
                act,
                on=["element", "team_n", "side"],
                how="inner",
                suffixes=("", "_act"),
            )
            if not j.empty and j.duplicated(["element", "fixture_key"]).any():
                # Ambiguous DGW cartesian — drop rather than double-count.
                j = pd.DataFrame()
        if j.empty:
            continue

        apps = (
            hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
            if len(hist)
            else pd.Series(dtype=float)
        )
        j["n_prior"] = j["element"].map(apps).fillna(0)
        j = j.loc[
            (pd.to_numeric(j["minutes"], errors="coerce") > MIN_MINUTES)
            & (j["n_prior"] >= MIN_HISTORY)
        ].copy()
        if j.empty:
            continue

        # Aggregate DGW to player-GW: sum points / xp; product of discrete logs.
        rows_out: list[dict[str, Any]] = []
        for eid, g in j.groupby("element"):
            xp = float(g["xp"].sum())
            xp_g = float(g["xp_goals"].sum())
            xp_a = float(g["xp_assists"].sum())
            xp_cs = float(g["xp_cs"].sum())
            xp_gc = float(g["xp_gc"].sum())
            p_g = float(g["P_G"].sum())
            p_a = float(g["P_A"].sum())
            p_cs = float(g["P_CS"].sum())
            p_gc = float(g["P_GC"].sum())
            y = float(g["total_points"].sum())
            pos = normalize_position(str(g["position"].iloc[0]))

            # Discrete factors: sum log over fixtures (independent under M).
            log_cs_sum = 0.0
            log_gc_sum = 0.0
            for fr in g.itertuples(index=False):
                cs_event = float(fr.P_CS) > 0.0
                if float(CS_POINTS.get(pos, 0) or 0) > 0:
                    log_cs_sum += _log_bernoulli(cs_event, float(fr.p_cs_m))
                gc_k = int(round(-float(fr.P_GC))) if pos in ("GKP", "DEF") else 0
                pmf = np.asarray(fr.gc_pmf, dtype=float)
                if pos in ("GKP", "DEF"):
                    log_gc_sum += _log_categorical(gc_k, pmf)

            # Rebuild factored score with aggregated Gaussian parts + summed discrete.
            log_g = float(
                gaussian_log_score(np.array([p_g]), np.array([xp_g]), SIGMA)[0]
            )
            log_a = float(
                gaussian_log_score(np.array([p_a]), np.array([xp_a]), SIGMA)[0]
            )
            y_rest = y - p_g - p_a - p_cs - p_gc
            mu_rest = xp - xp_g - xp_a - xp_cs - xp_gc
            log_rest = float(
                gaussian_log_score(np.array([y_rest]), np.array([mu_rest]), SIGMA)[0]
            )
            log_fact = log_g + log_a + log_cs_sum + log_gc_sum + log_rest
            log_base = float(
                gaussian_log_score(np.array([y]), np.array([xp]), SIGMA)[0]
            )
            rows_out.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "element": int(eid),
                    "position": pos,
                    "n_fx": int(len(g)),
                    "minutes": float(g["minutes"].sum()),
                    "total_points": y,
                    "xp": xp,
                    "xp_goals": xp_g,
                    "xp_assists": xp_a,
                    "xp_cs": xp_cs,
                    "xp_gc": xp_gc,
                    "P_G": p_g,
                    "P_A": p_a,
                    "P_CS": p_cs,
                    "P_GC": p_gc,
                    "log_base": log_base,
                    "log_fact": log_fact,
                    "log_g": log_g,
                    "log_a": log_a,
                    "log_cs": log_cs_sum,
                    "log_gc": log_gc_sum,
                    "log_rest": log_rest,
                    "d_log": log_fact - log_base,
                }
            )
        if rows_out:
            parts.append(pd.DataFrame(rows_out))
        if gw % 5 == 0:
            n = len(rows_out)
            print(f"  [{season}] GW{gw} rows={n}", flush=True)

    if not parts:
        raise RuntimeError(f"no eligible rows for {season}")
    rows = pd.concat(parts, ignore_index=True)
    gw = (
        rows.groupby(["season", "gw"], as_index=False)
        .agg(
            d_log=("d_log", "mean"),
            log_base=("log_base", "mean"),
            log_fact=("log_fact", "mean"),
            n=("element", "count"),
        )
    )
    return rows, gw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("m_discrete_event_likelihood_screen") or {}
    print("[m-disc] as-of audit…", flush=True)
    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )

    t0 = _now()
    row_parts: list[pd.DataFrame] = []
    gw_parts: list[pd.DataFrame] = []
    season_summaries: list[dict[str, Any]] = []

    for season in CLOSED_SEASONS:
        print(f"[m-disc] {season} …", flush=True)
        rows, gw = run_season(season)
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log": float(gw["d_log"].mean()),
                "mean_log_base": float(rows["log_base"].mean()),
                "mean_log_fact": float(rows["log_fact"].mean()),
            }
        )
        print(
            f"  n={len(rows)} d_log={season_summaries[-1]['mean_d_log']:+.4f}",
            flush=True,
        )

    rows_all = pd.concat(row_parts, ignore_index=True)
    gw_all = pd.concat(gw_parts, ignore_index=True)
    deltas = {
        season: gw_all.loc[gw_all["season"] == season, "d_log"].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    interval = cluster_interval(deltas, seasons=CLOSED_SEASONS)
    for season, arr in deltas.items():
        if arr.size < 20:
            raise RuntimeError(f"{season} has fewer than 20 gameweeks")

    key = comparison_key(TREATMENT, BASELINE)
    comparisons = {
        **_certified_protocol_comparison(),
        key: {
            **interval,
            "incomplete_seasons": {},
            "n_rows": int(len(rows_all)),
        },
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "m_discrete_event_likelihood_screen",
    }

    lo = float(interval["lo"])
    hi = float(interval["hi"])
    if lo > 0:
        verdict = "promote"
    elif hi < 0:
        verdict = "park_clear_lose"
    else:
        verdict = "inconclusive"

    summary = {
        "task": "m_discrete_event_likelihood_screen",
        "protocol": "m_discrete_event_likelihood_screen",
        "pre_registered": "2026-10-30",
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "gw_range": [GW_START, GW_END],
        "pool": {"min_minutes": MIN_MINUTES, "min_history": MIN_HISTORY},
        "seasons": season_summaries,
        "cluster_interval": interval,
        "promote_if": screen.get("promote_if"),
        "covers_zero": bool(lo <= 0 <= hi),
        "verdict": verdict,
        "default_unchanged": True,
        "mean_layer_unchanged": True,
        "elapsed_s": _now() - t0,
        "component_mean_logs": {
            "log_g": float(rows_all["log_g"].mean()),
            "log_a": float(rows_all["log_a"].mean()),
            "log_cs": float(rows_all["log_cs"].mean()),
            "log_gc": float(rows_all["log_gc"].mean()),
            "log_rest": float(rows_all["log_rest"].mean()),
        },
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows_all.to_csv(PROCESSED / "m_discrete_event_likelihood_rows.csv", index=False)
    gw_all.to_csv(PROCESSED / "m_discrete_event_likelihood_gw.csv", index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (PROCESSED / "m_discrete_event_likelihood_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n"
    )

    lines = [
        "# Discrete M-event predictive likelihood vs Gaussian-on-total",
        "",
        "Pre-registered: `protocol.m_discrete_event_likelihood_screen`.",
        "",
        "Baseline: Gaussian(`total_points` | FD centreline `score_xp_v2` μ, σ=3).",
        "Treatment: factored log score from the **same** locked μ — Gaussian G/A/rest,",
        "Bernoulli CS via P(CS) from M, categorical floor(GA/2) from M.",
        "",
        f"GW{GW_START}–{GW_END} · wall {summary['elapsed_s']:.1f}s · n={len(rows_all)}",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Result (treatment − baseline)",
        "",
        f"- Mean Δ log-score / GW: **{interval['mean']:+.4f}** "
        f"[{interval['lo']:+.4f}, {interval['hi']:+.4f}]",
        f"- Covers 0: **{summary['covers_zero']}**",
        f"- Verdict: **{summary['verdict']}**",
        "- Mean layer / defaults **unchanged**.",
        "",
        "| Season | n | mean Δ log | mean log_base | mean log_fact |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log']:+.4f} | "
            f"{s['mean_log_base']:+.4f} | {s['mean_log_fact']:+.4f} |"
        )
    cl = summary["component_mean_logs"]
    lines += [
        "",
        "## Component mean logs (treatment, pooled)",
        "",
        f"- log_G: {cl['log_g']:+.4f}",
        f"- log_A: {cl['log_a']:+.4f}",
        f"- log_CS: {cl['log_cs']:+.4f}",
        f"- log_GC: {cl['log_gc']:+.4f}",
        f"- log_rest: {cl['log_rest']:+.4f}",
        "",
        "## Notes",
        "",
        "- Positive Δ = factored M-event likelihood beats Gaussian-on-total.",
        "- Mode/median boards were not used.",
        "- Eligible: minutes>60, ≥3 prior apps, FD-priced weeks.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.m_discrete_event_likelihood_screen",
        "```",
        "",
    ]
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "m_discrete_event_likelihood_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated factored M-event likelihood vs Gaussian-on-total",
                "verdict": {
                    "verdict": verdict,
                    "mean_d_log": interval["mean"],
                    "lo": lo,
                    "hi": hi,
                    "covers_zero": summary["covers_zero"],
                    "default_unchanged": True,
                    "mean_layer_unchanged": True,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )

    # Update protocol status in-place for the batch record.
    proto_path = ROOT / "experiments" / "protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    entry = proto.get("m_discrete_event_likelihood_screen") or {}
    entry["status"] = (
        "done_promote"
        if verdict == "promote"
        else ("done_park_clear_lose" if verdict == "park_clear_lose" else "done_inconclusive")
    )
    entry["result"] = (
        f"Δ log {interval['mean']:+.4f} [{lo:+.4f},{hi:+.4f}] → {verdict}; "
        "mean layer unchanged"
    )
    proto["m_discrete_event_likelihood_screen"] = entry
    proto_path.write_text(json.dumps(proto, indent=2) + "\n", encoding="utf-8")

    print(
        f"[m-disc] done verdict={verdict} Δ={interval['mean']:+.4f} "
        f"[{lo:+.4f},{hi:+.4f}] → {REPORT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
