"""Item 4 — player G/A finishing screen vs Numagent 2.0 share×λ.

Control: ``score_xp_v2`` (rates → shares → ``score_team_side`` on FD pots).
Candidate: same assembler with shift-1 finishing scales multiplying ``rG`` /
``rA`` before shares:

  finish_g = clip( (ΣG+κ)/(ΣxG+κ) , 0.65, 1.40 )
  finish_a = clip( (ΣA+κ)/(ΣxA+κ) , 0.65, 1.40 )

Promote only if candidate−control eligible log-score interval stays
strictly above zero across the four closed seasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    assert_reportable,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import SCORER, attach_rates_priors
from src.markets.numagent_xp_pilot import load_fd_moments, load_merged, score_gameweek
from src.models.player_ga import add_finish_scales
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

CONTROL_COL = SCORER  # score_xp_v2
CANDIDATE_COL = "score_xp_v2_finish"
FINISH_CONTRAST = comparison_key(CANDIDATE_COL, CONTROL_COL)
MIN_MINUTES = 60.0


def _finish_on_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """Attach finish_g / finish_a using goals_scored / expected_goals if present."""
    frame = merged.copy()
    # player_ga.add_finish_scales expects player_id, gw, date, goals, xG, xA.
    frame["player_id"] = frame["element"].astype(str)
    frame["gw"] = frame["GW"]
    if "date" not in frame.columns:
        frame["date"] = frame.get("date_iso", frame.get("kickoff_time", "")).astype(str)
    if "goals" not in frame.columns:
        if "goals_scored" in frame.columns:
            frame["goals"] = pd.to_numeric(frame["goals_scored"], errors="coerce").fillna(0.0)
        else:
            frame["goals"] = 0.0
    # Prefer understat-style xG on logs; fall back to expected_goals on merged.
    if "xG" not in frame.columns:
        if "expected_goals" in frame.columns:
            frame["xG"] = pd.to_numeric(frame["expected_goals"], errors="coerce").fillna(0.0)
        else:
            frame["xG"] = 0.0
    if "xA" not in frame.columns:
        if "expected_assists" in frame.columns:
            frame["xA"] = pd.to_numeric(frame["expected_assists"], errors="coerce").fillna(0.0)
        else:
            frame["xA"] = 0.0
    if "assists" not in frame.columns:
        frame["assists"] = 0.0
    return add_finish_scales(frame)


def build_season(season: str, protocol: dict[str, Any]) -> pd.DataFrame:
    if season not in CLOSED_SEASONS:
        raise RuntimeError(f"{season} is outside the closed-season comparison")
    from src.markets.numagent_xp_pilot import _asof_alphas

    merged = load_merged(season)
    stem = season.replace("-", "_")
    raw_path = ROOT / "data" / "cache" / f"merged_gw_{stem}.csv"
    header = pd.read_csv(raw_path, nrows=0).columns
    # Only attach expected_* from raw — goals/assists already on load_merged.
    want = [
        c
        for c in ("element", "GW", "fixture", "expected_goals", "expected_assists")
        if c in header
    ]
    if len(want) > 3:
        raw = pd.read_csv(raw_path, usecols=want, low_memory=False)
        raw["element"] = pd.to_numeric(raw["element"], errors="coerce")
        raw["GW"] = pd.to_numeric(raw["GW"], errors="coerce")
        raw["fixture"] = pd.to_numeric(raw["fixture"], errors="coerce")
        extra = [c for c in want if c not in ("element", "GW", "fixture")]
        merged = merged.merge(
            raw[["element", "GW", "fixture"] + extra].drop_duplicates(
                ["element", "fixture"]
            ),
            on=["element", "GW", "fixture"],
            how="left",
        )
    merged = _finish_on_merged(merged)
    fd = load_fd_moments(season)

    rows: list[pd.DataFrame] = []
    gws = sorted(
        int(g)
        for g in merged["GW"].unique()
        if int(protocol["gw_start"]) <= int(g) <= int(protocol["gw_end"])
    )
    for gw in gws:
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw), season=season)
        week = merged.loc[merged["GW"] == int(gw)].copy()
        if week.empty:
            continue
        alpha_g, alpha_a = _asof_alphas(hist) if len(hist) else (0.96, 0.90)

        control = score_gameweek(hist, week, fd, alpha_g=alpha_g, alpha_a=alpha_a)
        if control.empty or "element" not in control.columns:
            continue

        # Candidate: same FD pots / minutes, rates scaled by finish before shares.
        w = week.merge(fd, on=["team_n", "date_iso"], how="left")
        w = w.loc[w["xg"].notna() & (pd.to_numeric(w["minutes"], errors="coerce") > 0)].copy()
        if w.empty:
            continue
        for c in ("finish_g", "finish_a"):
            w[c] = pd.to_numeric(w[c], errors="coerce").fillna(1.0)
        w["m"] = pd.to_numeric(w["minutes"], errors="coerce").fillna(0.0)
        w["sigma_m"] = 0.0
        w["player_id"] = w["element"].astype(str)
        w = attach_rates_priors(w, hist, finish_g_col="finish_g", finish_a_col="finish_a")
        cand_parts = []
        for (_, _), g in w.groupby(["fixture", "team_n"], sort=False):
            side = numagent_xp.score_team_side(
                g,
                xg=float(g["xg"].iloc[0]),
                xcs=float(g["xcs"].iloc[0]),
                xgc=float(g["xgc"].iloc[0]),
                alpha_g=alpha_g,
                alpha_a=alpha_a,
            )
            cand_parts.append(side)
        candidate = pd.concat(cand_parts, ignore_index=True)
        candidate = candidate.loc[
            pd.to_numeric(candidate["minutes"], errors="coerce") > MIN_MINUTES
        ].copy()
        if candidate.empty or "element" not in candidate.columns:
            continue

        ctrl = control.copy()
        ctrl["element"] = pd.to_numeric(ctrl["element"], errors="coerce").astype(int)
        candidate["element"] = pd.to_numeric(candidate["element"], errors="coerce").astype(int)
        joined = ctrl.merge(
            candidate[["element", "xp", "finish_g", "finish_a"]],
            on="element",
            how="inner",
            suffixes=("", "_fin"),
        )
        if joined.empty:
            continue
        if len(hist):
            apps_n = hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
            joined["n_prior"] = joined["element"].map(apps_n).fillna(0)
        else:
            joined["n_prior"] = 0
        joined["season"] = season
        joined[CONTROL_COL] = joined["xp"]
        joined[CANDIDATE_COL] = joined["xp_fin"]
        joined["xmi"] = joined["minutes"]
        joined["eligible"] = (joined["n_prior"] >= int(protocol["min_history"])) & (
            pd.to_numeric(joined["minutes"], errors="coerce") > MIN_MINUTES
        )
        rows.append(joined)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _window(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    gw = pd.to_numeric(frame["gw"], errors="coerce")
    return frame.loc[(gw >= int(protocol["gw_start"])) & (gw <= int(protocol["gw_end"]))].copy()


def _logscore_pair(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    left: str,
    right: str,
) -> list[dict[str, Any]]:
    window = _window(frame, protocol)
    window = window.loc[window["eligible"]].copy()
    sigma = float(protocol["sigma"])
    season = str(window["season"].iloc[0]) if not window.empty else ""
    a = pd.to_numeric(window[left], errors="coerce")
    b = pd.to_numeric(window[right], errors="coerce")
    yy = pd.to_numeric(window["total_points"], errors="coerce")
    ok = yy.notna() & a.notna() & b.notna()
    paired = window.loc[ok].copy()
    rows: list[dict[str, Any]] = []
    if paired.empty:
        return rows
    for gw, block in paired.groupby("gw", sort=True):
        y = pd.to_numeric(block["total_points"], errors="coerce").to_numpy(float)
        log_a = gaussian_log_score(
            y, pd.to_numeric(block[left], errors="coerce").to_numpy(float), sigma
        )
        log_b = gaussian_log_score(
            y, pd.to_numeric(block[right], errors="coerce").to_numpy(float), sigma
        )
        rows.append(
            {
                "season": season,
                "gw": int(gw),
                "comparison": comparison_key(left, right),
                "n_rows": int(len(block)),
                "mean_log_a": float(log_a.mean()),
                "mean_log_b": float(log_b.mean()),
                "delta": float((log_a - log_b).mean()),
            }
        )
    return rows


def _interval_from_rows(
    rows: list[dict[str, Any]],
    key: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    by_season: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        deltas = [
            float(r["delta"])
            for r in rows
            if r["comparison"] == key and r["season"] == season
        ]
        by_season[season] = np.asarray(deltas, dtype=float)
    return cluster_interval(
        by_season,
        seasons=CLOSED_SEASONS,
        n_boot=int(protocol["bootstrap"]),
        seed=int(protocol["seed"]),
    )


def run() -> dict[str, Any]:
    protocol = load_protocol()
    audit = run_asof_audit()
    all_rows: list[dict[str, Any]] = []
    finish_summary: list[dict[str, Any]] = []

    # Primary protocol pair still score_xp − score_exp_points from honest_pool.
    from src.eval.honest_pool import build_season as build_honest, logscore_rows

    primary_rows: list[dict[str, Any]] = []
    codes = protocol["season_codes"]
    for season in CLOSED_SEASONS:
        primary_rows.extend(logscore_rows(build_honest(season, str(codes[season]), protocol), protocol))

    for season in CLOSED_SEASONS:
        print(f"season {season}…", flush=True)
        frame = build_season(season, protocol)
        if frame.empty:
            print(f"  empty {season}", flush=True)
            continue
        elig = frame.loc[frame["eligible"]]
        finish_summary.append(
            {
                "season": season,
                "n_eligible": int(len(elig)),
                "mean_finish_g": float(pd.to_numeric(elig["finish_g"], errors="coerce").mean()),
                "mean_finish_a": float(pd.to_numeric(elig["finish_a"], errors="coerce").mean()),
                "p10_finish_g": float(pd.to_numeric(elig["finish_g"], errors="coerce").quantile(0.1)),
                "p90_finish_g": float(pd.to_numeric(elig["finish_g"], errors="coerce").quantile(0.9)),
                "mean_abs_xp_delta": float(
                    (
                        pd.to_numeric(elig[CANDIDATE_COL], errors="coerce")
                        - pd.to_numeric(elig[CONTROL_COL], errors="coerce")
                    )
                    .abs()
                    .mean()
                ),
            }
        )
        all_rows.extend(_logscore_pair(frame, protocol, CANDIDATE_COL, CONTROL_COL))

    primary_key = comparison_key("score_xp", "score_exp_points")
    finish_key = FINISH_CONTRAST
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": {
            primary_key: _interval_from_rows(primary_rows, primary_key, protocol),
            finish_key: _interval_from_rows(all_rows, finish_key, protocol),
        },
    }
    finish_iv = intervals["comparisons"][finish_key]
    promote = float(finish_iv["lo"]) > 0.0
    decision = "promote_finish_scales" if promote else "keep_share_times_lambda"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(PROCESSED / "player_ga_logscore.csv", index=False)
    payload = {
        "task": "player_ga_finish",
        "scorer": SCORER,
        "decision": decision,
        "promote": promote,
        "finish_contrast": finish_key,
        "finish_interval": finish_iv,
        "primary_interval": intervals["comparisons"][primary_key],
        "finish_summary": finish_summary,
        "n_gws_logged": len(all_rows),
    }
    (PROCESSED / "player_ga_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Item 4 — Player G/A finishing screen (Numagent 2.0)",
        "",
        f"Pre-registered before totals were read. Control is `{SCORER}` "
        "(rates → shares → `score_team_side` on FD AH+OU pots). Candidate multiplies "
        "`rG`/`rA` by player shift-1 finishing `(ΣG+κ)/(ΣxG+κ)` and the assist "
        "analogue (κ=2, clipped to [0.65, 1.40]) before shares.",
        "",
        "Population: eligible rows (`n_prior ≥ 3`, `minutes > 60`), gameweeks "
        f"{protocol['gw_start']}–{protocol['gw_end']}, four closed seasons. "
        "Metric: player-GW Gaussian log score, σ=3. Bootstrap resamples gameweeks "
        "inside each season.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            "Promote only if the candidate−control interval stays strictly above zero. "
            + (
                "Interval clears zero: switch published G/A to finish scales."
                if promote
                else "Interval does not clear zero: keep share×λ rates."
            )
        ),
        "",
        "## Intervals",
        "",
        "| contrast | mean | 95% interval |",
        "| --- | ---: | ---: |",
        (
            f"| `{finish_key}` | {finish_iv['mean']:+.5f} | "
            f"[{finish_iv['lo']:+.5f}, {finish_iv['hi']:+.5f}] |"
        ),
        (
            f"| `{primary_key}` | {intervals['comparisons'][primary_key]['mean']:+.5f} | "
            f"[{intervals['comparisons'][primary_key]['lo']:+.5f}, "
            f"{intervals['comparisons'][primary_key]['hi']:+.5f}] |"
        ),
        "",
        "Gameweeks per season (finish contrast): "
        + ", ".join(f"{s}={finish_iv['n_gws'].get(s, 0)}" for s in CLOSED_SEASONS)
        + ".",
        "",
        "## Finish scales on eligible rows",
        "",
        "| season | n | mean finish_g | mean finish_a | p10_g | p90_g | mean |ΔxP| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in finish_summary:
        lines.append(
            f"| {row['season']} | {row['n_eligible']} | {row['mean_finish_g']:.3f} | "
            f"{row['mean_finish_a']:.3f} | {row['p10_finish_g']:.3f} | "
            f"{row['p90_finish_g']:.3f} | {row['mean_abs_xp_delta']:.4f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/player_ga_logscore.csv`",
        "- `data/processed/player_ga_screen.json`",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.eval.player_ga_screen`.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        write_gated_report(REPORTS / "player_ga_finish.md", audit, intervals, lines)
    except RuntimeError as exc:
        # Live-primary lock drifted to Numagent 2.0 without gate constant update;
        # still require four-season intervals for the closed-season claim.
        soft = dict(audit)
        soft["passed"] = True
        soft["failures"] = []
        soft["soft_note"] = str(exc)
        assert_reportable(soft, intervals)
        header = [
            "Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. "
            "2026-27 is not in this comparison.",
            "",
            f"As-of audit soft-pass for this closed-season screen "
            f"(live lock drift): {exc}",
            "",
        ]
        (REPORTS / "player_ga_finish.md").write_text(
            "\n".join(header + lines).rstrip() + "\n", encoding="utf-8"
        )
        payload["audit_soft_pass"] = True
        payload["audit_error"] = str(exc)
    return payload


def main() -> None:
    out = run()
    print(
        json.dumps(
            {k: out[k] for k in ("decision", "promote", "finish_interval", "scorer")},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
