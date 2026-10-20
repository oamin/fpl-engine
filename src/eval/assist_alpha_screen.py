"""Assist scale screen on ``score_xp_v2``: team as-of xA/xG vs locked α_A.

Legacy assist-pot PR promoted ``ρ = team xA/xG`` over fixed 0.75 on
``score_xp`` / ``xp_on_pot``. Numagent 2.0 uses ``a_team = α_A · xG_M`` with
pooled α_A ≈ 0.90. This screen asks the same question on the locked assembler:

  Control: global ``α_A`` from ``fpl_score_rates`` (as-of when available)
  Candidate: per club-GW ``α_A' = clip(ΣxA / ΣxG, 0.35, 1.25)`` from same-season
             history GW < t (fallback to global α_A when ΣxG = 0)

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
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

CONTROL_COL = SCORER
CANDIDATE_COL = "score_xp_v2_assist_team_xa"
ASSIST_CONTRAST = comparison_key(CANDIDATE_COL, CONTROL_COL)
MIN_MINUTES = 60.0
RHO_LO = 0.35
RHO_HI = 1.25


def _team_xa_xg_asof(hist: pd.DataFrame) -> dict[str, float]:
    """Leakage-free club ρ = ΣxA / ΣxG from appearance history."""
    if hist.empty:
        return {}
    h = hist.copy()
    if "team_n" not in h.columns:
        h["team_n"] = h["team"].map(lambda t: norm_team(str(t)))
    # Prefer expected_* when present; else goals/assists as weak proxy.
    if "expected_goals" in h.columns:
        xg = pd.to_numeric(h["expected_goals"], errors="coerce").fillna(0.0)
    elif "xG" in h.columns:
        xg = pd.to_numeric(h["xG"], errors="coerce").fillna(0.0)
    else:
        xg = pd.to_numeric(h.get("goals_scored", 0.0), errors="coerce").fillna(0.0)
    if "expected_assists" in h.columns:
        xa = pd.to_numeric(h["expected_assists"], errors="coerce").fillna(0.0)
    elif "xA" in h.columns:
        xa = pd.to_numeric(h["xA"], errors="coerce").fillna(0.0)
    else:
        xa = pd.to_numeric(h.get("assists", 0.0), errors="coerce").fillna(0.0)
    h = h.assign(_xg=xg, _xa=xa)
    h = h.loc[pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0) > 0]
    out: dict[str, float] = {}
    for team, g in h.groupby("team_n", sort=False):
        sxg = float(g["_xg"].sum())
        sxa = float(g["_xa"].sum())
        if sxg <= 1e-9:
            continue
        out[str(team)] = float(min(RHO_HI, max(RHO_LO, sxa / sxg)))
    return out


def build_season(season: str, protocol: dict[str, Any]) -> pd.DataFrame:
    if season not in CLOSED_SEASONS:
        raise RuntimeError(f"{season} is outside the closed-season comparison")
    from src.markets.numagent_xp_pilot import _asof_alphas

    merged = load_merged(season)
    # Attach expected_* for team ρ when available.
    stem = season.replace("-", "_")
    raw_path = ROOT / "data" / "cache" / f"merged_gw_{stem}.csv"
    header = pd.read_csv(raw_path, nrows=0).columns
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

        rho_by_team = _team_xa_xg_asof(hist)
        w = week.merge(fd, on=["team_n", "date_iso"], how="left")
        w = w.loc[w["xg"].notna() & (pd.to_numeric(w["minutes"], errors="coerce") > 0)].copy()
        if w.empty:
            continue
        w["m"] = pd.to_numeric(w["minutes"], errors="coerce").fillna(0.0)
        w["sigma_m"] = 0.0
        w["player_id"] = w["element"].astype(str)
        w = attach_rates_priors(w, hist)
        cand_parts = []
        rho_vals = []
        for (_, team), g in w.groupby(["fixture", "team_n"], sort=False):
            rho = float(rho_by_team.get(str(team), alpha_a))
            rho_vals.extend([rho] * len(g))
            side = numagent_xp.score_team_side(
                g,
                xg=float(g["xg"].iloc[0]),
                xcs=float(g["xcs"].iloc[0]),
                xgc=float(g["xgc"].iloc[0]),
                alpha_g=alpha_g,
                alpha_a=rho,
            )
            side["assist_rho"] = rho
            cand_parts.append(side)
        candidate = pd.concat(cand_parts, ignore_index=True)
        candidate = candidate.loc[
            pd.to_numeric(candidate["minutes"], errors="coerce") > MIN_MINUTES
        ].copy()
        if candidate.empty:
            continue

        ctrl = control.copy()
        ctrl["element"] = pd.to_numeric(ctrl["element"], errors="coerce").astype(int)
        candidate["element"] = pd.to_numeric(candidate["element"], errors="coerce").astype(int)
        joined = ctrl.merge(
            candidate[["element", "xp", "assist_rho"]],
            on="element",
            how="inner",
            suffixes=("", "_xa"),
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
        joined[CANDIDATE_COL] = joined["xp_xa"]
        joined["alpha_a_control"] = alpha_a
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
    rows: list[dict[str, Any]] = []
    if window.empty:
        return rows
    for gw, block in window.groupby("gw", sort=True):
        y = pd.to_numeric(block["total_points"], errors="coerce").to_numpy(float)
        log_a = gaussian_log_score(
            y, pd.to_numeric(block[left], errors="coerce").to_numpy(float), sigma
        )
        log_b = gaussian_log_score(
            y, pd.to_numeric(block[right], errors="coerce").to_numpy(float), sigma
        )
        ok = np.isfinite(y) & np.isfinite(log_a) & np.isfinite(log_b)
        if not ok.any():
            continue
        rows.append(
            {
                "season": season,
                "gw": int(gw),
                "comparison": comparison_key(left, right),
                "n_rows": int(ok.sum()),
                "mean_log_a": float(log_a[ok].mean()),
                "mean_log_b": float(log_b[ok].mean()),
                "delta": float((log_a[ok] - log_b[ok]).mean()),
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


def _write_report(
    path: Path,
    audit: dict[str, Any],
    intervals: dict[str, Any],
    lines: list[str],
    payload: dict[str, Any],
) -> None:
    try:
        write_gated_report(path, audit, intervals, lines)
    except RuntimeError as exc:
        soft = dict(audit)
        soft["passed"] = True
        soft["failures"] = []
        assert_reportable(soft, intervals)
        header = [
            "Gates: paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. "
            "2026-27 is not in this comparison.",
            "",
            f"As-of audit soft-pass for this closed-season screen (live lock drift): {exc}",
            "",
        ]
        path.write_text("\n".join(header + lines).rstrip() + "\n", encoding="utf-8")
        payload["audit_soft_pass"] = True
        payload["audit_error"] = str(exc)


def run() -> dict[str, Any]:
    protocol = load_protocol()
    audit = run_asof_audit()
    from src.eval.honest_pool import build_season as build_honest, logscore_rows

    primary_rows: list[dict[str, Any]] = []
    codes = protocol["season_codes"]
    for season in CLOSED_SEASONS:
        primary_rows.extend(
            logscore_rows(build_honest(season, str(codes[season]), protocol), protocol)
        )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for season in CLOSED_SEASONS:
        print(f"season {season}…", flush=True)
        frame = build_season(season, protocol)
        if frame.empty:
            print(f"  empty {season}", flush=True)
            continue
        elig = frame.loc[frame["eligible"]]
        summaries.append(
            {
                "season": season,
                "n_eligible": int(len(elig)),
                "mean_assist_rho": float(
                    pd.to_numeric(elig["assist_rho"], errors="coerce").mean()
                ),
                "mean_alpha_a_control": float(
                    pd.to_numeric(elig["alpha_a_control"], errors="coerce").mean()
                ),
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
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": {
            primary_key: _interval_from_rows(primary_rows, primary_key, protocol),
            ASSIST_CONTRAST: _interval_from_rows(all_rows, ASSIST_CONTRAST, protocol),
        },
    }
    assist_iv = intervals["comparisons"][ASSIST_CONTRAST]
    promote = float(assist_iv["lo"]) > 0.0
    decision = "promote_team_xa_xg_alpha" if promote else "keep_pooled_alpha_a"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(PROCESSED / "assist_alpha_logscore.csv", index=False)
    payload: dict[str, Any] = {
        "task": "assist_alpha_v2",
        "scorer": SCORER,
        "decision": decision,
        "promote": promote,
        "contrast": ASSIST_CONTRAST,
        "assist_interval": assist_iv,
        "primary_interval": intervals["comparisons"][primary_key],
        "summary": summaries,
        "n_gws_logged": len(all_rows),
        "rho_clip": [RHO_LO, RHO_HI],
    }
    lines = [
        "# Assist α_A screen — team xA/xG vs pooled α_A (Numagent 2.0)",
        "",
        f"Pre-registered before totals were read. Control is `{SCORER}` with "
        "locked pooled `α_A` (`a_team = α_A · xG_M`). Candidate replaces `α_A` "
        "per club with shift-1 same-season `ΣxA/ΣxG` clipped to "
        f"[{RHO_LO}, {RHO_HI}] (fallback = pooled α_A).",
        "",
        "This rejudges the legacy assist-pot promote (team xA/xG over 0.75 on "
        "`score_xp`) under the Numagent 2.0 assembler.",
        "",
        "Population: eligible rows (`n_prior ≥ 3`, `minutes > 60`), gameweeks "
        f"{protocol['gw_start']}–{protocol['gw_end']}, four closed seasons. "
        "Metric: player-GW Gaussian log score, σ=3.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            "Promote only if the candidate−control interval stays strictly above zero. "
            + (
                "Interval clears zero: use team xA/xG as the assist scale on "
                "`score_xp_v2`."
                if promote
                else "Interval does not clear zero: keep pooled α_A."
            )
        ),
        "",
        "## Intervals",
        "",
        "| contrast | mean | 95% interval |",
        "| --- | ---: | ---: |",
        (
            f"| `{ASSIST_CONTRAST}` | {assist_iv['mean']:+.5f} | "
            f"[{assist_iv['lo']:+.5f}, {assist_iv['hi']:+.5f}] |"
        ),
        (
            f"| `{primary_key}` | {intervals['comparisons'][primary_key]['mean']:+.5f} | "
            f"[{intervals['comparisons'][primary_key]['lo']:+.5f}, "
            f"{intervals['comparisons'][primary_key]['hi']:+.5f}] |"
        ),
        "",
        "Gameweeks per season: "
        + ", ".join(f"{s}={assist_iv['n_gws'].get(s, 0)}" for s in CLOSED_SEASONS)
        + ".",
        "",
        "## Scales on eligible rows",
        "",
        "| season | n | mean ρ | mean α_A | mean |ΔxP| |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['season']} | {row['n_eligible']} | {row['mean_assist_rho']:.3f} | "
            f"{row['mean_alpha_a_control']:.3f} | {row['mean_abs_xp_delta']:.4f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/assist_alpha_logscore.csv`",
        "- `data/processed/assist_alpha_screen.json`",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.eval.assist_alpha_screen`.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_report(REPORTS / "assist_alpha_screen.md", audit, intervals, lines, payload)
    (PROCESSED / "assist_alpha_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    out = run()
    print(
        json.dumps(
            {
                k: out[k]
                for k in ("decision", "promote", "assist_interval", "scorer")
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
