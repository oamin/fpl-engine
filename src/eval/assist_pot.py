"""Pre-registered screen: team xA/xG assist pot vs fixed ρ=0.75.

Control keeps ``score_xp`` (``lam_assist = 0.75 · lam_scored``). Candidate sets
``lam_assist = clip(exp_team_xa / exp_team_xg) · lam_scored`` from shift-1
team priors, then recomputes ``compute_xp``. Judged on eligible player-GW
Gaussian log score across the four closed seasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    COMPARISONS,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.eval.honest_pool import eligible_mask
from src.ingest.fpl_odds import join_players_to_fixtures, load_football_data, load_player_logs
from src.models.xp_engine import (
    ASSIST_RHO_FIXED,
    DEFCON_THRESH,
    MIN_MINUTES,
    add_market_pots,
    add_player_priors,
    apply_lam_assist,
    compute_xp,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

CONTROL_COL = "score_xp"
CANDIDATE_COL = "score_xp_assist_team_xa"
ASSIST_CONTRAST = comparison_key(CANDIDATE_COL, CONTROL_COL)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gw"] = pd.to_numeric(out["gw"], errors="coerce")
    out = out.dropna(subset=["gw", "position"]).copy()
    out["gw"] = out["gw"].astype(int)
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
    out["minutes"] = minutes
    out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0.0)
    out["xG"] = pd.to_numeric(out["xG"], errors="coerce").fillna(0.0)
    out["xA"] = pd.to_numeric(out["xA"], errors="coerce").fillna(0.0)
    out["goals"] = pd.to_numeric(out.get("goals", 0.0), errors="coerce").fillna(0.0)
    out["value"] = pd.to_numeric(out.get("value"), errors="coerce")
    raw = pd.to_numeric(out.get("defcon", 0.0), errors="coerce").fillna(0.0)
    thr = out["position"].map(DEFCON_THRESH)
    out["defcon_hit"] = (
        out["position"].isin(list(DEFCON_THRESH))
        & (minutes >= MIN_MINUTES)
        & (raw >= thr.fillna(999))
    ).astype(float)
    if "p_not_lose" not in out.columns:
        out["p_not_lose"] = np.nan
    out["p_not_lose"] = pd.to_numeric(out["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        out["p_draw"], errors="coerce"
    )
    if "attack_strength" not in out.columns:
        out["attack_strength"] = np.nan
    if "defend_threat" not in out.columns:
        out["defend_threat"] = np.nan
    return out


def build_season(season: str, code: str, protocol: dict[str, Any]) -> pd.DataFrame:
    if season not in CLOSED_SEASONS:
        raise RuntimeError(f"{season} is outside the closed-season comparison")
    players = load_player_logs(season=season)
    fixtures = load_football_data(code=code)
    joined, _, _stats = join_players_to_fixtures(
        players, fixtures, retain_sheet_rows=True
    )
    frame = _prepare(joined)
    frame = add_market_pots(frame)
    frame = add_player_priors(frame)

    control = apply_lam_assist(frame, mode="fixed")
    control = compute_xp(control)
    candidate = apply_lam_assist(frame, mode="team_xa_xg")
    candidate = compute_xp(candidate)

    out = control.copy()
    out["season"] = season
    out[CONTROL_COL] = control["xp"]
    out[CANDIDATE_COL] = candidate["xp"].to_numpy(float)
    out["score_exp_points"] = out["exp_points"]
    out["assist_rho_control"] = ASSIST_RHO_FIXED
    out["assist_rho_candidate"] = candidate["assist_rho"].to_numpy(float)
    out["lam_assist_candidate"] = candidate["lam_assist"].to_numpy(float)
    out["eligible"] = eligible_mask(
        out,
        min_history=int(protocol["min_history"]),
        min_xmi=float(protocol["min_xmi"]),
    )
    return out


def _window(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    gw = pd.to_numeric(frame["gw"], errors="coerce")
    return frame.loc[(gw >= int(protocol["gw_start"])) & (gw <= int(protocol["gw_end"]))].copy()


def _logscore_pair(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    left: str,
    right: str,
    *,
    eligible_only: bool,
) -> list[dict[str, Any]]:
    window = _window(frame, protocol)
    if eligible_only:
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
        log_a = gaussian_log_score(y, pd.to_numeric(block[left], errors="coerce").to_numpy(float), sigma)
        log_b = gaussian_log_score(y, pd.to_numeric(block[right], errors="coerce").to_numpy(float), sigma)
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
    codes = protocol["season_codes"]
    all_rows: list[dict[str, Any]] = []
    rho_summary: list[dict[str, Any]] = []

    for season in CLOSED_SEASONS:
        frame = build_season(season, str(codes[season]), protocol)
        elig = frame.loc[frame["eligible"]]
        rho_summary.append(
            {
                "season": season,
                "n_eligible": int(len(elig)),
                "mean_assist_rho": float(pd.to_numeric(elig["assist_rho_candidate"], errors="coerce").mean()),
                "p10_assist_rho": float(pd.to_numeric(elig["assist_rho_candidate"], errors="coerce").quantile(0.1)),
                "p90_assist_rho": float(pd.to_numeric(elig["assist_rho_candidate"], errors="coerce").quantile(0.9)),
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
        all_rows.extend(
            _logscore_pair(frame, protocol, CONTROL_COL, "score_exp_points", eligible_only=True)
        )
        all_rows.extend(
            _logscore_pair(frame, protocol, CANDIDATE_COL, CONTROL_COL, eligible_only=True)
        )

    primary_key = comparison_key(CONTROL_COL, "score_exp_points")
    assist_key = ASSIST_CONTRAST
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": {
            primary_key: _interval_from_rows(all_rows, primary_key, protocol),
            assist_key: _interval_from_rows(all_rows, assist_key, protocol),
        },
    }
    assist_iv = intervals["comparisons"][assist_key]
    promote = float(assist_iv["lo"]) > 0.0
    decision = "promote_team_xa_xg" if promote else "keep_fixed_0.75"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(PROCESSED / "assist_pot_logscore.csv", index=False)
    payload = {
        "decision": decision,
        "promote": promote,
        "assist_contrast": assist_key,
        "assist_interval": assist_iv,
        "primary_interval": intervals["comparisons"][primary_key],
        "rho_summary": rho_summary,
        "control_rho": ASSIST_RHO_FIXED,
        "n_gws_logged": len(all_rows),
    }
    (PROCESSED / "assist_pot_screen.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Assist pot screen — team xA/xG vs fixed ρ=0.75",
        "",
        "Pre-registered before totals were read. Control is published `score_xp` "
        f"(`lam_assist = {ASSIST_RHO_FIXED} · lam_scored`). Candidate replaces ρ with "
        "shift-1 `exp_team_xa / exp_team_xg` (clipped), then recomputes `compute_xp`.",
        "",
        "Population: eligible rows (`n_prior ≥ 3`, `xmi ≥ 45`), gameweeks "
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
                "Interval clears zero: switch the published assist pot to `team_xa_xg`."
                if promote
                else "Interval does not clear zero: keep the fixed 0.75 control."
            )
        ),
        "",
        "## Intervals",
        "",
        f"| contrast | mean | 95% interval |",
        f"| --- | ---: | ---: |",
        (
            f"| `{assist_key}` | {assist_iv['mean']:+.5f} | "
            f"[{assist_iv['lo']:+.5f}, {assist_iv['hi']:+.5f}] |"
        ),
        (
            f"| `{primary_key}` | {intervals['comparisons'][primary_key]['mean']:+.5f} | "
            f"[{intervals['comparisons'][primary_key]['lo']:+.5f}, "
            f"{intervals['comparisons'][primary_key]['hi']:+.5f}] |"
        ),
        "",
        "Gameweeks per season (assist contrast): "
        + ", ".join(
            f"{s}={assist_iv['n_gws'][s]}" for s in CLOSED_SEASONS
        )
        + ".",
        "",
        "## Candidate ρ on eligible rows",
        "",
        "| season | n | mean ρ | p10 | p90 | mean |ΔxP| |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rho_summary:
        lines.append(
            f"| {row['season']} | {row['n_eligible']} | {row['mean_assist_rho']:.3f} | "
            f"{row['p10_assist_rho']:.3f} | {row['p90_assist_rho']:.3f} | "
            f"{row['mean_abs_xp_delta']:.4f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/assist_pot_logscore.csv`",
        "- `data/processed/assist_pot_screen.json`",
        "",
        "Reproduce: `python -m src.eval.assist_pot`.",
        "",
    ]
    write_gated_report(REPORTS / "assist_pot.md", audit, intervals, lines)
    return payload


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: out[k] for k in ("decision", "promote", "assist_interval")}, indent=2))
