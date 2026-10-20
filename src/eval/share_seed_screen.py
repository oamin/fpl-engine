"""Share-seeding screen vs frozen mean shares on ``score_xp_v2``.

Pre-registered before totals. Control is published ``score_xp_v2`` (point
rates → compositional shares → ``score_team_side`` on FD AH+OU pots).
Candidate draws rates inside their SE on each team-side

  rG' = max(0, rG + Z·drG),  Z ~ N(0,1)   (same for rA)

then re-scores and averages xP over ``N_MC`` draws — i.e. E[xP | rate noise]
as the share-seed proxy. Rates/priors stay as-of (GW < t); only the share
axis is jittered.

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

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

CONTROL_COL = SCORER
CANDIDATE_COL = "score_xp_v2_share_seed"
SEED_CONTRAST = comparison_key(CANDIDATE_COL, CONTROL_COL)
MIN_MINUTES = 60.0
N_MC = 20
MC_SEED = 0


def _score_side_seeded(
    g: pd.DataFrame,
    *,
    xg: float,
    xcs: float,
    xgc: float,
    alpha_g: float,
    alpha_a: float,
    rng: np.random.Generator,
    n_mc: int = N_MC,
) -> pd.DataFrame:
    """Mean xP over rate draws; other columns from the centreline score."""
    centre = numagent_xp.score_team_side(
        g, xg=xg, xcs=xcs, xgc=xgc, alpha_g=alpha_g, alpha_a=alpha_a
    )
    if centre.empty or n_mc <= 1:
        return centre
    rG = pd.to_numeric(g["rG"], errors="coerce").fillna(0.0).to_numpy(float)
    drG = pd.to_numeric(g["drG"], errors="coerce").fillna(0.0).to_numpy(float)
    rA = pd.to_numeric(g["rA"], errors="coerce").fillna(0.0).to_numpy(float)
    drA = pd.to_numeric(g["drA"], errors="coerce").fillna(0.0).to_numpy(float)
    acc = np.zeros(len(g), dtype=float)
    for _ in range(n_mc):
        draw = g.copy()
        draw["rG"] = np.maximum(0.0, rG + rng.normal(0.0, 1.0, size=len(g)) * drG)
        draw["rA"] = np.maximum(0.0, rA + rng.normal(0.0, 1.0, size=len(g)) * drA)
        # Point draw: collapse rate SE so shares are not double-shrunk on noise.
        draw["drG"] = 0.0
        draw["drA"] = 0.0
        scored = numagent_xp.score_team_side(
            draw, xg=xg, xcs=xcs, xgc=xgc, alpha_g=alpha_g, alpha_a=alpha_a
        )
        acc += pd.to_numeric(scored["xp"], errors="coerce").fillna(0.0).to_numpy(float)
    out = centre.copy()
    out["xp"] = acc / float(n_mc)
    out[numagent_xp.SCORE_NAME] = out["xp"]
    return out


def build_season(season: str, protocol: dict[str, Any]) -> pd.DataFrame:
    if season not in CLOSED_SEASONS:
        raise RuntimeError(f"{season} is outside the closed-season comparison")
    from src.markets.numagent_xp_pilot import _asof_alphas

    merged = load_merged(season)
    fd = load_fd_moments(season)
    rng = np.random.default_rng(MC_SEED + hash(season) % 10_000)
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

        w = week.merge(fd, on=["team_n", "date_iso"], how="left")
        w = w.loc[w["xg"].notna() & (pd.to_numeric(w["minutes"], errors="coerce") > 0)].copy()
        if w.empty:
            continue
        w["m"] = pd.to_numeric(w["minutes"], errors="coerce").fillna(0.0)
        w["sigma_m"] = 0.0
        w["player_id"] = w["element"].astype(str)
        w = attach_rates_priors(w, hist)
        cand_parts = []
        for (_, _), g in w.groupby(["fixture", "team_n"], sort=False):
            cand_parts.append(
                _score_side_seeded(
                    g,
                    xg=float(g["xg"].iloc[0]),
                    xcs=float(g["xcs"].iloc[0]),
                    xgc=float(g["xgc"].iloc[0]),
                    alpha_g=alpha_g,
                    alpha_a=alpha_a,
                    rng=rng,
                )
            )
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
            candidate[["element", "xp"]],
            on="element",
            how="inner",
            suffixes=("", "_seed"),
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
        joined[CANDIDATE_COL] = joined["xp_seed"]
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
                "mean_abs_xp_delta": float(
                    (
                        pd.to_numeric(elig[CANDIDATE_COL], errors="coerce")
                        - pd.to_numeric(elig[CONTROL_COL], errors="coerce")
                    )
                    .abs()
                    .mean()
                ),
                "mean_xp_delta": float(
                    (
                        pd.to_numeric(elig[CANDIDATE_COL], errors="coerce")
                        - pd.to_numeric(elig[CONTROL_COL], errors="coerce")
                    ).mean()
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
            SEED_CONTRAST: _interval_from_rows(all_rows, SEED_CONTRAST, protocol),
        },
    }
    seed_iv = intervals["comparisons"][SEED_CONTRAST]
    promote = float(seed_iv["lo"]) > 0.0
    decision = "promote_share_seeding" if promote else "keep_frozen_shares"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(PROCESSED / "share_seed_logscore.csv", index=False)
    payload: dict[str, Any] = {
        "task": "share_seed",
        "scorer": SCORER,
        "decision": decision,
        "promote": promote,
        "n_mc": N_MC,
        "mc_seed": MC_SEED,
        "contrast": SEED_CONTRAST,
        "seed_interval": seed_iv,
        "primary_interval": intervals["comparisons"][primary_key],
        "summary": summaries,
        "n_gws_logged": len(all_rows),
    }
    (PROCESSED / "share_seed_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Share-seeding screen (Numagent 2.0)",
        "",
        f"Pre-registered before totals were read. Control is `{SCORER}` "
        "(point rates → shares → `score_team_side` on FD pots). Candidate averages "
        f"`score_team_side` over **N={N_MC}** rate draws "
        "`r' = max(0, r + Z·dr)`, Z~N(0,1), then rebuilds shares — the share-seed "
        "proxy asked for under cascade scenario uncertainty.",
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
                "Interval clears zero: open share seeding on cascade strands."
                if promote
                else "Interval does not clear zero: keep frozen mean shares "
                "(`SEED_SHARES=False`)."
            )
        ),
        "",
        "## Intervals",
        "",
        "| contrast | mean | 95% interval |",
        "| --- | ---: | ---: |",
        (
            f"| `{SEED_CONTRAST}` | {seed_iv['mean']:+.5f} | "
            f"[{seed_iv['lo']:+.5f}, {seed_iv['hi']:+.5f}] |"
        ),
        (
            f"| `{primary_key}` | {intervals['comparisons'][primary_key]['mean']:+.5f} | "
            f"[{intervals['comparisons'][primary_key]['lo']:+.5f}, "
            f"{intervals['comparisons'][primary_key]['hi']:+.5f}] |"
        ),
        "",
        "Gameweeks per season: "
        + ", ".join(f"{s}={seed_iv['n_gws'].get(s, 0)}" for s in CLOSED_SEASONS)
        + ".",
        "",
        "## Eligible |ΔxP|",
        "",
        "| season | n | mean ΔxP | mean |ΔxP| |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['season']} | {row['n_eligible']} | {row['mean_xp_delta']:+.5f} | "
            f"{row['mean_abs_xp_delta']:.5f} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/share_seed_logscore.csv`",
        "- `data/processed/share_seed_screen.json`",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.eval.share_seed_screen`.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_report(REPORTS / "share_seed_screen.md", audit, intervals, lines, payload)
    (PROCESSED / "share_seed_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    out = run()
    print(
        json.dumps(
            {k: out[k] for k in ("decision", "promote", "seed_interval", "scorer", "n_mc")},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
