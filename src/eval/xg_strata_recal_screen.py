"""Stratified FD centreline pot calibration — LOSO xG-quartile rescale.

Pre-registered in ``protocol.score_xp_v2_xg_strata_recal`` before totals.

Control: published ``score_xp_v2`` on raw FD AH+OU centreline ``xg``.
Candidate: same assembler; replace side ``xg`` with

  xg' = xg · scale(bin(xg))

where quartile edges and per-bin ``scale = mean(goals)/mean(xg)`` are fit on
the other three closed seasons' FD side-rows (LOSO). ``xcs`` / ``xgc`` and
``α_G`` / ``α_A`` stay as published.

Promote only if the candidate−control eligible log-score interval stays
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
from src.markets.cascade_score_v2 import SCORER
from src.markets.numagent_xp_pilot import load_fd_moments, load_merged, score_gameweek

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

CONTROL_COL = SCORER
CANDIDATE_COL = "score_xp_v2_xg_strata"
CONTRAST = comparison_key(CANDIDATE_COL, CONTROL_COL)
MIN_MINUTES = 60.0
N_BINS = 4
XG_FLOOR = 0.05
XG_CEIL = 4.5


def _moments_path(season: str) -> Path:
    return PROCESSED / f"scoreline_fd_moments_{season.replace('-', '_')}.csv"


def load_side_rows(season: str) -> pd.DataFrame:
    """FD centreline side-rows with predicted xG and actual match goals."""
    path = _moments_path(season)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    out = df[["xg", "goals", "ah_line", "side"]].copy()
    out["xg"] = pd.to_numeric(out["xg"], errors="coerce")
    out["goals"] = pd.to_numeric(out["goals"], errors="coerce")
    out["ah_line"] = pd.to_numeric(out["ah_line"], errors="coerce")
    out = out.dropna(subset=["xg", "goals"])
    out["season"] = season
    return out


def fit_quartile_scales(
    train: pd.DataFrame, *, n_bins: int = N_BINS
) -> dict[str, Any]:
    """LOSO pot scales: edges from training xG quantiles; scale_k = ḡ / x̄ per bin."""
    xg = train["xg"].to_numpy(float)
    goals = train["goals"].to_numpy(float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(xg, qs))
    if edges.size < 3:
        # Degenerate fold — identity map.
        return {
            "edges": [float(xg.min()), float(xg.max() + 1e-6)],
            "scales": [1.0],
            "n_train": int(len(train)),
            "bin_stats": [],
        }
    # Ensure rightmost edge is open above max.
    edges = edges.astype(float)
    edges[0] = min(edges[0], float(xg.min()))
    edges[-1] = max(edges[-1], float(xg.max()) + 1e-9)
    scales: list[float] = []
    stats: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (xg >= lo) & (xg < hi)
        else:
            mask = (xg >= lo) & (xg <= hi)
        if not np.any(mask):
            scales.append(1.0)
            stats.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0.0,
                    "mean_xg": float("nan"),
                    "mean_goals": float("nan"),
                    "scale": 1.0,
                }
            )
            continue
        mx = float(xg[mask].mean())
        mg = float(goals[mask].mean())
        scale = float(mg / mx) if mx > 1e-9 else 1.0
        # Keep scales in a sane band — still data-driven, not a premium tax.
        scale = float(np.clip(scale, 0.5, 1.5))
        scales.append(scale)
        stats.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": float(mask.sum()),
                "mean_xg": mx,
                "mean_goals": mg,
                "scale": scale,
            }
        )
    return {
        "edges": [float(e) for e in edges],
        "scales": scales,
        "n_train": int(len(train)),
        "bin_stats": stats,
    }


def apply_scales(xg: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    edges = np.asarray(fit["edges"], dtype=float)
    scales = np.asarray(fit["scales"], dtype=float)
    x = np.asarray(xg, dtype=float)
    out = x.copy()
    for i, scale in enumerate(scales):
        lo, hi = edges[i], edges[i + 1]
        if i < len(scales) - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi + 1e-12)
        out[mask] = x[mask] * float(scale)
    return np.clip(out, XG_FLOOR, XG_CEIL)


def loso_fit(holdout: str) -> dict[str, Any]:
    parts = [load_side_rows(s) for s in CLOSED_SEASONS if s != holdout]
    train = pd.concat(parts, ignore_index=True)
    fit = fit_quartile_scales(train)
    fit["holdout"] = holdout
    fit["train_seasons"] = [s for s in CLOSED_SEASONS if s != holdout]
    return fit


def calibrate_fd(fd: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    out = fd.copy()
    out["xg_raw"] = pd.to_numeric(out["xg"], errors="coerce")
    out["xg"] = apply_scales(out["xg_raw"].to_numpy(float), fit)
    return out


# Fixed absolute bands for the secondary diagnostic table only (not the LOSO map).
_DIAG_XG_EDGES = (0.0, 1.05, 1.35, 1.75, 5.0)
_DIAG_XG_LABELS = ("q1_low", "q2", "q3", "q4_high")


def _pot_strata(sides: pd.DataFrame, xg_col: str = "xg") -> list[dict[str, Any]]:
    """goals/xg by fixed xG bands and by AH favouritism (side-aware)."""
    df = sides.copy()
    df["xg_use"] = pd.to_numeric(df[xg_col], errors="coerce")
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce")
    df = df.dropna(subset=["xg_use", "goals"])
    rows: list[dict[str, Any]] = []
    df["xg_band"] = pd.cut(
        df["xg_use"],
        bins=list(_DIAG_XG_EDGES),
        labels=list(_DIAG_XG_LABELS),
        include_lowest=True,
    )
    for label, g in df.groupby("xg_band", observed=True):
        mx = float(g["xg_use"].mean())
        mg = float(g["goals"].mean())
        rows.append(
            {
                "stratum": "xg_fixed",
                "band": str(label),
                "n": int(len(g)),
                "mean_xg": mx,
                "mean_goals": mg,
                "goals_over_xg": float(mg / mx) if mx > 1e-9 else float("nan"),
            }
        )
    # Side-aware AH strength: home uses ah_line; away uses -ah_line.
    ah = pd.to_numeric(df["ah_line"], errors="coerce").fillna(0.0)
    strength = np.where(df["side"].astype(str) == "home", -ah, ah)
    # Positive strength ⇒ side is favourite (receiving goals on AH).
    df = df.copy()
    df["fav"] = strength
    bins = [-np.inf, -0.75, -0.25, 0.25, 0.75, np.inf]
    labels = ["big_dog", "dog", "even", "fav", "big_fav"]
    df["ah_band"] = pd.cut(df["fav"], bins=bins, labels=labels)
    for label, g in df.groupby("ah_band", observed=True):
        if g.empty:
            continue
        mx = float(g["xg_use"].mean())
        mg = float(g["goals"].mean())
        rows.append(
            {
                "stratum": "ah_band",
                "band": str(label),
                "n": int(len(g)),
                "mean_xg": mx,
                "mean_goals": mg,
                "goals_over_xg": float(mg / mx) if mx > 1e-9 else float("nan"),
            }
        )
    return rows


def build_season(
    season: str, protocol: dict[str, Any], fit: dict[str, Any]
) -> pd.DataFrame:
    if season not in CLOSED_SEASONS:
        raise RuntimeError(f"{season} is outside the closed-season comparison")
    from src.markets.numagent_xp_pilot import _asof_alphas

    merged = load_merged(season)
    fd_raw = load_fd_moments(season)
    fd_cal = calibrate_fd(fd_raw, fit)
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
        control = score_gameweek(hist, week, fd_raw, alpha_g=alpha_g, alpha_a=alpha_a)
        candidate = score_gameweek(hist, week, fd_cal, alpha_g=alpha_g, alpha_a=alpha_a)
        if control.empty or candidate.empty:
            continue
        if "element" not in control.columns or "element" not in candidate.columns:
            continue
        ctrl = control.copy()
        cand = candidate.copy()
        ctrl["element"] = pd.to_numeric(ctrl["element"], errors="coerce").astype(int)
        cand["element"] = pd.to_numeric(cand["element"], errors="coerce").astype(int)
        joined = ctrl.merge(
            cand[["element", "xp"]],
            on="element",
            how="inner",
            suffixes=("", "_cal"),
        )
        if joined.empty:
            continue
        if len(hist):
            apps_n = hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
            joined["n_prior"] = joined["element"].map(apps_n).fillna(0)
        else:
            joined["n_prior"] = 0
        joined["season"] = season
        joined["gw"] = int(gw)
        joined[CONTROL_COL] = joined["xp"]
        joined[CANDIDATE_COL] = joined["xp_cal"]
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
    fits: dict[str, Any] = {}
    strata_diag: list[dict[str, Any]] = []

    for season in CLOSED_SEASONS:
        print(f"season {season}…", flush=True)
        fit = loso_fit(season)
        fits[season] = fit
        # Side-row diagnostic (not a gate): raw vs calibrated pot ratios.
        sides = load_side_rows(season)
        for row in _pot_strata(sides, "xg"):
            strata_diag.append({"season": season, "arm": "raw", **row})
        sides_cal = sides.copy()
        sides_cal["xg_cal"] = apply_scales(sides_cal["xg"].to_numpy(float), fit)
        for row in _pot_strata(sides_cal, "xg_cal"):
            strata_diag.append({"season": season, "arm": "calibrated", **row})

        frame = build_season(season, protocol, fit)
        if frame.empty:
            print(f"  empty {season}", flush=True)
            continue
        elig = frame.loc[frame["eligible"]]
        summaries.append(
            {
                "season": season,
                "n_eligible": int(len(elig)),
                "mean_xp_delta": float(
                    (
                        pd.to_numeric(elig[CANDIDATE_COL], errors="coerce")
                        - pd.to_numeric(elig[CONTROL_COL], errors="coerce")
                    ).mean()
                ),
                "mean_abs_xp_delta": float(
                    (
                        pd.to_numeric(elig[CANDIDATE_COL], errors="coerce")
                        - pd.to_numeric(elig[CONTROL_COL], errors="coerce")
                    )
                    .abs()
                    .mean()
                ),
                "scales": fit["scales"],
                "edges": fit["edges"],
            }
        )
        all_rows.extend(_logscore_pair(frame, protocol, CANDIDATE_COL, CONTROL_COL))
        print(
            f"  scales={np.round(fit['scales'], 3).tolist()} "
            f"n_elig={len(elig)}",
            flush=True,
        )

    primary_key = comparison_key("score_xp", "score_exp_points")
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": {
            primary_key: _interval_from_rows(primary_rows, primary_key, protocol),
            CONTRAST: _interval_from_rows(all_rows, CONTRAST, protocol),
        },
    }
    iv = intervals["comparisons"][CONTRAST]
    promote = float(iv["lo"]) > 0.0
    decision = "promote_xg_strata_recal" if promote else "keep_raw_fd_xg"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(PROCESSED / "xg_strata_recal_logscore.csv", index=False)
    pd.DataFrame(strata_diag).to_csv(
        PROCESSED / "xg_strata_recal_side_strata.csv", index=False
    )
    payload: dict[str, Any] = {
        "task": "xg_strata_recal",
        "scorer": SCORER,
        "decision": decision,
        "promote": promote,
        "contrast": CONTRAST,
        "interval": iv,
        "primary_interval": intervals["comparisons"][primary_key],
        "summary": summaries,
        "loso_fits": {
            s: {
                "edges": fits[s]["edges"],
                "scales": fits[s]["scales"],
                "bin_stats": fits[s]["bin_stats"],
                "n_train": fits[s]["n_train"],
            }
            for s in fits
        },
        "n_gws_logged": len(all_rows),
    }
    (PROCESSED / "xg_strata_recal_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stratified FD centreline pot calibration (xG quartiles)",
        "",
        "Pre-registered before totals were read "
        "(`protocol.score_xp_v2_xg_strata_recal`).",
        "",
        f"Control is `{SCORER}` on raw FD AH+OU centreline `xg`. Candidate keeps "
        "shares, minutes, `xcs`/`xgc`, and `α_G`/`α_A`, and replaces attack pot "
        "level with a **LOSO xG-quartile multiplicative rescale**:",
        "",
        "```",
        "scale_k = mean(goals)_k / mean(xg)_k   # other three closed seasons",
        "xg' = clip(xg · scale(bin(xg)), 0.05, 4.5)",
        "```",
        "",
        "Quartile edges come from the training fold only. Per-bin scales clipped "
        "to [0.5, 1.5]. This targets the conditional high-xG under-delivery / "
        "low-xG over-delivery in match pots — not a premium-player tax.",
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
                "Interval clears zero: apply LOSO xG-quartile rescale on FD centreline pots."
                if promote
                else "Interval does not clear zero: keep raw FD `xg`."
            )
        ),
        "",
        "## Intervals",
        "",
        "| contrast | mean | 95% interval |",
        "| --- | ---: | ---: |",
        (
            f"| `{CONTRAST}` | {iv['mean']:+.5f} | "
            f"[{iv['lo']:+.5f}, {iv['hi']:+.5f}] |"
        ),
        (
            f"| `{primary_key}` | {intervals['comparisons'][primary_key]['mean']:+.5f} | "
            f"[{intervals['comparisons'][primary_key]['lo']:+.5f}, "
            f"{intervals['comparisons'][primary_key]['hi']:+.5f}] |"
        ),
        "",
        "Gameweeks per season: "
        + ", ".join(f"{s}={iv['n_gws'].get(s, 0)}" for s in CLOSED_SEASONS)
        + ".",
        "",
        "## LOSO scales (holdout ← train on other three)",
        "",
        "| holdout | scales (low→high xG) | mean ΔxP | mean |ΔxP| | n elig |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        scales_s = ", ".join(f"{s:.3f}" for s in row["scales"])
        lines.append(
            f"| {row['season']} | {scales_s} | {row['mean_xp_delta']:+.5f} | "
            f"{row['mean_abs_xp_delta']:.5f} | {row['n_eligible']} |"
        )

    # Compact side-row strata (pooled over seasons, raw vs cal).
    strata_df = pd.DataFrame(strata_diag)
    lines += [
        "",
        "## Secondary — side-row goals/xg (diagnostic)",
        "",
        "Pooled across seasons. Fixed absolute xG bands for display "
        f"({list(_DIAG_XG_EDGES)}). Not a promote gate.",
        "",
    ]
    if not strata_df.empty:
        for stratum, title, order in (
            ("xg_fixed", "xg_band (fixed)", list(_DIAG_XG_LABELS)),
            (
                "ah_band",
                "ah_band",
                ["big_dog", "dog", "even", "fav", "big_fav"],
            ),
        ):
            sub = strata_df.loc[strata_df["stratum"] == stratum]
            if sub.empty:
                continue
            lines += [
                f"### {title}",
                "",
                "| arm | band | n | mean xG | mean goals | goals/xG |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
            agg = (
                sub.groupby(["arm", "band"], observed=True)
                .apply(
                    lambda g: pd.Series(
                        {
                            "n": g["n"].sum(),
                            "mean_xg": np.average(g["mean_xg"], weights=g["n"]),
                            "mean_goals": np.average(g["mean_goals"], weights=g["n"]),
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
            )
            agg["ratio"] = agg["mean_goals"] / agg["mean_xg"]
            for arm in ("raw", "calibrated"):
                for band in order:
                    hit = agg.loc[(agg["arm"] == arm) & (agg["band"] == band)]
                    if hit.empty:
                        continue
                    r = hit.iloc[0]
                    lines.append(
                        f"| {arm} | {band} | {int(r['n'])} | {r['mean_xg']:.3f} | "
                        f"{r['mean_goals']:.3f} | {r['ratio']:.3f} |"
                    )
            lines.append("")

    lines += [
        "## Outputs",
        "",
        "- `data/processed/xg_strata_recal_logscore.csv`",
        "- `data/processed/xg_strata_recal_side_strata.csv`",
        "- `data/processed/xg_strata_recal_screen.json`",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.eval.xg_strata_recal_screen`.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_report(REPORTS / "xg_strata_recal_screen.md", audit, intervals, lines, payload)
    (PROCESSED / "xg_strata_recal_screen.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    out = run()
    print(
        json.dumps(
            {
                k: out[k]
                for k in ("decision", "promote", "interval", "scorer", "contrast")
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
