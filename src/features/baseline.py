"""Stage 4 — player baseline points and residual / z-score."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

MIN_MINUTES = 60.0
MIN_PRIOR = 3


def add_baseline_residual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["player_id", "date", "fixture_id"], kind="mergesort")
    out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0.0)
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)

    # Baseline on all appearances; analysis often filters ≥60'.
    out["baseline_points"] = out.groupby("player_id")["total_points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    out["n_prior_apps"] = out.groupby("player_id").cumcount()
    pos_mean = out.groupby("position")["total_points"].transform("mean")
    out["baseline_points"] = out["baseline_points"].fillna(pos_mean)

    out["residual_points"] = out["total_points"] - out["baseline_points"]
    # Expanding residual std for z (prior only).
    out["resid_std"] = out.groupby("player_id")["residual_points"].transform(
        lambda s: s.shift(1).expanding().std()
    )
    # Cold start: position residual std.
    pos_std = out.groupby("position")["residual_points"].transform("std")
    out["resid_std"] = out["resid_std"].fillna(pos_std).clip(lower=0.5)
    out["z_resid"] = out["residual_points"] / out["resid_std"]
    return out


def residual_lift(df: pd.DataFrame) -> dict[str, Any]:
    """Top vs bottom decile of baseline: do high-baseline players score more?"""
    sub = df.loc[
        (df["minutes"] >= MIN_MINUTES) & (df["n_prior_apps"] >= MIN_PRIOR)
    ].copy()
    stats: dict[str, Any] = {"n": int(len(sub))}
    if len(sub) < 50:
        stats["error"] = "too few rows"
        return stats

    # Predictive check: prior baseline rank vs realised residual (should be ~0 if well calibrated)
    # and vs realised points (should be positive).
    sub["base_bin"] = pd.qcut(sub["baseline_points"], 10, duplicates="drop")
    g = sub.groupby("base_bin", observed=True).agg(
        mean_points=("total_points", "mean"),
        mean_resid=("residual_points", "mean"),
        mean_base=("baseline_points", "mean"),
        n=("total_points", "count"),
    )
    top = g.iloc[-1]["mean_points"]
    bot = g.iloc[0]["mean_points"]
    stats["top_decile_points"] = float(top)
    stats["bottom_decile_points"] = float(bot)
    stats["lift_points"] = float(top - bot)
    stats["corr_base_points"] = float(sub["baseline_points"].corr(sub["total_points"]))
    stats["corr_base_resid"] = float(sub["baseline_points"].corr(sub["residual_points"]))
    stats["mean_abs_resid"] = float(sub["residual_points"].abs().mean())
    by_pos = {}
    for pos, chunk in sub.groupby("position"):
        by_pos[str(pos)] = {
            "n": int(len(chunk)),
            "corr_base_points": float(chunk["baseline_points"].corr(chunk["total_points"])),
            "mean_resid": float(chunk["residual_points"].mean()),
        }
    stats["by_position"] = by_pos
    return stats


def write_stage4_report(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# Stage 4 — Player baseline + residual",
        "",
        f"- Rows (≥60′, ≥{MIN_PRIOR} prior apps): **{stats.get('n', 0)}**",
        f"- Corr(baseline, points): **{stats.get('corr_base_points', float('nan')):.3f}**",
        f"- Corr(baseline, residual): **{stats.get('corr_base_resid', float('nan')):.3f}** (≈0 = calibrated)",
        f"- Top−bottom baseline decile points: **{stats.get('lift_points', float('nan')):.2f}**",
        f"- Mean |residual|: **{stats.get('mean_abs_resid', float('nan')):.2f}**",
        "",
        "## By position",
        "",
    ]
    for pos, s in (stats.get("by_position") or {}).items():
        lines.append(
            f"- {pos}: n={s['n']} corr(base,pts)={s['corr_base_points']:.3f} "
            f"mean_resid={s['mean_resid']:.3f}"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/player_baseline.csv`",
        "",
        "Baseline = expanding mean of prior total_points; z = residual / prior resid std.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline(shares_path: Path | None = None) -> dict[str, Any]:
    df = pd.read_csv(shares_path or (PROCESSED / "player_shares.csv"))
    out = add_baseline_residual(df)
    out.to_csv(PROCESSED / "player_baseline.csv", index=False)
    stats = residual_lift(out)
    write_stage4_report(REPORTS / "stage_4_baseline.md", stats)
    return stats
