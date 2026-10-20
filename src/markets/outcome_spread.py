"""Per-player predictive outcome spread (σ_out) — not estimation SE.

``xp_se`` = uncertainty of μ → EB shrink only.
``sigma_out`` = SD of points around μ → scenario idio noise only.

As-of expanding stats from ``score_xp_v2`` residuals (GW < t, minutes≥60):

- sigma_out: expanding residual SD, shrunk toward position pooled SD until n≥N_FULL
- p90_pts: expanding 90th percentile of total_points (ceiling diagnostic)
- blank_rate: expanding share of weeks with total_points ≤ BLANK_THRESHOLD
- haul_rate: expanding share of weeks with total_points ≥ HAUL_THRESHOLD

  python -m src.markets.outcome_spread
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

MIN_MINUTES = 60.0
BLANK_THRESHOLD = 2.0  # ≤2 pts counts as blank (when played ≥60')
HAUL_THRESHOLD = 10.0
N_FULL = 6  # full weight on player SD once this many prior ≥60' residuals
SIGMA_FLOOR = 0.5
SIGMA_CEIL = 8.0
DEFAULT_POS_SIGMA = {"GKP": 2.7, "DEF": 3.0, "MID": 3.1, "FWD": 3.7}


def _player_key(df: pd.DataFrame) -> pd.Series:
    """Season-scoped player id so priors do not leak across years."""
    return df["season"].astype(str) + ":" + df["element"].astype(str)


def expand_outcome_spread(
    v2: pd.DataFrame,
    *,
    min_minutes: float = MIN_MINUTES,
    blank_threshold: float = BLANK_THRESHOLD,
    haul_threshold: float = HAUL_THRESHOLD,
    n_full: int = N_FULL,
) -> pd.DataFrame:
    """Add as-of sigma_out / p90 / blank_rate / haul_rate on unique element×fixture rows.

    Uses only prior GWs (shift-1 expanding). Same-GW points never enter the prior.
    μ in this module is always raw ``score_xp_v2`` (non-shrunk).
    """
    df = v2.copy()
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["score_xp_v2"] = pd.to_numeric(df["score_xp_v2"], errors="coerce")
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    df = df.dropna(subset=["total_points", "score_xp_v2", "gw", "element", "season"])
    df = df.drop_duplicates(["season", "fixture_id", "element"], keep="first")
    df = df.sort_values(["season", "element", "gw", "fixture_id"], kind="mergesort")
    df = df.reset_index(drop=True)
    df["pkey"] = _player_key(df)
    df["resid"] = df["total_points"] - df["score_xp_v2"]
    df["eligible"] = df["minutes"] >= float(min_minutes)

    # Position pooled residual SD on eligible rows (full-sample cold start only;
    # each player's path is still shift-1 / as-of).
    pos_sigma: dict[str, float] = {}
    for pos, g in df.loc[df["eligible"]].groupby("position"):
        s = float(g["resid"].std(ddof=1)) if len(g) > 2 else float("nan")
        if not np.isfinite(s) or s <= 0:
            s = float(DEFAULT_POS_SIGMA.get(str(pos), 3.0))
        pos_sigma[str(pos)] = s

    sigma_out = np.full(len(df), np.nan)
    p90_pts = np.full(len(df), np.nan)
    blank_rate = np.full(len(df), np.nan)
    haul_rate = np.full(len(df), np.nan)
    n_out = np.zeros(len(df), dtype=int)
    pos_arr = df["position"].astype(str).to_numpy()
    elig = df["eligible"].to_numpy()
    resid = df["resid"].to_numpy(float)
    pts = df["total_points"].to_numpy(float)

    for _, ix in df.groupby("pkey", sort=False).indices.items():
        r_hist: list[float] = []
        p_hist: list[float] = []
        for row_i in ix:
            n = len(r_hist)
            n_out[row_i] = n
            pos = pos_arr[row_i]
            prior_s = pos_sigma.get(pos, DEFAULT_POS_SIGMA.get(pos, 3.0))
            if n >= 2:
                player_s = float(np.std(r_hist, ddof=1))
                w = min(1.0, n / float(n_full))
                s = w * player_s + (1.0 - w) * prior_s
            elif n == 1:
                s = 0.3 * abs(r_hist[0]) + 0.7 * prior_s
            else:
                s = prior_s
            sigma_out[row_i] = float(np.clip(s, SIGMA_FLOOR, SIGMA_CEIL))
            if n >= 1:
                arr = np.asarray(p_hist, dtype=float)
                p90_pts[row_i] = float(np.quantile(arr, 0.90))
                blank_rate[row_i] = float(np.mean(arr <= blank_threshold))
                haul_rate[row_i] = float(np.mean(arr >= haul_threshold))
            else:
                p90_pts[row_i] = float("nan")
                blank_rate[row_i] = float("nan")
                haul_rate[row_i] = float("nan")
            if elig[row_i]:
                r_hist.append(float(resid[row_i]))
                p_hist.append(float(pts[row_i]))

    out = df.copy()
    out["mu"] = out["score_xp_v2"]  # non-shrunk alias
    out["sigma_out"] = sigma_out
    out["p90_pts"] = p90_pts
    out["blank_rate"] = blank_rate
    out["haul_rate"] = haul_rate
    out["n_out"] = n_out
    out["pos_sigma_prior"] = out["position"].astype(str).map(pos_sigma)
    return out


def summarise_spread(spread: pd.DataFrame) -> dict[str, Any]:
    """Quick closed-season diagnostics (not a forecast gate)."""
    sub = spread.loc[spread["n_out"] >= 3].copy()
    by_pos = (
        sub.groupby("position", sort=True)
        .agg(
            n=("sigma_out", "size"),
            mean_sigma_out=("sigma_out", "mean"),
            median_sigma_out=("sigma_out", "median"),
            mean_p90=("p90_pts", "mean"),
            mean_blank_rate=("blank_rate", "mean"),
            mean_haul_rate=("haul_rate", "mean"),
        )
        .reset_index()
    )
    # Illustrative: high vs low sigma_out within MID (same μ band)
    mid = sub.loc[sub["position"] == "MID"].copy()
    mid["mu"] = mid["score_xp_v2"]
    band = mid.loc[(mid["mu"] >= 3.5) & (mid["mu"] <= 5.5)]
    if len(band) > 40:
        lo = band["sigma_out"].quantile(0.2)
        hi = band["sigma_out"].quantile(0.8)
        low_s = band.loc[band["sigma_out"] <= lo]
        high_s = band.loc[band["sigma_out"] >= hi]
        contrast = {
            "mu_band": [3.5, 5.5],
            "low_sigma_mean": float(low_s["sigma_out"].mean()),
            "high_sigma_mean": float(high_s["sigma_out"].mean()),
            "low_realised_sd": float(low_s["resid"].std(ddof=1)),
            "high_realised_sd": float(high_s["resid"].std(ddof=1)),
            "n_low": int(len(low_s)),
            "n_high": int(len(high_s)),
        }
    else:
        contrast = None
    return {
        "n_rows": int(len(spread)),
        "n_with_prior3": int(len(sub)),
        "by_position": by_pos.to_dict(orient="records"),
        "mid_mu_band_contrast": contrast,
        "defs": {
            "resid": "total_points − score_xp_v2",
            "eligible": f"minutes≥{int(MIN_MINUTES)}",
            "blank_threshold": BLANK_THRESHOLD,
            "haul_threshold": HAUL_THRESHOLD,
            "n_full": N_FULL,
            "mu": "score_xp_v2 (non-shrunk)",
            "note": "xp_se untouched; sigma_out / rates are draws-only",
        },
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Player outcome spread (σ_out)",
        "",
        "`xp_se` = estimation SE of μ (shrink only). "
        "`sigma_out` = predictive SD of points around μ (scenario idio only). "
        "Do not mix.",
        "",
        f"- Rows: **{summary['n_rows']}**; with ≥3 prior eligible residuals: "
        f"**{summary['n_with_prior3']}**",
        f"- Blank / haul: **≤{summary['defs']['blank_threshold']:.0f}** / "
        f"**≥{summary['defs']['haul_threshold']:.0f}** pts (played ≥60′)",
        f"- Player SD full weight at n≥**{summary['defs']['n_full']}**",
        f"- μ: **{summary['defs']['mu']}**",
        "",
        "| position | n | mean σ_out | median σ_out | mean P90 | blank rate | haul rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary["by_position"]:
        lines.append(
            f"| {r['position']} | {r['n']} | {r['mean_sigma_out']:.2f} | "
            f"{r['median_sigma_out']:.2f} | {r['mean_p90']:.2f} | "
            f"{r['mean_blank_rate']:.3f} | {r['mean_haul_rate']:.3f} |"
        )
    c = summary.get("mid_mu_band_contrast")
    if c:
        lines += [
            "",
            "## MID μ∈[3.5, 5.5]: low vs high σ_out (sanity)",
            "",
            f"- Low-σ mean σ_out **{c['low_sigma_mean']:.2f}** "
            f"(realised resid SD **{c['low_realised_sd']:.2f}**, n={c['n_low']})",
            f"- High-σ mean σ_out **{c['high_sigma_mean']:.2f}** "
            f"(realised resid SD **{c['high_realised_sd']:.2f}**, n={c['n_high']})",
            "",
            "High-σ bucket should show larger realised residual SD if σ_out ranks volatility. "
            f"Here Δ realised SD = {c['high_realised_sd'] - c['low_realised_sd']:+.2f} "
            "(modest in this μ band — σ_out is still the right draws object when "
            "player histories differ sharply).",
        ]
    lines += [
        "",
        "## Anti-conflation",
        "",
        "- `xp_se` → EB shrink only (unchanged).",
        "- `sigma_out` → scenario idio only (`draw_fixture`).",
        "- Do not subtract either from μ; do not fold σ_out into `xp_se`.",
        "",
        "Not a forecast gate.",
        "",
        f"CSV: `{(PROCESSED / 'outcome_spread_closed.csv').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", type=Path, default=CLOSED_V2)
    args = ap.parse_args(argv)

    v2 = pd.read_csv(args.v2)
    spread = expand_outcome_spread(v2)
    summary = summarise_spread(spread)
    out_csv = PROCESSED / "outcome_spread_closed.csv"
    cols = [
        "season",
        "gw",
        "element",
        "name",
        "team",
        "position",
        "fixture_id",
        "minutes",
        "total_points",
        "score_xp_v2",
        "xp_se",
        "resid",
        "mu",
        "sigma_out",
        "p90_pts",
        "blank_rate",
        "haul_rate",
        "n_out",
        "pos_sigma_prior",
    ]
    spread[cols].to_csv(out_csv, index=False)
    out_json = PROCESSED / "outcome_spread_summary.json"
    out_md = REPORTS / "outcome_spread.md"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(summary, out_md)
    print(json.dumps({"summary": summary, "csv": str(out_csv)}, indent=2, default=str))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
