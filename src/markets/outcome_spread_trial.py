"""Trial: non-shrunk μ + σ_out + blank/haul rates in a scenario-style model.

Pool: minutes≥60, n_out≥3, four closed seasons. μ = raw score_xp_v2 (not μ_post).

Arms
----
1. **Gaussian** — Y ~ N(μ, σ_out²)
2. **Rates** — predicted blank_rate / haul_rate vs realised I(Y≤2) / I(Y≥10)
3. **Mixture** — with p_b draw blank level, with p_h draw haul level, else N(μ,σ²)
   (p_b, p_h clipped so p_b+p_h≤0.95). Scenario hit rates from Monte Carlo.

Not a forecast gate / not write_gated_report. Diagnostic only.

  python -m src.markets.outcome_spread_trial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.markets.outcome_spread import (
    BLANK_THRESHOLD,
    HAUL_THRESHOLD,
    MIN_MINUTES,
    expand_outcome_spread,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

MIN_N_OUT = 3
N_DRAW = 200
BLANK_LEVEL = 1.0  # mixture blank outcome (points)
HAUL_LEVEL_FLOOR = 12.0  # mixture haul floor; else max(p90, μ+σ)


def _pool(spread: pd.DataFrame) -> pd.DataFrame:
    df = spread.copy()
    df = df.loc[
        (df["minutes"] >= MIN_MINUTES)
        & (df["n_out"] >= MIN_N_OUT)
        & np.isfinite(df["mu"])
        & np.isfinite(df["sigma_out"])
        & np.isfinite(df["blank_rate"])
        & np.isfinite(df["haul_rate"])
    ].copy()
    df["y"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["is_blank"] = (df["y"] <= BLANK_THRESHOLD).astype(float)
    df["is_haul"] = (df["y"] >= HAUL_THRESHOLD).astype(float)
    df["z"] = (df["y"] - df["mu"]) / df["sigma_out"].clip(lower=0.5)
    return df.dropna(subset=["y", "z"])


def gaussian_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    z = df["z"].to_numpy(float)
    mu = df["mu"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sig = df["sigma_out"].to_numpy(float)
    # Implied blank/haul probs under Gaussian
    p_blank_g = norm.cdf(BLANK_THRESHOLD, loc=mu, scale=sig)
    p_haul_g = 1.0 - norm.cdf(HAUL_THRESHOLD - 1e-9, loc=mu, scale=sig)
    return {
        "n": int(len(df)),
        "mu_bias": float(np.mean(y - mu)),
        "mu_mae": float(np.mean(np.abs(y - mu))),
        "z_sd": float(np.std(z, ddof=1)),
        "coverage_1sd": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_2sd": float(np.mean(np.abs(z) <= 2.0)),
        "gaussian_mean_p_blank": float(np.mean(p_blank_g)),
        "realised_blank_rate": float(df["is_blank"].mean()),
        "gaussian_mean_p_haul": float(np.mean(p_haul_g)),
        "realised_haul_rate": float(df["is_haul"].mean()),
        "blank_brier_gaussian": float(np.mean((p_blank_g - df["is_blank"].to_numpy()) ** 2)),
        "haul_brier_gaussian": float(np.mean((p_haul_g - df["is_haul"].to_numpy()) ** 2)),
    }


def rate_calibration(df: pd.DataFrame, *, col: str, outcome: str, n_bins: int = 10) -> dict[str, Any]:
    """Reliability of as-of blank_rate / haul_rate (deciles of predicted rate)."""
    sub = df[[col, outcome]].dropna().copy()
    try:
        sub["bin"] = pd.qcut(sub[col], n_bins, duplicates="drop")
    except ValueError:
        return {"n": int(len(sub)), "bins": [], "error": "qcut_failed"}
    rows = []
    for b, g in sub.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(b),
                "n": int(len(g)),
                "pred_mean": float(g[col].mean()),
                "realised_rate": float(g[outcome].mean()),
            }
        )
    pred = sub[col].to_numpy(float)
    y = sub[outcome].to_numpy(float)
    return {
        "n": int(len(sub)),
        "brier": float(np.mean((pred - y) ** 2)),
        "base_rate": float(y.mean()),
        "pred_mean": float(pred.mean()),
        "bins": rows,
    }


def mixture_scenarios(df: pd.DataFrame, *, n_draw: int = N_DRAW, seed: int = 0) -> dict[str, Any]:
    """Monte Carlo mixture vs Gaussian: mean error and blank/haul hit rates."""
    rng = np.random.default_rng(seed)
    mu = df["mu"].to_numpy(float)
    sig = df["sigma_out"].to_numpy(float)
    p_b = df["blank_rate"].to_numpy(float)
    p_h = df["haul_rate"].to_numpy(float)
    p90 = df["p90_pts"].to_numpy(float)
    y = df["y"].to_numpy(float)
    n = len(df)

    # Clip joint tail mass
    p_b = np.clip(p_b, 0.0, 0.95)
    p_h = np.clip(p_h, 0.0, 0.95)
    overflow = p_b + p_h > 0.95
    scale = np.where(overflow, 0.95 / np.maximum(p_b + p_h, 1e-9), 1.0)
    p_b = p_b * scale
    p_h = p_h * scale
    haul_level = np.maximum(np.nan_to_num(p90, nan=HAUL_LEVEL_FLOOR), HAUL_LEVEL_FLOOR)
    haul_level = np.maximum(haul_level, mu + sig)

    # Gaussian draws
    g = rng.normal(mu.reshape(-1, 1), sig.reshape(-1, 1), size=(n, n_draw))
    # Mixture
    u = rng.random((n, n_draw))
    m = rng.normal(mu.reshape(-1, 1), sig.reshape(-1, 1), size=(n, n_draw))
    blank_m = u < p_b.reshape(-1, 1)
    haul_m = (~blank_m) & (u < (p_b + p_h).reshape(-1, 1))
    m = np.where(blank_m, BLANK_LEVEL, m)
    m = np.where(haul_m, haul_level.reshape(-1, 1), m)

    def _summ(draws: np.ndarray, label: str) -> dict[str, float]:
        mean_draw = draws.mean(axis=1)
        p_blank = (draws <= BLANK_THRESHOLD).mean(axis=1)
        p_haul = (draws >= HAUL_THRESHOLD).mean(axis=1)
        return {
            f"{label}_mean_bias": float(np.mean(y - mean_draw)),
            f"{label}_mae_vs_y": float(np.mean(np.abs(y - mean_draw))),
            f"{label}_blank_rate_pred": float(p_blank.mean()),
            f"{label}_haul_rate_pred": float(p_haul.mean()),
            f"{label}_blank_brier": float(np.mean((p_blank - df["is_blank"].to_numpy()) ** 2)),
            f"{label}_haul_brier": float(np.mean((p_haul - df["is_haul"].to_numpy()) ** 2)),
        }

    out = {"n": n, "n_draw": n_draw}
    out.update(_summ(g, "gaussian"))
    out.update(_summ(m, "mixture"))
    out["realised_blank_rate"] = float(df["is_blank"].mean())
    out["realised_haul_rate"] = float(df["is_haul"].mean())
    out["empirical_blank_rate_mean"] = float(df["blank_rate"].mean())
    out["empirical_haul_rate_mean"] = float(df["haul_rate"].mean())
    return out


def by_season(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for season, g in df.groupby("season", sort=True):
        gd = gaussian_diagnostics(g)
        mix = mixture_scenarios(g, seed=hash(season) % 10_000)
        rows.append(
            {
                "season": season,
                "n": gd["n"],
                "mu_bias": gd["mu_bias"],
                "z_sd": gd["z_sd"],
                "coverage_1sd": gd["coverage_1sd"],
                "blank_brier_rates": float(
                    np.mean((g["blank_rate"] - g["is_blank"]) ** 2)
                ),
                "haul_brier_rates": float(np.mean((g["haul_rate"] - g["is_haul"]) ** 2)),
                "mixture_blank_brier": mix["mixture_blank_brier"],
                "gaussian_blank_brier": mix["gaussian_blank_brier"],
                "mixture_haul_brier": mix["mixture_haul_brier"],
                "gaussian_haul_brier": mix["gaussian_haul_brier"],
            }
        )
    return rows


def write_report(doc: dict[str, Any], path: Path) -> None:
    g = doc["gaussian"]
    mix = doc["mixture"]
    br = doc["blank_rates"]
    hr = doc["haul_rates"]
    lines = [
        "# Outcome model trial — μ (raw) + σ_out + blank/haul",
        "",
        "Diagnostic only (not a gated forecast screen). Pool: minutes≥60, "
        f"n_out≥{MIN_N_OUT}, μ = **score_xp_v2** (non-shrunk).",
        "",
        f"- Blank = points ≤ **{BLANK_THRESHOLD:.0f}**; haul = points ≥ **{HAUL_THRESHOLD:.0f}**",
        f"- Mixture blank level = {BLANK_LEVEL:.0f}; haul level = max(P90, {HAUL_LEVEL_FLOOR:.0f}, μ+σ)",
        "",
        "## 1. Non-shrunk μ + Gaussian σ_out",
        "",
        f"- n = **{g['n']}**",
        f"- μ bias (y−μ): **{g['mu_bias']:+.3f}**; MAE **{g['mu_mae']:.3f}**",
        f"- z-SD: **{g['z_sd']:.3f}** (target ≈1); 1σ coverage **{g['coverage_1sd']:.1%}**; "
        f"2σ **{g['coverage_2sd']:.1%}**",
        f"- Gaussian mean P(blank) **{g['gaussian_mean_p_blank']:.3f}** vs realised "
        f"**{g['realised_blank_rate']:.3f}**",
        f"- Gaussian mean P(haul) **{g['gaussian_mean_p_haul']:.3f}** vs realised "
        f"**{g['realised_haul_rate']:.3f}**",
        "",
        "## 2. Empirical blank / haul rates (as-of)",
        "",
        f"- blank_rate: pred mean **{br['pred_mean']:.3f}**, realised **{br['base_rate']:.3f}**, "
        f"Brier **{br['brier']:.4f}**",
        f"- haul_rate: pred mean **{hr['pred_mean']:.3f}**, realised **{hr['base_rate']:.3f}**, "
        f"Brier **{hr['brier']:.4f}**",
        "",
        "### Blank-rate deciles",
        "",
        "| bin | n | pred | realised |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in br.get("bins", []):
        lines.append(
            f"| {r['bin']} | {r['n']} | {r['pred_mean']:.3f} | {r['realised_rate']:.3f} |"
        )
    lines += [
        "",
        "### Haul-rate deciles",
        "",
        "| bin | n | pred | realised |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in hr.get("bins", []):
        lines.append(
            f"| {r['bin']} | {r['n']} | {r['pred_mean']:.3f} | {r['realised_rate']:.3f} |"
        )
    lines += [
        "",
        "## 3. Scenario arms (Monte Carlo)",
        "",
        f"n_draw = {mix['n_draw']}. Lower Brier is better for blank/haul hits.",
        "",
        "| arm | mean bias | MAE | P(blank) | blank Brier | P(haul) | haul Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Realised | — | — | {mix['realised_blank_rate']:.3f} | — | "
        f"{mix['realised_haul_rate']:.3f} | — |",
        f"| Empirical rates (mean) | — | — | {mix['empirical_blank_rate_mean']:.3f} | "
        f"{br['brier']:.4f} | {mix['empirical_haul_rate_mean']:.3f} | {hr['brier']:.4f} |",
        f"| Gaussian N(μ,σ) | {mix['gaussian_mean_bias']:+.3f} | {mix['gaussian_mae_vs_y']:.3f} | "
        f"{mix['gaussian_blank_rate_pred']:.3f} | {mix['gaussian_blank_brier']:.4f} | "
        f"{mix['gaussian_haul_rate_pred']:.3f} | {mix['gaussian_haul_brier']:.4f} |",
        f"| Mixture (rates+N) | {mix['mixture_mean_bias']:+.3f} | {mix['mixture_mae_vs_y']:.3f} | "
        f"{mix['mixture_blank_rate_pred']:.3f} | {mix['mixture_blank_brier']:.4f} | "
        f"{mix['mixture_haul_rate_pred']:.3f} | {mix['mixture_haul_brier']:.4f} |",
        "",
        "## By season",
        "",
        "| season | n | μ bias | z-SD | 1σ cov | blank Brier (rates) | haul Brier (rates) | "
        "Δ blank Brier (mix−gauss) | Δ haul Brier (mix−gauss) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n']} | {r['mu_bias']:+.3f} | {r['z_sd']:.2f} | "
            f"{r['coverage_1sd']:.1%} | {r['blank_brier_rates']:.4f} | "
            f"{r['haul_brier_rates']:.4f} | "
            f"{r['mixture_blank_brier'] - r['gaussian_blank_brier']:+.4f} | "
            f"{r['mixture_haul_brier'] - r['gaussian_haul_brier']:+.4f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        doc["verdict"],
        "",
        f"JSON: `{(PROCESSED / 'outcome_spread_trial.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", type=Path, default=CLOSED_V2)
    ap.add_argument("--n-draw", type=int, default=N_DRAW)
    args = ap.parse_args(argv)

    v2 = pd.read_csv(args.v2)
    spread = expand_outcome_spread(v2)
    df = _pool(spread)
    g = gaussian_diagnostics(df)
    br = rate_calibration(df, col="blank_rate", outcome="is_blank")
    hr = rate_calibration(df, col="haul_rate", outcome="is_haul")
    mix = mixture_scenarios(df, n_draw=args.n_draw, seed=0)
    seasons = by_season(df)

    # Verdict heuristics (diagnostic language, not a gate)
    z_ok = 0.85 <= g["z_sd"] <= 1.25
    blank_rates_beat_gauss = br["brier"] < g["blank_brier_gaussian"]
    haul_rates_beat_gauss = hr["brier"] < g["haul_brier_gaussian"]
    mix_blank_better = mix["mixture_blank_brier"] < mix["gaussian_blank_brier"]
    mix_haul_better = mix["mixture_haul_brier"] < mix["gaussian_haul_brier"]
    bits = [
        f"Raw μ bias {g['mu_bias']:+.2f} pts; z-SD {g['z_sd']:.2f} "
        f"({'near 1' if z_ok else 'off target — σ_out mis-scaled'}).",
        f"Empirical blank_rate Brier {br['brier']:.4f} vs Gaussian-implied "
        f"{g['blank_brier_gaussian']:.4f} "
        f"({'rates win' if blank_rates_beat_gauss else 'Gaussian competitive'}).",
        f"Empirical haul_rate Brier {hr['brier']:.4f} vs Gaussian-implied "
        f"{g['haul_brier_gaussian']:.4f} "
        f"({'rates win' if haul_rates_beat_gauss else 'Gaussian competitive'}).",
        f"Mixture vs Gaussian blank Brier Δ "
        f"{mix['mixture_blank_brier'] - mix['gaussian_blank_brier']:+.4f}; haul Δ "
        f"{mix['mixture_haul_brier'] - mix['gaussian_haul_brier']:+.4f} "
        f"({'mixture helps tails' if (mix_blank_better or mix_haul_better) else 'mixture does not help'}).",
    ]
    verdict = " ".join(bits)

    doc = {
        "defs": {
            "mu": "score_xp_v2 non-shrunk",
            "blank": f"points≤{BLANK_THRESHOLD}",
            "haul": f"points≥{HAUL_THRESHOLD}",
            "pool": f"minutes≥{int(MIN_MINUTES)}, n_out≥{MIN_N_OUT}",
        },
        "gaussian": g,
        "blank_rates": br,
        "haul_rates": hr,
        "mixture": mix,
        "by_season": seasons,
        "verdict": verdict,
    }
    out_json = PROCESSED / "outcome_spread_trial.json"
    out_md = REPORTS / "outcome_spread_trial.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(json.dumps({"verdict": verdict, "gaussian": g, "mixture": {
        k: mix[k] for k in mix if "brier" in k or "rate" in k or k.startswith("gaussian_mean") or k.startswith("mixture_mean")
    }}, indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
