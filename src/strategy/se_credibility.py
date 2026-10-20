"""Credibility checks for score_xp_v2 estimation SE + shrinkage arms.

Walk-forward within each closed season (GW ≥ 5), moments on GW < t only:

1. Calibration z-SD for raw SE and SE+σ_model.
2. Forecast arms vs actual points (Gaussian log-score σ=3):
   - none: raw μ
   - constant: position K = τ²/(τ²+mean_SE²)  [locked candidate]
   - se: player K = τ²/(τ²+SE_i²)
   - se_rescaled: one pre-registered extra arm — SE × walk-forward raw z-SD,
     then SE-dependent K. Dropped if it does not beat constant.
3. Paired GW Δlog intervals (cluster bootstrap) + leave-one-season-out.

Prior: **one mean per position** (not position × price). Constant K therefore
preserves within-position order; it mainly rescales gains vs hit/chip thresholds.

  python -m src.strategy.se_credibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.strategy.shrinkage import (
    calibration_z,
    estimate_cohort_moments,
    gaussian_log_score,
    moments_to_dict,
    shrink_mu,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED = PROCESSED / "score_xp_v2_closed.csv"
SIGMA_LOG = 3.0
GW_START = 5
N_BOOT = 1000
SEED = 0
ARMS = ("none", "constant", "se", "se_rescaled")


def load_closed() -> pd.DataFrame:
    df = pd.read_csv(CLOSED)
    df["mu"] = pd.to_numeric(df["score_xp_v2"], errors="coerce")
    df["se"] = pd.to_numeric(df["xp_se"], errors="coerce").fillna(0.0)
    df["y"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype(int)
    df["position"] = df["position"].astype(str)
    return df.dropna(subset=["mu", "y"]).copy()


def _k_constant_from_mom(mom) -> float:
    den = mom.tau2 + mom.mean_se2
    if den <= 0 or mom.tau2 <= 0:
        return 0.0
    return float(mom.tau2 / den)


def walk_forward_arms(df: pd.DataFrame) -> dict[str, Any]:
    """As-of shrinkage arms; pool metrics across seasons."""
    rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    k_rows: list[dict[str, Any]] = []
    # Expanding mean of raw z-SD within season for se_rescaled (as-of, no peek).
    z_hist: dict[str, list[float]] = {}

    for season, sdf in df.groupby("season", sort=True):
        sdf = sdf.sort_values("gw")
        z_hist[str(season)] = []
        gws = [int(g) for g in sorted(sdf["gw"].unique()) if int(g) >= GW_START]
        for gw in gws:
            hist = sdf.loc[sdf["gw"] < gw]
            cur = sdf.loc[sdf["gw"] == gw]
            if hist.empty or cur.empty:
                continue
            # Calibration scale from prior weeks only (walk-forward).
            prior_zs = z_hist[str(season)]
            se_scale = float(np.mean(prior_zs)) if prior_zs else 1.0

            moments: dict[str, Any] = {}
            preds: dict[str, np.ndarray] = {
                "none": cur["mu"].to_numpy(float),
                "constant": np.zeros(len(cur), dtype=float),
                "se": np.zeros(len(cur), dtype=float),
                "se_rescaled": np.zeros(len(cur), dtype=float),
            }
            for p, h in hist.groupby("position"):
                mom = estimate_cohort_moments(
                    h["mu"].to_numpy(float),
                    h["se"].to_numpy(float),
                    h["y"].to_numpy(float),
                )
                moments[p] = mom
                msk = cur["position"].to_numpy() == p
                if not msk.any():
                    continue
                mu_c = cur.loc[msk, "mu"].to_numpy(float)
                se_c = cur.loc[msk, "se"].to_numpy(float)
                k_c = _k_constant_from_mom(mom)
                k_rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "position": p,
                        "prior": mom.prior,
                        "tau2": mom.tau2,
                        "mean_se2": mom.mean_se2,
                        "k_constant": k_c,
                        "n_hist": mom.n,
                        "se_scale_asof": se_scale,
                    }
                )
                # K uses SE_calc only (σ_model out of K until multi-week estimate).
                preds["constant"][msk] = shrink_mu(
                    mu_c, se_c, prior=mom.prior, tau2=mom.tau2, sigma_model2=0.0,
                    mode="constant", k_constant=k_c,
                )
                preds["se"][msk] = shrink_mu(
                    mu_c, se_c, prior=mom.prior, tau2=mom.tau2, sigma_model2=0.0,
                    mode="se",
                )
                # Rescaled SE arm: moments on scaled SE for τ² consistency.
                se_h = h["se"].to_numpy(float) * se_scale
                mom_r = estimate_cohort_moments(
                    h["mu"].to_numpy(float), se_h, h["y"].to_numpy(float)
                )
                preds["se_rescaled"][msk] = shrink_mu(
                    mu_c,
                    se_c * se_scale,
                    prior=mom_r.prior,
                    tau2=mom_r.tau2,
                    sigma_model2=0.0,
                    mode="se",
                )

            y = cur["y"].to_numpy(float)
            for arm, mu_hat in preds.items():
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "arm": arm,
                        "n": int(len(y)),
                        "mse": float(np.mean((y - mu_hat) ** 2)),
                        "mae": float(np.mean(np.abs(y - mu_hat))),
                        "log_score": gaussian_log_score(y, mu_hat, sigma=SIGMA_LOG),
                    }
                )

            sm2 = float(np.mean([m.sigma_model2 for m in moments.values()])) if moments else 0.0
            cal = calibration_z(
                y, cur["mu"].to_numpy(float), cur["se"].to_numpy(float), sigma_model2=sm2
            )
            cal_raw = calibration_z(
                y, cur["mu"].to_numpy(float), cur["se"].to_numpy(float), sigma_model2=0.0
            )
            if np.isfinite(cal_raw["z_sd"]):
                z_hist[str(season)].append(float(cal_raw["z_sd"]))
            cal_rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "raw_z_sd": cal_raw["z_sd"],
                    "raw_cover_1": cal_raw["cover_1"],
                    "tot_z_sd": cal["z_sd"],
                    "tot_cover_1": cal["cover_1"],
                    "sigma_model2": sm2,
                    "se_scale_asof": se_scale,
                }
            )

    arm = pd.DataFrame(rows)
    cal = pd.DataFrame(cal_rows)
    k_df = pd.DataFrame(k_rows)
    summary_arms = (
        arm.groupby("arm", as_index=False)
        .agg(
            n=("n", "sum"),
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            log_score=("log_score", "mean"),
        )
        .sort_values("log_score", ascending=False)
    )
    none_ls = float(summary_arms.loc[summary_arms["arm"] == "none", "log_score"].iloc[0])
    summary_arms["d_log_vs_none"] = summary_arms["log_score"] - none_ls

    intervals = _paired_intervals(arm)
    loso = _loso(arm)
    k_summary = _k_summary(k_df)
    verd = _verdict(summary_arms, intervals)

    return {
        "prior_definition": (
            "one mean per position within season, estimated on GW < t only "
            "(not position × price band)"
        ),
        "k_definition": (
            "K_pos = τ²_pos / (τ²_pos + mean_SE²_pos) from GW < t; "
            "μ_post = prior + K · (μ − prior); σ_model not in K"
        ),
        "arms": summary_arms.to_dict(orient="records"),
        "arms_by_season": (
            arm.groupby(["season", "arm"], as_index=False)
            .agg(mse=("mse", "mean"), log_score=("log_score", "mean"), n=("n", "sum"))
            .to_dict(orient="records")
        ),
        "calibration": {
            "mean_raw_z_sd": float(cal["raw_z_sd"].mean()) if len(cal) else None,
            "mean_raw_cover_1": float(cal["raw_cover_1"].mean()) if len(cal) else None,
            "mean_tot_z_sd": float(cal["tot_z_sd"].mean()) if len(cal) else None,
            "mean_tot_cover_1": float(cal["tot_cover_1"].mean()) if len(cal) else None,
            "mean_sigma_model2": float(cal["sigma_model2"].mean()) if len(cal) else None,
            "n_gw": int(len(cal)),
        },
        "k_by_position": k_summary,
        "intervals": intervals,
        "loso": loso,
        "verdict": verd,
        "decision_effects": {
            "within_position_order": "unchanged under constant K (one prior per position)",
            "cross_position": "small shifts via different (prior, K) per position",
            "hit_chip": (
                "predicted transfer / chip gains shrink toward the prior; "
                "calibrate hit and chip thresholds on μ_post — do not also apply "
                "the ~0.5 decision-week haircut on top"
            ),
        },
    }


def _gw_deltas(arm: pd.DataFrame, left: str, right: str) -> dict[str, np.ndarray]:
    """Per-season arrays of (left − right) GW mean log-scores."""
    piv = arm.pivot_table(index=["season", "gw"], columns="arm", values="log_score")
    out: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if left not in block.columns or right not in block.columns:
            continue
        d = (block[left] - block[right]).dropna().to_numpy(float)
        out[season] = d
    return out


def _paired_intervals(arm: pd.DataFrame) -> dict[str, Any]:
    contrasts = [
        ("constant", "none"),
        ("se", "constant"),
        ("se_rescaled", "constant"),
        ("se", "none"),
    ]
    out: dict[str, Any] = {}
    for left, right in contrasts:
        deltas = _gw_deltas(arm, left, right)
        if set(deltas) != set(CLOSED_SEASONS):
            out[f"{left}_minus_{right}"] = {"error": "missing seasons", "seasons": list(deltas)}
            continue
        iv = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
        iv["covers_zero"] = bool(iv["lo"] <= 0.0 <= iv["hi"])
        iv["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}
        out[f"{left}_minus_{right}"] = iv
    return out


def _loso(arm: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-season-out mean Δlog (constant − none) on remaining seasons."""
    deltas = _gw_deltas(arm, "constant", "none")
    folds = []
    for held in CLOSED_SEASONS:
        kept = [s for s in CLOSED_SEASONS if s != held]
        parts = [deltas[s] for s in kept if s in deltas]
        if len(parts) < 2:
            continue
        mean = float(np.concatenate(parts).mean())
        # Bootstrap within kept seasons
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT, dtype=float)
        for b in range(N_BOOT):
            taken = [arr[rng.integers(0, arr.size, size=arr.size)] for arr in parts]
            boots[b] = float(np.concatenate(taken).mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        folds.append(
            {
                "held_out": held,
                "mean_d_log": mean,
                "lo": float(lo),
                "hi": float(hi),
                "covers_zero": bool(lo <= 0.0 <= hi),
                "n_gws": {s: int(deltas[s].size) for s in kept},
            }
        )
    return {"contrast": "constant_minus_none", "folds": folds}


def _k_summary(k_df: pd.DataFrame) -> dict[str, Any]:
    if k_df.empty:
        return {}
    by_pos = (
        k_df.groupby("position", as_index=False)
        .agg(
            mean_k=("k_constant", "mean"),
            median_k=("k_constant", "median"),
            mean_prior=("prior", "mean"),
            mean_tau2=("tau2", "mean"),
            mean_se2=("mean_se2", "mean"),
            n_gw=("gw", "count"),
        )
        .sort_values("position")
    )
    # Deploy snapshot: last GW of 2025-26 walk-forward (as-of moments)
    snap = {}
    s = k_df.loc[k_df["season"] == "2025-26"]
    if not s.empty:
        last = int(s["gw"].max())
        for _, r in s.loc[s["gw"] == last].iterrows():
            snap[str(r["position"])] = {
                "gw": last,
                "prior": float(r["prior"]),
                "tau2": float(r["tau2"]),
                "mean_se2": float(r["mean_se2"]),
                "k_constant": float(r["k_constant"]),
            }
    return {
        "walk_forward_mean": by_pos.to_dict(orient="records"),
        "snapshot_2025_26_last_gw": snap,
    }


def _verdict(summary_arms: pd.DataFrame, intervals: dict[str, Any]) -> dict[str, Any]:
    by = {r["arm"]: r for r in summary_arms.to_dict(orient="records")}
    se = by.get("se", {})
    const = by.get("constant", {})
    none = by.get("none", {})
    resc = by.get("se_rescaled", {})
    se_beats_const = float(se.get("log_score", -1e9)) > float(const.get("log_score", 0)) + 1e-6
    se_beats_none = float(se.get("log_score", -1e9)) > float(none.get("log_score", 0)) + 1e-6
    const_beats_none = float(const.get("log_score", -1e9)) > float(none.get("log_score", 0)) + 1e-6
    resc_beats_const = float(resc.get("log_score", -1e9)) > float(const.get("log_score", 0)) + 1e-6

    dropped: list[str] = []
    if not resc_beats_const:
        dropped.append("se_rescaled")
    if not se_beats_const:
        dropped.append("se")

    if resc_beats_const and float(resc.get("log_score", -1e9)) > float(se.get("log_score", 0)):
        use = "se_rescaled"
        note = "SE rescaled by walk-forward z-SD beats constant — promote that arm."
    elif se_beats_const and se_beats_none:
        use = "se"
        note = "SE-dependent K beats constant and raw μ on walk-forward log-score."
    elif const_beats_none:
        use = "constant"
        note = (
            "Constant position K beats raw μ; SE-dependent and se_rescaled add "
            "nothing beyond constant (dropped)."
        )
    else:
        use = "none"
        note = "Shrinkage does not beat raw μ on this screen — Layer 2 should pass through μ."

    iv = intervals.get("constant_minus_none", {})
    return {
        "recommended_mode": use,
        "note": note,
        "dropped_arms": dropped,
        "se_beats_constant": se_beats_const,
        "se_beats_none": se_beats_none,
        "constant_beats_none": const_beats_none,
        "se_rescaled_beats_constant": resc_beats_const,
        "constant_minus_none_interval": {
            "mean": iv.get("mean"),
            "lo": iv.get("lo"),
            "hi": iv.get("hi"),
            "covers_zero": iv.get("covers_zero"),
        },
    }


def bin_check(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Slope of y~μ within SE terciles (diagnostic)."""
    work = df.loc[df["gw"] >= GW_START].copy()
    work["se_tercile"] = pd.qcut(work["se"].rank(method="first"), 3, labels=["low", "mid", "high"])
    rows = []
    for t, g in work.groupby("se_tercile", observed=True):
        if len(g) < 50:
            continue
        x = g["mu"].to_numpy(float)
        y = g["y"].to_numpy(float)
        x = x - x.mean()
        y = y - y.mean()
        slope = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else float("nan")
        rows.append(
            {
                "se_tercile": str(t),
                "n": int(len(g)),
                "mean_se": float(g["se"].mean()),
                "slope_y_on_mu": slope,
                "corr": float(g["mu"].corr(g["y"])),
            }
        )
    return rows


def write_report(doc: dict[str, Any], path: Path) -> None:
    cal = doc["calibration"]
    verd = doc["verdict"]
    iv = doc["intervals"].get("constant_minus_none", {})
    lines = [
        "# `score_xp_v2` SE credibility + shrinkage screen",
        "",
        "Estimation SE (`xp_se`) is **not** predictive σ for points. Use it to **shrink**",
        "μ toward a position prior before the MILP — do not subtract η·SE from the objective.",
        "",
        "## Prior and K (locked method)",
        "",
        f"- **Prior:** {doc['prior_definition']}.",
        f"- **K:** {doc['k_definition']}.",
        "- Constant K leaves **within-position order unchanged**; it rescales gains vs",
        "  the −4 hit and chip thresholds. Calibrate those on μ_post; do **not** stack",
        "  the ~0.5 decision-week haircut on top.",
        "",
        "### Walk-forward mean K by position",
        "",
        "| position | mean K | mean prior | mean τ² | mean SE² |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in doc.get("k_by_position", {}).get("walk_forward_mean", []):
        lines.append(
            f"| {r['position']} | {r['mean_k']:.3f} | {r['mean_prior']:.2f} | "
            f"{r['mean_tau2']:.3f} | {r['mean_se2']:.3f} |"
        )
    snap = doc.get("k_by_position", {}).get("snapshot_2025_26_last_gw", {})
    if snap:
        lines += [
            "",
            "### Snapshot (2025-26 last walk-forward GW, as-of)",
            "",
            "| position | K | prior | τ² |",
            "| --- | ---: | ---: | ---: |",
        ]
        for p in sorted(snap):
            r = snap[p]
            lines.append(
                f"| {p} | {r['k_constant']:.3f} | {r['prior']:.2f} | {r['tau2']:.3f} |"
            )

    lines += [
        "",
        "## Calibration (walk-forward GW≥5)",
        "",
        f"- Raw SE: mean z-SD **{cal['mean_raw_z_sd']:.2f}**, 1σ coverage **{cal['mean_raw_cover_1']:.1%}**",
        f"- SE + σ_model: mean z-SD **{cal['mean_tot_z_sd']:.2f}**, 1σ coverage **{cal['mean_tot_cover_1']:.1%}**",
        f"- Mean σ_model² (from prior weeks): **{cal['mean_sigma_model2']:.3f}**",
        "",
        "Target z-SD ≈ 1. Raw SE is too narrow; single-GW residual σ_model brings z-SD≈1",
        "but is **not** folded into K. Multi-week σ_model (5–10 GW windows, Poisson noise",
        "subtracted) stays pending.",
        "",
        "## Arms (as-of; Gaussian log-score σ=3)",
        "",
        "| arm | mean log-score | Δ vs none | MSE | MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["arms"]:
        lines.append(
            f"| {r['arm']} | {r['log_score']:.5f} | {r['d_log_vs_none']:+.5f} | "
            f"{r['mse']:.3f} | {r['mae']:.3f} |"
        )

    lines += [
        "",
        "## Size and uncertainty (constant − none)",
        "",
        f"- Paired GW mean Δlog: **{iv.get('mean', float('nan')):+.5f}** "
        f"[{iv.get('lo', float('nan')):+.5f}, {iv.get('hi', float('nan')):+.5f}] "
        f"(cluster bootstrap B={N_BOOT}, seed={SEED})",
        f"- Covers zero: **{iv.get('covers_zero')}**",
        "",
        "| season | mean Δlog |",
        "| --- | ---: |",
    ]
    for s, m in (iv.get("by_season_mean") or {}).items():
        lines.append(f"| {s} | {m:+.5f} |")

    lines += [
        "",
        "### Leave-one-season-out (constant − none)",
        "",
        "| held out | mean Δlog | 95% CI | covers 0 |",
        "| --- | ---: | --- | --- |",
    ]
    for f in doc.get("loso", {}).get("folds", []):
        lines.append(
            f"| {f['held_out']} | {f['mean_d_log']:+.5f} | "
            f"[{f['lo']:+.5f}, {f['hi']:+.5f}] | {f['covers_zero']} |"
        )

    # Other intervals in brief
    lines += ["", "### Other paired intervals", ""]
    for key in ("se_minus_constant", "se_rescaled_minus_constant", "se_minus_none"):
        row = doc["intervals"].get(key, {})
        if "mean" not in row:
            continue
        lines.append(
            f"- `{key}`: {row['mean']:+.5f} [{row['lo']:+.5f}, {row['hi']:+.5f}] "
            f"(covers 0: {row['covers_zero']})"
        )

    lines += [
        "",
        f"**Verdict:** use mode `{verd['recommended_mode']}`. {verd['note']}",
        f"Dropped arms: {', '.join(verd.get('dropped_arms') or []) or 'none'}.",
        "",
        "## SE tercile slopes (y on μ)",
        "",
        "| tercile | n | mean SE | slope | corr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["bin_check"]:
        lines.append(
            f"| {r['se_tercile']} | {r['n']} | {r['mean_se']:.3f} | "
            f"{r['slope_y_on_mu']:.3f} | {r['corr']:.3f} |"
        )
    lines += [
        "",
        "## Implication for Layer 2",
        "",
        f"- Shrinkage runs **before** the solve; MILP on μ_post (`mode={verd['recommended_mode']}`).",
        "- Forecast credibility is this screen (realised points). The L2 audit is a",
        "  **solver legality smoke** on closed μ boards (as-of moments GW < t) — it does",
        "  not re-prove the forecast gain.",
        "- When transfers/chips land: μ_post everywhere in gain accounting; no double haircut.",
        "",
        f"JSON: `{(PROCESSED / 'se_credibility.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    df = load_closed()
    doc = walk_forward_arms(df)
    doc["bin_check"] = bin_check(df)
    s = df.loc[df["season"] == "2025-26"]
    moments = {}
    for p, g in s.groupby("position"):
        moments[p] = estimate_cohort_moments(
            g["mu"].to_numpy(float), g["se"].to_numpy(float), g["y"].to_numpy(float)
        )
    doc["moments_2025_26"] = moments_to_dict(moments)
    out_json = PROCESSED / "se_credibility.json"
    out_md = REPORTS / "se_credibility.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(json.dumps(doc["verdict"], indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
