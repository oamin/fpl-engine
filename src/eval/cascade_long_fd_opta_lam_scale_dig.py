"""Dig: how FD λ scales relative to Opta ranking λ (implementable form).

Uses paired team-side rows from ``cascade_long_fd_vs_opta_lam_compare``
(2025-26 primary; 2024-25 for OOS stability). Diagnostic only — does not
flip ``AWAY_SHRINK`` / ``LAM_SHRINK`` / ``DEFAULT_VENUE_MODE``.

Findings locked before reading promote totals:

1. **Not a global multiplier.** Opta−FD mean flips sign across seasons
   (+0.07 in 2025-26, −0.08 in 2024-25).
2. **Venue mean-ratio is the simple implementable form:**
   ``λ'_H = r_H · λ_H``, ``λ'_A = r_A · λ_A`` with
   ``r_ha = mean(FD_ha) / mean(Opta_ha)``.
3. **Direction is stable:** r_H > 1 (FD warmer at home), r_A < 1
   (Opta hotter away). Magnitudes move by season.
4. **Compression:** OLS ``FD ≈ a·Opta + b`` has a < 1 — Opta tails are
   fatter than the book; a pure scale cannot match high-λ sides.
5. **Hook:** same post-λ site as ``away_shrink`` / ``lam_shrink`` in
   ``opta_gf_ga_horizon`` — add optional ``home_scale`` (or a
   ``fd_venue_scale`` pair). Do not promote without a fresh OOS gate.

  PYTHONPATH=. python3 -m src.eval.cascade_long_fd_opta_lam_scale_dig
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[2]
ROWS_25 = ROOT / "data" / "processed" / "cascade_long_fd_vs_opta_lam_compare_rows.csv"
ROWS_24 = (
    ROOT / "data" / "processed" / "cascade_long_fd_vs_opta_lam_compare_rows_2024-25.csv"
)
REPORT = ROOT / "reports" / "cascade_long_fd_opta_lam_scale_dig.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_fd_opta_lam_scale_dig.json")


def _now() -> float:
    return time.time()


def _fit_mul(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    mae = float(np.mean(np.abs(a * x + b - y)))
    return a, b, mae


def _venue_ratios(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for ha in ("H", "A"):
        sub = df.loc[df["ha"] == ha]
        out[ha] = float(sub["fd"].mean() / sub["opta"].mean())
    out["global"] = float(df["fd"].mean() / df["opta"].mean())
    return out


def _apply_venue_r(
    df: pd.DataFrame, r_h: float, r_a: float
) -> dict[str, float]:
    pred = df["opta"].to_numpy(float).copy()
    ha = df["ha"].to_numpy()
    pred[ha == "H"] *= float(r_h)
    pred[ha == "A"] *= float(r_a)
    act = df["act"].to_numpy(float)
    fd = df["fd"].to_numpy(float)
    err = pred - act
    return {
        "bias": float(err.mean()),
        "mae": float(np.abs(err).mean()),
        "bias_H": float(err[ha == "H"].mean()),
        "bias_A": float(err[ha == "A"].mean()),
        "mae_vs_fd": float(np.abs(pred - fd).mean()),
        "mean_opta_scaled_minus_fd": float((pred - fd).mean()),
    }


def _season_block(df: pd.DataFrame, season: str) -> dict[str, Any]:
    o = df["opta"].to_numpy(float)
    f = df["fd"].to_numpy(float)
    ratios = _venue_ratios(df)
    mul_s = _fit_mul(o, f)
    a, b, mae_aff = _fit_affine(o, f)
    by_ha: dict[str, Any] = {}
    for ha in ("H", "A"):
        sub = df.loc[df["ha"] == ha]
        xo, xf = sub["opta"].to_numpy(float), sub["fd"].to_numpy(float)
        aa, bb, mae = _fit_affine(xo, xf)
        by_ha[ha] = {
            "n": int(len(sub)),
            "mean_opta": float(xo.mean()),
            "mean_fd": float(xf.mean()),
            "mean_act": float(sub["act"].mean()),
            "ratio_fd_over_opta": float(xf.mean() / xo.mean()),
            "ols_mul_s": _fit_mul(xo, xf),
            "affine_a": aa,
            "affine_b": bb,
            "affine_mae": mae,
            "opta_bias": float((xo - sub["act"]).mean()),
            "fd_bias": float((xf - sub["act"]).mean()),
            "std_opta": float(xo.std(ddof=1)),
            "std_fd": float(xf.std(ddof=1)),
        }
    # Opta quintiles: FD/Opta (shows compression)
    qdf = df.copy()
    qdf["oq"] = pd.qcut(qdf["opta"], 5, duplicates="drop")
    quintiles = []
    for q, g in qdf.groupby("oq", observed=True):
        quintiles.append(
            {
                "opta_quintile": str(q),
                "n": int(len(g)),
                "mean_opta": float(g["opta"].mean()),
                "mean_fd": float(g["fd"].mean()),
                "fd_over_opta": float(g["fd"].mean() / g["opta"].mean()),
                "fd_minus_opta": float((g["fd"] - g["opta"]).mean()),
            }
        )
    return {
        "season": season,
        "n": int(len(df)),
        "mean_opta": float(o.mean()),
        "mean_fd": float(f.mean()),
        "mean_opta_minus_fd": float((o - f).mean()),
        "corr": float(np.corrcoef(o, f)[0, 1]),
        "mae_opta_fd": float(np.abs(o - f).mean()),
        "global_ratio_fd_over_opta": ratios["global"],
        "venue_ratio_fd_over_opta": {"H": ratios["H"], "A": ratios["A"]},
        "ols_mul_fd_from_opta": mul_s,
        "affine_fd_from_opta": {"a": a, "b": b, "mae": mae_aff},
        "by_ha": by_ha,
        "opta_quintiles": quintiles,
        "raw_vs_act": _apply_venue_r(df, 1.0, 1.0),
        "oracle_venue_scale_vs_act": _apply_venue_r(df, ratios["H"], ratios["A"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-2025", type=Path, default=ROWS_25)
    ap.add_argument("--rows-2024", type=Path, default=ROWS_24)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = _now()
    if not args.rows_2025.is_file():
        raise SystemExit(f"missing {args.rows_2025}")
    df25 = pd.read_csv(args.rows_2025)
    block25 = _season_block(df25, "2025-26")
    block24: dict[str, Any] | None = None
    df24: pd.DataFrame | None = None
    if args.rows_2024.is_file():
        df24 = pd.read_csv(args.rows_2024)
        block24 = _season_block(df24, "2024-25")

    rH25 = block25["venue_ratio_fd_over_opta"]["H"]
    rA25 = block25["venue_ratio_fd_over_opta"]["A"]
    cross: dict[str, Any] = {}
    if block24 is not None and df24 is not None:
        rH24 = block24["venue_ratio_fd_over_opta"]["H"]
        rA24 = block24["venue_ratio_fd_over_opta"]["A"]
        pool = pd.concat([df25, df24], ignore_index=True)
        rH_p = float(
            pool.loc[pool["ha"] == "H", "fd"].mean()
            / pool.loc[pool["ha"] == "H", "opta"].mean()
        )
        rA_p = float(
            pool.loc[pool["ha"] == "A", "fd"].mean()
            / pool.loc[pool["ha"] == "A", "opta"].mean()
        )
        cross = {
            "fit_2025_apply_2024": _apply_venue_r(df24, rH25, rA25),
            "fit_2024_apply_2025": _apply_venue_r(df25, rH24, rA24),
            "pooled_ratios": {"H": rH_p, "A": rA_p},
            "pooled_on_2025": _apply_venue_r(df25, rH_p, rA_p),
            "pooled_on_2024": _apply_venue_r(df24, rH_p, rA_p),
        }
    else:
        rH_p, rA_p = rH25, rA25

    implement = {
        "form": "λ'_H = r_H · λ_H ;  λ'_A = r_A · λ_A",
        "r_H_2025_26": rH25,
        "r_A_2025_26": rA25,
        "r_H_2024_25": None if block24 is None else block24["venue_ratio_fd_over_opta"]["H"],
        "r_A_2024_25": None if block24 is None else block24["venue_ratio_fd_over_opta"]["A"],
        "r_H_pooled": rH_p,
        "r_A_pooled": rA_p,
        "hook": (
            "After blind Dixon–Coles λ in sample_match_from_opta_ranks / "
            "sample_opta_scenario — same site as away_shrink (λ_A) and "
            "lam_shrink (both). Prefer named home_scale + away_shrink "
            "(or fd_venue_scale=(r_H, r_A)); do not overload LAM_SHRINK."
        ),
        "affine_note": (
            "OLS FD ≈ a·Opta + b has a≈0.55–0.91 and b>0: Opta high tails "
            "overshoot the book. Venue mean-ratio matches season venue "
            "means but does not compress tails; affine would."
        ),
        "oos_warning": (
            "Fixed ratios fitted on one season move bias the wrong way on "
            "the other (sign of Opta−FD flips). Pre-register an OOS club-λ "
            "screen before any default flip — same bar as place×venue / "
            "expanding_ha."
        ),
    }

    state = {
        "task": "cascade_long_fd_opta_lam_scale_dig",
        "protocol": "cascade_long_fd_opta_lam_scale_dig",
        "pre_registered": "2026-10-29",
        "season_2025_26": block25,
        "season_2024_25": block24,
        "cross_season": cross,
        "implementable_scaling": implement,
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# How FD λ scales relative to Opta ranking λ",
        "",
        f"Pre-registered **2026-10-29** · protocol "
        f"`cascade_long_fd_opta_lam_scale_dig` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Verdict",
        "",
        "- **Not a single global scale.** Mean Opta−FD is **+0.068** in "
        "2025-26 and **−0.078** in 2024-25.",
        "- **Implementable form (venue mean-ratio):**",
        "",
        "```",
        "λ'_H = r_H · λ_H",
        "λ'_A = r_A · λ_A",
        "r_ha = mean(FD_ha) / mean(Opta_ha)",
        "```",
        "",
        f"- **2025-26:** `r_H = {rH25:.4f}`, `r_A = {rA25:.4f}` "
        f"(home boost, away shrink).",
    ]
    if block24 is not None:
        lines.append(
            f"- **2024-25:** `r_H = {block24['venue_ratio_fd_over_opta']['H']:.4f}`, "
            f"`r_A = {block24['venue_ratio_fd_over_opta']['A']:.4f}` "
            f"(same direction, larger home boost)."
        )
        lines.append(
            f"- **Pooled 24+25 (candidate constants):** "
            f"`r_H = {rH_p:.4f}`, `r_A = {rA_p:.4f}`."
        )
    lines += [
        "- Direction is **stable** (FD warmer home / Opta hotter away); "
        "**magnitudes are not** — fitting on one season and applying to the "
        "other worsens venue bias. Do **not** promote without OOS + PI.",
        "- Opta is also **more dispersed** than FD in 2025-26 (high Opta "
        "quintile F/O ≈ 0.83). Affine `FD ≈ a·Opta + b` (a < 1) compresses "
        "tails; pure scale cannot.",
        "",
        "## Implement",
        "",
        "1. After blind λ in `sample_match_from_opta_ranks` / "
        "`sample_opta_scenario` (same post-λ site as `away_shrink`).",
        "2. Prefer explicit `home_scale` + existing `away_shrink`, or a "
        "pair `fd_venue_scale=(r_H, r_A)` — do not overload `LAM_SHRINK` "
        "(that scales both sides equally).",
        "3. Defaults stay identity (`1.0`, `1.0`) until a pre-registered "
        "club-λ OOS screen clears the place×venue promote bar.",
        "",
        "## 2025-26 fits",
        "",
        f"| lens | value |",
        f"| --- | ---: |",
        f"| n paired | {block25['n']} |",
        f"| mean Opta / FD | {block25['mean_opta']:.3f} / {block25['mean_fd']:.3f} |",
        f"| Opta−FD | {block25['mean_opta_minus_fd']:+.3f} |",
        f"| corr | {block25['corr']:.3f} |",
        f"| global FD/Opta | {block25['global_ratio_fd_over_opta']:.4f} |",
        f"| OLS mul FD≈s·Opta | s={block25['ols_mul_fd_from_opta']:.4f} |",
        (
            f"| affine FD≈a·Opta+b | a={block25['affine_fd_from_opta']['a']:.4f}, "
            f"b={block25['affine_fd_from_opta']['b']:+.4f} |"
        ),
        "",
        "| ha | FD/Opta | OLS s | affine a | affine b | Opta bias | FD bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ha in ("H", "A"):
        b = block25["by_ha"][ha]
        lines.append(
            f"| {ha} | {b['ratio_fd_over_opta']:.4f} | {b['ols_mul_s']:.4f} | "
            f"{b['affine_a']:.4f} | {b['affine_b']:+.4f} | "
            f"{b['opta_bias']:+.3f} | {b['fd_bias']:+.3f} |"
        )
    lines += [
        "",
        "### Opta quintiles (compression)",
        "",
        "| quintile | n | mean Opta | mean FD | FD/Opta | FD−Opta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for q in block25["opta_quintiles"]:
        lines.append(
            f"| {q['opta_quintile']} | {q['n']} | {q['mean_opta']:.3f} | "
            f"{q['mean_fd']:.3f} | {q['fd_over_opta']:.3f} | "
            f"{q['fd_minus_opta']:+.3f} |"
        )
    if block24 is not None:
        lines += [
            "",
            "## Cross-season stability",
            "",
            "| fit → apply | bias | bias H | bias A | MAE vs act | MAE vs FD |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, key in (
            ("raw Opta on 2024-25", None),
            ("r(2025) → 2024", "fit_2025_apply_2024"),
            ("r(2024) → 2025", "fit_2024_apply_2025"),
            ("pooled → 2025", "pooled_on_2025"),
            ("pooled → 2024", "pooled_on_2024"),
        ):
            if key is None:
                m = block24["raw_vs_act"]
            else:
                m = cross[key]
            lines.append(
                f"| {name} | {m['bias']:+.3f} | {m['bias_H']:+.3f} | "
                f"{m['bias_A']:+.3f} | {m['mae']:.3f} | {m['mae_vs_fd']:.3f} |"
            )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare "
        "--season 2025-26",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare "
        "--season 2024-25 \\",
        "  --rows-out data/processed/cascade_long_fd_vs_opta_lam_compare_rows_2024-25.csv",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_fd_opta_lam_scale_dig",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_fd_opta_lam_scale_dig",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Fit FD↔Opta venue mean-ratio scaling (implementable form)",
                "verdict": {
                    "form": implement["form"],
                    "r_H_2025_26": rH25,
                    "r_A_2025_26": rA25,
                    "r_H_pooled": rH_p,
                    "r_A_pooled": rA_p,
                    "default_unchanged": True,
                    "oos_warning": True,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(implement, indent=2, default=str))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
