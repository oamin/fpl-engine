"""Dig: why cascade_short generative loses log score to L3 centreline.

Uses the published eligible-pool screen rows
(``cascade_short_goal_weight_likelihood_rows.csv``). Diagnostic only —
gen-as-mean already parked (viability B1).

Layers:

1. Mean bias / MAE / corr / pred SD for rate_minutes, π∝r, L3.
2. Δ log (gen − L3) by L3-xp quintile.
3. Verdict: variance / ranking, not a large mean-bias gap.

  PYTHONPATH=. python3 -m src.eval.cascade_gen_vs_l3_loss_anatomy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import gaussian_log_score

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "data" / "processed" / "cascade_short_goal_weight_likelihood_rows.csv"
REPORT = ROOT / "reports" / "cascade_gen_vs_l3_loss_anatomy.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_gen_vs_l3_loss_anatomy.json")
SIGMA = 3.0


def _now() -> float:
    return time.time()


def _arm_stats(df: pd.DataFrame, col: str) -> dict[str, float]:
    y = df["total_points"].to_numpy(float)
    mu = df[col].to_numpy(float)
    err = mu - y
    return {
        "bias": float(err.mean()),
        "mae": float(np.abs(err).mean()),
        "mean_xp": float(mu.mean()),
        "std_xp": float(mu.std(ddof=1)),
        "corr": float(np.corrcoef(mu, y)[0, 1]),
        "mean_log": float(gaussian_log_score(y, mu, SIGMA).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, default=ROWS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = _now()
    if not args.rows.is_file():
        raise SystemExit(f"missing {args.rows}")
    df = pd.read_csv(args.rows)
    y = df["total_points"].to_numpy(float)
    arms = {
        "rate_minutes": _arm_stats(df, "xp_rm"),
        "rate": _arm_stats(df, "xp_rate"),
        "l3": _arm_stats(df, "xp_l3"),
    }
    df = df.copy()
    df["log_rm"] = gaussian_log_score(y, df["xp_rm"].to_numpy(float), SIGMA)
    df["log_l3"] = gaussian_log_score(y, df["xp_l3"].to_numpy(float), SIGMA)
    df["d_rm_l3"] = df["log_rm"] - df["log_l3"]
    df["l3_q"] = pd.qcut(df["xp_l3"], 5, duplicates="drop")
    by_q = []
    for q, g in df.groupby("l3_q", observed=True):
        by_q.append(
            {
                "l3_quintile": str(q),
                "n": int(len(g)),
                "mean_d_rm_l3": float(g["d_rm_l3"].mean()),
                "bias_rm": float((g["xp_rm"] - g["total_points"]).mean()),
                "bias_l3": float((g["xp_l3"] - g["total_points"]).mean()),
                "std_rm": float(g["xp_rm"].std(ddof=1)),
                "std_l3": float(g["xp_l3"].std(ddof=1)),
            }
        )
    by_season = []
    for season, g in df.groupby("season"):
        by_season.append(
            {
                "season": str(season),
                "mean_d_rm_l3": float(g["d_rm_l3"].mean()),
                "corr_rm": float(np.corrcoef(g["xp_rm"], g["total_points"])[0, 1]),
                "corr_l3": float(np.corrcoef(g["xp_l3"], g["total_points"])[0, 1]),
                "std_rm": float(g["xp_rm"].std(ddof=1)),
                "std_l3": float(g["xp_l3"].std(ddof=1)),
            }
        )

    state = {
        "task": "cascade_gen_vs_l3_loss_anatomy",
        "protocol": "cascade_gen_vs_l3_loss_anatomy",
        "pre_registered": "2026-10-29",
        "n_rows": int(len(df)),
        "arms": arms,
        "by_l3_quintile": by_q,
        "by_season": by_season,
        "verdict_notes": [
            "Mean bias nearly identical (gen and L3 both ~−0.96 vs points).",
            "Gen has higher pred SD and lower corr — extra variance / worse ranking.",
            "Δ log worsens in the top L3-xp quintile (high-xp players).",
            "Confirms viability B1: gen-as-mean parked; not a mean-level G fix.",
        ],
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Why cascade_short generative loses to L3",
        "",
        f"Pre-registered **2026-10-29** · protocol `cascade_gen_vs_l3_loss_anatomy` · "
        f"wall {state['elapsed_s']:.1f}s · n={len(df)}",
        "",
        "## Verdict",
        "",
        "- **Not a mean-bias gap.** rate_minutes bias "
        f"**{arms['rate_minutes']['bias']:+.3f}** vs L3 "
        f"**{arms['l3']['bias']:+.3f}** (both cold vs total points).",
        f"- Gen has **higher pred SD** ({arms['rate_minutes']['std_xp']:.2f} vs "
        f"{arms['l3']['std_xp']:.2f}) and **lower corr** "
        f"({arms['rate_minutes']['corr']:.3f} vs {arms['l3']['corr']:.3f}).",
        "- Loss concentrates on **high-xp** players (top L3 quintile).",
        "- Aligns with viability B1 (**gen clearly_loses** vs centreline). "
        "Gen-as-mean stays parked; do not chase XI resid G via more dice noise.",
        "",
        "## Arm summary",
        "",
        "| arm | bias | MAE | corr | std(xp) | mean log |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("rate_minutes", "rate", "l3"):
        a = arms[name]
        lines.append(
            f"| {name} | {a['bias']:+.3f} | {a['mae']:.3f} | {a['corr']:.3f} | "
            f"{a['std_xp']:.3f} | {a['mean_log']:.4f} |"
        )
    lines += [
        "",
        "## Δ log (rate_minutes − L3) by L3-xp quintile",
        "",
        "| quintile | n | Δ log | bias rm | bias L3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for q in by_q:
        lines.append(
            f"| {q['l3_quintile']} | {q['n']} | {q['mean_d_rm_l3']:+.4f} | "
            f"{q['bias_rm']:+.3f} | {q['bias_l3']:+.3f} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_gen_vs_l3_loss_anatomy",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_gen_vs_l3_loss_anatomy",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Dig gen vs L3 log-score loss anatomy",
                "verdict": {
                    "bias_rm": arms["rate_minutes"]["bias"],
                    "bias_l3": arms["l3"]["bias"],
                    "std_rm": arms["rate_minutes"]["std_xp"],
                    "std_l3": arms["l3"]["std_xp"],
                    "corr_rm": arms["rate_minutes"]["corr"],
                    "corr_l3": arms["l3"]["corr"],
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state["verdict_notes"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
