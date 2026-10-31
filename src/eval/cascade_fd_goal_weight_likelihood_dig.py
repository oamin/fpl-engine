"""Dig: cascade generative rate_minutes vs π∝r on player-GW log score.

Uses the published FD shared-L3 decision-GW maps (GW8–38):

- π∝r: ``cascade_vs_l3_component_xp_pi_r_gw8-38``
- rate_minutes: ``cascade_vs_l3_component_xp_rate_minutes_fd_gw8-38``

XI resid G already showed rate_minutes much hotter (+100 vs +14). This dig
asks whether **Gaussian log score** on the scored components still prefers
the default (as the gated ``score_xp_v2_goal_share_rate`` screen did for
centreline score_xp_v2 — clear lose for π∝r).

Metrics (diagnostic; not a four-season gated promote):

- Log score of ``casc_xP_G`` vs ``P_G`` (σ=3)
- Log score of ``casc_xP_sum`` (G+CS+GA) vs ``P_sum``
- Paired Δ by GW: rate − rate_minutes (positive ⇒ π∝r better)

Does **not** flip ``goal_weight`` default.

  PYTHONPATH=. python3 -m src.eval.cascade_fd_goal_weight_likelihood_dig
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
REPORT = ROOT / "reports" / "cascade_fd_goal_weight_likelihood_dig.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_fd_goal_weight_likelihood_dig.json")

RATE_PLAYERS = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_gw8-38_players.json"
)
RM_PLAYERS = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_rate_minutes_fd_gw8-38_players.json"
)
RATE_SUMMARY = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_gw8-38.json"
)
RM_SUMMARY = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_rate_minutes_fd_gw8-38.json"
)
SIGMA = 3.0


def _now() -> float:
    return time.time()


def _pack(players: pd.DataFrame, *, label: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    pl = players.copy()
    y_g = pl["P_G"].to_numpy(float)
    mu_g = pl["casc_xP_G"].to_numpy(float)
    y_sum = pl["P_sum"].to_numpy(float)
    mu_sum = pl["casc_xP_sum"].to_numpy(float)
    log_g = gaussian_log_score(y_g, mu_g, SIGMA)
    log_sum = gaussian_log_score(y_sum, mu_sum, SIGMA)
    pl = pl.assign(log_G=log_g, log_sum=log_sum, resid_G=pl["casc_xP_G"] - pl["P_G"])
    by_gw = (
        pl.groupby("gw", as_index=False)
        .agg(
            mean_log_G=("log_G", "mean"),
            mean_log_sum=("log_sum", "mean"),
            resid_G=("resid_G", "sum"),
            n=("log_G", "size"),
        )
        .sort_values("gw")
    )
    by_pos = {
        str(pos): {
            "mean_log_G": float(g["log_G"].mean()),
            "resid_G": float(g["resid_G"].sum()),
            "n": int(len(g)),
        }
        for pos, g in pl.groupby("position")
    }
    s = (summary or {}).get("summary") or {}
    return {
        "label": label,
        "goal_weight": (summary or {}).get("goal_weight", label),
        "n": int(len(pl)),
        "mean_log_G": float(log_g.mean()),
        "mean_log_sum": float(log_sum.mean()),
        "resid_G": float(pl["resid_G"].sum()),
        "sum_casc_xP_G": float(s.get("sum_casc_xP_G", pl["casc_xP_G"].sum())),
        "sum_P_G": float(s.get("sum_P_G", pl["P_G"].sum())),
        "mae_casc_G": float((pl["casc_xP_G"] - pl["P_G"]).abs().mean()),
        "by_position": by_pos,
        "by_gw": by_gw.to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = _now()
    for path in (RATE_PLAYERS, RM_PLAYERS):
        if not path.is_file():
            raise SystemExit(f"missing artifact {path}")

    rate_sum = json.loads(RATE_SUMMARY.read_text()) if RATE_SUMMARY.is_file() else {}
    rm_sum = json.loads(RM_SUMMARY.read_text()) if RM_SUMMARY.is_file() else {}
    rate_pl = pd.DataFrame(json.loads(RATE_PLAYERS.read_text()))
    rm_pl = pd.DataFrame(json.loads(RM_PLAYERS.read_text()))

    # Align on player_id × gw for paired Δ.
    key = ["gw", "player_id"]
    rate_pl = rate_pl.copy()
    rm_pl = rm_pl.copy()
    rate_pl["log_G"] = gaussian_log_score(
        rate_pl["P_G"].to_numpy(float), rate_pl["casc_xP_G"].to_numpy(float), SIGMA
    )
    rm_pl["log_G"] = gaussian_log_score(
        rm_pl["P_G"].to_numpy(float), rm_pl["casc_xP_G"].to_numpy(float), SIGMA
    )
    rate_pl["log_sum"] = gaussian_log_score(
        rate_pl["P_sum"].to_numpy(float), rate_pl["casc_xP_sum"].to_numpy(float), SIGMA
    )
    rm_pl["log_sum"] = gaussian_log_score(
        rm_pl["P_sum"].to_numpy(float), rm_pl["casc_xP_sum"].to_numpy(float), SIGMA
    )
    merged = rate_pl[key + ["log_G", "log_sum", "position"]].merge(
        rm_pl[key + ["log_G", "log_sum"]],
        on=key,
        suffixes=("_rate", "_rm"),
        how="inner",
    )
    merged["d_log_G"] = merged["log_G_rate"] - merged["log_G_rm"]
    merged["d_log_sum"] = merged["log_sum_rate"] - merged["log_sum_rm"]
    by_gw = (
        merged.groupby("gw", as_index=False)
        .agg(
            d_log_G=("d_log_G", "mean"),
            d_log_sum=("d_log_sum", "mean"),
            n=("d_log_G", "size"),
        )
        .sort_values("gw")
    )
    # GW-cluster interval (simple normal approx on GW means; diagnostic).
    gw_d = by_gw["d_log_G"].to_numpy(float)
    gw_mean = float(gw_d.mean()) if len(gw_d) else float("nan")
    gw_se = float(gw_d.std(ddof=1) / np.sqrt(len(gw_d))) if len(gw_d) > 1 else float("nan")
    lo = gw_mean - 1.96 * gw_se
    hi = gw_mean + 1.96 * gw_se

    rate = _pack(rate_pl, label="rate", summary=rate_sum)
    rm = _pack(rm_pl, label="rate_minutes", summary=rm_sum)

    # Reference: gated score_xp_v2 π∝r screen (already parked).
    ref = {
        "protocol": "score_xp_v2_goal_share_rate",
        "verdict": "park_clear_lose",
        "mean_d_log": -0.000549,
        "note": "centreline score_xp_v2 four-season gate; π∝r lost to rate_minutes",
    }

    prefer = (
        "rate_minutes"
        if gw_mean < 0
        else ("rate" if gw_mean > 0 else "tie")
    )
    covers_zero = bool(lo <= 0.0 <= hi) if np.isfinite(lo) and np.isfinite(hi) else True

    state = {
        "task": "cascade_fd_goal_weight_likelihood_dig",
        "protocol": "cascade_fd_goal_weight_likelihood_dig",
        "pre_registered": "2026-10-28",
        "sigma": SIGMA,
        "n_paired": int(len(merged)),
        "rate": rate,
        "rate_minutes": rm,
        "paired": {
            "mean_d_log_G_rate_minus_rm": float(merged["d_log_G"].mean()),
            "mean_d_log_sum_rate_minus_rm": float(merged["d_log_sum"].mean()),
            "gw_mean_d_log_G": gw_mean,
            "gw_interval_95": [float(lo), float(hi)],
            "covers_zero": covers_zero,
            "prefer_on_log_G": prefer,
            "by_gw": by_gw.to_dict(orient="records"),
        },
        "score_xp_v2_reference": ref,
        "verdict_notes": [
            "XI resid G hotter under rate_minutes does not by itself overturn the likelihood gate.",
            "score_xp_v2 already parked π∝r as clear lose on four closed seasons.",
            "This dig is cascade generative components on one shared XI map (diagnostic).",
        ],
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Cascade FD goal_weight likelihood dig",
        "",
        f"Pre-registered **2026-10-28** · protocol "
        f"`cascade_fd_goal_weight_likelihood_dig` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Verdict",
        "",
        f"- XI resid G: π∝r **{rate['resid_G']:+.1f}** vs rate_minutes "
        f"**{rm['resid_G']:+.1f}** (minutes hotter).",
        f"- Mean log score on G (σ={SIGMA}): π∝r **{rate['mean_log_G']:.4f}** vs "
        f"rate_minutes **{rm['mean_log_G']:.4f}**.",
        f"- Paired GW Δ (rate − rate_minutes) on G: **{gw_mean:+.4f}** "
        f"[{lo:+.4f}, {hi:+.4f}] · covers 0: **{covers_zero}** · "
        f"prefer **{prefer}**.",
        f"- Gated centreline reference (`score_xp_v2_goal_share_rate`): "
        f"**park_clear_lose** for π∝r (mean Δ {ref['mean_d_log']:+.4f}).",
        "- Default `goal_weight=rate_minutes` **unchanged**.",
        "",
        "## Position resid G (diagnostic)",
        "",
        "| pos | π∝r resid G | rate_minutes resid G | π∝r mean log G | rm mean log G |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        r = rate["by_position"].get(pos, {})
        m = rm["by_position"].get(pos, {})
        lines.append(
            f"| {pos} | {r.get('resid_G', float('nan')):+.1f} | "
            f"{m.get('resid_G', float('nan')):+.1f} | "
            f"{r.get('mean_log_G', float('nan')):.4f} | "
            f"{m.get('mean_log_G', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Log scores here are on **component** G (and G+CS+GA vs P_sum), not "
        "full `score_xp_v2` totals — diagnostic for the cascade generative map.",
        "- Hotter mean resid can still win likelihood if variance/calibration "
        "improves; compare Δ log, not Σ resid alone.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_fd_goal_weight_likelihood_dig",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_fd_goal_weight_likelihood_dig",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Dig cascade rate_minutes vs π∝r on component log score",
                "verdict": {
                    "resid_G_rate": rate["resid_G"],
                    "resid_G_rate_minutes": rm["resid_G"],
                    "gw_mean_d_log_G": gw_mean,
                    "prefer": prefer,
                    "covers_zero": covers_zero,
                    "default_unchanged": True,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state["paired"], indent=2, default=str))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
