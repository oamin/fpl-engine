"""Horizon-MILP re-screen: L5 beam K ∈ {1,2,4,8,16} vs L4 priced no-chip.

Same setup as ``layer5_vs_l4_trial --rescreen``; primary = L5(K) − L4 per K.
Priced t:t+2 hard-stop only. Depth-0 = solve_horizon_transfers.

  python -m src.strategy.layer5_vs_l4_k_trial
  python -m src.strategy.layer5_vs_l4_k_trial --ks 1,2,4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_mupost_trial import climb_season as climb_l4
from src.strategy.layer5_beam import DEFAULT_BEAM_K
from src.strategy.layer5_beam_k_trial import climb_k
from src.strategy.shrink_climb import SeasonStore, _stem

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
TIME_LIMIT = 45.0
DEFAULT_KS = (1, 2, 4, 8, 16)
assert DEFAULT_BEAM_K == 8


def _interval(df: pd.DataFrame, arm_a: str, arm_b: str) -> dict[str, Any] | None:
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if arm_a not in block.columns or arm_b not in block.columns:
            continue
        d = (block[arm_a] - block[arm_b]).dropna().to_numpy(float)
        if d.size:
            deltas[season] = d
    if set(deltas) != set(CLOSED_SEASONS):
        return None
    iv = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
    iv["covers_zero"] = bool(iv["lo"] <= 0.0 <= iv["hi"])
    iv["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}
    return iv


def summarise(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    vs_l4 = {f"l5_k{k}": _interval(df, f"l5_k{k}", "l4") for k in ks}
    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "season": season,
            "n_gw": int(sub.loc[sub["arm"] == "l4", "gw"].nunique()),
            "l4_pts": float(sub.loc[sub["arm"] == "l4", "pts_net"].sum()),
            "l4_hits": int(sub.loc[sub["arm"] == "l4", "hit_points"].sum()),
        }
        for k in ks:
            arm = f"l5_k{k}"
            pts = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            row[f"{arm}_pts"] = pts
            row[f"{arm}_d"] = pts - row["l4_pts"]
            row[f"{arm}_hits"] = int(sub.loc[sub["arm"] == arm, "hit_points"].sum())
            chip = sub.loc[sub["arm"] == arm, "chip"].fillna("none")
            row[f"{arm}_chip_weeks"] = int((chip != "none").sum())
        by_season.append(row)
    totals: dict[str, Any] = {
        "total_l4": float(sum(r["l4_pts"] for r in by_season)),
    }
    for k in ks:
        arm = f"l5_k{k}"
        totals[f"total_{arm}"] = float(sum(r[f"{arm}_pts"] for r in by_season))
        totals[f"total_d_{arm}"] = float(sum(r[f"{arm}_d"] for r in by_season))
    return {
        "by_season": by_season,
        **totals,
        "paired_vs_l4": vs_l4,
        "weeks": df.to_dict(orient="records"),
    }


def _fmt_iv(iv: dict[str, Any] | None) -> list[str]:
    if not iv:
        return ["- Interval unavailable.", ""]
    lines = [
        f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
        f"- Covers zero: **{iv.get('covers_zero')}**",
        "",
        "| season | mean Δ/GW |",
        "| --- | ---: |",
    ]
    for s, m in (iv.get("by_season_mean") or {}).items():
        lines.append(f"| {s} | {m:+.3f} |")
    lines.append("")
    return lines


def _verdict(iv: dict[str, Any] | None, name: str) -> str:
    if not iv:
        return f"{name}: no interval."
    if iv["lo"] > 0:
        return f"{name} **beats** L4."
    if iv["hi"] < 0:
        return f"{name} **loses** to L4."
    return f"{name} **inconclusive** vs L4."


def write_report(doc: dict[str, Any], path: Path, ks: list[int]) -> None:
    defs = doc.get("defs") or {}
    vs = doc.get("paired_vs_l4") or {}
    lines = [
        "# Layer 5 beam K grid vs L4 priced — horizon-MILP re-screen",
        "",
        f"K ∈ {', '.join(str(k) for k in ks)}. L4 = priced t:t+2 no chips (γ=1). "
        "L5 = joint chip beam; depth-0 = `solve_horizon_transfers`. "
        f"time_limit = {defs.get('time_limit')}s.",
        "",
        "Promote a K only if L5(K)−L4 interval is clear of 0 above.",
        "",
        "## Paired vs L4 by K",
        "",
    ]
    for k in ks:
        arm = f"l5_k{k}"
        lines += [f"### K={k}", ""] + _fmt_iv(vs.get(arm))

    # summary table
    lines += [
        "## Season totals",
        "",
        "| season | L4 | "
        + " | ".join(f"K={k}" for k in ks)
        + " | "
        + " | ".join(f"ΔK={k}" for k in ks)
        + " |",
        "| --- | ---: | "
        + " | ".join(["---:"] * len(ks))
        + " | "
        + " | ".join(["---:"] * len(ks))
        + " |",
    ]
    for r in doc["by_season"]:
        pts = " | ".join(f"{r[f'l5_k{k}_pts']:.0f}" for k in ks)
        ds = " | ".join(f"{r[f'l5_k{k}_d']:+.0f}" for k in ks)
        lines.append(f"| {r['season']} | {r['l4_pts']:.0f} | {pts} | {ds} |")
    pooled = " · ".join(
        f"K={k} **{doc[f'total_d_l5_k{k}']:+.0f}**" for k in ks
    )
    lines += [
        "",
        f"**Pooled Δ vs L4:** {pooled}",
        "",
        "## Verdict",
        "",
    ]
    for k in ks:
        lines.append(_verdict(vs.get(f"l5_k{k}"), f"K={k}"))
        lines.append("")
    # best by pooled point estimate
    best_k = max(ks, key=lambda k: float(doc[f"total_d_l5_k{k}"]))
    lines += [
        f"Best pooled point estimate: **K={best_k}** "
        f"(Δ {doc[f'total_d_l5_k{best_k}']:+.0f}). "
        f"Default beam width locked **K={DEFAULT_BEAM_K}** (PI, 2026-10-16); "
        "L4 remains the strategiser default (L5 not promoted).",
        "",
        "JSON: `data/processed/layer5_vs_l4_k_rescreen.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    if not ks:
        raise SystemExit("need at least one K")

    print("loading …", flush=True)
    v2 = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []

    for season in seasons:
        merged = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{_stem(season)}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2)

        print(f"climbing {season} arm=l4 …", flush=True)
        l4_rows = climb_l4(
            store,
            arm="l4_mupost",
            merged=merged,
            fd=fd,
            prices=prices,
            gw_start=args.gw_start,
            gw_end=args.gw_end,
            time_limit=args.time_limit,
            gamma=1.0,
        )
        for r in l4_rows:
            all_rows.append(
                {
                    "season": r["season"],
                    "gw": r["gw"],
                    "arm": "l4",
                    "chip": "none",
                    "pts_net": r["pts_net"],
                    "hit_points": r.get("hit_points", 0),
                }
            )

        for k in ks:
            print(f"climbing {season} arm=l5 K={k} …", flush=True)
            l5_rows = climb_k(
                store,
                K=int(k),
                merged=merged,
                fd=fd,
                prices=prices,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
                board_mode="priced",
            )
            for r in l5_rows:
                all_rows.append(
                    {
                        "season": r["season"],
                        "gw": r["gw"],
                        "arm": f"l5_k{k}",
                        "chip": r.get("chip", "none"),
                        "pts_net": r["pts_net"],
                        "hit_points": r.get("hit_points", 0),
                        "K": k,
                    }
                )

    doc = summarise(all_rows, ks)
    doc["defs"] = {
        "ks": ks,
        "gamma": 1.0,
        "time_limit": float(args.time_limit),
        "board": "priced hard-stop",
        "horizon_milp": True,
        "default_beam_k": DEFAULT_BEAM_K,
    }
    out_json = PROCESSED / "layer5_vs_l4_k_rescreen.json"
    out_md = REPORTS / "layer5_vs_l4_k_rescreen.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md, ks)
    print(
        json.dumps(
            {
                "totals_d": {f"K{k}": doc[f"total_d_l5_k{k}"] for k in ks},
                "vs_l4": {
                    f"K{k}": {
                        key: ((doc["paired_vs_l4"] or {}).get(f"l5_k{k}") or {}).get(key)
                        for key in ("mean", "lo", "hi", "covers_zero")
                    }
                    for k in ks
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
