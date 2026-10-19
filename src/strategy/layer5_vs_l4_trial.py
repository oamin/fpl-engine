"""L5 beam (K=2) vs L4 priced no-chip — priced boards only.

Both arms: shared μ_post wildcard open, γ=1 priced t:t+2, unconstrained hits.
L4 = horizon transfers only (no chips). L5 = joint chip+transfer beam DEFAULT_BEAM_K
with depth-0 horizon MILP (locked 2026-10-16).

  python -m src.strategy.layer5_vs_l4_trial
  python -m src.strategy.layer5_vs_l4_trial --rescreen   # dated horizon-MILP re-screen
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
TIME_LIMIT = 45.0  # match L4 priced MILP; prior Step 5 used 10s


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if "l5" not in block.columns or "l4" not in block.columns:
            continue
        d = (block["l5"] - block["l4"]).dropna().to_numpy(float)
        if d.size:
            deltas[season] = d

    interval = None
    if set(deltas) == set(CLOSED_SEASONS):
        interval = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
        interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
        interval["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}

    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        l4 = float(sub.loc[sub["arm"] == "l4", "pts_net"].sum())
        l5 = float(sub.loc[sub["arm"] == "l5", "pts_net"].sum())
        by_season.append(
            {
                "season": season,
                "n_gw": int(sub.loc[sub["arm"] == "l4", "gw"].nunique()),
                "l4_pts": l4,
                "l5_pts": l5,
                "d_pts": l5 - l4,
                "l4_hits": int(sub.loc[sub["arm"] == "l4", "hit_points"].sum()),
                "l5_hits": int(sub.loc[sub["arm"] == "l5", "hit_points"].sum()),
                "l5_chip_weeks": int(
                    (sub.loc[sub["arm"] == "l5", "chip"].fillna("none") != "none").sum()
                )
                if "chip" in sub.columns
                else 0,
            }
        )
    return {
        "by_season": by_season,
        "total_l4": float(sum(r["l4_pts"] for r in by_season)),
        "total_l5": float(sum(r["l5_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "paired_gw_interval": interval,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    iv = doc.get("paired_gw_interval") or {}
    defs = doc.get("defs") or {}
    rescreen = bool(defs.get("horizon_milp_rescreen"))
    title = (
        "# Layer 5 beam (K=2) vs L4 priced — horizon-MILP re-screen"
        if rescreen
        else "# Layer 5 beam (K=2) vs Layer 4 priced no-chip"
    )
    lines = [
        title,
        "",
        "Shared μ_post wildcard open. L4 = priced t:t+2 transfers (γ=1), no chips. "
        f"L5 = joint chip+transfer beam with **K={defs.get('beam_K')}**. "
        "Unconstrained hits (−4/hit). Promote L5 only if L5−L4 paired interval clears 0.",
        "",
    ]
    if rescreen:
        lines += [
            "Re-screen after depth-0 none/TC/BB = `solve_horizon_transfers` "
            "(prior Step 5 was myopic weekly transfers).",
            f"time_limit = {defs.get('time_limit')}s · board = priced hard-stop only.",
            "",
        ]
    lines += [
        "## Paired GW Δpoints (L5 − L4)",
        "",
    ]
    if iv:
        lines += [
            f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
            f"- Covers zero: **{iv.get('covers_zero')}**",
            "",
            "| season | mean Δ/GW |",
            "| --- | ---: |",
        ]
        for s, m in (iv.get("by_season_mean") or {}).items():
            lines.append(f"| {s} | {m:+.3f} |")
    else:
        lines += ["- Interval unavailable.", ""]

    lines += [
        "",
        "## Season totals",
        "",
        "| season | n GW | L4 | L5 | Δ | L4 hits | L5 hits | L5 chip weeks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['l4_pts']:.0f} | {r['l5_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['l4_hits']} | {r['l5_hits']} | {r['l5_chip_weeks']} |"
        )
    lines += [
        "",
        f"**Pooled:** L4 {doc['total_l4']:.0f} · L5 {doc['total_l5']:.0f} · "
        f"Δ **{doc['total_d']:+.0f}**",
        "",
        "## Verdict",
        "",
    ]
    if iv:
        if iv["lo"] > 0:
            lines.append(
                "L5 beam **beats** L4 no-chip — promote as default strategiser path."
            )
        elif iv["hi"] < 0:
            lines.append(
                "L5 beam **loses** to L4 no-chip — keep L4; chips stay advisory."
            )
        else:
            lines.append(
                "L5 vs L4 **inconclusive** — keep L4 default; optional live shadow of L5."
            )
    else:
        lines.append("No paired interval.")
    json_name = (
        "layer5_vs_l4_rescreen.json"
        if (doc.get("defs") or {}).get("horizon_milp_rescreen")
        else "layer5_vs_l4.json"
    )
    lines += ["", f"JSON: `data/processed/{json_name}`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--beam-k", type=int, default=DEFAULT_BEAM_K)
    ap.add_argument(
        "--rescreen",
        action="store_true",
        help="dated horizon-MILP re-screen; write layer5_vs_l4_rescreen.*",
    )
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

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
                    "phase": r.get("phase"),
                    "chip": "none",
                    "pts_net": r["pts_net"],
                    "hit_points": r.get("hit_points", 0),
                    "n_transfers": r.get("n_transfers", 0),
                }
            )

        print(f"climbing {season} arm=l5 K={args.beam_k} …", flush=True)
        l5_rows = climb_k(
            store,
            K=int(args.beam_k),
            merged=merged,
            fd=fd,
            prices=prices,
            gw_start=args.gw_start,
            gw_end=args.gw_end,
            time_limit=args.time_limit,
        )
        for r in l5_rows:
            all_rows.append(
                {
                    "season": r["season"],
                    "gw": r["gw"],
                    "arm": "l5",
                    "phase": r.get("phase"),
                    "chip": r.get("chip", "none"),
                    "pts_net": r["pts_net"],
                    "hit_points": r.get("hit_points", 0),
                    "n_transfers": r.get("n_transfers", 0),
                }
            )

    doc = summarise(all_rows)
    doc["defs"] = {
        "beam_K": int(args.beam_k),
        "gamma": 1.0,
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
        "time_limit": float(args.time_limit),
        "l4": "priced t:t+2 no chips",
        "l5": "joint beam chips+transfers; depth-0 = solve_horizon_transfers",
        "horizon_milp_rescreen": bool(args.rescreen),
        "board": "priced hard-stop only",
    }
    if args.rescreen:
        out_json = PROCESSED / "layer5_vs_l4_rescreen.json"
        out_md = REPORTS / "layer5_vs_l4_rescreen.md"
    else:
        out_json = PROCESSED / "layer5_vs_l4.json"
        out_md = REPORTS / "layer5_vs_l4.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    iv = doc.get("paired_gw_interval") or {}
    print(
        json.dumps(
            {
                "total_d": doc["total_d"],
                "paired": {
                    "mean": iv.get("mean"),
                    "lo": iv.get("lo"),
                    "hi": iv.get("hi"),
                    "covers_zero": iv.get("covers_zero"),
                }
                if iv
                else None,
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
