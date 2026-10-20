"""Screen: L5 beam on calendar boards (depth=8) vs L4 priced no-chip.

L4 unchanged (priced t:t+2, γ=1, no chips). L5 uses option-3 extended
as-of boards (FD when joined else strength fill) + DEFAULT_BEAM_K.

Does not overwrite ``reports/layer5_vs_l4.md`` (priced-board Step 5).

  python -m src.strategy.layer5_calendar_trial
  python -m src.strategy.layer5_calendar_trial --seasons 2025-26
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
from src.strategy.layer5_calendar_boards import DEFAULT_CALENDAR_DEPTH
from src.strategy.shrink_climb import SeasonStore, _stem

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
TIME_LIMIT = 10.0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if "l5_cal" not in block.columns or "l4" not in block.columns:
            continue
        d = (block["l5_cal"] - block["l4"]).dropna().to_numpy(float)
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
        l5 = float(sub.loc[sub["arm"] == "l5_cal", "pts_net"].sum())
        by_season.append(
            {
                "season": season,
                "n_gw": int(sub.loc[sub["arm"] == "l4", "gw"].nunique()),
                "l4_pts": l4,
                "l5_cal_pts": l5,
                "d_pts": l5 - l4,
                "l4_hits": int(sub.loc[sub["arm"] == "l4", "hit_points"].sum()),
                "l5_cal_hits": int(sub.loc[sub["arm"] == "l5_cal", "hit_points"].sum()),
                "l5_cal_chip_weeks": int(
                    (sub.loc[sub["arm"] == "l5_cal", "chip"].fillna("none") != "none").sum()
                )
                if "chip" in sub.columns
                else 0,
                "mean_H": float(
                    pd.to_numeric(
                        sub.loc[sub["arm"] == "l5_cal", "H"], errors="coerce"
                    ).mean()
                )
                if "H" in sub.columns
                else float("nan"),
            }
        )
    return {
        "by_season": by_season,
        "total_l4": float(sum(r["l4_pts"] for r in by_season)),
        "total_l5_cal": float(sum(r["l5_cal_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "paired_gw_interval": interval,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    iv = doc.get("paired_gw_interval") or {}
    lines = [
        "# Layer 5 calendar boards vs Layer 4 priced no-chip",
        "",
        "Option 3 screen. Shared μ_post wildcard open. "
        "L4 = priced t:t+2 transfers (γ=1), no chips. "
        f"L5 = joint beam **K={doc['defs']['beam_K']}** on calendar boards "
        f"**depth={doc['defs']['calendar_depth']}** "
        "(FD pots when joined; else as-of team-strength fill). "
        "L4 `BOOKIE_DEPTH` unchanged. Promote only if L5_cal−L4 paired lo > 0.",
        "",
        "## Paired GW Δpoints (L5_cal − L4)",
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
        lines += ["- Interval unavailable (need all four closed seasons).", ""]

    lines += [
        "",
        "## Season totals",
        "",
        "| season | n GW | L4 | L5_cal | Δ | L4 hits | L5 hits | chip weeks | mean H |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['l4_pts']:.0f} | {r['l5_cal_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['l4_hits']} | {r['l5_cal_hits']} | "
            f"{r['l5_cal_chip_weeks']} | {r.get('mean_H', float('nan')):.1f} |"
        )
    lines += [
        "",
        f"**Pooled:** L4 {doc['total_l4']:.0f} · L5_cal {doc['total_l5_cal']:.0f} · "
        f"Δ **{doc['total_d']:+.0f}**",
        "",
        "## Verdict",
        "",
    ]
    if iv:
        if iv["lo"] > 0:
            lines.append(
                "L5 on calendar boards **beats** L4 — dated promote arm for L5 default."
            )
        elif iv["hi"] < 0:
            lines.append(
                "L5 on calendar boards **loses** to L4 — keep L4; longer boards alone "
                "do not rescue chip greed."
            )
        else:
            lines.append(
                "L5_cal vs L4 **inconclusive** — keep L4 default; optional live shadow."
            )
    else:
        lines.append("No paired interval.")
    lines += [
        "",
        "Priced-board Step 5 (frozen): `reports/layer5_vs_l4.md`.",
        "",
        "JSON: `data/processed/layer5_calendar_vs_l4.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--beam-k", type=int, default=DEFAULT_BEAM_K)
    ap.add_argument("--depth", type=int, default=DEFAULT_CALENDAR_DEPTH)
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
                    "H": r.get("H"),
                }
            )

        print(
            f"climbing {season} arm=l5_cal K={args.beam_k} depth={args.depth} …",
            flush=True,
        )
        l5_rows = climb_k(
            store,
            K=int(args.beam_k),
            merged=merged,
            fd=fd,
            prices=prices,
            gw_start=args.gw_start,
            gw_end=args.gw_end,
            time_limit=args.time_limit,
            board_mode="calendar",
            calendar_depth=int(args.depth),
        )
        for r in l5_rows:
            all_rows.append(
                {
                    "season": r["season"],
                    "gw": r["gw"],
                    "arm": "l5_cal",
                    "phase": r.get("phase"),
                    "chip": r.get("chip", "none"),
                    "pts_net": r["pts_net"],
                    "hit_points": r.get("hit_points", 0),
                    "n_transfers": r.get("n_transfers", 0),
                    "H": r.get("H"),
                }
            )

    doc = summarise(all_rows)
    doc["defs"] = {
        "beam_K": int(args.beam_k),
        "calendar_depth": int(args.depth),
        "gamma": 1.0,
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
        "l4": "priced t:t+2 no chips",
        "l5_cal": "joint beam on calendar boards",
        "board_mode": "calendar",
    }
    out_json = PROCESSED / "layer5_calendar_vs_l4.json"
    out_md = REPORTS / "layer5_calendar_vs_l4.md"
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
                "by_season": [
                    {"season": r["season"], "d": r["d_pts"]} for r in doc["by_season"]
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
