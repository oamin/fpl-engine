"""Diagnostic: each chip alone on calendar boards vs L4 priced no-chip.

Arms (L5 beam K=2, calendar depth=8; only listed chip + transfers):
  no_chips — transfers only (none)
  tc       — none + triple_captain
  bb       — none + bench_boost
  wc       — none + wildcard
  fh       — none + free_hit

Baseline L4 priced γ=1 no chips (shared). Promote nothing — diagnostic only.

  python -m src.strategy.layer5_chip_ablation_trial
  python -m src.strategy.layer5_chip_ablation_trial --seasons 2025-26 --arms no_chips,fh
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
from src.strategy.layer5_week import NO_CHIP
from src.strategy.shrink_climb import SeasonStore, _stem

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
TIME_LIMIT = 10.0

# name -> frozenset of chips the beam may play (none always kept in expand_node)
ABLATION_ARMS: dict[str, frozenset[str]] = {
    "no_chips": frozenset({NO_CHIP}),
    "tc": frozenset({NO_CHIP, "triple_captain"}),
    "bb": frozenset({NO_CHIP, "bench_boost"}),
    "wc": frozenset({NO_CHIP, "wildcard"}),
    "fh": frozenset({NO_CHIP, "free_hit"}),
}
DEFAULT_ARMS = tuple(ABLATION_ARMS.keys())


def summarise(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    by_arm: dict[str, Any] = {}
    for arm in arms:
        deltas: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            if season not in piv.index.get_level_values(0):
                continue
            block = piv.loc[season]
            if arm not in block.columns or "l4" not in block.columns:
                continue
            d = (block[arm] - block["l4"]).dropna().to_numpy(float)
            if d.size:
                deltas[season] = d
        interval = None
        if set(deltas) == set(CLOSED_SEASONS):
            interval = cluster_interval(
                deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
            )
            interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
            interval["by_season_mean"] = {
                s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS
            }
        season_rows = []
        for season in CLOSED_SEASONS:
            sub = df.loc[df["season"] == season]
            if sub.empty or arm not in sub["arm"].values:
                continue
            l4 = float(sub.loc[sub["arm"] == "l4", "pts_net"].sum())
            a = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            chip_weeks = int(
                (sub.loc[sub["arm"] == arm, "chip"].fillna("none") != "none").sum()
            )
            season_rows.append(
                {
                    "season": season,
                    "l4_pts": l4,
                    "arm_pts": a,
                    "d_pts": a - l4,
                    "chip_weeks": chip_weeks,
                    "hits": int(sub.loc[sub["arm"] == arm, "hit_points"].sum()),
                }
            )
        by_arm[arm] = {
            "by_season": season_rows,
            "total_d": float(sum(r["d_pts"] for r in season_rows)),
            "total_arm": float(sum(r["arm_pts"] for r in season_rows)),
            "total_l4": float(sum(r["l4_pts"] for r in season_rows)),
            "paired_gw_interval": interval,
            "chip_weeks": int(sum(r["chip_weeks"] for r in season_rows)),
            "hits": int(sum(r["hits"] for r in season_rows)),
        }
    return {"by_arm": by_arm, "weeks": df.to_dict(orient="records")}


def write_report(doc: dict[str, Any], path: Path, *, arms: list[str]) -> None:
    lines = [
        "# Layer 5 chip ablation vs L4",
        "",
        "Diagnostic only — not a promote screen. Calendar boards depth=8, beam K=2. "
        "Each arm may play **only** the named chip (+ transfers); L4 = priced t:t+2 no chips.",
        "",
        "Prior joint-climb dissection (`layer5_calendar_vs_l4`): chip **play** weeks "
        "were net **+114** vs L4; **none** weeks **−413**. Ablation asks which chip "
        "poisons the later path.",
        "",
        "## Arm vs L4 (paired GW)",
        "",
        "| arm | mean Δ/GW | lo | hi | covers 0 | pooled Δ | chip weeks | hits |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for arm in arms:
        block = doc["by_arm"][arm]
        iv = block.get("paired_gw_interval") or {}
        lines.append(
            f"| `{arm}` | {iv.get('mean', float('nan')):+.3f} | "
            f"{iv.get('lo', float('nan')):+.3f} | {iv.get('hi', float('nan')):+.3f} | "
            f"{iv.get('covers_zero', 'n/a')} | {block['total_d']:+.0f} | "
            f"{block['chip_weeks']} | {block['hits']} |"
        )
    lines += ["", "## By season (Δ pts)", "", "| season | " + " | ".join(f"`{a}`" for a in arms) + " |",
              "| --- | " + " | ".join("---:" for _ in arms) + " |"]
    seasons = CLOSED_SEASONS
    for season in seasons:
        cells = []
        for arm in arms:
            rows = {r["season"]: r for r in doc["by_arm"][arm]["by_season"]}
            if season in rows:
                cells.append(f"{rows[season]['d_pts']:+.0f}")
            else:
                cells.append("—")
        lines.append(f"| {season} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Reading",
        "",
        "- `no_chips` ≈ L5 transfer search on calendar boards with chips locked out — "
        "isolates board/horizon vs L4 priced path.",
        "- A single-chip arm that clears `no_chips` helps; one that loses further is the poison.",
        "",
        "JSON: `data/processed/layer5_chip_ablation.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--beam-k", type=int, default=DEFAULT_BEAM_K)
    ap.add_argument("--depth", type=int, default=DEFAULT_CALENDAR_DEPTH)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ABLATION_ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {list(ABLATION_ARMS)}")

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
                    "n_transfers": r.get("n_transfers", 0),
                }
            )

        for arm in arms:
            allow = ABLATION_ARMS[arm]
            print(
                f"climbing {season} arm={arm} chips={sorted(allow)} …",
                flush=True,
            )
            rows = climb_k(
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
                allowed_chips=allow,
            )
            for r in rows:
                all_rows.append(
                    {
                        "season": r["season"],
                        "gw": r["gw"],
                        "arm": arm,
                        "chip": r.get("chip", "none"),
                        "pts_net": r["pts_net"],
                        "hit_points": r.get("hit_points", 0),
                        "n_transfers": r.get("n_transfers", 0),
                    }
                )

    doc = summarise(all_rows, arms)
    doc["defs"] = {
        "beam_K": int(args.beam_k),
        "calendar_depth": int(args.depth),
        "arms": {a: sorted(ABLATION_ARMS[a]) for a in arms},
        "role": "diagnostic ablation — not promote",
    }
    out_json = PROCESSED / "layer5_chip_ablation.json"
    out_md = REPORTS / "layer5_chip_ablation.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md, arms=arms)
    summary = {
        a: {
            "total_d": doc["by_arm"][a]["total_d"],
            "mean": (doc["by_arm"][a].get("paired_gw_interval") or {}).get("mean"),
            "covers_zero": (doc["by_arm"][a].get("paired_gw_interval") or {}).get(
                "covers_zero"
            ),
        }
        for a in arms
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
