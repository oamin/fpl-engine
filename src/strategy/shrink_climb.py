"""Side-by-side weekly MILP climb: raw μ vs constant μ_post.

Each gameweek rebuilds a legal 15+XI+captain (no transfer link — weekly wildcard).
Arms choose on forecast μ / μ_post; scored on realised total_points (XI + captain).

This is a **points sanity check**, not the locked forecast gate
(``se_credibility``). Free-transfer linking is Layer 3 (not built).

  python -m src.strategy.shrink_climb
  python -m src.strategy.shrink_climb --seasons 2025-26 --gw-start 10 --gw-end 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.rules.fpl_2026 import BUDGET_TENTHS, normalize_position
from src.strategy.layer1_week import solve_week
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.shrinkage import estimate_cohort_moments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 6  # need GW < t history within season (closed files start at 5)
N_BOOT = 1000
SEED = 0


def _stem(season: str) -> str:
    return season.replace("-", "_")


class SeasonStore:
    """Cache closed v2 + prices for one season."""

    def __init__(self, season: str, v2_all: pd.DataFrame) -> None:
        self.season = season
        self.v2 = v2_all.loc[v2_all["season"] == season].copy()
        self.v2["element"] = pd.to_numeric(self.v2["element"], errors="coerce")
        self.v2["score_xp_v2"] = pd.to_numeric(self.v2["score_xp_v2"], errors="coerce")
        self.v2["xp_se"] = pd.to_numeric(self.v2["xp_se"], errors="coerce").fillna(0.0)
        self.v2["total_points"] = pd.to_numeric(self.v2["total_points"], errors="coerce").fillna(0.0)
        self.v2["gw"] = pd.to_numeric(self.v2["gw"], errors="coerce").astype(int)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{_stem(season)}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["element"] = pd.to_numeric(prices["element"], errors="coerce")
        prices["value"] = pd.to_numeric(prices["value"], errors="coerce")
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        self.prices = prices.drop_duplicates(["element", "gw"], keep="first")

    def gws(self, gw_start: int, gw_end: int) -> list[int]:
        avail = sorted(int(g) for g in self.v2["gw"].unique())
        return [g for g in avail if gw_start <= g <= gw_end]

    def pool(self, gw: int) -> pd.DataFrame:
        cur = self.v2.loc[self.v2["gw"] == int(gw)]
        agg = (
            cur.groupby("element", as_index=False)
            .agg(
                score_xp_v2=("score_xp_v2", "sum"),
                xp_se=("xp_se", lambda s: float((s.astype(float) ** 2).sum() ** 0.5)),
                total_points=("total_points", "sum"),
                name=("name", "first"),
                position=("position", "first"),
                team=("team", "first"),
            )
        )
        pr = self.prices.loc[self.prices["gw"] == int(gw)]
        pool = agg.merge(
            pr[["element", "value", "team_norm", "player_id"]],
            on="element",
            how="inner",
        )
        pool = pool.loc[np.isfinite(pool["score_xp_v2"]) & np.isfinite(pool["value"])].copy()
        pool["player_id"] = pool["player_id"].fillna(pool["element"]).astype(str)
        pool["club"] = pool["team_norm"].fillna(pool["team"]).astype(str)
        pool["position"] = pool["position"].map(normalize_position)
        pool["mu"] = pool["score_xp_v2"]
        return pool.reset_index(drop=True)

    def moments_asof(self, gw: int) -> dict[str, Any] | None:
        hist = self.v2.loc[self.v2["gw"] < int(gw)]
        if hist.empty:
            return None
        moments = {}
        for p, g in hist.groupby("position"):
            moments[str(normalize_position(p) or p)] = estimate_cohort_moments(
                g["score_xp_v2"].to_numpy(float),
                g["xp_se"].to_numpy(float),
                g["total_points"].to_numpy(float),
            )
        return moments


def _realised(plan_xi: pd.DataFrame, captain_id: str, pts_map: pd.Series) -> float:
    ids = plan_xi["player_id"].astype(str)
    pts = ids.map(pts_map).astype(float)
    cap = float(pts_map.get(str(captain_id), 0.0))
    return float(pts.sum() + cap)


def climb_season(
    store: SeasonStore,
    *,
    gw_start: int,
    gw_end: int,
    time_limit: float = 15.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gw in store.gws(gw_start, gw_end):
        pool = store.pool(gw)
        moments = store.moments_asof(gw)
        if moments is None:
            continue
        pts_map = pool.set_index(pool["player_id"].astype(str))["total_points"]

        raw = solve_week(
            pool, mu_col="score_xp_v2", budget=BUDGET_TENTHS, time_limit=time_limit
        )
        shr = solve_week_sigma(
            pool,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="constant",
            moments_by_pos=moments,
            budget=BUDGET_TENTHS,
            time_limit=time_limit,
        )
        raw_pts = _realised(raw.xi, raw.captain_id, pts_map)
        shr_pts = _realised(shr.plan.xi, shr.plan.captain_id, pts_map)
        overlap = len(
            set(raw.xi["player_id"].astype(str))
            & set(shr.plan.xi["player_id"].astype(str))
        )
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "raw_pts": raw_pts,
                "shrink_pts": shr_pts,
                "d_pts": shr_pts - raw_pts,
                "raw_obj_mu": float(raw.objective),
                "shrink_obj_post": float(shr.post_objective),
                "shrink_obj_abs": float(shr.abs_objective),
                "xi_overlap": overlap,
                "same_xi": overlap == 11 and raw.captain_id == shr.plan.captain_id,
            }
        )
        print(
            f"  {store.season} GW{gw:02d}  raw={raw_pts:.0f}  "
            f"shrink={shr_pts:.0f}  Δ={shr_pts - raw_pts:+.0f}  "
            f"overlap={overlap}/11",
            flush=True,
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    by_season = (
        df.groupby("season", as_index=False)
        .agg(
            n_gw=("gw", "count"),
            raw_pts=("raw_pts", "sum"),
            shrink_pts=("shrink_pts", "sum"),
            d_pts=("d_pts", "sum"),
            mean_d=("d_pts", "mean"),
            frac_same_xi=("same_xi", "mean"),
            mean_overlap=("xi_overlap", "mean"),
        )
        .sort_values("season")
    )
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        block = df.loc[df["season"] == season, "d_pts"].to_numpy(float)
        if block.size:
            deltas[season] = block
    interval = None
    if set(deltas) == set(CLOSED_SEASONS):
        interval = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
        interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
    return {
        "by_season": by_season.to_dict(orient="records"),
        "total_raw": float(df["raw_pts"].sum()),
        "total_shrink": float(df["shrink_pts"].sum()),
        "total_d": float(df["d_pts"].sum()),
        "n_gw": int(len(df)),
        "frac_same_xi": float(df["same_xi"].mean()),
        "mean_overlap": float(df["xi_overlap"].mean()),
        "paired_gw_interval": interval,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    iv = doc.get("paired_gw_interval") or {}
    lines = [
        "# Weekly MILP climb — raw μ vs constant μ_post",
        "",
        "Each GW: joint MILP rebuild of 15+XI+captain (no transfer link). Choose on",
        "`score_xp_v2` (raw) or constant-K μ_post; score = realised XI + captain points.",
        "",
        "**Not** the locked forecast gate (`se_credibility`). Not a free-transfer climb",
        "(Layer 3). Sanity check only.",
        "",
        f"Gameweeks: {doc['n_gw']} across four closed seasons (from GW{GW_START}).",
        f"Same XI+captain: **{doc['frac_same_xi']:.1%}**; mean XI overlap **{doc['mean_overlap']:.1f}/11**.",
        "",
        "## Season totals",
        "",
        "| season | n GW | raw pts | shrink pts | Δ | mean Δ/GW |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['raw_pts']:.0f} | {r['shrink_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['mean_d']:+.2f} |"
        )
    lines += [
        "",
        f"**Pooled:** raw {doc['total_raw']:.0f} · shrink {doc['total_shrink']:.0f} · "
        f"Δ **{doc['total_d']:+.0f}**",
        "",
        "## Paired GW Δpoints (shrink − raw)",
        "",
    ]
    if iv:
        lines += [
            f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}] "
            f"(cluster bootstrap B={N_BOOT}, seed={SEED})",
            f"- Covers zero: **{iv.get('covers_zero')}**",
            "",
        ]
    else:
        lines += ["- Interval unavailable (missing season).", ""]
    lines += [
        "## Read",
        "",
        "- Positive Δ ⇒ constant shrink gained realised points on the weekly rebuild.",
        "- A small / zero-covering interval is expected: within-position order is mostly",
        "  unchanged under constant K, so XIs often match; the forecast win is on",
        "  player-GW log-score, not necessarily on optimiser totals.",
        "",
        f"JSON: `{(PROCESSED / 'shrink_climb.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=15.0)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print("loading closed score_xp_v2 …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []
    for season in seasons:
        print(f"climbing {season} …", flush=True)
        store = SeasonStore(season, v2_all)
        all_rows.extend(
            climb_season(
                store,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
            )
        )
    doc = summarise(all_rows)
    out_json = PROCESSED / "shrink_climb.json"
    out_md = REPORTS / "shrink_climb.md"
    # Drop per-week rows from printed summary size in md path; keep full JSON
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_report(doc, out_md)
    print(json.dumps({k: doc[k] for k in ("by_season", "total_d", "paired_gw_interval", "frac_same_xi")}, indent=2, default=str))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
