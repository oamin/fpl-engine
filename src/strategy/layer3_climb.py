"""Layer-3 rolling climb sanity: raw μ vs μ_post transfers.

Each arm:
  GW0: wildcard 15 (L1 raw / L2 constant shrink)
  thereafter: ``solve_week_transfers`` on that arm's μ, ft bank + hits
Score = realised XI + captain − hit points.

Sanity check only — not a protocol gate. Chips / horizon out.

  python -m src.strategy.layer3_climb
  python -m src.strategy.layer3_climb --seasons 2025-26 --gw-start 6 --gw-end 15
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
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer1_week import solve_week
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_week import (
    DEFAULT_MAX_HITS,
    HeldState,
    held_from_frame,
    solve_week_transfers,
)
from src.strategy.shrink_climb import SeasonStore, _realised
from src.strategy.shrinkage import CohortMoments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 6
N_BOOT = 1000
SEED = 0
MAX_HITS = DEFAULT_MAX_HITS


def _attach_post(
    pool: pd.DataFrame, moments: dict[str, CohortMoments] | None
) -> pd.DataFrame:
    out, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    return out


def _ensure_held_rows(pool: pd.DataFrame, held: HeldState, prev_meta: pd.DataFrame) -> pd.DataFrame:
    """Inject stub rows for owned players missing from this GW's pool."""
    have = set(pool["player_id"].astype(str))
    missing = [p for p in held.player_ids if p not in have]
    if not missing:
        return pool
    stubs = []
    prev = prev_meta.set_index(prev_meta["player_id"].astype(str))
    for pid in missing:
        if pid not in prev.index:
            raise RuntimeError(f"cannot stub missing held player {pid}")
        row = prev.loc[pid]
        stubs.append(
            {
                "element": row.get("element", pid),
                "player_id": pid,
                "name": row.get("name", pid),
                "position": normalize_position(str(row["position"])),
                "club": str(row.get("club", row.get("team", "UNK"))),
                "team": str(row.get("team", row.get("club", "UNK"))),
                "value": int(row["value"]),
                "score_xp_v2": 0.0,
                "xp_se": 0.0,
                "total_points": 0.0,
                "mu": 0.0,
                "mu_post": 0.0,
            }
        )
    return pd.concat([pool, pd.DataFrame(stubs)], ignore_index=True)


def _open_squad(
    pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    *,
    arm: str,
    time_limit: float,
) -> tuple[HeldState, pd.DataFrame]:
    work = _attach_post(pool, moments)
    if arm == "raw":
        plan = solve_week(work, mu_col="score_xp_v2", budget=BUDGET_TENTHS, time_limit=time_limit)
        squad = plan.squad.copy()
    elif arm == "shrink":
        splan = solve_week_sigma(
            work,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            budget=BUDGET_TENTHS,
            time_limit=time_limit,
        )
        squad = splan.plan.squad.copy()
    else:
        raise ValueError(arm)
    squad["purchase"] = squad["value"].astype(int)
    bank = int(BUDGET_TENTHS - squad["purchase"].sum())
    held = held_from_frame(squad, bank=max(0, bank), ft=1)
    return held, work


def _step_held(held: HeldState, plan, pool: pd.DataFrame) -> HeldState:
    purchase = dict(held.purchase)
    for pid in plan.sold:
        purchase.pop(pid, None)
    for pid in plan.bought:
        purchase[pid] = int(pool.loc[pool["player_id"] == pid, "value"].iloc[0])
    # Keep purchase for retained; values may have moved but purchase stays
    new_ids = tuple(str(p) for p in plan.squad["player_id"].tolist())
    return HeldState(new_ids, purchase, bank=plan.bank_after, ft=plan.ft_after)


def climb_arm(
    store: SeasonStore,
    *,
    arm: str,
    gw_start: int,
    gw_end: int,
    time_limit: float,
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    if not gws:
        return []
    rows: list[dict[str, Any]] = []
    held: HeldState | None = None
    prev_meta: pd.DataFrame | None = None

    for i, gw in enumerate(gws):
        moments = store.moments_asof(gw)
        if moments is None and i == 0:
            continue
        pool = store.pool(gw)
        pts_map = pool.set_index(pool["player_id"].astype(str))["total_points"]

        if held is None:
            held, work = _open_squad(pool, moments, arm=arm, time_limit=time_limit)
            # Score opening week with L1/L2 XI (wildcard — no hits)
            if arm == "raw":
                plan0 = solve_week(work, mu_col="score_xp_v2", time_limit=time_limit)
                xi, cap = plan0.xi, plan0.captain_id
            else:
                splan = solve_week_sigma(
                    work,
                    mu_col="score_xp_v2",
                    se_col="xp_se",
                    mode="none",
                    moments_by_pos=moments,
                    time_limit=time_limit,
                )
                xi, cap = splan.plan.xi, splan.plan.captain_id
            pts = _realised(xi, cap, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "wildcard",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "pts_xi_cap": pts,
                    "pts_net": pts,
                    "bank": held.bank,
                    "ft": held.ft,
                }
            )
            prev_meta = work.copy()
            # Advance FT as if 0 transfers played on wildcard week
            from src.rules.fpl_2026 import advance_ft

            held = HeldState(held.player_ids, held.purchase, bank=held.bank, ft=advance_ft(1, 0))
            print(
                f"  {store.season} {arm} GW{gw:02d} WC  pts={pts:.0f} bank={held.bank} ft={held.ft}",
                flush=True,
            )
            continue

        work = _attach_post(pool, moments)
        work = _ensure_held_rows(work, held, prev_meta if prev_meta is not None else work)
        # Refresh pts map after stubs
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)
        mu_col = "score_xp_v2" if arm == "raw" else "mu_post"
        if arm == "raw":
            work["score_xp_v2"] = pd.to_numeric(work["score_xp_v2"], errors="coerce").fillna(0.0)
        plan = solve_week_transfers(
            work, held, mu_col=mu_col, max_hits=MAX_HITS, time_limit=time_limit
        )
        pts = _realised(plan.xi, plan.captain_id, pts_map)
        net = pts - float(plan.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "n_transfers": int(plan.n_transfers),
                "hits": int(plan.hits),
                "hit_points": int(plan.hit_points),
                "pts_xi_cap": pts,
                "pts_net": net,
                "bank": plan.bank_after,
                "ft": plan.ft_after,
                "sold": list(plan.sold),
                "bought": list(plan.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={plan.n_transfers} "
            f"hits={plan.hits} pts={net:.0f} (xi={pts:.0f})",
            flush=True,
        )
        held = _step_held(held, plan, work)
        prev_meta = work.copy()
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    # Wide by season-gw
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if "shrink" not in block.columns or "raw" not in block.columns:
            continue
        d = (block["shrink"] - block["raw"]).dropna().to_numpy(float)
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
        raw = float(sub.loc[sub["arm"] == "raw", "pts_net"].sum())
        shr = float(sub.loc[sub["arm"] == "shrink", "pts_net"].sum())
        by_season.append(
            {
                "season": season,
                "n_gw": int(sub.loc[sub["arm"] == "raw", "gw"].nunique()),
                "raw_pts": raw,
                "shrink_pts": shr,
                "d_pts": shr - raw,
                "raw_hits": int(sub.loc[sub["arm"] == "raw", "hit_points"].sum()),
                "shrink_hits": int(sub.loc[sub["arm"] == "shrink", "hit_points"].sum()),
                "raw_transfers": int(sub.loc[sub["arm"] == "raw", "n_transfers"].sum()),
                "shrink_transfers": int(sub.loc[sub["arm"] == "shrink", "n_transfers"].sum()),
            }
        )
    return {
        "by_season": by_season,
        "total_raw": float(sum(r["raw_pts"] for r in by_season)),
        "total_shrink": float(sum(r["shrink_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "paired_gw_interval": interval,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    iv = doc.get("paired_gw_interval") or {}
    lines = [
        "# Layer 3 climb sanity — raw μ vs μ_post transfers",
        "",
        "Each arm wildcards GW0 on its μ, then rolls `solve_week_transfers`",
        "(FT bank, max 2 hits). Score = realised XI + captain − hit points.",
        "",
        "**Sanity only** — not a protocol gate. No chips / horizon.",
        "",
        "## Season totals (net of hits)",
        "",
        "| season | n GW | raw | shrink | Δ | raw hits | shrink hits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['raw_pts']:.0f} | {r['shrink_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['raw_hits']} | {r['shrink_hits']} |"
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
            "| season | mean Δ/GW |",
            "| --- | ---: |",
        ]
        for s, m in (iv.get("by_season_mean") or {}).items():
            lines.append(f"| {s} | {m:+.3f} |")
    else:
        lines += ["- Interval unavailable.", ""]
    lines += [
        "",
        f"JSON: `{(PROCESSED / 'layer3_climb.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=20.0)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print("loading closed score_xp_v2 …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []
    for season in seasons:
        store = SeasonStore(season, v2_all)
        for arm in ("raw", "shrink"):
            print(f"climbing {season} arm={arm} …", flush=True)
            all_rows.extend(
                climb_arm(
                    store,
                    arm=arm,
                    gw_start=args.gw_start,
                    gw_end=args.gw_end,
                    time_limit=args.time_limit,
                )
            )
    doc = summarise(all_rows)
    (PROCESSED / "layer3_climb.json").write_text(
        json.dumps(doc, indent=2, default=str), encoding="utf-8"
    )
    write_report(doc, REPORTS / "layer3_climb.md")
    print(
        json.dumps(
            {k: doc[k] for k in ("by_season", "total_d", "paired_gw_interval")},
            indent=2,
            default=str,
        )
    )
    print(f"wrote {REPORTS / 'layer3_climb.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
