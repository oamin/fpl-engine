"""Layer-4 climb sanity: horizon (H) vs Layer-3 one-week on μ_post.

Both arms wildcard GW0 on μ_post (constant shrink), then each deadline:
  - L3: ``solve_week_transfers``
  - L4: ``solve_horizon_transfers``

Default **path=repeat**: future horizon boards copy decision-week μ_post
(as-of; no future-score leakage). Optional ``path=oracle`` uses later closed
μ boards as an optimistic upper bound only.

Score = realised XI + captain − hit points. Sanity only — not a protocol gate.

  python -m src.strategy.layer4_climb
  python -m src.strategy.layer4_climb --path oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.models.season_climb_ft import GAMMA
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import (
    DEFAULT_MAX_HITS,
    HeldState,
    TransferWeekPlan,
    held_from_frame,
    solve_week_transfers,
)
from src.strategy.layer4_week import (
    DEFAULT_HORIZON,
    DEFAULT_TERMINAL_MU_WEIGHT,
    DEFAULT_TERMINAL_VALUE_WEIGHT,
    solve_horizon_transfers,
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


def _step_held(held: HeldState, plan: TransferWeekPlan) -> HeldState:
    """Advance held state using prices on the plan squad (may include future-only buys)."""
    purchase = dict(held.purchase)
    for pid in plan.sold:
        purchase.pop(pid, None)
    squad = plan.squad.set_index(plan.squad["player_id"].astype(str))
    for pid in plan.bought:
        purchase[pid] = int(squad.loc[pid, "value"])
    new_ids = tuple(str(p) for p in plan.squad["player_id"].tolist())
    return HeldState(new_ids, purchase, bank=plan.bank_after, ft=plan.ft_after)


def _horizon_pools(
    store: SeasonStore,
    gw: int,
    H: int,
    moments: dict[str, CohortMoments] | None,
    held: HeldState,
    prev_meta: pd.DataFrame,
    decision_work: pd.DataFrame,
    *,
    path: str = "repeat",
) -> list[pd.DataFrame]:
    """Build H weekly boards for the horizon solver.

    ``repeat`` (default): copy decision-week μ_post H times — as-of safe.
    ``oracle``: later closed μ boards (optimistic; not as-of).
    """
    if path == "repeat":
        return [decision_work.copy() for _ in range(H)]
    if path != "oracle":
        raise ValueError(f"unknown path {path!r}")
    gws = store.gws(gw, 38)
    out: list[pd.DataFrame] = []
    last = decision_work
    for t in range(H):
        if t < len(gws):
            g = gws[t]
            pool = store.pool(g)
            work = _attach_post(pool, moments)
            work = _ensure_held_rows(work, held, prev_meta)
            last = work
            out.append(work)
        else:
            out.append(last.copy())
    return out


def climb_arm(
    store: SeasonStore,
    *,
    arm: str,
    horizon: int,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    path: str = "repeat",
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

        if held is None:
            work = _attach_post(pool, moments)
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
            squad["purchase"] = squad["value"].astype(int)
            bank = int(BUDGET_TENTHS - squad["purchase"].sum())
            held = held_from_frame(squad, bank=max(0, bank), ft=1)
            pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
            pts = _realised(splan.plan.xi, splan.plan.captain_id, pts_map)
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
                }
            )
            prev_meta = work.copy()
            held = HeldState(
                held.player_ids, held.purchase, bank=held.bank, ft=advance_ft(1, 0)
            )
            print(
                f"  {store.season} {arm} GW{gw:02d} WC pts={pts:.0f}",
                flush=True,
            )
            continue

        assert prev_meta is not None
        work = _attach_post(pool, moments)
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)

        if arm == "l3":
            plan = solve_week_transfers(
                work, held, mu_col="mu_post", max_hits=MAX_HITS, time_limit=time_limit
            )
            week0 = plan
            horizon_obj = float(plan.objective)
        elif arm == "l4":
            pools = _horizon_pools(
                store,
                gw,
                horizon,
                moments,
                held,
                prev_meta,
                work,
                path=path,
            )
            hplan = solve_horizon_transfers(
                pools,
                held,
                mu_col="mu_post",
                horizon=horizon,
                gamma=GAMMA,
                terminal_mu_weight=DEFAULT_TERMINAL_MU_WEIGHT,
                terminal_value_weight=DEFAULT_TERMINAL_VALUE_WEIGHT,
                max_hits=MAX_HITS,
                time_limit=time_limit,
            )
            week0 = hplan.week0
            horizon_obj = float(hplan.objective)
        else:
            raise ValueError(arm)

        pts = _realised(week0.xi, week0.captain_id, pts_map)
        net = pts - float(week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "n_transfers": int(week0.n_transfers),
                "hits": int(week0.hits),
                "hit_points": int(week0.hit_points),
                "pts_xi_cap": pts,
                "pts_net": net,
                "horizon_obj": horizon_obj,
                "sold": list(week0.sold),
                "bought": list(week0.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} pts={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        # Merge plan squad into meta so future stubs keep bought players
        prev_meta = pd.concat([work, week0.squad], ignore_index=True)
        prev_meta = prev_meta.drop_duplicates("player_id", keep="last")
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if "l4" not in block.columns or "l3" not in block.columns:
            continue
        d = (block["l4"] - block["l3"]).dropna().to_numpy(float)
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
        l3 = float(sub.loc[sub["arm"] == "l3", "pts_net"].sum())
        l4 = float(sub.loc[sub["arm"] == "l4", "pts_net"].sum())
        by_season.append(
            {
                "season": season,
                "n_gw": int(sub.loc[sub["arm"] == "l3", "gw"].nunique()),
                "l3_pts": l3,
                "l4_pts": l4,
                "d_pts": l4 - l3,
                "l3_hits": int(sub.loc[sub["arm"] == "l3", "hit_points"].sum()),
                "l4_hits": int(sub.loc[sub["arm"] == "l4", "hit_points"].sum()),
            }
        )
    return {
        "by_season": by_season,
        "total_l3": float(sum(r["l3_pts"] for r in by_season)),
        "total_l4": float(sum(r["l4_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "paired_gw_interval": interval,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    iv = doc.get("paired_gw_interval") or {}
    lines = [
        "# Layer 4 climb sanity — horizon vs Layer 3 (μ_post)",
        "",
        f"Both arms wildcard on μ_post, then L3 one-week vs L4 H={doc['defs']['horizon']} "
        f"(γ={doc['defs']['gamma']}, terminal_μ={doc['defs']['terminal_mu_weight']}, "
        f"path={doc['defs']['path']}).",
        "Score = realised XI + captain − hits. **Sanity only** — not a gate.",
        "",
        f"- Path **repeat**: future boards = decision-week μ (as-of).",
        f"- Path **oracle**: future closed μ boards (optimistic upper bound).",
        "",
        "| season | n GW | L3 | L4 | Δ | L3 hits | L4 hits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['l3_pts']:.0f} | {r['l4_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['l3_hits']} | {r['l4_hits']} |"
        )
    lines += [
        "",
        f"**Pooled:** L3 {doc['total_l3']:.0f} · L4 {doc['total_l4']:.0f} · "
        f"Δ **{doc['total_d']:+.0f}**",
        "",
        "## Paired GW Δpoints (L4 − L3)",
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
        lines += ["- Interval unavailable (need all four seasons).", ""]
    lines += [
        "",
        "Does **not** overturn L3; informs whether horizon helps before Layer 5.",
        "",
        f"JSON: `data/processed/layer4_climb.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument(
        "--path",
        choices=("repeat", "oracle"),
        default="repeat",
        help="future μ path for L4 (default: repeat = as-of safe)",
    )
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print("loading closed score_xp_v2 …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []
    for season in seasons:
        store = SeasonStore(season, v2_all)
        for arm in ("l3", "l4"):
            print(f"climbing {season} arm={arm} path={args.path} …", flush=True)
            all_rows.extend(
                climb_arm(
                    store,
                    arm=arm,
                    horizon=args.horizon,
                    gw_start=args.gw_start,
                    gw_end=args.gw_end,
                    time_limit=args.time_limit,
                    path=args.path,
                )
            )
    doc = summarise(all_rows)
    doc["defs"] = {
        "horizon": args.horizon,
        "gamma": GAMMA,
        "terminal_mu_weight": DEFAULT_TERMINAL_MU_WEIGHT,
        "terminal_value_weight": DEFAULT_TERMINAL_VALUE_WEIGHT,
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
        "path": args.path,
    }
    out_json = PROCESSED / "layer4_climb.json"
    out_md = REPORTS / "layer4_climb.md"
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
