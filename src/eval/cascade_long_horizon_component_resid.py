"""cascade_long horizon residual: E[G,CS,GA] on t+2:end vs actual.

Product scope for cascade_long is **t+2 … end**, not the decision GW.
This dig is separate from the decision-GW Opta A/B in
``cascade_vs_l3_component_xp --match-source opta``.

At each decision GW ``t``:

1. Refresh ≥60′ selection + full-club share pools from as-of hist.
2. Roll generative mean boards with **Opta/oracle λ** (empty priced index)
   and ``goal_weight=rate`` (π∝r) over plan GWs ``t+2 … horizon_end``.
3. Lock XI from the L3 owned path (seed WC XI at open; else
   ``pick_xi`` on ``score_xp_v2``) — **no captain**.
4. Residual = Σ_XI (xP − P) on each target GW; horizon totals Σ over
   ``t+2:end``. Track how those horizon residuals **shift** as ``t`` steps.

  PYTHONPATH=. python3 -m src.eval.cascade_long_horizon_component_resid \\
      --n 40 --goal-weight rate --steps 8,9,...,36 --horizon-end 38
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import (
    DEFAULT_SEASON,
    SEASON_E0,
    SEED,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_xi_ft_component_resid import actual_components
from src.eval.cascade_mean_xi_ft_rolling import _ensure_owned_in_pool
from src.eval.cascade_mean_xi_ft_step_pool_resid import (
    _inject_owned_from_share,
    _load_seed_owned,
    mean_component_boards_split,
)
from src.eval.cascade_vs_l3_component_xp import (
    L4_JSON,
    SEED_JSON,
    _force_owned_from_hist,
    _l3_owned_by_gw,
    _lookup,
    _v2_component_board,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.season_climb import pick_xi
from src.models.short_horizon_cascade import load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_horizon_component_resid.md"
STATE = ROOT / "REPORT_STATE.json"
N_STRANDS = 40
# Need t+2 ≤ end: default steps leave ≥2 GWs of long horizon.
STEP_GWS = tuple(range(8, 37))
HORIZON_END = 38


def _now() -> float:
    return time.time()


def _pid_element(pid: str) -> int:
    return int(str(pid).split(":")[-1])


def _per_gw(steps: list[dict[str, Any]], resid_key: str, n_key: str) -> list[float]:
    out: list[float] = []
    for s in steps:
        n = float(s.get(n_key) or 0)
        out.append(float(s[resid_key]) / n if n > 0 else float("nan"))
    return out


def _plot_horizon_shift(
    steps: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    """Per-GW mean resid: long Opta t+2:end vs short FD t:t+1 (same XI)."""
    gws = [int(s["decision_gw"]) for s in steps]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    series = (
        ("resid_G", "short_resid_G", "xP_G − P_G  (per GW)"),
        ("resid_CS", "short_resid_CS", "xP_CS − P_CS  (per GW)"),
        ("resid_GA", "short_resid_GA", "xP_GA − P_GA  (per GW)"),
    )
    has_short = all(f"short_resid_G" in s for s in steps)
    for ax, (long_k, short_k, lab) in zip(axes, series):
        ax.axhline(0.0, color="#888", lw=0.8)
        y_long = _per_gw(steps, long_k, "n_plan")
        ax.plot(
            gws,
            y_long,
            "o-",
            color="#c0392b",
            lw=1.5,
            ms=3.5,
            label="long Opta  t+2:end",
        )
        if has_short and short_k in steps[0]:
            y_short = _per_gw(steps, short_k, "short_n_plan")
            ax.plot(
                gws,
                y_short,
                "s--",
                color="#2980b9",
                lw=1.4,
                ms=3.5,
                label="short FD  t:t+1",
            )
        ax.set_ylabel(lab)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(framealpha=0.95, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("decision GW t")
    axes[0].set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _score_xi_plan(
    *,
    boards: dict[int, pd.DataFrame],
    plan_gws: list[int],
    xi_ids: list[str],
    merged: pd.DataFrame,
    decision_gw: int,
    arm: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Σ_XI components vs actual over plan_gws; return totals + per-target cells."""
    sum_c = {"G": 0.0, "CS": 0.0, "GA": 0.0}
    sum_p = {"G": 0.0, "CS": 0.0, "GA": 0.0}
    cells: list[dict[str, Any]] = []
    for gw in plan_gws:
        casc_board = boards[int(gw)]
        act = actual_components(merged, int(gw))
        act_by_e = act.set_index("element") if not act.empty else None
        tg_c = {"G": 0.0, "CS": 0.0, "GA": 0.0}
        tg_p = {"G": 0.0, "CS": 0.0, "GA": 0.0}
        for pid in xi_ids:
            eid = _pid_element(pid)
            casc = _lookup(casc_board, pid, ("xp_goals", "xp_cs", "xp_gc", "xp"))
            cg = float(casc["xp_goals"]) if np.isfinite(casc["xp_goals"]) else 0.0
            ccs = float(casc["xp_cs"]) if np.isfinite(casc["xp_cs"]) else 0.0
            cga = float(casc["xp_gc"]) if np.isfinite(casc["xp_gc"]) else 0.0
            if act_by_e is not None and eid in act_by_e.index:
                a = act_by_e.loc[eid]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                pg, pcs, pga = float(a["P_G"]), float(a["P_CS"]), float(a["P_GC"])
            else:
                pg = pcs = pga = 0.0
            tg_c["G"] += cg
            tg_c["CS"] += ccs
            tg_c["GA"] += cga
            tg_p["G"] += pg
            tg_p["CS"] += pcs
            tg_p["GA"] += pga
        cell = {
            "arm": arm,
            "decision_gw": int(decision_gw),
            "target_gw": int(gw),
            "horizon_offset": int(gw) - int(decision_gw),
            "casc_G": tg_c["G"],
            "casc_CS": tg_c["CS"],
            "casc_GA": tg_c["GA"],
            "P_G": tg_p["G"],
            "P_CS": tg_p["CS"],
            "P_GA": tg_p["GA"],
            "resid_G": tg_c["G"] - tg_p["G"],
            "resid_CS": tg_c["CS"] - tg_p["CS"],
            "resid_GA": tg_c["GA"] - tg_p["GA"],
        }
        cells.append(cell)
        for k in sum_c:
            sum_c[k] += tg_c[k]
            sum_p[k] += tg_p[k]
    totals = {
        "casc_G": sum_c["G"],
        "casc_CS": sum_c["CS"],
        "casc_GA": sum_c["GA"],
        "P_G": sum_p["G"],
        "P_CS": sum_p["CS"],
        "P_GA": sum_p["GA"],
        "resid_G": sum_c["G"] - sum_p["G"],
        "resid_CS": sum_c["CS"] - sum_p["CS"],
        "resid_GA": sum_c["GA"] - sum_p["GA"],
        "n_plan": len(plan_gws),
    }
    return totals, cells


def _plot_target_revision(
    cells: list[dict[str, Any]],
    path: Path,
    *,
    targets: list[int],
    title: str,
) -> None:
    """For fixed target GWs, how resid_G revises as decision t approaches."""
    if not cells or not targets:
        return
    df = pd.DataFrame(cells)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0.0, color="#888", lw=0.8)
    for tg in targets:
        sub = df.loc[df["target_gw"] == int(tg)].sort_values("decision_gw")
        if sub.empty:
            continue
        ax.plot(
            sub["decision_gw"],
            sub["resid_G"],
            "o-",
            lw=1.3,
            ms=3,
            label=f"target GW{tg}",
        )
    ax.set_xlabel("decision GW t")
    ax.set_ylabel("XI Σ (xP_G − P_G) at target")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(framealpha=0.95, ncol=min(4, len(targets)), fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--steps", type=str, default=",".join(str(g) for g in STEP_GWS))
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--seed-owned", type=Path, default=SEED_JSON)
    ap.add_argument("--l4-json", type=Path, default=L4_JSON)
    ap.add_argument(
        "--goal-weight",
        choices=("rate_minutes", "rate"),
        default="rate",
        help="Default rate (π∝r) for this long-horizon dig.",
    )
    ap.add_argument(
        "--reuse-long",
        type=Path,
        default=None,
        help=(
            "Reuse prior long Opta JSON (skip long boards); recompute cascade_short "
            "t:t+1 on the same XI/pools and refresh plots."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/opt/cursor/artifacts/cascade_long_horizon_component_resid.json"
        ),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    goal_weight = str(args.goal_weight)
    step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    horizon_end = int(args.horizon_end)
    n_strands = int(args.n)
    seed0 = int(args.seed)
    seed_meta = _load_seed_owned(args.seed_owned)
    owned_by_gw = _l3_owned_by_gw(
        seed_meta["player_ids"],
        args.l4_json,
        season=season,
        gws=step_gws,
        seed_gw=int(seed_meta.get("gw") or min(step_gws)),
    )

    print(
        f"[long-horiz] season={season} steps={step_gws} "
        f"horizon_end={horizon_end} N={n_strands} goal_weight={goal_weight} "
        f"match=opta (t+2:end) seed={seed_meta['path']}",
        flush=True,
    )
    t_all = _now()
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    steps_out: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    short_cells: list[dict[str, Any]] = []
    stash: pd.DataFrame | None = None

    reused: dict[int, dict[str, Any]] = {}
    reused_cells: list[dict[str, Any]] = []
    if args.reuse_long is not None:
        prior = json.loads(Path(args.reuse_long).read_text(encoding="utf-8"))
        reused = {int(s["decision_gw"]): s for s in prior.get("steps") or []}
        cpath = prior.get("cells_path")
        if cpath and Path(cpath).exists():
            reused_cells = json.loads(Path(cpath).read_text(encoding="utf-8"))
        print(
            f"[long-horiz] reuse-long={args.reuse_long} "
            f"({len(reused)} steps, {len(reused_cells)} long cells)",
            flush=True,
        )

    for t in step_gws:
        plan_lo = int(t) + 2
        if plan_lo > horizon_end and int(t) not in reused:
            print(f"[long-horiz] skip GW{t}: t+2={plan_lo} > end={horizon_end}", flush=True)
            continue
        print(
            f"[long-horiz] === decision GW{t} → long {plan_lo}…{horizon_end} "
            f"+ short {t}…{min(int(t) + 1, horizon_end)} ===",
            flush=True,
        )
        owned = list(owned_by_gw[int(t)])
        hist = numagent_xp.asof_share_history(merged, before_gw=int(t), season=season)
        alpha_g, alpha_a = _asof_alphas(hist)
        team_hist = hist_by_team_norm(hist)

        v2_board = _v2_component_board(
            hist=hist,
            enriched=enriched,
            fd=fd,
            prices=prices,
            season=season,
            decision_gw=int(t),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            force_ids=owned,
        )
        if v2_board.empty:
            raise RuntimeError(f"empty v2 board at GW{t}")

        # Prefer locked XI from reused long step (exact match for overlay).
        if int(t) in reused and reused[int(t)].get("xi_ids"):
            xi_ids = [str(x) for x in reused[int(t)]["xi_ids"]]
            xi_source = str(reused[int(t)].get("xi_source") or "reused_long")
        elif int(t) == int(seed_meta["gw"] or step_gws[0]) and seed_meta.get("xi_ids"):
            xi_ids = list(seed_meta["xi_ids"])
            xi_source = "seed_l3_l4_wc"
        else:
            squad = v2_board.loc[
                v2_board["player_id"].astype(str).isin(set(owned))
            ].copy()
            missing = [p for p in owned if p not in set(squad["player_id"].astype(str))]
            if missing:
                raise RuntimeError(f"GW{t} owned missing from v2 board: {missing}")
            xi_df, _form = pick_xi(squad, "score_xp_v2")
            xi_ids = [str(x) for x in xi_df["player_id"].astype(str).tolist()]
            xi_source = "pick_xi_score_xp_v2"

        select_pool = _build_hist_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        share_pool = _build_hist_share_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        force_ids = list(dict.fromkeys(list(owned) + list(xi_ids)))
        share_pool = _force_owned_from_hist(
            share_pool,
            hist,
            prices,
            owned=force_ids,
            decision_gw=int(t),
            season=season,
        )
        if stash is not None:
            in_stash = set(stash["player_id"].astype(str))
            select_pool = _ensure_owned_in_pool(
                select_pool, stash, [p for p in force_ids if p in in_stash]
            )
            share_pool = _ensure_owned_in_pool(
                share_pool, stash, [p for p in force_ids if p in in_stash]
            )
        select_pool = _inject_owned_from_share(select_pool, share_pool, force_ids)
        stash = select_pool.copy()

        id_to_name = {
            str(r.player_id): str(r.name)
            for r in select_pool.itertuples(index=False)
        }
        for r in v2_board.itertuples(index=False):
            id_to_name.setdefault(str(r.player_id), str(r.name))

        step_seed = seed0 if t == step_gws[0] else seed0 + 10_000 * int(t)

        # --- long Opta t+2:end ---
        if int(t) in reused:
            long_tot = {
                "casc_G": float(reused[int(t)]["casc_G"]),
                "casc_CS": float(reused[int(t)]["casc_CS"]),
                "casc_GA": float(reused[int(t)]["casc_GA"]),
                "P_G": float(reused[int(t)]["P_G"]),
                "P_CS": float(reused[int(t)]["P_CS"]),
                "P_GA": float(reused[int(t)]["P_GA"]),
                "resid_G": float(reused[int(t)]["resid_G"]),
                "resid_CS": float(reused[int(t)]["resid_CS"]),
                "resid_GA": float(reused[int(t)]["resid_GA"]),
                "n_plan": int(reused[int(t)]["n_plan"]),
            }
            plan_gws = list(reused[int(t)].get("plan_gws") or [])
            board_s_long = float(reused[int(t)].get("board_s") or 0.0)
            long_cells_t = [
                c
                for c in reused_cells
                if int(c.get("decision_gw", -1)) == int(t)
                and c.get("arm", "long") != "short"
            ]
            for c in long_cells_t:
                c = dict(c)
                c["arm"] = "long"
                cells.append(c)
            print(
                f"[long-horiz] reused long n_plan={long_tot['n_plan']} "
                f"resid G/CS/GA={long_tot['resid_G']:+.1f}/"
                f"{long_tot['resid_CS']:+.1f}/{long_tot['resid_GA']:+.1f}",
                flush=True,
            )
        else:
            fixtures_gw = _hist_fixtures_by_gw(
                enriched, start=plan_lo, end=horizon_end
            )
            plan_gws = sorted(fixtures_gw)
            if not plan_gws:
                print(
                    f"[long-horiz] GW{t}: no fixtures in {plan_lo}…{horizon_end}",
                    flush=True,
                )
                continue
            print(
                f"[long-horiz] select={len(select_pool)} share={len(share_pool)} "
                f"xi_src={xi_source} long_plan={plan_gws[0]}…{plan_gws[-1]} "
                f"n_plan={len(plan_gws)}",
                flush=True,
            )
            boards, _scores, board_meta = mean_component_boards_split(
                select_pool=select_pool,
                share_pool=share_pool,
                fixtures_gw=fixtures_gw,
                priced_by_gw={},
                ranks=ranks,
                profile=profile,
                league_gf=league_gf,
                hist_by_team=team_hist,
                alpha_g=alpha_g,
                alpha_a=alpha_a,
                plan_gws=plan_gws,
                n_strands=n_strands,
                seed0=step_seed,
                goal_weight=goal_weight,
            )
            board_s_long = float(board_meta["elapsed_s"])
            print(f"[long-horiz] long boards {board_s_long:.1f}s", flush=True)
            long_tot, long_cells_t = _score_xi_plan(
                boards=boards,
                plan_gws=plan_gws,
                xi_ids=xi_ids,
                merged=merged,
                decision_gw=int(t),
                arm="long",
            )
            cells.extend(long_cells_t)

        # --- short FD t:t+1 (same XI / pools) ---
        short_hi = min(int(t) + 1, horizon_end)
        short_fx = _hist_fixtures_by_gw(enriched, start=int(t), end=short_hi)
        short_plan = sorted(short_fx)
        if not short_plan:
            raise RuntimeError(f"GW{t}: empty short plan t:t+1")
        short_priced = _fd_priced_by_gw(enriched, fd, short_plan)
        print(
            f"[long-horiz] short_plan={short_plan[0]}…{short_plan[-1]} "
            f"n={len(short_plan)} priced_gws={sorted(short_priced)}",
            flush=True,
        )
        short_boards, _ss, short_meta = mean_component_boards_split(
            select_pool=select_pool,
            share_pool=share_pool,
            fixtures_gw=short_fx,
            priced_by_gw=short_priced,
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            plan_gws=short_plan,
            n_strands=n_strands,
            seed0=step_seed + 777,
            goal_weight=goal_weight,
        )
        print(f"[long-horiz] short boards {short_meta['elapsed_s']:.1f}s", flush=True)
        short_tot, short_cells_t = _score_xi_plan(
            boards=short_boards,
            plan_gws=short_plan,
            xi_ids=xi_ids,
            merged=merged,
            decision_gw=int(t),
            arm="short",
        )
        short_cells.extend(short_cells_t)

        step_rec = {
            "decision_gw": int(t),
            "xi_source": xi_source,
            "xi_ids": xi_ids,
            "xi_names": [id_to_name.get(x, x) for x in xi_ids],
            "owned": owned,
            "plan_gws": plan_gws,
            "n_plan": int(long_tot["n_plan"]),
            "n_selection_pool": int(len(select_pool)),
            "n_share_pool": int(len(share_pool)),
            "board_s": float(board_s_long),
            "casc_G": float(long_tot["casc_G"]),
            "casc_CS": float(long_tot["casc_CS"]),
            "casc_GA": float(long_tot["casc_GA"]),
            "P_G": float(long_tot["P_G"]),
            "P_CS": float(long_tot["P_CS"]),
            "P_GA": float(long_tot["P_GA"]),
            "resid_G": float(long_tot["resid_G"]),
            "resid_CS": float(long_tot["resid_CS"]),
            "resid_GA": float(long_tot["resid_GA"]),
            "short_plan_gws": short_plan,
            "short_n_plan": int(short_tot["n_plan"]),
            "short_board_s": float(short_meta["elapsed_s"]),
            "short_casc_G": float(short_tot["casc_G"]),
            "short_casc_CS": float(short_tot["casc_CS"]),
            "short_casc_GA": float(short_tot["casc_GA"]),
            "short_P_G": float(short_tot["P_G"]),
            "short_P_CS": float(short_tot["P_CS"]),
            "short_P_GA": float(short_tot["P_GA"]),
            "short_resid_G": float(short_tot["resid_G"]),
            "short_resid_CS": float(short_tot["resid_CS"]),
            "short_resid_GA": float(short_tot["resid_GA"]),
        }
        steps_out.append(step_rec)
        print(
            f"[long-horiz] GW{t} long resid/n G="
            f"{step_rec['resid_G']/max(step_rec['n_plan'],1):+.2f} "
            f"short resid/n G="
            f"{step_rec['short_resid_G']/max(step_rec['short_n_plan'],1):+.2f}",
            flush=True,
        )

    if not steps_out:
        raise RuntimeError("no decision steps produced boards")

    plot_shift = Path(
        "/opt/cursor/artifacts/cascade_long_horizon_component_resid_shift.png"
    )
    _plot_horizon_shift(
        steps_out,
        plot_shift,
        title=(
            f"{season} π∝r per-GW resid: long Opta t+2…{horizon_end} "
            f"vs short FD t:t+1 (same L3 XI)"
        ),
    )
    # Fixed targets near mid / late season for revision paths (long arm).
    want_targets = [g for g in (16, 20, 24, 28, 32, 36) if g <= horizon_end]
    plot_rev = Path(
        "/opt/cursor/artifacts/cascade_long_horizon_component_resid_revision.png"
    )
    _plot_target_revision(
        [c for c in cells if c.get("arm", "long") == "long"],
        plot_rev,
        targets=want_targets,
        title=f"{season} long resid_G revision for fixed targets as t → target−2",
    )

    # Summaries: early / mid / late decision blocks
    def _block(lo: int, hi: int) -> dict[str, Any] | None:
        sub = [s for s in steps_out if lo <= int(s["decision_gw"]) <= hi]
        if not sub:
            return None
        return {
            "block": f"t={lo}-{hi}",
            "n_steps": len(sub),
            "mean_resid_G": float(np.mean([s["resid_G"] for s in sub])),
            "mean_resid_CS": float(np.mean([s["resid_CS"] for s in sub])),
            "mean_resid_GA": float(np.mean([s["resid_GA"] for s in sub])),
            "mean_n_plan": float(np.mean([s["n_plan"] for s in sub])),
            "mean_resid_G_per_gw": float(
                np.mean([s["resid_G"] / max(s["n_plan"], 1) for s in sub])
            ),
            "mean_resid_CS_per_gw": float(
                np.mean([s["resid_CS"] / max(s["n_plan"], 1) for s in sub])
            ),
            "mean_short_resid_G_per_gw": float(
                np.mean(
                    [s["short_resid_G"] / max(s["short_n_plan"], 1) for s in sub]
                )
            ),
            "mean_short_resid_CS_per_gw": float(
                np.mean(
                    [s["short_resid_CS"] / max(s["short_n_plan"], 1) for s in sub]
                )
            ),
        }

    blocks = [
        b
        for b in (
            _block(8, 12),
            _block(13, 19),
            _block(20, 26),
            _block(27, 33),
            _block(34, 36),
        )
        if b is not None
    ]

    # Per target-GW: mean resid across all decisions that forecast it (long arm)
    by_target: list[dict[str, Any]] = []
    if cells:
        cdf = pd.DataFrame(cells)
        if "arm" in cdf.columns:
            cdf = cdf.loc[cdf["arm"].fillna("long") == "long"]
        for tg, g in cdf.groupby("target_gw"):
            by_target.append(
                {
                    "target_gw": int(tg),
                    "n_forecasts": int(len(g)),
                    "mean_resid_G": float(g["resid_G"].mean()),
                    "mean_resid_CS": float(g["resid_CS"].mean()),
                    "mean_casc_G": float(g["casc_G"].mean()),
                    "mean_P_G": float(g["P_G"].mean()),
                }
            )

    state = {
        "task": "cascade_long_horizon_component_resid",
        "season": season,
        "step_gws": step_gws,
        "horizon_end": horizon_end,
        "plan_rule": "long=t+2:horizon_end; short=t:t+1",
        "match_source": "opta+fd",
        "goal_weight": goal_weight,
        "n_strands": n_strands,
        "seed": seed0,
        "seed_owned": seed_meta,
        "reuse_long": str(args.reuse_long) if args.reuse_long else None,
        "shared_team": (
            "L3 owned path; XI = seed WC or pick_xi(score_xp_v2); no captain. "
            "Selection/share pools refresh every decision GW. "
            "Long = Opta t+2:end; short = FD t:t+1 on the same XI."
        ),
        "elapsed_s": _now() - t_all,
        "blocks": blocks,
        "by_target_gw": by_target,
        "steps": steps_out,
        "plot_shift": str(plot_shift),
        "plot_revision": str(plot_rev),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells_path = Path(str(args.out).replace(".json", "_cells.json"))
    all_cells = cells + short_cells
    cells_path.write_text(
        json.dumps(all_cells, indent=2, default=str) + "\n", encoding="utf-8"
    )
    state["cells_path"] = str(cells_path)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    long_pg = [s["resid_G"] / max(s["n_plan"], 1) for s in steps_out]
    short_pg = [s["short_resid_G"] / max(s["short_n_plan"], 1) for s in steps_out]
    lines = [
        "# cascade_long vs cascade_short — per-GW E[G,CS,GA] residuals",
        "",
        f"Season **{season}** · decision steps GW{step_gws[0]}–{step_gws[-1]} · "
        f"N={n_strands} · goal_weight=`{goal_weight}` · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## What this is",
        "",
        "- **long** — Opta/oracle λ on **t+2 … end** (product cascade_long).",
        "- **short** — FD centreline λ on **t:t+1** (product cascade_short), "
        "same refreshed pools + same L3 XI, no captain.",
        "- Plot = **per-GW** mean resid (Σ resid / n_plan) so horizons compare.",
        "",
        f"![per-GW shift]({plot_shift})",
        "",
        f"![long target revision]({plot_rev})",
        "",
        f"- Mean per-GW resid_G: long **{float(np.mean(long_pg)):+.2f}**, "
        f"short **{float(np.mean(short_pg)):+.2f}**",
        f"- Mean per-GW resid_CS: long **"
        f"{float(np.mean([s['resid_CS']/max(s['n_plan'],1) for s in steps_out])):+.2f}**, "
        f"short **"
        f"{float(np.mean([s['short_resid_CS']/max(s['short_n_plan'],1) for s in steps_out])):+.2f}**",
        "",
        "## Per decision GW (per-GW resid)",
        "",
        "| t | long n | long G/n | short n | short G/n | long CS/n | short CS/n | "
        "long GA/n | short GA/n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in steps_out:
        n = max(int(s["n_plan"]), 1)
        sn = max(int(s["short_n_plan"]), 1)
        lines.append(
            f"| {s['decision_gw']} | {s['n_plan']} | {s['resid_G']/n:+.2f} | "
            f"{s['short_n_plan']} | {s['short_resid_G']/sn:+.2f} | "
            f"{s['resid_CS']/n:+.2f} | {s['short_resid_CS']/sn:+.2f} | "
            f"{s['resid_GA']/n:+.2f} | {s['short_resid_GA']/sn:+.2f} |"
        )
    lines += [
        "",
        "## Mean per-GW resid by decision block",
        "",
        "| block | n | long G/n | short G/n | long CS/n | short CS/n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in blocks:
        lines.append(
            f"| {b['block']} | {b['n_steps']} | {b['mean_resid_G_per_gw']:+.2f} | "
            f"{b['mean_short_resid_G_per_gw']:+.2f} | "
            f"{b['mean_resid_CS_per_gw']:+.2f} | "
            f"{b['mean_short_resid_CS_per_gw']:+.2f} |"
        )
    repro = (
        f"PYTHONPATH=. python3 -m src.eval.cascade_long_horizon_component_resid "
        f"--n {n_strands} --goal-weight {goal_weight} "
        f"--steps {','.join(str(g) for g in step_gws)} "
        f"--horizon-end {horizon_end}"
    )
    if args.reuse_long:
        repro += f" --reuse-long {args.reuse_long}"
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        repro,
        "```",
        "",
        f"Cells: `{cells_path}`",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_horizon_component_resid",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": (
                    f"long Opta t+2…{horizon_end} + short FD t:t+1 per-GW resid "
                    f"overlay (π∝r, shared L3 XI)"
                ),
                "verdict": {
                    "n_steps": len(steps_out),
                    "mean_long_resid_G_per_gw": float(np.mean(long_pg)),
                    "mean_short_resid_G_per_gw": float(np.mean(short_pg)),
                    "blocks": blocks,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
                "plot_shift": str(plot_shift),
                "plot_revision": str(plot_rev),
                "goal_weight": goal_weight,
                "match_source": "opta+fd",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_blocks": blocks,
                "n_steps": len(steps_out),
                "mean_long_G_per_gw": float(np.mean(long_pg)),
                "mean_short_G_per_gw": float(np.mean(short_pg)),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} {REPORT} {plot_shift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
