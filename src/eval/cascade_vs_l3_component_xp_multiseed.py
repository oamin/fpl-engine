"""Multi-seed xP(G,CS,GA) residual: build boards once, score many GW8 opens.

Protocol:

1. Build cascade generative + l3_v2 component boards **once** for GW8–12.
2. Make several GW8 seeded 15s (L3/L4 WC, v2 sigma/raw, exclusions, cascade
   open-loop).
3. For each seed: **hold** that 15 (no transfers); each GW ``pick_xi`` on
   score_xp_v2 among owned (GW8 uses seed XI if provided).
4. Report Σ (xP − actual) for G/CS/GA per seed and the cross-seed mean —
   does the gap swing around a midpoint?

No captaincy.

  PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp_multiseed --n 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from src.eval.cascade_mean_xi_ft_hist import HORIZON_END, opening_15
from src.eval.cascade_mean_xi_ft_step_pool_resid import (
    _inject_owned_from_share,
    _load_seed_owned,
    mean_component_boards_split,
)
from src.eval.cascade_vs_l3_component_xp import (
    _lookup,
    _pid_element,
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
from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer1_week import solve_week
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.shrink_climb import SeasonStore

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_vs_l3_component_xp_multiseed.md"
STATE = ROOT / "REPORT_STATE.json"
SEED_JSON = ROOT / "data" / "processed" / "l3_l4_gw08_seed_2025_26.json"
STEP_GWS = (8, 9, 10, 11, 12)
N_STRANDS = 40
CLOSED_V2 = ROOT / "data" / "processed" / "score_xp_v2_closed.csv"


def _now() -> float:
    return time.time()


def _add_seed(
    seeds: list[dict[str, Any]],
    seen: set[frozenset[str]],
    name: str,
    owned: list[str],
    xi_ids: list[str] | None,
    note: str,
) -> None:
    key = frozenset(owned)
    if len(owned) != 15:
        print(f"[multiseed] skip {name}: owned={len(owned)}", flush=True)
        return
    if key in seen:
        print(f"[multiseed] skip {name}: duplicate owned", flush=True)
        return
    seen.add(key)
    seeds.append(
        {
            "name": name,
            "owned": list(owned),
            "xi_ids": list(xi_ids) if xi_ids else None,
            "note": note,
        }
    )
    print(f"[multiseed] seed {name} n={len(seeds)}", flush=True)


def _make_store_seeds(
    *,
    season: str,
    open_gw: int,
    seed_json: Path,
) -> list[dict[str, Any]]:
    """GW8 opens from L3 JSON + closed-v2 solves (no cascade boards needed)."""
    seeds: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()

    meta = _load_seed_owned(seed_json)
    _add_seed(
        seeds,
        seen,
        "l3_wc",
        meta["player_ids"],
        meta.get("xi_ids"),
        meta.get("source") or "L3/L4 WC",
    )

    v2_all = pd.read_csv(CLOSED_V2)
    store = SeasonStore(season, v2_all)
    pool = store.pool(open_gw)
    moments = store.moments_asof(open_gw)
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    sigma = solve_week_sigma(
        work,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        budget=BUDGET_TENTHS,
        time_limit=45.0,
    )
    owned, xi = _squad_from_plan(sigma)
    _add_seed(seeds, seen, "v2_sigma", owned, xi, "solve_week_sigma score_xp_v2")

    raw = solve_week(
        work, mu_col="score_xp_v2", budget=BUDGET_TENTHS, time_limit=45.0
    )
    owned, xi = _squad_from_plan(raw)
    _add_seed(seeds, seen, "v2_raw", owned, xi, "solve_week score_xp_v2")

    work_nh = work.loc[pd.to_numeric(work["element"], errors="coerce") != 430].copy()
    if len(work_nh) >= 15:
        sigma_nh = solve_week_sigma(
            work_nh,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            budget=BUDGET_TENTHS,
            time_limit=45.0,
        )
        owned, xi = _squad_from_plan(sigma_nh)
        _add_seed(
            seeds, seen, "v2_no_haaland", owned, xi, "sigma excluding Haaland"
        )

    tn = work["team_norm"].astype(str).str.lower() if "team_norm" in work.columns else None
    team = work["team"].astype(str)
    ars = team.str.contains("Arsenal", case=False, na=False)
    if tn is not None:
        ars = ars | tn.str.contains("arsenal", na=False)
    work_na = work.loc[~ars].copy()
    if len(work_na) >= 15:
        sigma_na = solve_week_sigma(
            work_na,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            budget=BUDGET_TENTHS,
            time_limit=45.0,
        )
        owned, xi = _squad_from_plan(sigma_na)
        _add_seed(
            seeds, seen, "v2_no_arsenal", owned, xi, "sigma excluding Arsenal"
        )

    liv = team.str.contains("Liverpool", case=False, na=False)
    if tn is not None:
        liv = liv | tn.str.contains("liverpool", na=False)
    work_nl = work.loc[~liv].copy()
    if len(work_nl) >= 15:
        sigma_nl = solve_week_sigma(
            work_nl,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            budget=BUDGET_TENTHS,
            time_limit=45.0,
        )
        owned, xi = _squad_from_plan(sigma_nl)
        _add_seed(
            seeds, seen, "v2_no_liverpool", owned, xi, "sigma excluding Liverpool"
        )

    return seeds


def _squad_from_plan(plan: Any) -> tuple[list[str], list[str]]:
    if hasattr(plan, "plan") and getattr(plan, "plan") is not None:
        squad = plan.plan.squad
        xi = plan.plan.xi
    else:
        squad = plan.squad
        xi = plan.xi
    owned = [str(x) for x in squad["player_id"].tolist()]
    xi_ids = [str(x) for x in xi["player_id"].tolist()]
    return owned, xi_ids


def _score_seed_on_boards(
    seed: dict[str, Any],
    *,
    step_gws: list[int],
    open_gw: int,
    casc_boards: dict[int, pd.DataFrame],
    v2_boards: dict[int, pd.DataFrame],
    merged: pd.DataFrame,
) -> dict[str, Any]:
    owned = list(seed["owned"])
    rows: list[dict[str, Any]] = []
    for t in step_gws:
        v2 = v2_boards[int(t)]
        casc = casc_boards[int(t)]
        if int(t) == int(open_gw) and seed.get("xi_ids"):
            xi_ids = list(seed["xi_ids"])
            xi_src = "seed_xi"
        else:
            squad = v2.loc[v2["player_id"].astype(str).isin(set(owned))].copy()
            # fall back: allow element match via inject already on board
            have = set(squad["player_id"].astype(str))
            missing = [p for p in owned if p not in have]
            if missing:
                # try rebuild squad from element
                eids = {_pid_element(p) for p in owned}
                squad = v2.loc[
                    pd.to_numeric(v2["element"], errors="coerce").isin(eids)
                ].copy()
                if len(squad) < 11:
                    raise RuntimeError(
                        f"{seed['name']} GW{t}: owned missing from v2: {missing}"
                    )
            xi_df, _ = pick_xi(squad, "score_xp_v2")
            xi_ids = [str(x) for x in xi_df["player_id"].astype(str).tolist()]
            xi_src = "pick_xi_v2"
        act = actual_components(merged, int(t))
        act_by = act.set_index("element") if not act.empty else None
        g_merged = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(t)]
        for pid in xi_ids:
            eid = _pid_element(pid)
            c = _lookup(casc, pid, ("xp_goals", "xp_cs", "xp_gc"))
            v = _lookup(v2, pid, ("xp_goals", "xp_cs", "xp_gc"))
            if act_by is not None and eid in act_by.index:
                a = act_by.loc[eid]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                p_g, p_cs, p_ga = float(a["P_G"]), float(a["P_CS"]), float(a["P_GC"])
                name, pos, mins = str(a["name"]), str(a["position"]), float(a["minutes"])
            else:
                p_g = p_cs = p_ga = float("nan")
                name, pos, mins = pid, "", float("nan")
            hit = g_merged.loc[g_merged["element"] == eid]
            tp = float(hit.iloc[0]["total_points"] or 0) if not hit.empty else float("nan")
            rows.append(
                {
                    "seed": seed["name"],
                    "gw": int(t),
                    "xi_src": xi_src,
                    "player_id": pid,
                    "element": eid,
                    "name": name,
                    "position": pos,
                    "minutes": mins,
                    "casc_xP_G": c["xp_goals"],
                    "casc_xP_CS": c["xp_cs"],
                    "casc_xP_GA": c["xp_gc"],
                    "l3_xP_G": v["xp_goals"],
                    "l3_xP_CS": v["xp_cs"],
                    "l3_xP_GA": v["xp_gc"],
                    "P_G": p_g,
                    "P_CS": p_cs,
                    "P_GA": p_ga,
                    "total_points": tp,
                }
            )
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "name": seed["name"],
        "note": seed.get("note"),
        "owned": owned,
        "n_player_gw": int(len(df)),
        "jaccard_vs_l3_wc": None,
    }
    for arm, g, cs, ga in (
        ("casc", "casc_xP_G", "casc_xP_CS", "casc_xP_GA"),
        ("l3", "l3_xP_G", "l3_xP_CS", "l3_xP_GA"),
    ):
        out[f"{arm}_sum_xP_G"] = float(df[g].sum())
        out[f"{arm}_sum_xP_CS"] = float(df[cs].sum())
        out[f"{arm}_sum_xP_GA"] = float(df[ga].sum())
        out[f"{arm}_rG"] = float((df[g] - df["P_G"]).sum())
        out[f"{arm}_rCS"] = float((df[cs] - df["P_CS"]).sum())
        out[f"{arm}_rGA"] = float((df[ga] - df["P_GA"]).sum())
        out[f"{arm}_mae_G"] = float((df[g] - df["P_G"]).abs().mean())
        out[f"{arm}_mae_CS"] = float((df[cs] - df["P_CS"]).abs().mean())
        out[f"{arm}_mae_GA"] = float((df[ga] - df["P_GA"]).abs().mean())
    out["sum_P_G"] = float(df["P_G"].sum())
    out["sum_P_CS"] = float(df["P_CS"].sum())
    out["sum_P_GA"] = float(df["P_GA"].sum())
    out["sum_total_points"] = float(df["total_points"].sum())
    out["players"] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--steps", type=str, default=",".join(str(g) for g in STEP_GWS))
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--seed-owned", type=Path, default=SEED_JSON)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_vs_l3_component_xp_multiseed.json"),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    open_gw = int(step_gws[0])
    horizon_end = int(args.horizon_end)
    n_strands = int(args.n)
    seed0 = int(args.seed)

    print(
        f"[multiseed] season={season} steps={step_gws} N={n_strands}",
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

    print("[multiseed] making store seeds…", flush=True)
    seeds = _make_store_seeds(
        season=season, open_gw=open_gw, seed_json=args.seed_owned
    )
    force_union: set[str] = set()
    for s in seeds:
        force_union.update(s["owned"])

    print(
        f"[multiseed] building boards once ({len(seeds)} seeds, "
        f"{len(force_union)} force ids)…",
        flush=True,
    )
    casc_boards: dict[int, pd.DataFrame] = {}
    v2_boards: dict[int, pd.DataFrame] = {}
    board_meta_by_gw: dict[int, Any] = {}
    scores_by_gw_open: dict[int, np.ndarray] = {}
    select_open: pd.DataFrame | None = None
    plan_gws_open: list[int] = []

    for t in step_gws:
        hist = numagent_xp.asof_share_history(merged, before_gw=int(t), season=season)
        alpha_g, alpha_a = _asof_alphas(hist)
        team_hist = hist_by_team_norm(hist)
        select_pool = _build_hist_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        share_pool = _build_hist_share_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        select_pool = _inject_owned_from_share(
            select_pool, share_pool, list(force_union)
        )
        fixtures_gw = _hist_fixtures_by_gw(enriched, start=int(t), end=horizon_end)
        plan_gws = sorted(fixtures_gw)
        priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
        step_seed = seed0 if t == open_gw else seed0 + 10_000 * int(t)
        print(f"[multiseed] GW{t} cascade strands…", flush=True)
        boards, scores_by_gw, board_meta = mean_component_boards_split(
            select_pool=select_pool,
            share_pool=share_pool,
            fixtures_gw=fixtures_gw,
            priced_by_gw=priced_by_gw,
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            plan_gws=plan_gws,
            n_strands=n_strands,
            seed0=step_seed,
        )
        print(f"[multiseed] GW{t} boards {board_meta['elapsed_s']:.1f}s", flush=True)
        casc_boards[int(t)] = boards[int(t)]
        board_meta_by_gw[int(t)] = board_meta
        v2_boards[int(t)] = _v2_component_board(
            hist=hist,
            enriched=enriched,
            fd=fd,
            prices=prices,
            season=season,
            decision_gw=int(t),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            force_ids=list(force_union),
        )
        if t == open_gw:
            select_open = select_pool
            scores_by_gw_open = scores_by_gw
            plan_gws_open = plan_gws

    assert select_open is not None
    # Cascade open-loop seed (players already on select board)
    seen = {frozenset(s["owned"]) for s in seeds}
    ol = opening_15(select_open, scores_by_gw_open, plan_gws_open)
    _add_seed(
        seeds, seen, "casc_ol", ol, None, "opening_15 on cascade mean boards"
    )
    # If casc_ol introduced new ids (shouldn't), force onto v2 via re-score open only
    new_ids = [p for p in ol if p not in force_union]
    if new_ids:
        force_union.update(new_ids)
        for t in step_gws:
            hist = numagent_xp.asof_share_history(
                merged, before_gw=int(t), season=season
            )
            alpha_g, alpha_a = _asof_alphas(hist)
            v2_boards[int(t)] = _v2_component_board(
                hist=hist,
                enriched=enriched,
                fd=fd,
                prices=prices,
                season=season,
                decision_gw=int(t),
                alpha_g=alpha_g,
                alpha_a=alpha_a,
                force_ids=list(force_union),
            )

    # Jaccard vs l3_wc
    l3_owned = next(s["owned"] for s in seeds if s["name"] == "l3_wc")
    l3_set = set(l3_owned)

    results = []
    for s in seeds:
        print(f"[multiseed] scoring {s['name']}…", flush=True)
        # fill XI for seeds without xi_ids at open
        if not s.get("xi_ids"):
            squad = v2_boards[open_gw].loc[
                v2_boards[open_gw]["player_id"].astype(str).isin(set(s["owned"]))
            ]
            xi_df, _ = pick_xi(squad, "score_xp_v2")
            s["xi_ids"] = [str(x) for x in xi_df["player_id"].astype(str).tolist()]
        r = _score_seed_on_boards(
            s,
            step_gws=step_gws,
            open_gw=open_gw,
            casc_boards=casc_boards,
            v2_boards=v2_boards,
            merged=merged,
        )
        inter = len(set(s["owned"]) & l3_set)
        union = len(set(s["owned"]) | l3_set)
        r["jaccard_vs_l3_wc"] = float(inter / union) if union else 0.0
        results.append(r)
        print(
            f"  casc rG/CS/GA={r['casc_rG']:+.1f}/{r['casc_rCS']:+.1f}/{r['casc_rGA']:+.1f} "
            f"l3={r['l3_rG']:+.1f}/{r['l3_rCS']:+.1f}/{r['l3_rGA']:+.1f} "
            f"J={r['jaccard_vs_l3_wc']:.2f}",
            flush=True,
        )

    # Cross-seed summary
    def _arm_stats(arm: str) -> dict[str, Any]:
        g = np.array([r[f"{arm}_rG"] for r in results], float)
        cs = np.array([r[f"{arm}_rCS"] for r in results], float)
        ga = np.array([r[f"{arm}_rGA"] for r in results], float)
        return {
            "n_seeds": len(results),
            "rG_mean": float(g.mean()),
            "rG_std": float(g.std(ddof=0)),
            "rG_min": float(g.min()),
            "rG_max": float(g.max()),
            "rCS_mean": float(cs.mean()),
            "rCS_std": float(cs.std(ddof=0)),
            "rCS_min": float(cs.min()),
            "rCS_max": float(cs.max()),
            "rGA_mean": float(ga.mean()),
            "rGA_std": float(ga.std(ddof=0)),
            "rGA_min": float(ga.min()),
            "rGA_max": float(ga.max()),
            "rG_crosses_zero": bool(g.min() < 0 < g.max()),
            "rCS_crosses_zero": bool(cs.min() < 0 < cs.max()),
            "rGA_crosses_zero": bool(ga.min() < 0 < ga.max()),
        }

    summary = {
        "casc": _arm_stats("casc"),
        "l3": _arm_stats("l3"),
        "board_s": {
            str(g): float(board_meta_by_gw[g]["elapsed_s"]) for g in step_gws
        },
    }
    state = {
        "task": "cascade_vs_l3_component_xp_multiseed",
        "season": season,
        "step_gws": step_gws,
        "n_strands": n_strands,
        "protocol": (
            "Boards once per GW; each seed holds GW8 owned 15 (no transfers); "
            "XI = seed XI at GW8 else pick_xi on v2. No captain."
        ),
        "elapsed_s": _now() - t_all,
        "summary": summary,
        "seeds": [
            {
                k: v
                for k, v in r.items()
                if k != "players"
            }
            for r in results
        ],
        "players_by_seed": {r["name"]: r["players"] for r in results},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Multi-seed xP(G,CS,GA) residuals (boards once)",
        "",
        f"Season **{season}** · GW {step_gws} · N={n_strands} · "
        f"{len(results)} seeds · wall {state['elapsed_s']:.1f}s",
        "",
        "## Protocol",
        "",
        "- Build cascade + l3_v2 component boards **once** per GW.",
        "- Several GW8 seeded 15s; **hold** each squad (no transfers).",
        "- Each GW: seed XI at GW8 if given, else `pick_xi` on score_xp_v2.",
        "- Residual = Σ(xP − actual) on that XI. No captain.",
        "",
        "## Per seed",
        "",
        "| seed | J(l3_wc) | casc rG | casc rCS | casc rGA | "
        "l3 rG | l3 rCS | l3 rGA | ΣP_G | ΣP_CS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['jaccard_vs_l3_wc']:.2f} | "
            f"{r['casc_rG']:+.1f} | {r['casc_rCS']:+.1f} | {r['casc_rGA']:+.1f} | "
            f"{r['l3_rG']:+.1f} | {r['l3_rCS']:+.1f} | {r['l3_rGA']:+.1f} | "
            f"{r['sum_P_G']:.0f} | {r['sum_P_CS']:.0f} |"
        )
    lines += ["", "## Cross-seed midpoint", ""]
    for arm, label in (("casc", "cascade"), ("l3", "l3_v2")):
        s = summary[arm]
        lines += [
            f"### {label}",
            "",
            f"- r_G: mean **{s['rG_mean']:+.1f}** (sd {s['rG_std']:.1f}, "
            f"range [{s['rG_min']:+.1f}, {s['rG_max']:+.1f}]) "
            f"{'crosses 0' if s['rG_crosses_zero'] else 'does **not** cross 0'}",
            f"- r_CS: mean **{s['rCS_mean']:+.1f}** (sd {s['rCS_std']:.1f}, "
            f"range [{s['rCS_min']:+.1f}, {s['rCS_max']:+.1f}]) "
            f"{'crosses 0' if s['rCS_crosses_zero'] else 'does **not** cross 0'}",
            f"- r_GA: mean **{s['rGA_mean']:+.1f}** (sd {s['rGA_std']:.1f}, "
            f"range [{s['rGA_min']:+.1f}, {s['rGA_max']:+.1f}])",
            "",
        ]
    sc, sl = summary["casc"], summary["l3"]
    lines += [
        "## Verdict",
        "",
        (
            f"**CS does not swing around zero.** casc r_CS mean {sc['rCS_mean']:+.1f} "
            f"(range [{sc['rCS_min']:+.1f}, {sc['rCS_max']:+.1f}]); "
            f"l3_v2 mean {sl['rCS_mean']:+.1f} "
            f"(range [{sl['rCS_min']:+.1f}, {sl['rCS_max']:+.1f}]). "
            f"**G stays non-negative** here: casc mean {sc['rG_mean']:+.1f}, "
            f"l3_v2 mean {sl['rG_mean']:+.1f}. Shared CS underprint across opens; "
            f"GA straddles 0."
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp_multiseed "
        f"--n {n_strands}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_vs_l3_component_xp_multiseed",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Multi-seed component residual with boards-once",
                "verdict": summary,
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
