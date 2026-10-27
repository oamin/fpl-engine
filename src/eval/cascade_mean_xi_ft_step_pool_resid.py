"""Stepped selection-pool mean xi_ft: xP(G, CS, GA) vs actual.

Horizon GW8–19. At each decision GW ``t ∈ {8,…,12}``:

1. Rebuild **selection** pool = as-of mean minutes > 60 (n_apps≥2) before ``t``.
2. Rebuild **share** pool = full club ``m>0`` (impact subs on the dice).
3. Freeze that as-of state and project it forward on strands for ``t…19``.
4. Mean of N generative boards → one ``xi_ft`` plan; **commit** week ``t``.
5. Compare generative mean ``xp_goals`` / ``xp_cs`` / ``xp_gc`` on the XI
   to actual FPL G / CS / GA (GC) points.

Optional ``--seed-owned``: start from a fixed 15 (e.g. L3/L4 GW8 WC). At the
seed GW the squad is held (0 transfers); XI = ``pick_xi`` on the generative
board; FT bank advances via ``advance_ft(ft, 0)``. Later steps use normal
``xi_ft`` from that owned squad.

GA here is the goals-against points component (``xp_gc`` / FPL GC deductions).

  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid --n 40
  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid --n 40 \\
      --seed-owned /opt/cursor/artifacts/l3_l4_gw08_seed_2025_26.json
"""

from __future__ import annotations

import argparse
import json
import random
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
from src.eval.cascade_mean_xi_ft_hist import (
    CAND_TOP_N,
    HORIZON_END,
    MILP_TIME,
    opening_15,
)
from src.eval.cascade_mean_xi_ft_rolling import (
    _ensure_owned_in_pool,
    _plan_from_owned,
    _transfer_narrative,
    _week_by_gw,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import (
    hist_by_team_norm,
    score_fixture_generative,
)
from src.models.multistage_mc_milp import (
    DEFAULT_LINEUP_COV,
    ClubPath,
    _priced_index,
    init_paths_from_opta,
    sample_week_scenario,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.season_climb import pick_xi
from src.models.short_horizon_cascade import load_prices
from src.rules.fpl_2026 import advance_ft
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_mean_xi_ft_step_pool_resid.md"
STATE = ROOT / "REPORT_STATE.json"

START_GW = 8
STEP_GWS = (8, 9, 10, 11, 12)
N_STRANDS = 40
COMP_KEYS = ("xp_goals", "xp_cs", "xp_gc", "xp")


def _now() -> float:
    return time.time()


def _element(pid: str) -> int | None:
    text = str(pid)
    if ":" in text:
        text = text.split(":")[-1]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _load_seed_owned(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(x) for x in (raw.get("player_ids") or [])]
    if len(ids) != 15:
        raise ValueError(f"seed-owned needs 15 player_ids, got {len(ids)} from {path}")
    return {
        "path": str(path),
        "player_ids": ids,
        "gw": int(raw["gw"]) if raw.get("gw") is not None else None,
        "xi_ids": [str(x) for x in (raw.get("xi_ids") or [])] or None,
        "captain_id": str(raw["captain_id"]) if raw.get("captain_id") else None,
        "source": str(raw.get("source") or ""),
    }


def _inject_owned_from_share(
    select_pool: pd.DataFrame,
    share_pool: pd.DataFrame,
    owned: list[str],
) -> pd.DataFrame:
    """Pull thin-gate misses from the full-club share pool into selection."""
    have = set(select_pool["player_id"].astype(str))
    missing = [p for p in owned if p not in have]
    if not missing:
        return select_pool
    add = share_pool.loc[share_pool["player_id"].astype(str).isin(missing)].copy()
    got = set(add["player_id"].astype(str))
    still = [p for p in missing if p not in got]
    if still:
        raise RuntimeError(
            f"seed owned missing from select and share pools: {still}"
        )
    return pd.concat([select_pool, add], ignore_index=True)


def _hold_week_from_owned(
    board: pd.DataFrame,
    owned: list[str],
    *,
    gw: int,
    ft_before: int,
    seed_xi_ids: list[str] | None = None,
    seed_captain_id: str | None = None,
) -> dict[str, Any]:
    """0-transfer commit; advance_ft(ft, 0).

    If ``seed_xi_ids`` is provided (L3/L4 open XI), use it so the hold week
    matches the seeded climb. Otherwise ``pick_xi`` on generative ``xp``.
    """
    squad = board.loc[board["player_id"].astype(str).isin(set(owned))].copy()
    if len(squad) < 15:
        have = set(squad["player_id"].astype(str))
        raise RuntimeError(
            f"GW{gw} hold: board missing owned {[p for p in owned if p not in have]}"
        )
    # Deduplicate if inject overlapped
    squad = squad.drop_duplicates("player_id", keep="first")
    form: list[int] | tuple[int, int, int] | None = None
    if seed_xi_ids:
        xi_ids = [str(x) for x in seed_xi_ids]
        missing = [p for p in xi_ids if p not in set(owned)]
        if missing:
            raise RuntimeError(f"GW{gw} seed XI not in owned: {missing}")
        if len(xi_ids) != 11:
            raise RuntimeError(f"GW{gw} seed XI needs 11 ids, got {len(xi_ids)}")
        xi_df = squad.loc[squad["player_id"].astype(str).isin(xi_ids)]
    else:
        xi_df, form = pick_xi(squad, "xp")
        xi_ids = [str(x) for x in xi_df["player_id"].astype(str).tolist()]
    cap = str(seed_captain_id) if seed_captain_id else None
    if cap and cap not in set(xi_ids):
        raise RuntimeError(f"GW{gw} seed captain {cap} not in XI")
    return {
        "gw": int(gw),
        "player_ids": list(owned),
        "xi_ids": xi_ids,
        "captain_id": cap,
        "formation": list(form) if form is not None else None,
        "n_transfers": 0,
        "hits": 0,
        "ft_before": int(ft_before),
        "ft_after": int(advance_ft(int(ft_before), 0)),
        "xi_score": float(pd.to_numeric(xi_df["xp"], errors="coerce").fillna(0).sum()),
        "week_score": float(pd.to_numeric(xi_df["xp"], errors="coerce").fillna(0).sum()),
        "held": True,
    }


def mean_component_boards_split(
    *,
    select_pool: pd.DataFrame,
    share_pool: pd.DataFrame,
    fixtures_gw: dict[int, list[dict[str, str]]],
    priced_by_gw: dict[int, list[dict[str, Any]]],
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    hist_by_team: dict[str, pd.DataFrame],
    alpha_g: float,
    alpha_a: float,
    plan_gws: list[int],
    n_strands: int,
    seed0: int,
    goal_weight: str = "rate_minutes",
) -> tuple[dict[int, pd.DataFrame], dict[int, np.ndarray], dict[str, Any]]:
    """Generative mean components on share pool; project onto selection rows."""
    priced_index = _priced_index(priced_by_gw)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    n_sel = len(select_pool)
    sel_idx = {
        str(p): i for i, p in enumerate(select_pool["player_id"].astype(str))
    }
    acc = {g: {k: np.zeros(n_sel, float) for k in COMP_KEYS} for g in plan_gws}
    t0 = _now()
    for s in range(int(n_strands)):
        if s == 0 or (s + 1) % 20 == 0 or s + 1 == int(n_strands):
            print(f"  [step-pool] strand {s + 1}/{n_strands}", flush=True)
        rng = random.Random(int(seed0) + s)
        paths = {c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()}
        rolled: dict[int, list[dict[str, Any]]] = {}
        for gw in plan_gws:
            week = []
            for fx in fixtures_gw.get(int(gw), []):
                if fx["home"] not in paths or fx["away"] not in paths:
                    continue
                sc = sample_week_scenario(
                    fx["home"],
                    fx["away"],
                    gw=int(gw),
                    priced_index=priced_index,
                    paths=paths,
                    league_gf=league_gf,
                    rng=rng,
                    ranks=ranks,
                    profile=profile,
                    sd_league_gf=0.0,
                )
                paths[fx["home"]].observe(sc["hg"], sc["ag"])
                paths[fx["away"]].observe(sc["ag"], sc["hg"])
                week.append(sc)
            rolled[int(gw)] = week

        score_rng = random.Random(int(seed0) + 10_000 + s)
        for gw in plan_gws:
            g_acc = {k: np.zeros(n_sel, float) for k in COMP_KEYS}
            for sc in rolled[int(gw)]:
                hn = norm_team(str(sc["home"]))
                an = norm_team(str(sc["away"]))
                home = share_pool.loc[
                    share_pool["team_norm"].map(norm_team) == hn
                ].copy()
                away = share_pool.loc[
                    share_pool["team_norm"].map(norm_team) == an
                ].copy()
                if home.empty and away.empty:
                    continue
                home_xp, away_xp, _meta = score_fixture_generative(
                    home,
                    away,
                    lam_h=float(sc["lam_h"]),
                    lam_a=float(sc["lam_a"]),
                    rho=float(sc.get("rho") or 0.0),
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    rng=score_rng,
                    home_hist=hist_by_team.get(hn),
                    away_hist=hist_by_team.get(an),
                    home_props=sc.get("home_props")
                    if isinstance(sc.get("home_props"), dict)
                    else None,
                    away_props=sc.get("away_props")
                    if isinstance(sc.get("away_props"), dict)
                    else None,
                    lineup_cov=DEFAULT_LINEUP_COV,
                    goals_home=int(sc["hg"]),
                    goals_away=int(sc["ag"]),
                    goal_weight=goal_weight,
                )
                for scored in (home_xp, away_xp):
                    if scored is None or scored.empty:
                        continue
                    for i in range(len(scored)):
                        row = scored.iloc[i]
                        pid = (
                            str(row["player_id"])
                            if "player_id" in scored.columns
                            else str(int(row["element"]))
                        )
                        j = sel_idx.get(pid)
                        if j is None:
                            continue
                        g_acc["xp_goals"][j] += float(row.get("xp_goals") or 0.0)
                        g_acc["xp_cs"][j] += float(row.get("xp_cs") or 0.0)
                        g_acc["xp_gc"][j] += float(row.get("xp_gc") or 0.0)
                        g_acc["xp"][j] += float(row.get("xp") or 0.0)
            for k in COMP_KEYS:
                acc[int(gw)][k] += g_acc[k]

    frames: dict[int, pd.DataFrame] = {}
    scores_by_gw: dict[int, np.ndarray] = {}
    for gw in plan_gws:
        df = select_pool.copy()
        for k, arr in acc[int(gw)].items():
            df[k] = arr / float(n_strands)
        frames[int(gw)] = df
        scores_by_gw[int(gw)] = df["xp"].to_numpy(float)
    meta = {
        "n_strands": int(n_strands),
        "seed0": int(seed0),
        "elapsed_s": _now() - t0,
        "n_selection_pool": int(n_sel),
        "n_share_pool": int(len(share_pool)),
        "goal_weight": str(goal_weight),
    }
    return frames, scores_by_gw, meta


def score_xi_components(
    board: pd.DataFrame,
    actual: pd.DataFrame,
    xi_ids: list[str],
    *,
    gw: int,
) -> list[dict[str, Any]]:
    pid_col = board["player_id"].astype(str)
    act = actual.set_index("element") if not actual.empty else None
    rows: list[dict[str, Any]] = []
    for pid in xi_ids:
        hit = board.loc[pid_col == str(pid)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        eid = _element(pid)
        if eid is None or act is None or eid not in act.index:
            a_g = a_cs = a_ga = float("nan")
            name = str(r.get("name", pid))
            pos = str(r.get("position", ""))
            mins = float("nan")
        else:
            a = act.loc[eid]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            a_g, a_cs, a_ga = float(a["P_G"]), float(a["P_CS"]), float(a["P_GC"])
            name, pos, mins = str(a["name"]), str(a["position"]), float(a["minutes"])
        xg = float(r["xp_goals"])
        xcs = float(r["xp_cs"])
        xga = float(r["xp_gc"])  # GA points component
        rows.append(
            {
                "gw": int(gw),
                "player_id": str(pid),
                "element": eid,
                "name": name,
                "position": pos,
                "minutes": mins,
                "xP_G": xg,
                "xP_CS": xcs,
                "xP_GA": xga,
                "P_G": a_g,
                "P_CS": a_cs,
                "P_GA": a_ga,
                "resid_G": xg - a_g,
                "resid_CS": xcs - a_cs,
                "resid_GA": xga - a_ga,
            }
        )
    return rows


def _summarise(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out: dict[str, Any] = {"n_player_gw": int(len(df))}
    for lab in ("G", "CS", "GA"):
        r = df[f"resid_{lab}"]
        out[lab] = {
            "sum_xP": float(df[f"xP_{lab}"].sum()),
            "sum_P": float(df[f"P_{lab}"].sum()),
            "sum_resid": float(r.sum()),
            "mean_resid": float(r.mean()),
            "mae": float(r.abs().mean()),
        }
    return out


def _plot(per_gw: list[dict[str, Any]], path: Path, *, n: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=140)
    gws = [r["gw"] for r in per_gw]
    for key, label, color in (
        ("resid_G_per_match", "r_G", "#c0392b"),
        ("resid_CS_per_match", "r_CS", "#2980b9"),
        ("resid_GA_per_match", "r_GA", "#27ae60"),
    ):
        ax.plot(gws, [r[key] for r in per_gw], "o-", color=color, lw=1.6, ms=5, label=label)
    ax.axhline(0.0, color="#444", lw=0.8)
    ax.set_xlabel("committed GW")
    ax.set_ylabel("xP − actual (per XI player-match)")
    ax.set_title(f"Stepped ≥60′ selection pool · xP(G,CS,GA) resid  (N={n})")
    ax.set_xticks(gws)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--start-gw", type=int, default=START_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument(
        "--steps",
        type=str,
        default=",".join(str(g) for g in STEP_GWS),
    )
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cand-top-n", type=int, default=CAND_TOP_N)
    ap.add_argument("--milp-time", type=float, default=MILP_TIME)
    ap.add_argument(
        "--seed-owned",
        type=Path,
        default=None,
        help="JSON with player_ids (15). Hold that squad at seed GW; "
        "pick_xi on boards; FT via advance_ft(ft,0). Later GWs use xi_ft.",
    )
    ap.add_argument(
        "--goal-weight",
        choices=("rate_minutes", "rate"),
        default="rate_minutes",
        help="Goal share weights on generative dice: rate_minutes=w·r·m/90 "
        "(default); rate=π∝r (minutes gate only).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid.json"),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    goal_weight = str(args.goal_weight)
    e0 = SEASON_E0[season]
    step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    start_gw = int(args.start_gw)
    horizon_end = int(args.horizon_end)
    n_strands = int(args.n)
    seed0 = int(args.seed)
    seed_meta = _load_seed_owned(args.seed_owned) if args.seed_owned else None
    hold_gw = int(seed_meta["gw"] if seed_meta and seed_meta["gw"] is not None else start_gw)

    print(
        f"[step-pool] season={season} steps={step_gws} horizon={start_gw}-{horizon_end} "
        f"N={n_strands} goal_weight={goal_weight}"
        + (
            f" seed_owned={seed_meta['path']} hold_gw={hold_gw}"
            if seed_meta
            else ""
        ),
        flush=True,
    )
    t_all = _now()
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    owned: list[str] | None = list(seed_meta["player_ids"]) if seed_meta else None
    start_ft = 1
    stash: pd.DataFrame | None = None
    steps_out: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for t in step_gws:
        print(f"[step-pool] === decision GW{t} ===", flush=True)
        hist = numagent_xp.asof_share_history(merged, before_gw=int(t), season=season)
        alpha_g, alpha_a = _asof_alphas(hist)
        team_hist = hist_by_team_norm(hist)
        select_pool = _build_hist_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        share_pool = _build_hist_share_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        if owned:
            # Seed GW: thin select may miss WC picks — inject from share first.
            if stash is None:
                select_pool = _inject_owned_from_share(select_pool, share_pool, owned)
            else:
                select_pool = _ensure_owned_in_pool(select_pool, stash, owned)
                share_pool = _ensure_owned_in_pool(share_pool, stash, owned)
                select_pool = _inject_owned_from_share(select_pool, share_pool, owned)
        stash = select_pool.copy()

        fixtures_gw = _hist_fixtures_by_gw(enriched, start=int(t), end=horizon_end)
        plan_gws = sorted(fixtures_gw)
        priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
        step_seed = seed0 if t == start_gw else seed0 + 10_000 * int(t)
        print(
            f"[step-pool] select={len(select_pool)} share={len(share_pool)} "
            f"plan={plan_gws[0]}…{plan_gws[-1]} alphas=({alpha_g:.3f},{alpha_a:.3f})",
            flush=True,
        )
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
            goal_weight=goal_weight,
        )
        print(f"[step-pool] boards {board_meta['elapsed_s']:.1f}s", flush=True)

        held_open = bool(seed_meta and owned is not None and int(t) == hold_gw)
        if held_open:
            week = _hold_week_from_owned(
                boards[int(t)],
                owned,
                gw=int(t),
                ft_before=start_ft,
                seed_xi_ids=seed_meta.get("xi_ids"),
                seed_captain_id=seed_meta.get("captain_id"),
            )
            print(
                f"[step-pool] GW{t} HOLD seed squad+XI (0 xfers) "
                f"ft {week['ft_before']}→{week['ft_after']} "
                f"cap={week.get('captain_id')}",
                flush=True,
            )
        else:
            if owned is None:
                owned = opening_15(select_pool, scores_by_gw, plan_gws)
                start_ft = 1
            plan = _plan_from_owned(
                select_pool,
                scores_by_gw,
                plan_gws,
                owned,
                start_ft=start_ft,
                top_n=int(args.cand_top_n),
                time_limit=float(args.milp_time),
            )
            week = _week_by_gw(plan["weeks"])[int(t)]
        xi = [str(x) for x in (week.get("xi_ids") or week.get("player_ids") or [])]
        # Prefer true XI (11); fall back to squad ids if missing
        if len(xi) > 11 and week.get("xi_ids"):
            xi = [str(x) for x in week["xi_ids"]]

        act = actual_components(merged, int(t))
        rows = score_xi_components(boards[int(t)], act, xi, gw=int(t))
        all_rows.extend(rows)

        prev = set(owned)
        owned = [str(x) for x in week["player_ids"]]
        start_ft = int(week["ft_after"])
        buys = sorted(set(owned) - prev)
        sells = sorted(prev - set(owned))
        id_to_name = {
            str(r.player_id): str(r.name) for r in select_pool.itertuples(index=False)
        }
        # share pool names for any thin-gate injects already in select
        for r in share_pool.itertuples(index=False):
            id_to_name.setdefault(str(r.player_id), str(r.name))
        owned_names = [id_to_name.get(x, x) for x in owned]
        bench_ids = [x for x in owned if x not in set(xi)]
        bench_names = [id_to_name.get(x, x) for x in bench_ids]
        steps_out.append(
            {
                "gw": int(t),
                "n_selection_pool": int(len(select_pool)),
                "n_share_pool": int(len(share_pool)),
                "alpha_g": float(alpha_g),
                "alpha_a": float(alpha_a),
                "board_s": float(board_meta["elapsed_s"]),
                "plan_gws": plan_gws,
                "xi_ids": xi,
                "xi_names": [id_to_name.get(x, x) for x in xi],
                "bench_ids": bench_ids,
                "bench_names": bench_names,
                "owned": list(owned),
                "owned_names": owned_names,
                "buys": [id_to_name.get(x, x) for x in buys],
                "sells": [id_to_name.get(x, x) for x in sells],
                "n_transfers": int(week["n_transfers"]),
                "hits": int(week["hits"]),
                "ft_before": int(week["ft_before"]),
                "ft_after": int(week["ft_after"]),
                "xi_score": float(week.get("xi_score") or week.get("week_score") or 0.0),
                "held_seed": held_open,
                # Only persist captain on seed-hold GWs (realised scoring).
                "captain_id": week.get("captain_id") if held_open else None,
                "summary": _summarise(pd.DataFrame(rows)),
            }
        )
        print(
            f"[step-pool] GW{t} commit xfers={week['n_transfers']} hits={week['hits']} "
            f"resid_G={steps_out[-1]['summary']['G']['sum_resid']:+.1f}",
            flush=True,
        )

    df = pd.DataFrame(all_rows)
    per_gw = []
    for gw, sub in df.groupby("gw"):
        n = max(int(len(sub)), 1)
        per_gw.append(
            {
                "gw": int(gw),
                "n": n,
                "xP_G": float(sub["xP_G"].sum()),
                "P_G": float(sub["P_G"].sum()),
                "resid_G": float(sub["resid_G"].sum()),
                "xP_CS": float(sub["xP_CS"].sum()),
                "P_CS": float(sub["P_CS"].sum()),
                "resid_CS": float(sub["resid_CS"].sum()),
                "xP_GA": float(sub["xP_GA"].sum()),
                "P_GA": float(sub["P_GA"].sum()),
                "resid_GA": float(sub["resid_GA"].sum()),
                "resid_G_per_match": float(sub["resid_G"].sum()) / n,
                "resid_CS_per_match": float(sub["resid_CS"].sum()) / n,
                "resid_GA_per_match": float(sub["resid_GA"].sum()) / n,
                "n_selection_pool": next(
                    s["n_selection_pool"] for s in steps_out if s["gw"] == int(gw)
                ),
            }
        )

    plot_path = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid.png")
    if seed_meta:
        plot_path = Path(
            "/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid_seeded.png"
        )
    if goal_weight == "rate":
        plot_path = plot_path.with_name(plot_path.stem + "_pi_r" + plot_path.suffix)
    _plot(per_gw, plot_path, n=n_strands)
    summary = _summarise(df)
    state = {
        "task": "cascade_mean_xi_ft_step_pool_resid",
        "season": season,
        "horizon": [start_gw, horizon_end],
        "step_gws": step_gws,
        "n_strands": n_strands,
        "seed": seed0,
        "seed_owned": seed_meta,
        "hold_gw": hold_gw if seed_meta else None,
        "goal_weight": goal_weight,
        "selection_pool": "as-of mean minutes > 60, n_apps≥2; frozen until next step",
        "share_pool": "full club m>0 (impact subs on dice)",
        "elapsed_s": _now() - t_all,
        "summary": summary,
        "per_gw": per_gw,
        "steps": steps_out,
        "plot": str(plot_path),
    }
    for s in state["steps"]:
        s["n_owned"] = len(s.get("owned") or [])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    seed_line = ""
    if seed_meta:
        seed_line = (
            f"- **Seed owned**: `{seed_meta['path']}` "
            f"(hold GW{hold_gw}: 0 xfers; seed XI+captain if present else "
            f"`pick_xi`; `advance_ft(ft,0)`). "
            f"Source: {seed_meta.get('source') or 'n/a'}."
        )
    gw_note = (
        "π∝r (minutes gate only)"
        if goal_weight == "rate"
        else "π∝r·m/90 (default)"
    )
    lines = [
        "# Stepped ≥60′ selection pool — xP(G, CS, GA) vs actual",
        "",
        f"Season **{season}** · horizon {start_gw}–{horizon_end} · "
        f"commit steps {step_gws} · N={n_strands} · seed={seed0} · "
        f"goal_weight=`{goal_weight}` ({gw_note}) · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## Protocol",
        "",
        "- At each decision GW t: rebuild **selection** pool = players with "
        "as-of mean minutes > 60 (and ≥2 apps) before t.",
        "- **Share** pool = full club with m>0 (impact subs on generative dice).",
        f"- **Goal shares**: `{goal_weight}` — {gw_note}. Assists stay ∝rA·m/90.",
        "- That as-of state (rates, minutes, both pools) is **projected forward** "
        "on strands for t…horizon end until the next step refreshes it.",
        "- Mean of N generative boards → `xi_ft` plan; commit week-t XI.",
        "- Residual = xP − actual on that XI. **GA** = goals-against points "
        "(`xp_gc` / FPL GC).",
    ]
    if seed_line:
        lines += ["", seed_line]
    lines += [
        "",
        f"![per-match resid]({plot_path})",
        "",
        "## Overall (committed GW8–12 XI)",
        "",
    ]
    if summary:
        lines += [
            f"- Player-GWs: **{summary['n_player_gw']}**",
            "",
            "| component | Σ xP | Σ actual | Σ resid | mean / match | MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for lab in ("G", "CS", "GA"):
            c = summary[lab]
            lines.append(
                f"| {lab} | {c['sum_xP']:.1f} | {c['sum_P']:.1f} | "
                f"{c['sum_resid']:+.1f} | {c['mean_resid']:+.3f} | {c['mae']:.3f} |"
            )
    lines += [
        "",
        "## Per committed GW",
        "",
        "| GW | select n | share n | xfers | hits | r_G | r_CS | r_GA | ΣxP_G | ΣP_G |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, pg in zip(steps_out, per_gw, strict=True):
        lines.append(
            f"| {s['gw']} | {s['n_selection_pool']} | {s['n_share_pool']} | "
            f"{s['n_transfers']} | {s['hits']} | "
            f"{pg['resid_G_per_match']:+.2f} | {pg['resid_CS_per_match']:+.2f} | "
            f"{pg['resid_GA_per_match']:+.2f} | {pg['xP_G']:.1f} | {pg['P_G']:.1f} |"
        )
    lines += ["", "## Squads (XI + bench)", ""]
    for s in steps_out:
        lines.append(
            f"### GW{s['gw']} — buy {s['buys']} / sell {s['sells']} · "
            f"ft {s['ft_before']}→{s['ft_after']} · hits={s['hits']}"
        )
        lines.append("")
        lines.append("**XI**")
        for i, name in enumerate(s.get("xi_names") or [], 1):
            lines.append(f"{i}. {name}")
        lines.append("")
        lines.append("**Bench**")
        for i, name in enumerate(s.get("bench_names") or [], 1):
            lines.append(f"{i}. {name}")
        lines.append("")
    repro = (
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid "
        f"--n {n_strands} --steps {','.join(str(g) for g in step_gws)}"
    )
    if seed_meta:
        repro += f" --seed-owned {seed_meta['path']}"
        if args.out:
            repro += f" --out {args.out}"
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        repro,
        "```",
        "",
    ]
    report_path = REPORT
    if goal_weight == "rate":
        report_path = ROOT / "reports" / "cascade_mean_xi_ft_step_pool_pi_r.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    change = (
        "Step GW8–12: refresh ≥60′ selection pool each GW; "
        f"full shares; goal_weight={goal_weight}; xP(G,CS,GA) vs actual"
    )
    if seed_meta:
        change = (
            f"Stepped cascade π∝r goal shares; L3/L4 GW{hold_gw} WC seed "
            f"(hold open; xi_ft from GW{hold_gw + 1})"
            if goal_weight == "rate"
            else (
                f"Re-run stepped cascade seeded with L3/L4 GW{hold_gw} WC 15 "
                f"(hold open; xi_ft from GW{hold_gw + 1})"
            )
        )
    STATE.write_text(
        json.dumps(
            {
                "task": (
                    "cascade_mean_xi_ft_step_pool_pi_r"
                    if goal_weight == "rate"
                    else "cascade_mean_xi_ft_step_pool_resid"
                ),
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": change,
                "verdict": summary,
                "artifact": str(args.out),
                "report": str(report_path.relative_to(ROOT)),
                "goal_weight": goal_weight,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("wrote", args.out, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
