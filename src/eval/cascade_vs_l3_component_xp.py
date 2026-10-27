"""Player xP(G, CS, GA): cascade generative vs L3/L4 score_xp_v2 vs actual.

No captaincy. Same GW8 team = L3/L4 WC open
(``data/processed/l3_l4_gw08_seed_2025_26.json``). Later GWs follow the
locked **L3** owned path (buys/sells from ``layer4_mupost_gamma1.json``);
XI = seed XI at GW8, else ``pick_xi`` on score_xp_v2 among owned.

For each XI player-GW:

- **cascade** — stepped generative mean ``xp_goals`` / ``xp_cs`` / ``xp_gc``
  (full-club shares; ≥60′ selection refreshed each step; N strands).
- **l3_v2** — as-of ``score_xp_v2`` via FD centreline ``score_team_side``
  (same assembler as L3/L4 mean boards). L4 uses the same scorer when
  μ_post mode is none.
- **actual** — FPL component points P(G)/P(CS)/P(GC) and ``total_points``.

  PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp --n 40
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
from src.eval.cascade_mean_xi_ft_hist import HORIZON_END
from src.eval.cascade_mean_xi_ft_step_pool_resid import (
    _inject_owned_from_share,
    _load_seed_owned,
    mean_component_boards_split,
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
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    load_prices,
)
from src.rules.fpl_2026 import normalize_position
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_vs_l3_component_xp.md"
STATE = ROOT / "REPORT_STATE.json"
SEED_JSON = ROOT / "data" / "processed" / "l3_l4_gw08_seed_2025_26.json"
L4_JSON = ROOT / "data" / "processed" / "layer4_mupost_gamma1.json"
STEP_GWS = (8, 9, 10, 11, 12)
N_STRANDS = 40


def _now() -> float:
    return time.time()


def _pid_element(pid: str) -> int:
    return int(str(pid).split(":")[-1])


def _as_id_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def _l3_owned_by_gw(
    seed_ids: list[str],
    l4_path: Path,
    *,
    season: str,
    gws: list[int],
    seed_gw: int | None = None,
) -> dict[int, list[str]]:
    """Replay L3 buys/sells from the seed GW through each requested GW.

    Mid-season windows must apply all intervening transfers, not only weeks
    inside ``gws`` (else owned resets to the WC open).
    """
    want = {int(g) for g in gws}
    open_gw = int(seed_gw if seed_gw is not None else min(want))
    raw = json.loads(Path(l4_path).read_text(encoding="utf-8"))
    weeks = [
        w
        for w in (raw.get("weeks") or [])
        if str(w.get("season")) == season
        and str(w.get("arm")) == "l3"
        and int(w.get("gw") or 0) >= open_gw
        and int(w.get("gw") or 0) <= max(want)
    ]
    weeks.sort(key=lambda w: int(w["gw"]))
    owned = list(seed_ids)
    out: dict[int, list[str]] = {}
    for w in weeks:
        gw = int(w["gw"])
        if gw == open_gw:
            owned = list(seed_ids)
        else:
            sold = set(_as_id_list(w.get("sold")))
            bought = _as_id_list(w.get("bought"))
            owned = [p for p in owned if p not in sold] + bought
            if len(owned) != 15:
                raise RuntimeError(
                    f"L3 owned size {len(owned)} at GW{gw} after "
                    f"sell={sorted(sold)} buy={bought}"
                )
        if gw in want:
            out[gw] = list(owned)
    missing = sorted(want - set(out))
    if missing:
        raise RuntimeError(f"L3 path missing owned for GWs {missing}")
    return out


def _force_side_rows(
    side: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    force_elements: set[int],
) -> pd.DataFrame:
    """Append owned/XI players cut by MIN_MEAN_M so the shared XI can be scored."""
    if not force_elements:
        return side
    have = set(pd.to_numeric(side["element"], errors="coerce").dropna().astype(int))
    need = sorted(force_elements - have)
    if not need:
        return side
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    rows: list[dict[str, Any]] = []
    for eid in need:
        g = hist.loc[pd.to_numeric(hist["element"], errors="coerce") == eid]
        if g.empty:
            continue
        m = float(pd.to_numeric(g["minutes"], errors="coerce").fillna(0.0).mean())
        sig = float(pd.to_numeric(g["minutes"], errors="coerce").fillna(0.0).std())
        if not np.isfinite(sig) or sig < 5.0:
            sig = 5.0
        row: dict[str, Any] = {
            "element": int(eid),
            "name": str(g.iloc[0]["name"]),
            "position": normalize_position(str(g.iloc[0]["position"])),
            "team": str(g.iloc[0]["team"]),
            "m": m,
            "sigma_m": sig,
        }
        if not rates.empty:
            rr = rates.loc[rates["element"] == eid]
            if not rr.empty:
                for c in ("rG", "drG", "rA", "drA"):
                    if c in rr.columns:
                        row[c] = float(rr.iloc[0][c])
        if not priors.empty:
            pr = priors.loc[priors["element"] == eid]
            if not pr.empty:
                for c in (
                    "bonus_mu",
                    "bonus_se",
                    "card_mu",
                    "card_se",
                    "saves_mu",
                    "saves_se",
                    "defcon_mu",
                    "defcon_sig",
                    "defcon_n",
                ):
                    if c in pr.columns and pd.notna(pr.iloc[0][c]):
                        row[c] = pr.iloc[0][c]
        rows.append(row)
    if not rows:
        return side
    add = pd.DataFrame(rows)
    # Align columns with side defaults
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.2),
        ("bonus_se", 0.1),
        ("card_mu", -0.1),
        ("card_se", 0.05),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", numagent_xp.SIG_FLOOR),
        ("defcon_n", 5),
    ):
        if c not in add.columns:
            add[c] = default
        if c in side.columns and c not in add.columns:
            add[c] = default
    for c in side.columns:
        if c not in add.columns:
            add[c] = np.nan if side[c].dtype == object else 0.0
    add = add[side.columns]
    return pd.concat([side, add], ignore_index=True)


def _v2_component_board(
    *,
    hist: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    season: str,
    decision_gw: int,
    alpha_g: float,
    alpha_a: float,
    force_ids: list[str] | None = None,
) -> pd.DataFrame:
    """As-of score_xp_v2 board with xp_goals / xp_cs / xp_gc kept.

    ``score_week_on_fixture_moments`` drops components when aggregating; score
    each side here and sum G/CS/GA ourselves (DGW-safe).
    """
    side, _, _ = build_frozen_side_table(
        hist,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        season=season,
        historical_rate_baseline=True,
    )
    if side.empty:
        return pd.DataFrame()
    if force_ids:
        force_e = {_pid_element(p) for p in force_ids}
        side = _force_side_rows(side, hist, force_elements=force_e)
    fits = centreline_fixture_fits(fixtures_for_gw(enriched, int(decision_gw)), fd)
    if not fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fits:
        for team, team_n, prefix in (
            (str(fit["home"]), str(fit["home_n"]), "home"),
            (str(fit["away"]), str(fit["away_n"]), "away"),
        ):
            club = side.loc[side["team"].astype(str) == team].copy()
            if club.empty and "team_n" in side.columns:
                club = side.loc[side["team_n"] == team_n].copy()
            if club.empty:
                continue
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fit[f"xg_{prefix}"]),
                xcs=float(fit[f"xcs_{prefix}"]),
                xgc=float(fit[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            )
            scored = scored.copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    all_p = pd.concat(parts, ignore_index=True)
    board = (
        all_p.groupby("element", as_index=False)
        .agg(
            name=("name", "first"),
            position=("position", "first"),
            team=("team", "first"),
            xp_goals=("xp_goals", "sum"),
            xp_cs=("xp_cs", "sum"),
            xp_gc=("xp_gc", "sum"),
            xp=("xp", "sum"),
        )
    )
    board["score_xp_v2"] = board["xp"]
    pr = prices.loc[prices["gw"] == int(decision_gw)].drop_duplicates("element")
    board["element"] = pd.to_numeric(board["element"], errors="coerce")
    board = board.merge(
        pr[["element", "value", "player_id", "team_norm"]],
        on="element",
        how="left",
    )
    board["player_id"] = board["player_id"].fillna(
        board["element"].map(lambda e: f"{season}:{int(e)}" if pd.notna(e) else "")
    ).astype(str)
    board["position"] = board["position"].map(
        lambda p: normalize_position(str(p)) if pd.notna(p) else p
    )
    return board


def _lookup(
    board: pd.DataFrame,
    pid: str,
    cols: tuple[str, ...],
) -> dict[str, float]:
    hit = board.loc[board["player_id"].astype(str) == str(pid)]
    if hit.empty:
        eid = _pid_element(pid)
        if "element" in board.columns:
            hit = board.loc[pd.to_numeric(board["element"], errors="coerce") == eid]
    out = {c: float("nan") for c in cols}
    if hit.empty:
        return out
    r = hit.iloc[0]
    for c in cols:
        out[c] = float(r[c]) if c in r.index and pd.notna(r[c]) else float("nan")
    return out


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
        default="rate_minutes",
        help="Cascade generative goal shares: rate_minutes (default) or π∝r.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_vs_l3_component_xp.json"),
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
        f"[comp-xp] season={season} steps={step_gws} N={n_strands} "
        f"goal_weight={goal_weight} seed={seed_meta['path']}",
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

    rows: list[dict[str, Any]] = []
    steps_out: list[dict[str, Any]] = []

    for t in step_gws:
        print(f"[comp-xp] === GW{t} ===", flush=True)
        owned = list(owned_by_gw[int(t)])
        hist = numagent_xp.asof_share_history(merged, before_gw=int(t), season=season)
        alpha_g, alpha_a = _asof_alphas(hist)
        team_hist = hist_by_team_norm(hist)

        # --- L3/L4 score_xp_v2 component board (force owned onto side) ---
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

        # XI: seed at GW8; else pick_xi on score_xp_v2 among owned
        if int(t) == int(seed_meta["gw"] or step_gws[0]) and seed_meta.get("xi_ids"):
            xi_ids = list(seed_meta["xi_ids"])
            xi_source = "seed_l3_l4_wc"
        else:
            squad = v2_board.loc[
                v2_board["player_id"].astype(str).isin(set(owned))
            ].copy()
            have = set(squad["player_id"].astype(str))
            missing = [p for p in owned if p not in have]
            if missing:
                raise RuntimeError(f"GW{t} owned missing from v2 board: {missing}")
            xi_df, _form = pick_xi(squad, "score_xp_v2")
            xi_ids = [str(x) for x in xi_df["player_id"].astype(str).tolist()]
            xi_source = "pick_xi_score_xp_v2"

        # --- Cascade generative mean boards (same stepped pools) ---
        select_pool = _build_hist_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        share_pool = _build_hist_share_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        select_pool = _inject_owned_from_share(select_pool, share_pool, owned)
        # also ensure XI look-ups exist
        select_pool = _inject_owned_from_share(select_pool, share_pool, xi_ids)
        fixtures_gw = _hist_fixtures_by_gw(enriched, start=int(t), end=horizon_end)
        plan_gws = sorted(fixtures_gw)
        priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
        step_seed = seed0 if t == step_gws[0] else seed0 + 10_000 * int(t)
        print(
            f"[comp-xp] select={len(select_pool)} share={len(share_pool)} "
            f"xi_src={xi_source}",
            flush=True,
        )
        boards, _scores, board_meta = mean_component_boards_split(
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
        print(f"[comp-xp] cascade boards {board_meta['elapsed_s']:.1f}s", flush=True)
        casc_board = boards[int(t)]

        act = actual_components(merged, int(t))
        act_by_e = act.set_index("element") if not act.empty else None
        g_merged = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(t)]

        id_to_name = {
            str(r.player_id): str(r.name) for r in select_pool.itertuples(index=False)
        }
        for r in v2_board.itertuples(index=False):
            id_to_name.setdefault(str(r.player_id), str(r.name))

        gw_rows: list[dict[str, Any]] = []
        for pid in xi_ids:
            eid = _pid_element(pid)
            casc = _lookup(casc_board, pid, ("xp_goals", "xp_cs", "xp_gc", "xp"))
            v2 = _lookup(
                v2_board, pid, ("xp_goals", "xp_cs", "xp_gc", "score_xp_v2", "xp")
            )
            if act_by_e is not None and eid in act_by_e.index:
                a = act_by_e.loc[eid]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                p_g, p_cs, p_ga = float(a["P_G"]), float(a["P_CS"]), float(a["P_GC"])
                name, pos, mins = str(a["name"]), str(a["position"]), float(a["minutes"])
            else:
                p_g = p_cs = p_ga = float("nan")
                name, pos, mins = id_to_name.get(pid, pid), "", float("nan")
            hit = g_merged.loc[g_merged["element"] == eid]
            total_pts = (
                float(hit.iloc[0]["total_points"] or 0.0) if not hit.empty else float("nan")
            )
            row = {
                "gw": int(t),
                "player_id": str(pid),
                "element": eid,
                "name": name,
                "position": pos,
                "minutes": mins,
                "casc_xP_G": casc["xp_goals"],
                "casc_xP_CS": casc["xp_cs"],
                "casc_xP_GA": casc["xp_gc"],
                "casc_xP_sum": casc["xp_goals"] + casc["xp_cs"] + casc["xp_gc"],
                "l3_xP_G": v2["xp_goals"],
                "l3_xP_CS": v2["xp_cs"],
                "l3_xP_GA": v2["xp_gc"],
                "l3_xP_sum": v2["xp_goals"] + v2["xp_cs"] + v2["xp_gc"],
                "l3_score_xp_v2": v2.get("score_xp_v2", v2.get("xp")),
                "P_G": p_g,
                "P_CS": p_cs,
                "P_GA": p_ga,
                "P_sum": p_g + p_cs + p_ga,
                "total_points": total_pts,
            }
            gw_rows.append(row)
            rows.append(row)

        steps_out.append(
            {
                "gw": int(t),
                "xi_source": xi_source,
                "xi_ids": xi_ids,
                "xi_names": [id_to_name.get(x, x) for x in xi_ids],
                "owned": owned,
                "n_selection_pool": int(len(select_pool)),
                "n_share_pool": int(len(share_pool)),
                "alpha_g": float(alpha_g),
                "alpha_a": float(alpha_a),
                "cascade_board_s": float(board_meta["elapsed_s"]),
                "sum_casc_xP_G": float(np.nansum([r["casc_xP_G"] for r in gw_rows])),
                "sum_casc_xP_CS": float(np.nansum([r["casc_xP_CS"] for r in gw_rows])),
                "sum_casc_xP_GA": float(np.nansum([r["casc_xP_GA"] for r in gw_rows])),
                "sum_l3_xP_G": float(np.nansum([r["l3_xP_G"] for r in gw_rows])),
                "sum_l3_xP_CS": float(np.nansum([r["l3_xP_CS"] for r in gw_rows])),
                "sum_l3_xP_GA": float(np.nansum([r["l3_xP_GA"] for r in gw_rows])),
                "sum_P_G": float(np.nansum([r["P_G"] for r in gw_rows])),
                "sum_P_CS": float(np.nansum([r["P_CS"] for r in gw_rows])),
                "sum_P_GA": float(np.nansum([r["P_GA"] for r in gw_rows])),
                "sum_total_points": float(
                    np.nansum([r["total_points"] for r in gw_rows])
                ),
                "n_missing_casc": int(
                    sum(1 for r in gw_rows if not np.isfinite(r["casc_xP_G"]))
                ),
                "n_missing_l3": int(
                    sum(1 for r in gw_rows if not np.isfinite(r["l3_xP_G"]))
                ),
            }
        )
        print(
            f"[comp-xp] GW{t} Σ casc G/CS/GA="
            f"{steps_out[-1]['sum_casc_xP_G']:.1f}/"
            f"{steps_out[-1]['sum_casc_xP_CS']:.1f}/"
            f"{steps_out[-1]['sum_casc_xP_GA']:.1f} "
            f"l3={steps_out[-1]['sum_l3_xP_G']:.1f}/"
            f"{steps_out[-1]['sum_l3_xP_CS']:.1f}/"
            f"{steps_out[-1]['sum_l3_xP_GA']:.1f} "
            f"P={steps_out[-1]['sum_P_G']:.1f}/"
            f"{steps_out[-1]['sum_P_CS']:.1f}/"
            f"{steps_out[-1]['sum_P_GA']:.1f} "
            f"total_pts={steps_out[-1]['sum_total_points']:.0f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    summary = {
        "n_player_gw": int(len(df)),
        "sum_casc_xP_G": float(df["casc_xP_G"].sum()),
        "sum_casc_xP_CS": float(df["casc_xP_CS"].sum()),
        "sum_casc_xP_GA": float(df["casc_xP_GA"].sum()),
        "sum_l3_xP_G": float(df["l3_xP_G"].sum()),
        "sum_l3_xP_CS": float(df["l3_xP_CS"].sum()),
        "sum_l3_xP_GA": float(df["l3_xP_GA"].sum()),
        "sum_P_G": float(df["P_G"].sum()),
        "sum_P_CS": float(df["P_CS"].sum()),
        "sum_P_GA": float(df["P_GA"].sum()),
        "sum_total_points": float(df["total_points"].sum()),
        "mae_casc_G": float((df["casc_xP_G"] - df["P_G"]).abs().mean()),
        "mae_l3_G": float((df["l3_xP_G"] - df["P_G"]).abs().mean()),
        "mae_casc_CS": float((df["casc_xP_CS"] - df["P_CS"]).abs().mean()),
        "mae_l3_CS": float((df["l3_xP_CS"] - df["P_CS"]).abs().mean()),
        "mae_casc_GA": float((df["casc_xP_GA"] - df["P_GA"]).abs().mean()),
        "mae_l3_GA": float((df["l3_xP_GA"] - df["P_GA"]).abs().mean()),
    }
    gw_note = (
        "π∝r (minutes gate only)"
        if goal_weight == "rate"
        else "π∝r·m/90 (default)"
    )
    state = {
        "task": (
            "cascade_vs_l3_component_xp_pi_r"
            if goal_weight == "rate"
            else "cascade_vs_l3_component_xp"
        ),
        "season": season,
        "step_gws": step_gws,
        "n_strands": n_strands,
        "seed": seed0,
        "goal_weight": goal_weight,
        "seed_owned": seed_meta,
        "shared_team": (
            "GW8 = L3/L4 WC XI; GW9–12 = L3 owned path + pick_xi(score_xp_v2). "
            "No captain. Both scorers evaluated on that same XI."
        ),
        "elapsed_s": _now() - t_all,
        "summary": summary,
        "per_gw": steps_out,
        "players": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    title = (
        "# Cascade π∝r vs L3/L4 — player xP(G, CS, GA) + actual (no captain)"
        if goal_weight == "rate"
        else "# Cascade vs L3/L4 — player xP(G, CS, GA) + actual (no captain)"
    )
    lines = [
        title,
        "",
        f"Season **{season}** · GW {step_gws} · N={n_strands} · "
        f"goal_weight=`{goal_weight}` ({gw_note}) · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## What this is",
        "",
        "- **Same GW8 team** for both arms: L3/L4 wildcard open "
        f"(`{Path(seed_meta['path']).name}`).",
        "- GW9–12 follow the locked **L3** transfer path; XI = `pick_xi` on "
        "`score_xp_v2` among that owned 15.",
        "- Both scorers are read on **that same XI** (no captain, no hits):",
        f"  - **cascade** — generative mean `xp_goals` / `xp_cs` / `xp_gc` "
        f"(goal shares `{goal_weight}`)",
        "  - **l3_v2** — as-of `score_xp_v2` centreline components "
        "(L4 μ_post≡v2 when mode=none; default π∝r·m)",
        "  - **actual** — FPL `P_G` / `P_CS` / `P_GA` and full `total_points`",
        "",
        "## XI totals by GW (no captain)",
        "",
        "| GW | casc ΣG | casc ΣCS | casc ΣGA | l3 ΣG | l3 ΣCS | l3 ΣGA | "
        "P ΣG | P ΣCS | P ΣGA | Σ total_pts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in steps_out:
        lines.append(
            f"| {s['gw']} | {s['sum_casc_xP_G']:.1f} | {s['sum_casc_xP_CS']:.1f} | "
            f"{s['sum_casc_xP_GA']:.1f} | {s['sum_l3_xP_G']:.1f} | "
            f"{s['sum_l3_xP_CS']:.1f} | {s['sum_l3_xP_GA']:.1f} | "
            f"{s['sum_P_G']:.1f} | {s['sum_P_CS']:.1f} | {s['sum_P_GA']:.1f} | "
            f"{s['sum_total_points']:.0f} |"
        )
    lines += [
        "",
        f"**Overall** ({summary['n_player_gw']} player-GWs): "
        f"casc ΣG/CS/GA = {summary['sum_casc_xP_G']:.1f}/"
        f"{summary['sum_casc_xP_CS']:.1f}/{summary['sum_casc_xP_GA']:.1f}; "
        f"l3 = {summary['sum_l3_xP_G']:.1f}/{summary['sum_l3_xP_CS']:.1f}/"
        f"{summary['sum_l3_xP_GA']:.1f}; "
        f"actual = {summary['sum_P_G']:.1f}/{summary['sum_P_CS']:.1f}/"
        f"{summary['sum_P_GA']:.1f}; "
        f"Σ total_points = {summary['sum_total_points']:.0f}.",
        "",
        "| component | MAE casc | MAE l3_v2 |",
        "| --- | ---: | ---: |",
        f"| G | {summary['mae_casc_G']:.3f} | {summary['mae_l3_G']:.3f} |",
        f"| CS | {summary['mae_casc_CS']:.3f} | {summary['mae_l3_CS']:.3f} |",
        f"| GA | {summary['mae_casc_GA']:.3f} | {summary['mae_l3_GA']:.3f} |",
        "",
        "## Player rows",
        "",
    ]
    for s in steps_out:
        lines.append(
            f"### GW{s['gw']} — {s['xi_source']} · "
            f"Σ total_pts={s['sum_total_points']:.0f}"
        )
        lines.append("")
        lines.append(
            "| player | pos | m | casc G | casc CS | casc GA | "
            "l3 G | l3 CS | l3 GA | P_G | P_CS | P_GA | total_pts |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        )
        for r in df.loc[df["gw"] == s["gw"]].itertuples(index=False):
            lines.append(
                f"| {r.name} | {r.position} | {r.minutes:.0f} | "
                f"{r.casc_xP_G:.2f} | {r.casc_xP_CS:.2f} | {r.casc_xP_GA:.2f} | "
                f"{r.l3_xP_G:.2f} | {r.l3_xP_CS:.2f} | {r.l3_xP_GA:.2f} | "
                f"{r.P_G:.0f} | {r.P_CS:.0f} | {r.P_GA:.0f} | "
                f"{r.total_points:.0f} |"
            )
        lines.append("")
    lines += [
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_vs_l3_component_xp "
        f"--n {n_strands} --goal-weight {goal_weight}",
        "```",
        "",
    ]
    report_path = REPORT
    if goal_weight == "rate":
        report_path = ROOT / "reports" / "cascade_vs_l3_component_xp_pi_r.md"
    # Distinct report when the step window is not the GW8–12 default.
    if step_gws != list(STEP_GWS):
        tag = f"gw{step_gws[0]}-{step_gws[-1]}"
        stem = (
            "cascade_vs_l3_component_xp_pi_r"
            if goal_weight == "rate"
            else "cascade_vs_l3_component_xp"
        )
        report_path = ROOT / "reports" / f"{stem}_{tag}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": state["task"],
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": (
                    "π∝r cascade xP(G,CS,GA) vs L3/L4 on shared L3 path; "
                    "actuals; no captain"
                    if goal_weight == "rate"
                    else (
                        "Player xP(G,CS,GA) cascade vs L3/L4 on shared L3 path "
                        "(same GW8 WC XI); actuals; no captain"
                    )
                ),
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
