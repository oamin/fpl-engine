"""Per-GW residual: mean MILP vs stable cascade path (minutes oracle + autosub).

Protocol: ``cascade_mean_path_gw_resid`` (diagnostic).

  - **Full-side shares:** attack pots/shares built on the full minutes-oracle
    club (mean boards) / as-of share roster (path generative).
  - **Semi-oracle buy filter** at decision GW ``t``: only players with actual
    minutes ``> 60`` in GW ``t`` may be bought (applied *after* scoring).
  - **Minutes oracle** on residual/mean boards: actual ``m``, ``σ_m=0``.
  - **Bench autosub + VC** when banking actuals.

Arms:
  - **Mean:** ``xi_plus_captain`` on priced t:t+1 mean boards; held.
  - **Path:** strand-mean of held ``xi_plus_captain_ft`` lineups.

  PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid
  PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid --season 2025-26 --n 12
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import (
    DECISION_GW,
    DEFAULT_SEASON,
    HORIZON_END,
    N_STRANDS,
    SEASON_E0,
    SEED,
    SNAPSHOT_EVERY,
    START_OFFSET,
    _build_hist_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_GAMMA,
    DEFAULT_LINEUP_COV,
    OBJECTIVE_XI_CAPTAIN_FT,
    _pick_squad_xi_captain,
    init_paths_from_opta,
    run_strand,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.season_climb import apply_autosubs, ordered_bench
from src.models.short_horizon_cascade import (
    attach_prices_and_cap,
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    load_prices,
    score_week_on_fixture_moments,
)
from src.rules.fpl_2026 import normalize_position
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mean_path_gw_resid.md"
STATE_JSON = ROOT / "data/processed/cascade_mean_path_gw_resid" / "state.json"
MIN_ORACLE_MINUTES = 60.0


def _out_dir(season: str) -> Path:
    return ROOT / "data" / "predictions" / str(season) / "cascade_mean_path_gw_resid"


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def _element_from_pid(pid: str) -> int | None:
    text = str(pid)
    if ":" in text:
        text = text.split(":")[-1]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _pid_variants(pid: str, season: str) -> set[str]:
    """Match both ``430`` and ``2025-26:430`` id forms."""
    out = {str(pid)}
    eid = _element_from_pid(pid)
    if eid is not None:
        out.add(str(eid))
        out.add(f"{season}:{eid}")
    return out


def semi_oracle_elements(
    merged: pd.DataFrame, gw: int, *, min_minutes: float = MIN_ORACLE_MINUTES
) -> set[int]:
    """Players with actual minutes > ``min_minutes`` in GW ``gw``."""
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
    if g.empty:
        return set()
    mins = pd.to_numeric(g["minutes"], errors="coerce").fillna(0.0)
    keep = g.loc[mins > float(min_minutes), "element"]
    return {int(x) for x in pd.to_numeric(keep, errors="coerce").dropna().astype(int)}


def apply_minutes_oracle(
    side: pd.DataFrame, merged: pd.DataFrame, gw: int
) -> pd.DataFrame:
    """Overwrite ``m`` / ``xmi`` with actual minutes; ``σ_m=0``."""
    out = side.copy()
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
    by_el = {
        int(r.element): float(r.minutes or 0.0)
        for r in g.itertuples(index=False)
        if pd.notna(r.element)
    }
    m = []
    for eid in pd.to_numeric(out["element"], errors="coerce"):
        if pd.isna(eid):
            m.append(0.0)
        else:
            m.append(float(by_el.get(int(eid), 0.0)))
    out["m"] = m
    out["xmi"] = m
    out["sigma_m"] = 0.0
    return out


def _mean_board_oracle(
    side: pd.DataFrame,
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    season: str,
    decision_gw: int,
    gw: int,
    alpha_g: float,
    alpha_a: float,
    pool_ids: set[str],
) -> pd.DataFrame:
    """Minutes-oracle board on the **full** side; buy-filter to ``pool_ids``.

    Shares / attack pots are built on everyone with actual minutes that GW.
    Semi-oracle pool is applied only after scoring (selection filter).
    """
    oracle_side = apply_minutes_oracle(side, merged, gw)
    fits = centreline_fixture_fits(fixtures_for_gw(enriched, gw), fd)
    if not fits:
        return pd.DataFrame()
    board = score_week_on_fixture_moments(
        oracle_side, fits, alpha_g=alpha_g, alpha_a=alpha_a
    )
    if board.empty:
        return board
    board = attach_prices_and_cap(
        board,
        prices,
        season=season,
        decision_gw=decision_gw,
        selectable_ids=pool_ids,
        apply_player_pool=True,
        force_ids=None,
    )
    return board


def run_mean_arm_oracle(
    *,
    side: pd.DataFrame,
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    season: str,
    decision_gw: int,
    priced_gws: list[int],
    alpha_g: float,
    alpha_a: float,
    pool_ids: set[str],
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, Any]:
    """xi+captain MILP on discounted minutes-oracle mean boards (t:t+1)."""
    t0 = _now()
    if len(priced_gws) < 1:
        raise RuntimeError("no priced GWs for mean arm")
    boards: list[pd.DataFrame] = []
    for gw in priced_gws[:2]:
        b = _mean_board_oracle(
            side,
            merged,
            enriched,
            fd,
            prices,
            season=season,
            decision_gw=decision_gw,
            gw=int(gw),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            pool_ids=pool_ids,
        )
        if b.empty:
            raise RuntimeError(f"empty mean board GW{gw}")
        boards.append(b)

    base = boards[0].copy()
    base["mc_score"] = pd.to_numeric(base["score_xp_v2"], errors="coerce").fillna(0.0)
    if len(boards) > 1:
        b1 = boards[1].set_index(pd.to_numeric(boards[1]["element"], errors="coerce"))
        add = []
        for eid in pd.to_numeric(base["element"], errors="coerce"):
            if eid in b1.index:
                row = b1.loc[eid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                add.append(float(row["score_xp_v2"]) * float(gamma))
            else:
                add.append(0.0)
        base["mc_score"] = base["mc_score"] + np.asarray(add, float)

    def _club_key(row: pd.Series) -> str:
        for key in ("team_norm", "club", "team"):
            if key in row.index and pd.notna(row[key]):
                s = str(row[key]).strip()
                if s and s.lower() != "nan":
                    return norm_team(s)
        return "unk"

    base["team_norm"] = [_club_key(row) for _, row in base.iterrows()]
    if "name" not in base.columns:
        base["name"] = base["player_id"].astype(str)
    if "position" in base.columns:
        base["position"] = base["position"].map(
            lambda p: normalize_position(str(p)) if pd.notna(p) else "MID"
        )
    picked = _pick_squad_xi_captain(base, "mc_score", time_limit=20.0)
    squad = picked["squad"]
    return {
        "elapsed_s": _now() - t0,
        "priced_gws": list(priced_gws[:2]),
        "objective": "xi_plus_captain",
        "objective_value": float(picked["objective"]),
        "captain_id": picked["captain_id"],
        "xi_ids": list(picked["xi_ids"]),
        "player_ids": [str(x) for x in squad["player_id"].tolist()],
        "squad": [
            {
                "player_id": str(r.player_id),
                "name": str(r.name),
                "position": str(r.position),
                "team_norm": str(getattr(r, "team_norm", "")),
                "mc_score": float(r.mc_score),
                "in_xi": str(r.player_id) in set(picked["xi_ids"]),
                "captain": str(r.player_id) == str(picked["captain_id"]),
            }
            for r in squad.itertuples(index=False)
        ],
    }


def _held_lineups(
    strand_squads: list[dict[str, Any]], horizon_gws: list[int]
) -> dict[int, dict[str, Any]]:
    """Map each horizon GW → held 15 + XI/captain from latest snapshot ≤ GW."""
    by_snap = {int(s["gw"]): s for s in strand_squads if s.get("xi_ids")}
    snaps = sorted(by_snap)
    out: dict[int, dict[str, Any]] = {}
    if not snaps:
        return out
    si = 0
    cur = None
    for gw in horizon_gws:
        while si < len(snaps) and snaps[si] <= int(gw):
            cur = by_snap[snaps[si]]
            si += 1
        if cur is None:
            continue
        out[int(gw)] = {
            "player_ids": list(cur.get("player_ids") or []),
            "xi_ids": list(cur["xi_ids"]),
            "captain_id": cur.get("captain_id"),
            "snapshot_gw": int(cur["gw"]),
        }
    return out


def run_path_strands(
    *,
    pool: pd.DataFrame,
    fixtures_gw: dict[int, list[dict[str, str]]],
    priced_by_gw: dict[int, list[dict[str, Any]]],
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    hist_by_team: dict[str, pd.DataFrame],
    alpha_g: float,
    alpha_a: float,
    horizon_gws: list[int],
    snapshot_gws: list[int],
    n_strands: int,
    seed: int,
    selectable_ids: set[str] | None = None,
    ft_pe_nu: float | None = None,
    path_plan_mode: str = "open_loop",
    merged: pd.DataFrame | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    """Path strands. ``pool`` is the share/scoring roster; ``selectable_ids`` is the buy filter.

    ``ft_pe_nu`` defaults to ``DEFAULT_FT_PE_NU`` (=HIT_COST): terminal PE
    ``U=ν·f`` on the joint plan. ``path_plan_mode=deadline_causal`` requires
    ``merged`` + ``season`` (oracle minutes only at each deadline GW).
    """
    from src.models.multistage_mc_milp import DEFAULT_FT_PE_NU as _DEFAULT_FT_PE_NU

    if ft_pe_nu is None:
        ft_pe_nu = float(_DEFAULT_FT_PE_NU)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    rng = random.Random(int(seed))
    strands: list[dict[str, Any]] = []
    t0 = _now()
    for i in range(int(n_strands)):
        print(f"  [path] strand {i + 1}/{n_strands}", flush=True)
        strand_rng = random.Random(rng.randint(0, 2**31 - 1))
        res = run_strand(
            strand_id=i,
            fixtures_gw=fixtures_gw,
            snapshot_gws=snapshot_gws,
            horizon_gws=horizon_gws,
            pool=pool,
            paths0=paths0,
            league_gf=league_gf,
            rng=strand_rng,
            gamma=DEFAULT_GAMMA,
            objective=OBJECTIVE_XI_CAPTAIN_FT,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            hist_by_team=hist_by_team,
            generative_attack=True,
            lineup_cov=DEFAULT_LINEUP_COV,
            priced_by_gw=priced_by_gw,
            ranks=ranks,
            profile=profile,
            selectable_ids=selectable_ids,
            ft_pe_nu=float(ft_pe_nu),
            path_plan_mode=str(path_plan_mode),
            merged=merged,
            season=season,
        )
        strands.append(
            {
                "strand_id": i,
                "squads": list(res.squads),
                "held": _held_lineups(list(res.squads), horizon_gws),
            }
        )
    return {
        "elapsed_s": _now() - t0,
        "n_strands": int(n_strands),
        "strands_with_squads": sum(1 for s in strands if s["squads"]),
        "strands": strands,
    }


def _squad_frame_for_bank(
    *,
    player_ids: list[str],
    xi_ids: list[str],
    captain_id: str | None,
    board: pd.DataFrame,
    merged_gw: pd.DataFrame,
    season: str,
    pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a 15-row frame for autosub banking (μ, minutes, points, position)."""
    mu_col = "score_xp_v2" if "score_xp_v2" in board.columns else "mc_score"
    board_by: dict[str, pd.Series] = {}
    for row in board.itertuples(index=False):
        pid = str(getattr(row, "player_id", ""))
        for v in _pid_variants(pid, season):
            board_by[v] = row
        eid = getattr(row, "element", None)
        if eid is not None and pd.notna(eid):
            board_by[str(int(eid))] = row
            board_by[f"{season}:{int(eid)}"] = row

    pts_by: dict[int, tuple[float, float]] = {}
    pos_by: dict[int, str] = {}
    name_by: dict[int, str] = {}
    for row in merged_gw.itertuples(index=False):
        try:
            eid = int(row.element)
        except (TypeError, ValueError):
            continue
        pts_by[eid] = (float(row.minutes or 0.0), float(row.total_points or 0.0))
        pos_by[eid] = normalize_position(str(row.position)) if pd.notna(row.position) else "MID"
        name_by[eid] = str(row.name)

    pool_pos: dict[str, str] = {}
    pool_name: dict[str, str] = {}
    if pool is not None and not pool.empty:
        for row in pool.itertuples(index=False):
            pid = str(row.player_id)
            pool_pos[pid] = normalize_position(str(row.position))
            pool_name[pid] = str(getattr(row, "name", pid))
            for v in _pid_variants(pid, season):
                pool_pos[v] = pool_pos[pid]
                pool_name[v] = pool_name[pid]

    rows: list[dict[str, Any]] = []
    for pid in player_ids:
        eid = _element_from_pid(pid)
        mins, pts = pts_by.get(eid, (0.0, 0.0)) if eid is not None else (0.0, 0.0)
        br = None
        for v in _pid_variants(pid, season):
            if v in board_by:
                br = board_by[v]
                break
        mu = float(getattr(br, mu_col, 0.0) or 0.0) if br is not None else 0.0
        pos = pos_by.get(eid) if eid is not None else None
        if pos is None:
            for v in _pid_variants(pid, season):
                if v in pool_pos:
                    pos = pool_pos[v]
                    break
        if pos is None and br is not None and hasattr(br, "position"):
            pos = normalize_position(str(br.position))
        if pos is None:
            pos = "MID"
        name = name_by.get(eid) if eid is not None else None
        if name is None:
            for v in _pid_variants(pid, season):
                if v in pool_name:
                    name = pool_name[v]
                    break
        rows.append(
            {
                "player_id": str(pid),
                "name": name or str(pid),
                "position": pos,
                "mu": mu,
                "minutes": mins,
                "total_points": pts,
                "in_xi": any(
                    str(pid) in _pid_variants(x, season) or x in _pid_variants(pid, season)
                    for x in xi_ids
                ),
                "captain": bool(
                    captain_id
                    and (
                        str(pid) in _pid_variants(str(captain_id), season)
                        or str(captain_id) in _pid_variants(pid, season)
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def bank_locked_actual(
    squad_df: pd.DataFrame,
    *,
    xi_ids: list[str],
    captain_id: str | None,
    season: str,
) -> dict[str, Any]:
    """Actual points for a locked XI with FPL autosubs + VC if captain blanks."""
    if squad_df.empty or not xi_ids:
        return {"actual": 0.0, "n_autosubs": 0, "n_blank_intended": 0}

    # Resolve XI rows (id-form flexible).
    xi_set: set[str] = set()
    for x in xi_ids:
        xi_set |= _pid_variants(str(x), season)
    xi = squad_df.loc[squad_df["player_id"].astype(str).isin(xi_set)].copy()
    if len(xi) < 11:
        # fall back: match by element
        want = {_element_from_pid(x) for x in xi_ids}
        want.discard(None)
        xi = squad_df.loc[
            squad_df["player_id"].map(_element_from_pid).isin(want)
        ].copy()
    if len(xi) > 11:
        xi = xi.drop_duplicates("player_id", keep="first").head(11)

    n_blank = int((pd.to_numeric(xi["minutes"], errors="coerce").fillna(0.0) <= 0).sum())
    bench = ordered_bench(squad_df, xi, "mu")
    final_xi, n_subs = apply_autosubs(xi, bench, minutes_col="minutes")

    pts = pd.to_numeric(final_xi["total_points"], errors="coerce").fillna(0.0)
    pts_sum = float(pts.sum())

    # Captain / VC on *intended* XI by μ; VC doubles if captain blanked.
    xi_mu = pd.to_numeric(xi["mu"], errors="coerce").fillna(0.0)
    order = xi_mu.sort_values(ascending=False, kind="mergesort")
    if captain_id:
        cap_id = str(captain_id)
    elif len(order):
        cap_id = str(xi.loc[order.index[0], "player_id"])
    else:
        cap_id = ""
    vc_id = (
        str(xi.loc[order.index[1], "player_id"])
        if len(order) > 1
        else cap_id
    )
    squad_mins = {
        str(r.player_id): float(r.minutes or 0.0) for r in squad_df.itertuples(index=False)
    }
    squad_pts = {
        str(r.player_id): float(r.total_points or 0.0)
        for r in squad_df.itertuples(index=False)
    }
    # id-flexible lookup
    def _lookup(pid: str, table: dict[str, float]) -> float:
        for v in _pid_variants(pid, season):
            if v in table:
                return float(table[v])
        return 0.0

    if _lookup(cap_id, squad_mins) > 0:
        cap_extra = _lookup(cap_id, squad_pts)
    elif _lookup(vc_id, squad_mins) > 0:
        cap_extra = _lookup(vc_id, squad_pts)
    else:
        cap_extra = 0.0

    return {
        "actual": pts_sum + float(cap_extra),
        "n_autosubs": int(n_subs),
        "n_blank_intended": n_blank,
        "captain_id": cap_id,
        "vice_id": vc_id,
    }


def _xi_pred_from_board(
    board: pd.DataFrame, xi_ids: list[str], captain_id: str | None, season: str
) -> float:
    col = "score_xp_v2" if "score_xp_v2" in board.columns else "mu_post"
    mu: dict[str, float] = {}
    for row in board.itertuples(index=False):
        val = float(getattr(row, col, 0.0) or 0.0)
        pid = str(getattr(row, "player_id", ""))
        for v in _pid_variants(pid, season):
            mu[v] = val
        eid = getattr(row, "element", None)
        if eid is not None and pd.notna(eid):
            mu[str(int(eid))] = val
            mu[f"{season}:{int(eid)}"] = val
    total = 0.0
    for pid in xi_ids:
        for v in _pid_variants(str(pid), season):
            if v in mu:
                total += mu[v]
                break
    if captain_id:
        for v in _pid_variants(str(captain_id), season):
            if v in mu:
                total += mu[v]
                break
    return float(total)


def per_gw_residuals(
    *,
    mean: dict[str, Any],
    path: dict[str, Any],
    side: pd.DataFrame,
    pool: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    merged: pd.DataFrame,
    season: str,
    decision_gw: int,
    horizon_gws: list[int],
    alpha_g: float,
    alpha_a: float,
    pool_ids: set[str],
) -> list[dict[str, Any]]:
    mean_xi = list(mean.get("xi_ids") or [])
    mean_cap = mean.get("captain_id")
    mean_15 = list(mean.get("player_ids") or [s["player_id"] for s in mean.get("squad") or []])
    rows: list[dict[str, Any]] = []
    for gw in horizon_gws:
        board = _mean_board_oracle(
            side,
            merged,
            enriched,
            fd,
            prices,
            season=season,
            decision_gw=decision_gw,
            gw=int(gw),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            pool_ids=pool_ids,
        )
        gw_rows = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]

        mean_pred = _xi_pred_from_board(board, mean_xi, mean_cap, season)
        mean_squad = _squad_frame_for_bank(
            player_ids=mean_15,
            xi_ids=mean_xi,
            captain_id=mean_cap,
            board=board,
            merged_gw=gw_rows,
            season=season,
            pool=pool,
        )
        mean_bank = bank_locked_actual(
            mean_squad, xi_ids=mean_xi, captain_id=mean_cap, season=season
        )

        path_preds: list[float] = []
        path_acts: list[float] = []
        path_subs: list[int] = []
        for st in path.get("strands") or []:
            held = (st.get("held") or {}).get(int(gw))
            if not held:
                continue
            xi = list(held["xi_ids"])
            cap = held.get("captain_id")
            pids = list(held.get("player_ids") or xi)
            path_preds.append(_xi_pred_from_board(board, xi, cap, season))
            sdf = _squad_frame_for_bank(
                player_ids=pids,
                xi_ids=xi,
                captain_id=cap,
                board=board,
                merged_gw=gw_rows,
                season=season,
                pool=pool,
            )
            bank = bank_locked_actual(sdf, xi_ids=xi, captain_id=cap, season=season)
            path_acts.append(float(bank["actual"]))
            path_subs.append(int(bank["n_autosubs"]))
        if not path_preds:
            continue
        path_pred = float(np.mean(path_preds))
        path_act = float(np.mean(path_acts))
        rows.append(
            {
                "gw": int(gw),
                "mean_pred": mean_pred,
                "mean_actual": float(mean_bank["actual"]),
                "mean_resid": mean_pred - float(mean_bank["actual"]),
                "mean_n_autosubs": int(mean_bank["n_autosubs"]),
                "mean_n_blank_intended": int(mean_bank["n_blank_intended"]),
                "path_pred": path_pred,
                "path_actual": path_act,
                "path_resid": path_pred - path_act,
                "path_mean_autosubs": float(np.mean(path_subs)) if path_subs else 0.0,
                "delta_actual_path_minus_mean": path_act - float(mean_bank["actual"]),
                "delta_resid_path_minus_mean": (path_pred - path_act)
                - (mean_pred - float(mean_bank["actual"])),
                "n_path_strands": len(path_preds),
                "n_board": int(len(board)),
            }
        )
    return rows


def write_report(state: dict[str, Any], out_dir: Path) -> None:
    rows = state.get("per_gw") or []
    lines = [
        "# Mean vs stable cascade path — per-GW residual (oracle minutes + autosub)",
        "",
        f"Protocol: `cascade_mean_path_gw_resid` · season **{state['season']}** · "
        f"decision GW{state['decision_gw']} → horizon "
        f"{state['horizon_start']}–{state['horizon_end']} · "
        f"N={state['n_strands']} · seed={state['seed']}",
        "",
        "**Diagnostic.**",
        f"- **Shares** on the full minutes-oracle side / as-of share roster "
        f"(n_share={state.get('n_share_roster')}).",
        f"- **Buy filter (semi-oracle)** at GW{state['decision_gw']}: actual "
        f"minutes > {MIN_ORACLE_MINUTES:.0f} (n_buy={state.get('n_buy_pool')}).",
        "- **Minutes oracle** on residual/mean boards: actual `m`, `σ_m=0` "
        "for the scored GW (no GW(t+n) foresight in path generative minutes).",
        "- **Actuals:** locked XI + FPL bench autosub + VC if captain blanks.",
        "- Mean XI locked from priced t:t+1 MILP and held; path = strand-mean "
        "of held `xi_plus_captain_ft` lineups.",
        "- Pred = oracle-minutes FD `score_xp_v2` on planned XI (+ captain).",
        "- Residual = pred − actual.",
        "",
        f"- As-of rates/priors: GW `< {state['decision_gw']}` "
        "(historical rate baseline); minutes on boards are oracle per score GW",
        f"- Snapshots: {state['snapshot_gws']}",
        f"- Priced mean weeks: {state['priced_gws']}",
        f"- Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Per GW",
        "",
        "| GW | mean pred | mean actual | mean resid | mean autosubs | path pred | path actual | path resid | path−mean actual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['gw']} | {r['mean_pred']:.1f} | {r['mean_actual']:.1f} | "
            f"{r['mean_resid']:.1f} | {r.get('mean_n_autosubs', 0)} | "
            f"{r['path_pred']:.1f} | {r['path_actual']:.1f} | "
            f"{r['path_resid']:.1f} | {r['delta_actual_path_minus_mean']:.1f} |"
        )
    if rows:
        m_resid = float(np.mean([r["mean_resid"] for r in rows]))
        p_resid = float(np.mean([r["path_resid"] for r in rows]))
        m_mae = float(np.mean([abs(r["mean_resid"]) for r in rows]))
        p_mae = float(np.mean([abs(r["path_resid"]) for r in rows]))
        m_act = float(np.sum([r["mean_actual"] for r in rows]))
        p_act = float(np.sum([r["path_actual"] for r in rows]))
        lines += [
            "",
            "## Horizon summary",
            "",
            f"- Mean residual (bias): **{m_resid:.2f}** · MAE **{m_mae:.2f}** · "
            f"sum actual **{m_act:.1f}**",
            f"- Path residual (bias): **{p_resid:.2f}** · MAE **{p_mae:.2f}** · "
            f"sum actual **{p_act:.1f}**",
            f"- Path − mean sum actual: **{p_act - m_act:.1f}**",
            "",
        ]
    lines += [
        "## Artefacts",
        "",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{out_dir.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid "
        f"--season {state['season']} --n {state['n_strands']}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--decision-gw", type=int, default=DECISION_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}; choose from {sorted(SEASON_E0)}")
    e0 = SEASON_E0[season]
    decision_gw = int(args.decision_gw)
    horizon_end = int(args.horizon_end)
    start = decision_gw + START_OFFSET

    print(
        f"[resid] season={season} decision={decision_gw} "
        f"horizon={start}-{horizon_end} N={args.n} "
        f"(full-side shares; semi-oracle buy m>{MIN_ORACLE_MINUTES:.0f} @ GW{decision_gw}; "
        "minutes oracle boards; autosub actuals)",
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

    hist = numagent_xp.asof_share_history(
        merged, before_gw=decision_gw, season=season
    )
    if hist.empty:
        raise RuntimeError(f"no history before GW{decision_gw}")
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)

    # Share/scoring roster: as-of hist pool (not cut to GW-t m>60).
    score_pool = _build_hist_pool(
        hist, prices, decision_gw=decision_gw, season=season
    )
    oracle_els = semi_oracle_elements(merged, decision_gw)
    if len(oracle_els) < 15:
        raise RuntimeError(
            f"semi-oracle pool too small at GW{decision_gw}: {len(oracle_els)}"
        )
    # Buy filter only: actual m>60 at decision GW t.
    buy_pool = score_pool.loc[
        pd.to_numeric(score_pool["element"], errors="coerce").isin(oracle_els)
    ].copy()
    if len(buy_pool) < 15:
        raise RuntimeError(f"buy pool too small after semi-oracle cut: {len(buy_pool)}")
    pool_ids = set(buy_pool["player_id"].astype(str))
    pool_ids |= {
        str(int(e))
        for e in pd.to_numeric(buy_pool["element"], errors="coerce").dropna().astype(int)
    }

    # Path generative minutes: as-of m on the share roster (no GW(t+n) foresight).
    # Buy-eligible rows may use GW-t actual minutes (semi-oracle at t only).
    gw_t = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == decision_gw]
    m_t = {
        int(r.element): float(r.minutes or 0.0)
        for r in gw_t.itertuples(index=False)
        if pd.notna(r.element)
    }
    # Keep as-of m on share roster; overlay GW-t actual only for buy-eligible els.
    if "m" not in score_pool.columns:
        score_pool["m"] = pd.to_numeric(
            score_pool["xmi"] if "xmi" in score_pool.columns else 0.0,
            errors="coerce",
        ).fillna(0.0)
    score_pool["m"] = [
        float(m_t[int(e)]) if int(e) in oracle_els and int(e) in m_t else float(m)
        for e, m in zip(
            pd.to_numeric(score_pool["element"], errors="coerce").fillna(-1),
            pd.to_numeric(score_pool["m"], errors="coerce").fillna(0.0),
            strict=True,
        )
    ]
    score_pool["xmi"] = score_pool["m"]
    score_pool["sigma_m"] = 0.0
    print(
        f"[resid] share_roster={len(score_pool)} buy_pool={len(buy_pool)} "
        f"(semi-oracle buy m>{MIN_ORACLE_MINUTES:.0f} @ GW{decision_gw}; "
        f"{len(oracle_els)} GW players)",
        flush=True,
    )

    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start, end=horizon_end)
    horizon_gws = sorted(fixtures_gw)
    snapshot_gws = [
        g for i, g in enumerate(horizon_gws) if i % max(1, int(args.snapshot_every)) == 0
    ]
    priced_candidates = horizon_gws[:2]
    priced_by_gw = _fd_priced_by_gw(enriched, fd, priced_candidates)
    priced_gws = sorted(priced_by_gw)
    if len(priced_gws) < 1:
        raise RuntimeError("no FD priced fits in horizon head")

    # Full side for mean-board shares (minutes oracle applied per score GW).
    side, _, _ = build_frozen_side_table(
        hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        season=season,
        historical_rate_baseline=True,
    )

    print("[resid] mean arm (full-side shares → buy filter)…", flush=True)
    mean = run_mean_arm_oracle(
        side=side,
        merged=merged,
        enriched=enriched,
        fd=fd,
        prices=prices,
        season=season,
        decision_gw=decision_gw,
        priced_gws=priced_gws,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        pool_ids=pool_ids,
    )
    print(
        f"[resid] mean done cap={mean['captain_id']} ({mean['elapsed_s']:.1f}s)",
        flush=True,
    )

    print("[resid] path arm (full share roster → buy filter)…", flush=True)
    path = run_path_strands(
        pool=score_pool,
        fixtures_gw=fixtures_gw,
        priced_by_gw=priced_by_gw,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        hist_by_team=team_hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        horizon_gws=horizon_gws,
        snapshot_gws=snapshot_gws,
        n_strands=int(args.n),
        seed=int(args.seed),
        selectable_ids=pool_ids,
    )
    print(
        f"[resid] path done strands_with_squads={path['strands_with_squads']} "
        f"({path['elapsed_s']:.1f}s)",
        flush=True,
    )

    print("[resid] scoring per-GW residuals (oracle boards + autosub)…", flush=True)
    per_gw = per_gw_residuals(
        mean=mean,
        path=path,
        side=side,
        pool=buy_pool,
        enriched=enriched,
        fd=fd,
        prices=prices,
        merged=merged,
        season=season,
        decision_gw=decision_gw,
        horizon_gws=horizon_gws,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        pool_ids=pool_ids,
    )

    path_slim = {
        "elapsed_s": path["elapsed_s"],
        "n_strands": path["n_strands"],
        "strands_with_squads": path["strands_with_squads"],
        "strands": [
            {"strand_id": s["strand_id"], "squads": s["squads"]}
            for s in path["strands"]
        ],
    }

    summary: dict[str, Any] = {
        "full_side_shares": True,
        "semi_oracle_buy_filter": True,
        "min_oracle_minutes": MIN_ORACLE_MINUTES,
        "minutes_oracle_boards": True,
        "autosub_actuals": True,
        "n_share_roster": int(len(score_pool)),
        "n_buy_pool": int(len(buy_pool)),
    }
    if per_gw:
        summary.update(
            {
                "n_gws": len(per_gw),
                "mean_resid_bias": float(np.mean([r["mean_resid"] for r in per_gw])),
                "path_resid_bias": float(np.mean([r["path_resid"] for r in per_gw])),
                "mean_resid_mae": float(np.mean([abs(r["mean_resid"]) for r in per_gw])),
                "path_resid_mae": float(np.mean([abs(r["path_resid"]) for r in per_gw])),
                "mean_actual_sum": float(np.sum([r["mean_actual"] for r in per_gw])),
                "path_actual_sum": float(np.sum([r["path_actual"] for r in per_gw])),
                "path_minus_mean_actual_sum": float(
                    np.sum([r["delta_actual_path_minus_mean"] for r in per_gw])
                ),
                "mean_autosubs_sum": int(sum(r.get("mean_n_autosubs", 0) for r in per_gw)),
            }
        )

    state = {
        "task": "cascade_mean_path_gw_resid",
        "season": season,
        "e0": e0,
        "decision_gw": decision_gw,
        "horizon_start": start,
        "horizon_end": horizon_end,
        "n_strands": int(args.n),
        "seed": int(args.seed),
        "snapshot_gws": snapshot_gws,
        "priced_gws": priced_gws,
        "n_share_roster": int(len(score_pool)),
        "n_buy_pool": int(len(buy_pool)),
        "n_pool": int(len(buy_pool)),
        "semi_oracle_gw": decision_gw,
        "full_side_shares": True,
        "min_oracle_minutes": MIN_ORACLE_MINUTES,
        "minutes_oracle_boards": True,
        "autosub_actuals": True,
        "ranks": "oracle_post_gw38",
        "gf_ga_profile": "loso_prior_seasons",
        "mean": {
            "elapsed_s": mean.get("elapsed_s"),
            "captain_id": mean.get("captain_id"),
            "xi_ids": mean.get("xi_ids"),
            "player_ids": mean.get("player_ids"),
            "objective_value": mean.get("objective_value"),
        },
        "path": path_slim,
        "per_gw": per_gw,
        "summary": summary,
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    out_dir = _out_dir(season)
    _save(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_gw).to_csv(out_dir / "per_gw.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "mean_squad.json").write_text(
        json.dumps(mean, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(state, out_dir)
    print(json.dumps({"elapsed_s": state["elapsed_s"], "summary": summary}, indent=2))
    print(f"report={REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
