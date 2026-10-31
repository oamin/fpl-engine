"""Dig: why generative E[G] > centreline on the OL XI (full-club shares).

Same open-loop XI (GW8–19) as ``cascade_g_overprint_dig``. On each strand,
roll the path scenario then accumulate five XI E[G] arms:

1. **cl_fd** — ``sG_shrink · α · xG`` on full-horizon FD λ (prior dig)
2. **cl_path** — ``sG_shrink · α · xG(λ_path)`` (priced t:t+1 + Opta long)
3. **cl_raw** — ``sG_raw · α · xG(λ_path)`` (no share-SE shrink)
4. **mean_k** — ``sG_raw · α · k`` then **mean_πk** — ``π · α · k`` (props)
5. **gen** — attack-dice counts

Gaps:
  gen − cl_fd = (path−FD) + (raw−shrink) + (k−xG) + (π−sG) + (dice−πk)

  PYTHONPATH=. python3 -m src.eval.cascade_g_gen_centreline_gap --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_g_overprint_dig import _pid_key, _player_actual_g, _xi_by_gw
from src.eval.cascade_mc_vs_mean_hist import (
    SEASON_E0,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import side_moments_from_lams
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import (
    build_club_dice_from_side,
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
from src.models.short_horizon_cascade import load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_g_gen_centreline_gap.md"
STATE = ROOT / "REPORT_STATE.json"
OUT_DEFAULT = Path("/opt/cursor/artifacts/cascade_g_gen_centreline_gap.json")
ROLLING = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json")


def _now() -> float:
    return time.time()


def _element(pid: str) -> int | None:
    try:
        return int(float(_pid_key(pid)))
    except (TypeError, ValueError):
        return None


def _sG_maps(
    club: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
    xg: float,
    prop_p_by_element: dict[int, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """pid→ weight for shrink sG, raw sG (no props), and dice π (props if any)."""
    if club.empty:
        return {}, {}, {}
    g_team = float(alpha_g) * float(xg)
    scored_sh = numagent_xp.score_team_side(
        club,
        xg=float(xg),
        xcs=0.0,
        xgc=0.0,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        share_players=club,
        share_se_shrink=True,
    )
    scored_raw = numagent_xp.score_team_side(
        club,
        xg=float(xg),
        xcs=0.0,
        xgc=0.0,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        share_players=club,
        share_se_shrink=False,
    )
    shrink: dict[str, float] = {}
    raw: dict[str, float] = {}
    for i in range(len(club)):
        pid = _pid_key(str(club.iloc[i]["player_id"]))
        shrink[pid] = float(scored_sh.iloc[i]["sG"])
        raw[pid] = float(scored_raw.iloc[i]["sG"])
    pi: dict[str, float] = dict(raw)
    dice, ordered = build_club_dice_from_side(
        club,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        G_team=g_team,
        prop_p_by_element=prop_p_by_element,
    )
    if dice is not None and not ordered.empty:
        idc = "player_id" if "player_id" in ordered.columns else "element"
        for i, _name in enumerate(dice.names):
            pid = _pid_key(str(ordered.iloc[i][idc]))
            pi[pid] = float(dice.pi[i])
    return shrink, raw, pi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=ROLLING)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    t0 = _now()
    src = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    season = str(src["season"])
    e0 = SEASON_E0[season]
    seed0 = int(src["seed"])
    start_gw = int(src["horizon_start"])
    end_gw = int(src["horizon_end"])
    xi_by_gw = _xi_by_gw(src)

    print(
        f"[gap] season={season} N={args.n} XI GWs {start_gw}–{end_gw}",
        flush=True,
    )
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)
    hist = numagent_xp.asof_share_history(merged, before_gw=start_gw, season=season)
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)
    select = _build_hist_pool(hist, prices, decision_gw=start_gw, season=season)
    share = _build_hist_share_pool(hist, prices, decision_gw=start_gw, season=season)
    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start_gw, end=end_gw)
    plan_gws = sorted(fixtures_gw)
    # Product short = first two horizon GWs; long = Opta path λ (same as mean boards).
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    # Prior dig also scored centreline on full-horizon FD — keep that as cl_fd_all.
    priced_all = _fd_priced_by_gw(enriched, fd, plan_gws)
    priced_index = _priced_index(priced_by_gw)
    pact = _player_actual_g(merged)

    # Static FD centreline (shrunk) on full-horizon FD — matches prior dig
    print("[gap] static FD centreline (all GWs)…", flush=True)
    cl_fd: dict[tuple[int, str], float] = {}
    for gw in plan_gws:
        for sc in priced_all.get(int(gw), []):
            for team_key, lam_for, lam_against in (
                (norm_team(str(sc["home"])), float(sc["lam_h"]), float(sc["lam_a"])),
                (norm_team(str(sc["away"])), float(sc["lam_a"]), float(sc["lam_h"])),
            ):
                club = share.loc[share["team_norm"].map(norm_team) == team_key].copy()
                if club.empty:
                    continue
                mom = side_moments_from_lams(
                    lam_for, lam_against, rho=float(sc.get("rho") or 0.0)
                )
                xg = float(mom["xg"])
                scored = numagent_xp.score_team_side(
                    club,
                    xg=xg,
                    xcs=mom["xcs"],
                    xgc=mom["xgc"],
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    share_players=club,
                    share_se_shrink=True,
                )
                xi = xi_by_gw.get(int(gw), set())
                for i in range(len(scored)):
                    pid = _pid_key(str(club.iloc[i]["player_id"]))
                    if pid in xi:
                        cl_fd[(int(gw), pid)] = float(scored.iloc[i]["E_G"])

    # Strand path arms
    print("[gap] path strands…", flush=True)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    acc = {
        "cl_path": defaultdict(float),
        "cl_raw": defaultdict(float),
        "mean_k": defaultdict(float),
        "mean_k_pi": defaultdict(float),
        "gen": defaultdict(float),
    }
    # Team-GW scoreline diagnostics on XI clubs
    sum_k = 0.0
    sum_xg = 0.0
    sum_lam = 0.0
    n_side = 0
    sum_k_xi = 0.0
    sum_xg_xi = 0.0
    n_side_xi = 0

    for s in range(int(args.n)):
        if s == 0 or (s + 1) % 20 == 0 or s + 1 == int(args.n):
            print(f"  [gap] strand {s + 1}/{args.n}", flush=True)
        rng = random.Random(seed0 + s)
        paths = {c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()}
        score_rng = random.Random(seed0 + 10_000 + s)
        for gw in plan_gws:
            xi = xi_by_gw.get(int(gw), set())
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
                hn = norm_team(str(sc["home"]))
                an = norm_team(str(sc["away"]))
                home = share.loc[share["team_norm"].map(norm_team) == hn].copy()
                away = share.loc[share["team_norm"].map(norm_team) == an].copy()
                rho = float(sc.get("rho") or 0.0)
                mom_h = side_moments_from_lams(
                    float(sc["lam_h"]), float(sc["lam_a"]), rho=rho
                )
                mom_a = side_moments_from_lams(
                    float(sc["lam_a"]), float(sc["lam_h"]), rho=rho
                )
                k_h, k_a = int(sc["hg"]), int(sc["ag"])
                xg_h, xg_a = float(mom_h["xg"]), float(mom_a["xg"])

                sum_k += k_h + k_a
                sum_xg += xg_h + xg_a
                sum_lam += float(sc["lam_h"]) + float(sc["lam_a"])
                n_side += 2
                if any(
                    _pid_key(str(r.player_id)) in xi for r in home.itertuples(index=False)
                ):
                    sum_k_xi += k_h
                    sum_xg_xi += xg_h
                    n_side_xi += 1
                if any(
                    _pid_key(str(r.player_id)) in xi for r in away.itertuples(index=False)
                ):
                    sum_k_xi += k_a
                    sum_xg_xi += xg_a
                    n_side_xi += 1

                home_props = (
                    sc.get("home_props")
                    if isinstance(sc.get("home_props"), dict)
                    else None
                )
                away_props = (
                    sc.get("away_props")
                    if isinstance(sc.get("away_props"), dict)
                    else None
                )

                # Linear arms: shrink/raw on xG; mean_k on raw sG and on dice π
                for club, xg, k, props in (
                    (home, xg_h, k_h, home_props),
                    (away, xg_a, k_a, away_props),
                ):
                    if club.empty:
                        continue
                    shrink, raw, pi = _sG_maps(
                        club,
                        alpha_g=alpha_g,
                        alpha_a=alpha_a,
                        xg=xg,
                        prop_p_by_element=props,
                    )
                    for pid in xi:
                        if pid not in raw and pid not in shrink:
                            continue
                        s_sh = float(shrink.get(pid, 0.0))
                        s_rw = float(raw.get(pid, 0.0))
                        s_pi = float(pi.get(pid, s_rw))
                        key = (int(gw), pid)
                        acc["cl_path"][key] += s_sh * float(alpha_g) * float(xg)
                        acc["cl_raw"][key] += s_rw * float(alpha_g) * float(xg)
                        acc["mean_k"][key] += s_rw * float(alpha_g) * float(k)
                        acc["mean_k_pi"][key] += s_pi * float(alpha_g) * float(k)

                # Generative dice
                home_xp, away_xp, _ = score_fixture_generative(
                    home,
                    away,
                    lam_h=float(sc["lam_h"]),
                    lam_a=float(sc["lam_a"]),
                    rho=rho,
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    rng=score_rng,
                    home_hist=team_hist.get(hn),
                    away_hist=team_hist.get(an),
                    home_props=home_props,
                    away_props=away_props,
                    lineup_cov=DEFAULT_LINEUP_COV,
                    goals_home=k_h,
                    goals_away=k_a,
                )
                for scored in (home_xp, away_xp):
                    if scored is None or scored.empty:
                        continue
                    for i in range(len(scored)):
                        row = scored.iloc[i]
                        pid = _pid_key(str(row.get("player_id", "")))
                        if pid not in xi and "element" in scored.columns:
                            pid = _pid_key(str(int(row["element"])))
                        if pid not in xi:
                            continue
                        acc["gen"][(int(gw), pid)] += float(row.get("E_G") or 0.0)

    n = float(args.n)
    for arm in acc:
        for k in list(acc[arm]):
            acc[arm][k] /= n

    # Build XI player-GW table
    rows: list[dict[str, Any]] = []
    for gw, xi in xi_by_gw.items():
        act_gw = pact.loc[pact["GW"] == int(gw)].set_index("element")
        for pid in xi:
            eid = _element(pid)
            if eid is None:
                continue
            if eid in act_gw.index:
                a = act_gw.loc[eid]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                g = float(a["goals"])
                name = str(a["name"])
                pos = str(a["position"])
                team = norm_team(str(a["team_n"]))
            else:
                g, name, pos, team = 0.0, pid, "", ""
            key = (int(gw), str(pid))
            e_fd = float(cl_fd.get(key, 0.0))
            e_path = float(acc["cl_path"].get(key, 0.0))
            e_raw = float(acc["cl_raw"].get(key, 0.0))
            e_mk = float(acc["mean_k"].get(key, 0.0))
            e_mk_pi = float(acc["mean_k_pi"].get(key, 0.0))
            e_gen = float(acc["gen"].get(key, 0.0))
            rows.append(
                {
                    "gw": int(gw),
                    "player_id": str(pid),
                    "element": eid,
                    "name": name,
                    "position": pos,
                    "team_n": team,
                    "G": g,
                    "E_cl_fd": e_fd,
                    "E_cl_path": e_path,
                    "E_cl_raw": e_raw,
                    "E_mean_k": e_mk,
                    "E_mean_k_pi": e_mk_pi,
                    "E_gen": e_gen,
                    "gap_gen_fd": e_gen - e_fd,
                    "gap_path_fd": e_path - e_fd,
                    "gap_raw_path": e_raw - e_path,
                    "gap_k_raw": e_mk - e_raw,
                    "gap_pi_k": e_mk_pi - e_mk,
                    "gap_gen_k": e_gen - e_mk_pi,
                }
            )
    df = pd.DataFrame(rows)

    def _sum(col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

    totals = {
        "n_xi_player_gw": int(len(df)),
        "sum_G": _sum("G"),
        "sum_E_cl_fd": _sum("E_cl_fd"),
        "sum_E_cl_path": _sum("E_cl_path"),
        "sum_E_cl_raw": _sum("E_cl_raw"),
        "sum_E_mean_k": _sum("E_mean_k"),
        "sum_E_mean_k_pi": _sum("E_mean_k_pi"),
        "sum_E_gen": _sum("E_gen"),
        "resid_cl_fd": _sum("E_cl_fd") - _sum("G"),
        "resid_gen": _sum("E_gen") - _sum("G"),
        "gap_gen_fd": _sum("gap_gen_fd"),
        "gap_path_fd": _sum("gap_path_fd"),
        "gap_raw_path": _sum("gap_raw_path"),
        "gap_k_raw": _sum("gap_k_raw"),
        "gap_pi_k": _sum("gap_pi_k"),
        "gap_gen_k": _sum("gap_gen_k"),
        "alpha_g": float(alpha_g),
    }
    # identity: pieces sum to total gap
    totals["pieces_sum"] = (
        totals["gap_path_fd"]
        + totals["gap_raw_path"]
        + totals["gap_k_raw"]
        + totals["gap_pi_k"]
        + totals["gap_gen_k"]
    )
    totals["identity_err"] = totals["pieces_sum"] - totals["gap_gen_fd"]

    scoreline = {
        "n_sides_all": int(n_side),
        "mean_k_all": sum_k / max(n_side, 1),
        "mean_xg_all": sum_xg / max(n_side, 1),
        "mean_lam_all": sum_lam / max(n_side, 1),
        "bias_k_minus_xg_all": (sum_k - sum_xg) / max(n_side, 1),
        "n_sides_xi": int(n_side_xi),
        "mean_k_xi": sum_k_xi / max(n_side_xi, 1),
        "mean_xg_xi": sum_xg_xi / max(n_side_xi, 1),
        "bias_k_minus_xg_xi": (sum_k_xi - sum_xg_xi) / max(n_side_xi, 1),
    }

    by_player = (
        df.groupby(["name", "position"], as_index=False)
        .agg(
            n=("gap_gen_fd", "size"),
            gap=("gap_gen_fd", "sum"),
            path_fd=("gap_path_fd", "sum"),
            raw_path=("gap_raw_path", "sum"),
            k_raw=("gap_k_raw", "sum"),
            pi_k=("gap_pi_k", "sum"),
            gen_k=("gap_gen_k", "sum"),
            E_gen=("E_gen", "sum"),
            E_fd=("E_cl_fd", "sum"),
            G=("G", "sum"),
        )
        .sort_values("gap", ascending=False)
    )

    # Dominant piece
    pieces = {
        "path_λ_vs_FD": totals["gap_path_fd"],
        "raw_sG_vs_shrink": totals["gap_raw_path"],
        "E[k]_vs_xG": totals["gap_k_raw"],
        "props_π_vs_sG": totals["gap_pi_k"],
        "dice_vs_mean_πk": totals["gap_gen_k"],
    }
    dominant = max(pieces, key=lambda k: abs(pieces[k]))
    read = (
        f"gen−cl_fd Σ = **{totals['gap_gen_fd']:+.2f}** on XI "
        f"(gen resid {totals['resid_gen']:+.2f}, cl_fd resid {totals['resid_cl_fd']:+.2f}). "
        f"Pieces: pathλ−FD {totals['gap_path_fd']:+.2f}; "
        f"raw−shrink {totals['gap_raw_path']:+.2f}; "
        f"E[k]−xG {totals['gap_k_raw']:+.2f}; "
        f"props π−sG {totals['gap_pi_k']:+.2f}; "
        f"dice−πk {totals['gap_gen_k']:+.2f}. "
        f"**Dominant: {dominant}** ({pieces[dominant]:+.2f}). "
        f"XI-club scoreline bias k−xG {scoreline['bias_k_minus_xg_xi']:+.3f}."
    )

    state = {
        "task": "cascade_g_gen_centreline_gap",
        "season": season,
        "gws": plan_gws,
        "n_strands": int(args.n),
        "seed": seed0,
        "share_pool": "full_m_gt_0",
        "n_select": int(len(select)),
        "n_share": int(len(share)),
        "lineup_cov": bool(DEFAULT_LINEUP_COV),
        "elapsed_s": _now() - t0,
        "totals": totals,
        "pieces": pieces,
        "dominant": dominant,
        "scoreline": scoreline,
        "read": read,
        "top_gap_players": by_player.head(12).to_dict(orient="records"),
        "bottom_gap_players": by_player.tail(8).to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Generative − centreline G gap dig",
        "",
        f"Season **{season}** · OL XI GW{start_gw}–{end_gw} · full-club shares · "
        f"N={args.n} · wall {state['elapsed_s']:.1f}s · lineup_cov={DEFAULT_LINEUP_COV}",
        "",
        "## Arms (Σ E[G] on XI)",
        "",
        "| arm | Σ E[G] | resid vs G |",
        "| --- | ---: | ---: |",
        f"| cl_fd (shrink · FD xG) | {totals['sum_E_cl_fd']:.2f} | {totals['resid_cl_fd']:+.2f} |",
        f"| cl_path (shrink · path xG) | {totals['sum_E_cl_path']:.2f} | {totals['sum_E_cl_path'] - totals['sum_G']:+.2f} |",
        f"| cl_raw (raw · path xG) | {totals['sum_E_cl_raw']:.2f} | {totals['sum_E_cl_raw'] - totals['sum_G']:+.2f} |",
        f"| mean_k (raw · α · k) | {totals['sum_E_mean_k']:.2f} | {totals['sum_E_mean_k'] - totals['sum_G']:+.2f} |",
        f"| mean_πk (dice π · α · k) | {totals['sum_E_mean_k_pi']:.2f} | {totals['sum_E_mean_k_pi'] - totals['sum_G']:+.2f} |",
        f"| gen (dice) | {totals['sum_E_gen']:.2f} | {totals['resid_gen']:+.2f} |",
        f"| actual G | {totals['sum_G']:.2f} | 0 |",
        "",
        "## Gap pieces (gen − cl_fd)",
        "",
        f"Total **{totals['gap_gen_fd']:+.2f}** · pieces sum {totals['pieces_sum']:+.2f} "
        f"(err {totals['identity_err']:+.3f})",
        "",
        "| piece | Σ |",
        "| --- | ---: |",
        f"| path λ − FD | {totals['gap_path_fd']:+.2f} |",
        f"| raw sG − shrink | {totals['gap_raw_path']:+.2f} |",
        f"| E[k] − xG (linear) | {totals['gap_k_raw']:+.2f} |",
        f"| props π − sG | {totals['gap_pi_k']:+.2f} |",
        f"| dice − πk | {totals['gap_gen_k']:+.2f} |",
        "",
        f"**Dominant piece: {dominant}**",
        "",
        "## Scoreline on path (strand-mean)",
        "",
        f"- All sides: mean k {scoreline['mean_k_all']:.3f} vs xG "
        f"{scoreline['mean_xg_all']:.3f} (bias {scoreline['bias_k_minus_xg_all']:+.3f})",
        f"- XI clubs: mean k {scoreline['mean_k_xi']:.3f} vs xG "
        f"{scoreline['mean_xg_xi']:.3f} (bias {scoreline['bias_k_minus_xg_xi']:+.3f})",
        "",
        "## Top XI gap contributors (gen − cl_fd)",
        "",
        "| name | pos | n | gap | path−FD | raw−sh | k−xG | props | dice |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in by_player.head(12).itertuples(index=False):
        lines.append(
            f"| {r.name} | {r.position} | {int(r.n)} | {r.gap:+.2f} | "
            f"{r.path_fd:+.2f} | {r.raw_path:+.2f} | {r.k_raw:+.2f} | "
            f"{r.pi_k:+.2f} | {r.gen_k:+.2f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        read,
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_g_gen_centreline_gap --n {args.n}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_g_gen_centreline_gap",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Decompose gen−centreline G gap into pathλ / shrink / k / dice",
                "verdict": {
                    "gap_gen_fd": totals["gap_gen_fd"],
                    "pieces": pieces,
                    "dominant": dominant,
                    "read": read,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"totals": totals, "pieces": pieces, "dominant": dominant}, indent=2))
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
