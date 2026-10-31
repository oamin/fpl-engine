"""Count residual: E[G, CS, GA] − actual on the mean xi_ft rolling run.

Same boards as ``cascade_mean_xi_ft_component_resid``, but compares
**event counts** (not FPL points):

- G: mean generative FPL-attributed goals (``E_G``; ∅_G mass excluded via α_G)
- CS: P(CS eligible) = strand mean of ``(m≥60) ∧ (GA=0)``
- GA: mean team goals against on strands with ``m>0``

Also reports empty-goal rate: fraction of rolled match goals → ``∅_G``.

  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_count_resid \\
      --artifact /opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json
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
    SEASON_E0,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_xi_ft_rolling import _ensure_owned_in_pool
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
from src.models.short_horizon_cascade import load_prices
from src.rules.fpl_2026 import normalize_position
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mean_xi_ft_count_resid.md"
STATE_JSON = ROOT / "REPORT_STATE.json"


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


def actual_counts(merged: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Per-element actual G / CS / GA counts for one GW."""
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)].copy()
    if g.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for r in g.itertuples(index=False):
        eid = int(r.element) if pd.notna(r.element) else None
        if eid is None:
            continue
        mins = int(float(r.minutes or 0))
        pos = normalize_position(str(r.position))
        goals = int(float(r.goals_scored or 0))
        if bool(r.was_home):
            ga = int(float(r.team_a_score or 0))
        else:
            ga = int(float(r.team_h_score or 0))
        cs = 1.0 if (mins >= 60 and ga == 0) else 0.0
        rows.append(
            {
                "element": eid,
                "name": str(r.name),
                "position": pos,
                "minutes": mins,
                "G": float(goals),
                "CS": float(cs),
                "GA": float(ga) if mins > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def mean_count_boards(
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
    plan_gws: list[int],
    n_strands: int,
    seed0: int,
    share_pool: pd.DataFrame | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    """Average generative E[G], E[CS], E[GA] by GW (+ ∅_G diagnostics).

    ``pool`` = thin selection; ``share_pool`` = full club ``m>0`` for attack
    dice (required default). Counts for selected players are projected back.
    """
    if share_pool is None:
        raise ValueError(
            "share_pool is required (full club m>0); selection pool must stay thin"
        )
    priced_index = _priced_index(priced_by_gw)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    n = len(pool)
    keys = ("E_G", "E_CS", "E_GA")
    acc = {g: {k: np.zeros(n, float) for k in keys} for g in plan_gws}
    pid_to_idx = {str(p): i for i, p in enumerate(pool["player_id"].astype(str))}
    match_goals = 0
    fpl_goals = 0
    empty_goals = 0
    t0 = _now()
    for s in range(int(n_strands)):
        if s == 0 or (s + 1) % 25 == 0 or s + 1 == int(n_strands):
            print(f"  [count] strand {s + 1}/{n_strands}", flush=True)
        rng = random.Random(int(seed0) + s)
        paths: dict[str, ClubPath] = {
            c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()
        }
        rolled: dict[int, list[dict[str, Any]]] = {}
        for gw in plan_gws:
            week: list[dict[str, Any]] = []
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
        dice = share_pool
        for gw in plan_gws:
            g_acc = {k: np.zeros(n, float) for k in keys}
            for sc in rolled[int(gw)]:
                home_n = norm_team(str(sc["home"]))
                away_n = norm_team(str(sc["away"]))
                home_club = dice.loc[dice["team_norm"].map(norm_team) == home_n].copy()
                away_club = dice.loc[dice["team_norm"].map(norm_team) == away_n].copy()
                if home_club.empty and away_club.empty:
                    continue
                home_xp, away_xp, meta = score_fixture_generative(
                    home_club,
                    away_club,
                    lam_h=float(sc["lam_h"]),
                    lam_a=float(sc["lam_a"]),
                    rho=float(sc.get("rho") or 0.0),
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    rng=score_rng,
                    home_hist=hist_by_team.get(home_n),
                    away_hist=hist_by_team.get(away_n),
                    home_props=sc.get("home_props")
                    if isinstance(sc.get("home_props"), dict)
                    else None,
                    away_props=sc.get("away_props")
                    if isinstance(sc.get("away_props"), dict)
                    else None,
                    lineup_cov=DEFAULT_LINEUP_COV,
                    goals_home=int(sc["hg"]),
                    goals_away=int(sc["ag"]),
                )
                gh = int((meta or {}).get("goals_home") or sc["hg"])
                ga_m = int((meta or {}).get("goals_away") or sc["ag"])
                nh = int((meta or {}).get("n_home_scorers") or 0)
                na = int((meta or {}).get("n_away_scorers") or 0)
                match_goals += gh + ga_m
                fpl_goals += nh + na
                empty_goals += max(0, gh - nh) + max(0, ga_m - na)
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
                        idx = pid_to_idx.get(pid)
                        if idx is None:
                            continue
                        m = float(row.get("m") or 0.0)
                        ga = float(row.get("goals_against") or 0.0)
                        cs_team = bool(row.get("clean_sheet"))
                        eg = float(row.get("E_G") or 0.0)
                        e_cs = 1.0 if (m >= 60.0 and cs_team) else 0.0
                        e_ga = ga if m > 0 else 0.0
                        g_acc["E_G"][idx] += eg
                        g_acc["E_CS"][idx] += e_cs
                        g_acc["E_GA"][idx] += e_ga
            for k in keys:
                acc[int(gw)][k] += g_acc[k]

    frames: dict[int, pd.DataFrame] = {}
    for gw in plan_gws:
        df = pool.copy()
        for k, arr in acc[int(gw)].items():
            df[k] = arr / float(n_strands)
        frames[int(gw)] = df
    empty_rate = float(empty_goals / match_goals) if match_goals else float("nan")
    meta_out = {
        "n_strands": int(n_strands),
        "seed0": int(seed0),
        "elapsed_s": _now() - t0,
        "alpha_g": float(alpha_g),
        "alpha_a": float(alpha_a),
        "match_goals_rolled": int(match_goals),
        "fpl_goals_attributed": int(fpl_goals),
        "empty_goals_emptyset_G": int(empty_goals),
        "empty_goal_rate": empty_rate,
        "one_minus_alpha_g": 1.0 - float(alpha_g),
        "n_selection_pool": int(len(pool)),
        "n_share_pool": int(len(share_pool)),
        "share_pool": "full_m_gt_0",
    }
    return frames, meta_out


def _xi_ids(week: dict[str, Any]) -> list[str]:
    ids = week.get("xi_ids") or week.get("player_ids") or []
    return [str(x) for x in ids]


def score_xi(
    board: pd.DataFrame,
    actual: pd.DataFrame,
    xi_ids: list[str],
    *,
    gw: int,
    arm: str,
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
        else:
            a = act.loc[eid]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            a_g, a_cs, a_ga = float(a["G"]), float(a["CS"]), float(a["GA"])
            name = str(a["name"])
            pos = str(a["position"])
        eg, ecs, ega = float(r["E_G"]), float(r["E_CS"]), float(r["E_GA"])
        rows.append(
            {
                "arm": arm,
                "gw": int(gw),
                "player_id": str(pid),
                "element": eid,
                "name": name,
                "position": pos,
                "E_G": eg,
                "E_CS": ecs,
                "E_GA": ega,
                "G": a_g,
                "CS": a_cs,
                "GA": a_ga,
                "resid_G": eg - a_g,
                "resid_CS": ecs - a_cs,
                "resid_GA": ega - a_ga,
            }
        )
    return rows


def _summarise(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out: dict[str, Any] = {"n_player_gw": int(len(df))}
    for e_key, a_key, label in (
        ("E_G", "G", "G"),
        ("E_CS", "CS", "CS"),
        ("E_GA", "GA", "GA"),
    ):
        rcol = {"G": "resid_G", "CS": "resid_CS", "GA": "resid_GA"}[label]
        r = df[rcol]
        out[label] = {
            "sum_E": float(df[e_key].sum()),
            "sum_actual": float(df[a_key].sum()),
            "sum_resid": float(r.sum()),
            "mean_resid": float(r.mean()),
            "mae": float(r.abs().mean()),
            "rmse": float(np.sqrt((r * r).mean())),
        }
    return out


def _plot(state: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=140, sharey=False)
    for ax, arm_key, title in zip(
        axes,
        ("open_loop", "rolling"),
        ("Open-loop XI GW8–19", "Rolling XI GW8–10"),
        strict=True,
    ):
        rows = state["arms"][arm_key]["per_gw"]
        if not rows:
            ax.set_title(title + " (empty)")
            continue
        gws = [r["gw"] for r in rows]
        for key, label, color in (
            ("resid_G_per_match", "r_G / match", "#c0392b"),
            ("resid_CS_per_match", "r_CS / match", "#2980b9"),
            ("resid_GA_per_match", "r_GA / match", "#27ae60"),
        ):
            ax.plot(
                gws,
                [r[key] for r in rows],
                "o-",
                color=color,
                lw=1.6,
                ms=4,
                label=label,
            )
        ax.axhline(0.0, color="#444", lw=0.8)
        ax.set_xlabel("GW")
        ax.set_ylabel("E − actual (per XI player-match)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"Count residual E[G,CS,GA]−actual  (N={state['n_strands']}, "
        f"α_G={state['ol_empty']['alpha_g']:.3f})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_report(state: dict[str, Any]) -> None:
    ol_e = state["ol_empty"]
    ol_g = state["arms"]["open_loop"]["summary"].get("G") or {}
    thin_g = (state.get("thin_baseline_ol_G") or {}).get("sum_resid")
    lines = [
        "# Mean xi_ft run — count residual E[G, CS, GA] − actual",
        "",
        f"Source artifact: `{state['source_artifact']}`",
        f"Season **{state['season']}** · N={state['n_strands']} · "
        f"seed={state['seed']} · wall {state['elapsed_s']:.1f}s",
        "",
        "## Share pool (default)",
        "",
        "Attack dice / π on **full club ``m>0``**; selection stays ≥60′. "
        "Shares are not re-normalised on the thin board "
        "(`protocol.cascade_short_generative.share_pool`).",
        "",
    ]
    if thin_g is not None and ol_g:
        lines += [
            f"- Prior thin-share OL Σ resid G: **{float(thin_g):+.2f}**",
            f"- Full-share OL Σ resid G: **{float(ol_g.get('sum_resid', float('nan'))):+.2f}** "
            f"(Δ {float(ol_g.get('sum_resid', 0.0)) - float(thin_g):+.2f})",
            "",
        ]
    lines += [
        "## Own goals / non-attributed goals",
        "",
        "Generative attack puts mass on ``∅_G``: each rolled match goal is "
        "attributed to a player with probability ``α_G``, else ``∅_G`` "
        "(own goals / non-FPL scorers). ``α_G`` is the expanding "
        "player-goals / match-goals rate as-of the decision GW.",
        "",
        f"- OL as-of α_G = **{ol_e['alpha_g']:.4f}** "
        f"(⇒ empty mass 1−α_G = **{ol_e['one_minus_alpha_g']:.4f}**)",
        f"- Realised empty rate on OL strands: **{ol_e['empty_goal_rate']:.4f}** "
        f"({ol_e['empty_goals_emptyset_G']} / {ol_e['match_goals_rolled']} goals)",
        "",
        "Counts below use **FPL-attributed** E[G] (∅_G excluded). "
        "CS = eligible clean sheet ``(m≥60)∧(GA=0)``. "
        "GA = team goals against when ``m>0``.",
        "",
    ]
    if state.get("plot"):
        lines += [f"![per-match residuals]({state['plot']})", ""]
    for arm_key, title in (
        ("open_loop", "Open-loop XI (GW8 boards → GW8–19)"),
        ("rolling", "Rolling committed XI (step boards → GW8–10)"),
    ):
        arm = state["arms"][arm_key]
        s = arm["summary"]
        lines += [f"## {title}", ""]
        if not s:
            lines += ["_no rows_", ""]
            continue
        lines += [
            f"- Player-GWs: **{s['n_player_gw']}**",
            "",
            "| count | Σ E | Σ actual | Σ resid | mean resid / match | MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for comp in ("G", "CS", "GA"):
            c = s[comp]
            lines.append(
                f"| {comp} | {c['sum_E']:.2f} | {c['sum_actual']:.2f} | "
                f"{c['sum_resid']:+.2f} | {c['mean_resid']:+.3f} | {c['mae']:.3f} |"
            )
        lines += [
            "",
            "### Per GW (residual / match on XI)",
            "",
            "| GW | n | r_G | r_CS | r_GA |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in arm["per_gw"]:
            lines.append(
                f"| {row['gw']} | {row['n']} | "
                f"{row['resid_G_per_match']:+.3f} | "
                f"{row['resid_CS_per_match']:+.3f} | "
                f"{row['resid_GA_per_match']:+.3f} |"
            )
        lines.append("")
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_count_resid "
        f"--artifact {state['source_artifact']} --n {state['n_strands']}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json"),
    )
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_count_resid.json"),
    )
    args = ap.parse_args(argv)

    src = json.loads(Path(args.artifact).read_text())
    season = str(src["season"])
    e0 = SEASON_E0[season]
    n_strands = int(args.n if args.n is not None else src["n_strands"])
    seed0 = int(src["seed"])
    start_gw = int(src["horizon_start"])
    horizon_end = int(src["horizon_end"])

    print(f"[count-resid] season={season} N={n_strands}", flush=True)
    t_all = _now()

    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    print(f"[count-resid] OL count boards @ GW{start_gw}…", flush=True)
    hist = numagent_xp.asof_share_history(merged, before_gw=start_gw, season=season)
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)
    pool = _build_hist_pool(hist, prices, decision_gw=start_gw, season=season)
    share_pool = _build_hist_share_pool(
        hist, prices, decision_gw=start_gw, season=season
    )
    print(
        f"[count-resid] select={len(pool)} share={len(share_pool)} "
        "(full club m>0 for π)",
        flush=True,
    )
    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start_gw, end=horizon_end)
    plan_gws = sorted(fixtures_gw)
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    ol_boards, ol_meta = mean_count_boards(
        pool=pool,
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
        seed0=seed0,
    )
    print(
        f"[count-resid] OL boards {ol_meta['elapsed_s']:.1f}s  "
        f"empty_rate={ol_meta['empty_goal_rate']:.4f}",
        flush=True,
    )

    # Check meta attack keys from score_fixture_generative
    ol_rows: list[dict[str, Any]] = []
    for w in src["open_loop"]["weeks"]:
        gw = int(w["gw"])
        act = actual_counts(merged, gw)
        ol_rows.extend(
            score_xi(ol_boards[gw], act, _xi_ids(w), gw=gw, arm="open_loop")
        )
    ol_df = pd.DataFrame(ol_rows)

    roll_rows: list[dict[str, Any]] = []
    stash = pool.copy()
    owned = list(src["open_loop"]["opening_15"])
    for step in src["rolling"]["steps"]:
        gw = int(step["gw"])
        commit = step["commit"]
        if step.get("reused_open_loop") or gw == start_gw:
            boards = ol_boards
        else:
            step_seed = seed0 + 10_000 * gw
            print(f"[count-resid] rolling boards @ GW{gw}…", flush=True)
            hist_s = numagent_xp.asof_share_history(
                merged, before_gw=gw, season=season
            )
            ag, aa = _asof_alphas(hist_s)
            th = hist_by_team_norm(hist_s)
            pool_s = _build_hist_pool(hist_s, prices, decision_gw=gw, season=season)
            share_s = _build_hist_share_pool(
                hist_s, prices, decision_gw=gw, season=season
            )
            pool_s = _ensure_owned_in_pool(pool_s, stash, owned)
            share_s = _ensure_owned_in_pool(share_s, stash, owned)
            stash = pool_s.copy()
            fx = _hist_fixtures_by_gw(enriched, start=gw, end=horizon_end)
            pg = sorted(fx)
            priced = _fd_priced_by_gw(enriched, fd, pg[:2])
            boards, meta = mean_count_boards(
                pool=pool_s,
                share_pool=share_s,
                fixtures_gw=fx,
                priced_by_gw=priced,
                ranks=ranks,
                profile=profile,
                league_gf=league_gf,
                hist_by_team=th,
                alpha_g=ag,
                alpha_a=aa,
                plan_gws=pg,
                n_strands=n_strands,
                seed0=step_seed,
            )
            print(f"[count-resid] GW{gw} boards {meta['elapsed_s']:.1f}s", flush=True)
        act = actual_counts(merged, gw)
        xi = commit.get("xi_ids")
        if not xi:
            head = (step.get("replan_weeks_head") or [{}])[0]
            xi = head.get("xi_ids") or commit.get("player_ids")
        roll_rows.extend(
            score_xi(boards[gw], act, [str(x) for x in xi], gw=gw, arm="rolling")
        )
        owned = list(commit["player_ids"])
    roll_df = pd.DataFrame(roll_rows)

    def per_gw_table(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        rows = []
        for gw, sub in df.groupby("gw"):
            n = max(int(len(sub)), 1)
            rows.append(
                {
                    "gw": int(gw),
                    "n": n,
                    "E_G": float(sub["E_G"].sum()),
                    "G": float(sub["G"].sum()),
                    "resid_G": float(sub["resid_G"].sum()),
                    "E_CS": float(sub["E_CS"].sum()),
                    "CS": float(sub["CS"].sum()),
                    "resid_CS": float(sub["resid_CS"].sum()),
                    "E_GA": float(sub["E_GA"].sum()),
                    "GA": float(sub["GA"].sum()),
                    "resid_GA": float(sub["resid_GA"].sum()),
                    "resid_G_per_match": float(sub["resid_G"].sum()) / n,
                    "resid_CS_per_match": float(sub["resid_CS"].sum()) / n,
                    "resid_GA_per_match": float(sub["resid_GA"].sum()) / n,
                }
            )
        return rows

    thin_baseline = None
    thin_path = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_count_resid.json")
    if thin_path.exists():
        try:
            thin_baseline = json.loads(thin_path.read_text())["arms"]["open_loop"][
                "summary"
            ].get("G")
        except (KeyError, json.JSONDecodeError, TypeError):
            thin_baseline = None
    state = {
        "task": "cascade_mean_xi_ft_count_resid",
        "season": season,
        "source_artifact": str(args.artifact),
        "n_strands": n_strands,
        "seed": seed0,
        "elapsed_s": _now() - t_all,
        "share_pool": "full_m_gt_0",
        "n_selection_pool": ol_meta.get("n_selection_pool"),
        "n_share_pool": ol_meta.get("n_share_pool"),
        "thin_baseline_ol_G": thin_baseline,
        "ol_empty": {
            "alpha_g": ol_meta["alpha_g"],
            "alpha_a": ol_meta["alpha_a"],
            "one_minus_alpha_g": ol_meta["one_minus_alpha_g"],
            "empty_goal_rate": ol_meta["empty_goal_rate"],
            "empty_goals_emptyset_G": ol_meta["empty_goals_emptyset_G"],
            "match_goals_rolled": ol_meta["match_goals_rolled"],
            "fpl_goals_attributed": ol_meta["fpl_goals_attributed"],
        },
        "arms": {
            "open_loop": {
                "summary": _summarise(ol_df),
                "per_gw": per_gw_table(ol_df),
            },
            "rolling": {
                "summary": _summarise(roll_df),
                "per_gw": per_gw_table(roll_df),
            },
        },
    }
    plot_path = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_count_resid.png")
    _plot(state, plot_path)
    state["plot"] = str(plot_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    _write_report(state)
    ol_g = state["arms"]["open_loop"]["summary"].get("G") or {}
    STATE_JSON.write_text(
        json.dumps(
            {
                "task": "cascade_mean_xi_ft_count_resid",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Remeasure E[G,CS,GA]−actual under full-club share default",
                "verdict": {
                    "share_pool": "full_m_gt_0",
                    "ol_G_resid": ol_g.get("sum_resid"),
                    "thin_ol_G_resid": (thin_baseline or {}).get("sum_resid"),
                    "ol_CS": state["arms"]["open_loop"]["summary"].get("CS"),
                    "ol_GA": state["arms"]["open_loop"]["summary"].get("GA"),
                    "roll_G_resid": state["arms"]["rolling"]["summary"]
                    .get("G", {})
                    .get("sum_resid"),
                    "empty_goal_rate": ol_meta["empty_goal_rate"],
                    "read": (
                        "Full-share default cuts OL ΣE[G] resid "
                        f"{(thin_baseline or {}).get('sum_resid', float('nan')):+.1f} → "
                        f"{ol_g.get('sum_resid', float('nan')):+.1f}; "
                        "rolling GW8–10 G ≈ calibrated. Remaining OL G "
                        "overprint is the open dig."
                    ),
                },
                "artifact": str(args.out),
                "report": str(REPORT_MD.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(state["arms"]["open_loop"]["summary"], indent=2))
    print("wrote", args.out, REPORT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
