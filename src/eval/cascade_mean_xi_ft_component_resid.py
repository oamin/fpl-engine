"""G / CS / GC residual: actual P vs mean-board xP for the rolling run.

Rebuilds the same generative mean boards used by ``cascade_mean_xi_ft_rolling``
(N strands, oracle ranks, LOSO GF/GA, prior rate baseline), accumulates
``xp_goals`` / ``xp_cs`` / ``xp_gc``, and compares to FPL actual component
points on the planned XI.

Arms:
  - **open_loop:** GW8 boards → OL XI for GW8–19
  - **rolling:** step boards → committed XI for GW8–10 (GW8 reuses OL boards)

Residual = xP − actual (player-GW on the XI).

  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_component_resid \\
      --artifact /opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json
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
    SEASON_E0,
    _build_hist_pool,
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
from src.rules.fpl_2026 import (
    clean_sheet_points,
    goal_points,
    goals_conceded_points,
    normalize_position,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mean_xi_ft_component_resid.md"
STATE_JSON = ROOT / "data/processed/cascade_mean_xi_ft_component_resid" / "state.json"


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _element(pid: str) -> int | None:
    text = str(pid)
    if ":" in text:
        text = text.split(":")[-1]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def actual_components(merged: pd.DataFrame, gw: int) -> pd.DataFrame:
    """Per-element actual P(G), P(CS), P(GC) for one GW."""
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
            gc = int(float(r.team_a_score or 0))
        else:
            gc = int(float(r.team_h_score or 0))
        cs = mins >= 60 and gc == 0
        rows.append(
            {
                "element": eid,
                "name": str(r.name),
                "position": pos,
                "minutes": mins,
                "goals": goals,
                "goals_conceded": gc,
                "clean_sheet": bool(cs),
                "P_G": float(goal_points(pos, goals)),
                "P_CS": float(clean_sheet_points(pos, mins, cs)),
                "P_GC": float(goals_conceded_points(pos, mins, gc)),
            }
        )
    return pd.DataFrame(rows)


def mean_component_boards(
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
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    """Average generative ``xp_goals`` / ``xp_cs`` / ``xp_gc`` (+ total) by GW."""
    priced_index = _priced_index(priced_by_gw)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    n = len(pool)
    acc = {
        g: {
            "xp_goals": np.zeros(n, float),
            "xp_cs": np.zeros(n, float),
            "xp_gc": np.zeros(n, float),
            "xp": np.zeros(n, float),
        }
        for g in plan_gws
    }
    pid_to_idx = {str(p): i for i, p in enumerate(pool["player_id"].astype(str))}
    t0 = _now()
    for s in range(int(n_strands)):
        if s == 0 or (s + 1) % 25 == 0 or s + 1 == int(n_strands):
            print(f"  [comp] strand {s + 1}/{n_strands}", flush=True)
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
        out = pool.copy()
        if "team_n" not in out.columns and "team_norm" in out.columns:
            out["team_n"] = out["team_norm"].map(lambda t: norm_team(str(t)))
        for gw in plan_gws:
            g_acc = {
                "xp_goals": np.zeros(n, float),
                "xp_cs": np.zeros(n, float),
                "xp_gc": np.zeros(n, float),
                "xp": np.zeros(n, float),
            }
            for sc in rolled[int(gw)]:
                home_n = norm_team(str(sc["home"]))
                away_n = norm_team(str(sc["away"]))
                home_club = out.loc[out["team_norm"].map(norm_team) == home_n].copy()
                away_club = out.loc[out["team_norm"].map(norm_team) == away_n].copy()
                if home_club.empty and away_club.empty:
                    continue
                home_xp, away_xp, _meta = score_fixture_generative(
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
                        g_acc["xp_goals"][idx] += float(row["xp_goals"])
                        g_acc["xp_cs"][idx] += float(row["xp_cs"])
                        g_acc["xp_gc"][idx] += float(row["xp_gc"])
                        g_acc["xp"][idx] += float(row["xp"])
            for k in g_acc:
                acc[int(gw)][k] += g_acc[k]

    frames: dict[int, pd.DataFrame] = {}
    for gw in plan_gws:
        df = pool.copy()
        for k, arr in acc[int(gw)].items():
            df[k] = arr / float(n_strands)
        frames[int(gw)] = df
    meta = {"n_strands": int(n_strands), "seed0": int(seed0), "elapsed_s": _now() - t0}
    return frames, meta


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
            a_g = a_cs = a_gc = float("nan")
            name = str(r.get("name", pid))
            pos = str(r.get("position", ""))
        else:
            a = act.loc[eid]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            a_g, a_cs, a_gc = float(a["P_G"]), float(a["P_CS"]), float(a["P_GC"])
            name = str(a["name"])
            pos = str(a["position"])
        xg = float(r["xp_goals"])
        xcs = float(r["xp_cs"])
        xgc = float(r["xp_gc"])
        rows.append(
            {
                "arm": arm,
                "gw": int(gw),
                "player_id": str(pid),
                "element": eid,
                "name": name,
                "position": pos,
                "xP_G": xg,
                "xP_CS": xcs,
                "xP_GC": xgc,
                "P_G": a_g,
                "P_CS": a_cs,
                "P_GC": a_gc,
                "resid_G": xg - a_g,
                "resid_CS": xcs - a_cs,
                "resid_GC": xgc - a_gc,
            }
        )
    return rows


def _summarise(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out: dict[str, Any] = {"n_player_gw": int(len(df))}
    for comp in ("G", "CS", "GC"):
        r = df[f"resid_{comp}"]
        out[comp] = {
            "sum_xP": float(df[f"xP_{comp}"].sum()),
            "sum_P": float(df[f"P_{comp}"].sum()),
            "sum_resid": float(r.sum()),
            "mean_resid": float(r.mean()),
            "mae": float(r.abs().mean()),
            "rmse": float(np.sqrt((r * r).mean())),
        }
    out["combined"] = {
        "sum_xP": float((df["xP_G"] + df["xP_CS"] + df["xP_GC"]).sum()),
        "sum_P": float((df["P_G"] + df["P_CS"] + df["P_GC"]).sum()),
        "sum_resid": float(
            (df["resid_G"] + df["resid_CS"] + df["resid_GC"]).sum()
        ),
        "mae": float(
            (df["resid_G"].abs() + df["resid_CS"].abs() + df["resid_GC"].abs()).mean()
        ),
    }
    return out


def write_report(state: dict[str, Any]) -> None:
    lines = [
        "# Mean xi_ft run — G / CS / GC residual (xP − actual)",
        "",
        f"Source artifact: `{state['source_artifact']}`",
        f"Season **{state['season']}** · N={state['n_strands']} · "
        f"seed={state['seed']} · wall {state['elapsed_s']:.1f}s",
        "",
        "xP = generative mean-board component (`xp_goals` / `xp_cs` / `xp_gc`). "
        "Actual P from FPL goals / CS (≥60′ & 0 GA) / GC bins (−1 per 2 for GKP/DEF). "
        "Residual = **xP − P** on the planned XI.",
        "",
    ]
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
            "| component | Σ xP | Σ actual P | Σ resid | mean resid | MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for comp in ("G", "CS", "GC"):
            c = s[comp]
            lines.append(
                f"| {comp} | {c['sum_xP']:.1f} | {c['sum_P']:.1f} | "
                f"{c['sum_resid']:+.1f} | {c['mean_resid']:+.3f} | {c['mae']:.3f} |"
            )
        comb = s["combined"]
        lines += [
            f"| **G+CS+GC** | {comb['sum_xP']:.1f} | {comb['sum_P']:.1f} | "
            f"{comb['sum_resid']:+.1f} |  | {comb['mae']:.3f} |",
            "",
            "### Per GW (Σ over XI)",
            "",
            "| GW | xP_G | P_G | r_G | xP_CS | P_CS | r_CS | xP_GC | P_GC | r_GC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in arm["per_gw"]:
            lines.append(
                f"| {row['gw']} | {row['xP_G']:.1f} | {row['P_G']:.1f} | "
                f"{row['resid_G']:+.1f} | {row['xP_CS']:.1f} | {row['P_CS']:.1f} | "
                f"{row['resid_CS']:+.1f} | {row['xP_GC']:.1f} | {row['P_GC']:.1f} | "
                f"{row['resid_GC']:+.1f} |"
            )
        lines.append("")
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_component_resid "
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
    ap.add_argument("--n", type=int, default=None, help="override N strands")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_component_resid.json"),
    )
    args = ap.parse_args(argv)

    src = json.loads(Path(args.artifact).read_text())
    season = str(src["season"])
    e0 = SEASON_E0[season]
    n_strands = int(args.n if args.n is not None else src["n_strands"])
    seed0 = int(src["seed"])
    start_gw = int(src["horizon_start"])
    horizon_end = int(src["horizon_end"])
    step_gws = [int(g) for g in src["step_gws"]]

    print(
        f"[comp-resid] season={season} N={n_strands} from {args.artifact}",
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

    # --- Open-loop boards @ start_gw (same seed as rolling OL) ---
    print(f"[comp-resid] OL component boards @ GW{start_gw}…", flush=True)
    hist = numagent_xp.asof_share_history(
        merged, before_gw=start_gw, season=season
    )
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)
    pool = _build_hist_pool(hist, prices, decision_gw=start_gw, season=season)
    fixtures_gw = _hist_fixtures_by_gw(
        enriched, start=start_gw, end=horizon_end
    )
    plan_gws = sorted(fixtures_gw)
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    ol_boards, ol_meta = mean_component_boards(
        pool=pool,
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
    print(f"[comp-resid] OL boards {ol_meta['elapsed_s']:.1f}s", flush=True)

    ol_rows: list[dict[str, Any]] = []
    for w in src["open_loop"]["weeks"]:
        gw = int(w["gw"])
        act = actual_components(merged, gw)
        ol_rows.extend(
            score_xi(ol_boards[gw], act, _xi_ids(w), gw=gw, arm="open_loop")
        )
    ol_df = pd.DataFrame(ol_rows)

    # --- Rolling step boards ---
    roll_rows: list[dict[str, Any]] = []
    stash = pool.copy()
    owned = list(src["open_loop"]["opening_15"])
    for step in src["rolling"]["steps"]:
        gw = int(step["gw"])
        commit = step["commit"]
        if step.get("reused_open_loop") or gw == start_gw:
            boards = ol_boards
            board_s = 0.0
        else:
            step_seed = seed0 + 10_000 * gw
            print(f"[comp-resid] rolling boards @ GW{gw}…", flush=True)
            hist_s = numagent_xp.asof_share_history(
                merged, before_gw=gw, season=season
            )
            ag, aa = _asof_alphas(hist_s)
            th = hist_by_team_norm(hist_s)
            pool_s = _build_hist_pool(hist_s, prices, decision_gw=gw, season=season)
            pool_s = _ensure_owned_in_pool(pool_s, stash, owned)
            stash = pool_s.copy()
            fx = _hist_fixtures_by_gw(enriched, start=gw, end=horizon_end)
            pg = sorted(fx)
            priced = _fd_priced_by_gw(enriched, fd, pg[:2])
            boards, meta = mean_component_boards(
                pool=pool_s,
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
            board_s = float(meta["elapsed_s"])
            print(f"[comp-resid] GW{gw} boards {board_s:.1f}s", flush=True)
        act = actual_components(merged, gw)
        # Prefer XI; commit may only have player_ids — use xi from replan head if present
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
            rows.append(
                {
                    "gw": int(gw),
                    "xP_G": float(sub["xP_G"].sum()),
                    "P_G": float(sub["P_G"].sum()),
                    "resid_G": float(sub["resid_G"].sum()),
                    "xP_CS": float(sub["xP_CS"].sum()),
                    "P_CS": float(sub["P_CS"].sum()),
                    "resid_CS": float(sub["resid_CS"].sum()),
                    "xP_GC": float(sub["xP_GC"].sum()),
                    "P_GC": float(sub["P_GC"].sum()),
                    "resid_GC": float(sub["resid_GC"].sum()),
                }
            )
        return rows

    state = {
        "task": "cascade_mean_xi_ft_component_resid",
        "source_artifact": str(args.artifact),
        "season": season,
        "n_strands": n_strands,
        "seed": seed0,
        "residual_def": "xP - actual",
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
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    # player-level CSV
    out_dir = ROOT / "data/predictions" / season / "cascade_mean_xi_ft_component_resid"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ol_df.empty:
        ol_df.to_csv(out_dir / "open_loop_xi_components.csv", index=False)
    if not roll_df.empty:
        roll_df.to_csv(out_dir / "rolling_xi_components.csv", index=False)
    _save(state)
    write_report(state)
    Path(args.out).write_text(json.dumps(state, indent=2, default=str) + "\n")

    print(json.dumps({
        "elapsed_s": state["elapsed_s"],
        "open_loop": state["arms"]["open_loop"]["summary"],
        "rolling": state["arms"]["rolling"]["summary"],
        "report": str(REPORT_MD),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
