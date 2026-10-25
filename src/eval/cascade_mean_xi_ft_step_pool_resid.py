"""Stepped selection-pool mean xi_ft: xP(G, CS, GA) vs actual.

Horizon GW8–19. At each decision GW ``t ∈ {8,…,12}``:

1. Rebuild **selection** pool = as-of mean minutes > 60 (n_apps≥2) before ``t``.
2. Rebuild **share** pool = full club ``m>0`` (impact subs on the dice).
3. Freeze that as-of state and project it forward on strands for ``t…19``.
4. Mean of N generative boards → one ``xi_ft`` plan; **commit** week ``t``.
5. Compare generative mean ``xp_goals`` / ``xp_cs`` / ``xp_gc`` on the XI
   to actual FPL G / CS / GA (GC) points.

GA here is the goals-against points component (``xp_gc`` / FPL GC deductions).

  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid --n 40
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
from src.models.short_horizon_cascade import load_prices
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
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid.json"),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    e0 = SEASON_E0[season]
    step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    start_gw = int(args.start_gw)
    horizon_end = int(args.horizon_end)
    n_strands = int(args.n)
    seed0 = int(args.seed)

    print(
        f"[step-pool] season={season} steps={step_gws} horizon={start_gw}-{horizon_end} "
        f"N={n_strands}",
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

    owned: list[str] | None = None
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
            select_pool = _ensure_owned_in_pool(select_pool, stash, owned)
            share_pool = _ensure_owned_in_pool(share_pool, stash, owned)
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
        )
        print(f"[step-pool] boards {board_meta['elapsed_s']:.1f}s", flush=True)

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
                "owned": list(owned),
                "buys": [id_to_name.get(x, x) for x in buys],
                "sells": [id_to_name.get(x, x) for x in sells],
                "n_transfers": int(week["n_transfers"]),
                "hits": int(week["hits"]),
                "ft_before": int(week["ft_before"]),
                "ft_after": int(week["ft_after"]),
                "xi_score": float(week.get("xi_score") or week.get("week_score") or 0.0),
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
    _plot(per_gw, plot_path, n=n_strands)
    summary = _summarise(df)
    state = {
        "task": "cascade_mean_xi_ft_step_pool_resid",
        "season": season,
        "horizon": [start_gw, horizon_end],
        "step_gws": step_gws,
        "n_strands": n_strands,
        "seed": seed0,
        "selection_pool": "as-of mean minutes > 60, n_apps≥2; frozen until next step",
        "share_pool": "full club m>0 (impact subs on dice)",
        "elapsed_s": _now() - t_all,
        "summary": summary,
        "per_gw": per_gw,
        "steps": [
            {
                k: v
                for k, v in s.items()
                if k not in {"owned"}  # keep artifact smaller; xi_ids enough
            }
            for s in steps_out
        ],
        "plot": str(plot_path),
    }
    # attach owned only as counts
    for s, raw in zip(state["steps"], steps_out, strict=True):
        s["n_owned"] = len(raw["owned"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Stepped ≥60′ selection pool — xP(G, CS, GA) vs actual",
        "",
        f"Season **{season}** · horizon {start_gw}–{horizon_end} · "
        f"commit steps {step_gws} · N={n_strands} · seed={seed0} · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## Protocol",
        "",
        "- At each decision GW t: rebuild **selection** pool = players with "
        "as-of mean minutes > 60 (and ≥2 apps) before t.",
        "- **Share** pool = full club with m>0 (impact subs on generative dice).",
        "- That as-of state (rates, minutes, both pools) is **projected forward** "
        "on strands for t…horizon end until the next step refreshes it.",
        "- Mean of N generative boards → `xi_ft` plan; commit week-t XI.",
        "- Residual = xP − actual on that XI. **GA** = goals-against points "
        "(`xp_gc` / FPL GC).",
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
    lines += ["", "## Commits", ""]
    for s in steps_out:
        lines.append(
            f"- **GW{s['gw']}:** buy {s['buys']} / sell {s['sells']} · "
            f"ft {s['ft_before']}→{s['ft_after']} · hits={s['hits']}"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_step_pool_resid "
        f"--n {n_strands} --steps {','.join(str(g) for g in step_gws)}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_mean_xi_ft_step_pool_resid",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Step GW8–12: refresh ≥60′ selection pool each GW; "
                "full shares; xP(G,CS,GA) vs actual on committed XI",
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
