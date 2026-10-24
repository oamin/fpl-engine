"""Roll one short+long cascade strand and plot a player's xP through it.

Short: priced GW (default 8) via gw08 odds. Long: Opta rankmean λ → k∼M.
Player involvement: one dice allocate per fixture (no inner repeats).
Minutes frozen; as-of shares fixed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.cascade_score_v2 import build_frozen_pool, normalize_live_history
from src.markets.short_horizon_generative import (
    hist_by_team_norm,
    score_fixture_generative,
)
from src.models.multistage_mc_milp import (
    DEFAULT_LINEUP_COV,
    LOG_CSV,
    ClubPath,
    _priced_index,
    fixtures_by_gw,
    init_paths_from_opta,
    load_priced_week_fits,
    sample_week_scenario,
)
from src.models.opta_gf_ga_horizon import LIVE_E0, assessed_league_gf
from src.models.player_pool import POOL_JSON, load_player_pool_ids
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[1]


def _roll_components(row: pd.Series) -> dict[str, float]:
    goals = float(row.get("xp_goals") or 0.0)
    assists = float(row.get("xp_assists") or 0.0)
    cs = float(row.get("xp_cs") or 0.0)
    gc = float(row.get("xp_gc") or 0.0)
    appear = float(row.get("xp_appear") or 0.0)
    soft = (
        float(row.get("xp_defcon") or 0.0)
        + float(row.get("xp_bonus") or 0.0)
        + float(row.get("xp_cards") or 0.0)
        + float(row.get("xp_saves") or 0.0)
    )
    roll = goals + assists + cs + gc
    return {
        "xp_roll": roll,
        "xp_appear": appear,
        "xp_soft": soft,
        "xp_total": float(row.get("xp") or (roll + appear + soft)),
        "xp_goals": goals,
        "xp_assists": assists,
        "xp_cs": cs,
        "xp_gc": gc,
    }


def run_one_strand_player_xp(
    *,
    player_name: str = "Saka",
    decision_gw: int = 8,
    horizon_end: int = 14,
    priced_gw: int = 8,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bootstrap = json.loads((ROOT / "data/live/bootstrap.json").read_text())
    fixtures = json.loads((ROOT / "data/live/fixtures.json").read_text())
    team_names = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    selectable = load_player_pool_ids(POOL_JSON)
    logs = pd.read_csv(LOG_CSV)
    hist = numagent_xp.asof_share_history(
        normalize_live_history(logs),
        before_gw=int(decision_gw),
        season="2026-27",
    )
    team_hist = hist_by_team_norm(hist)
    pool = build_frozen_pool(
        bootstrap,
        selectable,
        logs,
        before_gw=int(decision_gw),
        season="2026-27",
    )
    hit = pool[pool["name"].astype(str).str.contains(player_name, case=False, na=False)]
    if hit.empty:
        raise SystemExit(f"player matching {player_name!r} not in pool")
    prow = hit.iloc[0]
    pid = str(prow["player_id"])
    pname = str(prow["name"])
    pteam = norm_team(str(prow["team_norm"]))

    by_gw = fixtures_by_gw(
        fixtures, team_names, start=int(decision_gw), end=int(horizon_end)
    )
    horizon_gws = sorted(g for g in by_gw if int(decision_gw) <= g <= int(horizon_end))
    if not horizon_gws:
        raise SystemExit("empty horizon")

    odds_path = ROOT / "data/scratch/gw08_odds_full.json"
    priced_by_gw: dict[int, list[dict[str, Any]]] = {}
    if odds_path.is_file() and int(priced_gw) in set(horizon_gws):
        fits = load_priced_week_fits(odds_path, bootstrap, gw=int(priced_gw))
        if fits:
            priced_by_gw[int(priced_gw)] = fits
    priced_index = _priced_index(priced_by_gw)

    league_state = assessed_league_gf(LIVE_E0)
    league_gf = float(league_state["league_gf"])
    sd_league_gf = float(league_state["se_league_gf"])
    paths: dict[str, ClubPath] = {
        c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in init_paths_from_opta().items()
    }
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()
    rng = random.Random(int(seed))
    score_rng = random.Random(int(seed) + 99)

    rows: list[dict[str, Any]] = []
    for gw in horizon_gws:
        week = by_gw.get(int(gw), [])
        player_fx = None
        for fx in week:
            if pteam in (fx["home"], fx["away"]):
                player_fx = fx
                break
        if player_fx is None:
            rows.append(
                {
                    "gw": int(gw),
                    "player": pname,
                    "player_id": pid,
                    "team": pteam,
                    "source": "blank",
                    "opponent": None,
                    "venue": None,
                    "hg": None,
                    "ag": None,
                    "lam_h": None,
                    "lam_a": None,
                    "xp_roll": 0.0,
                    "xp_appear": 0.0,
                    "xp_soft": 0.0,
                    "xp_total": 0.0,
                    "xp_goals": 0.0,
                    "xp_assists": 0.0,
                    "xp_cs": 0.0,
                    "xp_gc": 0.0,
                }
            )
            continue

        sc = sample_week_scenario(
            player_fx["home"],
            player_fx["away"],
            gw=int(gw),
            priced_index=priced_index,
            paths=paths,
            league_gf=league_gf,
            rng=rng,
            sd_league_gf=sd_league_gf,
        )
        # Path coherence for later Opta weeks (even though λ ignores ClubPath).
        if player_fx["home"] in paths and player_fx["away"] in paths:
            paths[player_fx["home"]].observe(sc["hg"], sc["ag"])
            paths[player_fx["away"]].observe(sc["ag"], sc["hg"])

        home_n = norm_team(sc["home"])
        away_n = norm_team(sc["away"])
        home_club = pool.loc[pool["team_norm"].map(norm_team) == home_n].copy()
        away_club = pool.loc[pool["team_norm"].map(norm_team) == away_n].copy()
        home_xp, away_xp, meta = score_fixture_generative(
            home_club,
            away_club,
            lam_h=float(sc["lam_h"]),
            lam_a=float(sc["lam_a"]),
            rho=float(sc.get("rho") or 0.0),
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            rng=score_rng,
            home_hist=team_hist.get(home_n),
            away_hist=team_hist.get(away_n),
            home_props=sc.get("home_props") if isinstance(sc.get("home_props"), dict) else None,
            away_props=sc.get("away_props") if isinstance(sc.get("away_props"), dict) else None,
            lineup_cov=DEFAULT_LINEUP_COV,
            goals_home=int(sc["hg"]),
            goals_away=int(sc["ag"]),
        )
        scored = home_xp if pteam == home_n else away_xp
        if scored is None or scored.empty:
            raise RuntimeError(f"no generative board for {pname} GW{gw}")
        if "player_id" in scored.columns:
            prow_sc = scored.loc[scored["player_id"].astype(str) == pid]
        else:
            prow_sc = scored.loc[scored["element"].astype(int).astype(str) == pid]
        if prow_sc.empty:
            # Player filtered m=0 on this side.
            comps = {
                "xp_roll": 0.0,
                "xp_appear": 0.0,
                "xp_soft": 0.0,
                "xp_total": 0.0,
                "xp_goals": 0.0,
                "xp_assists": 0.0,
                "xp_cs": 0.0,
                "xp_gc": 0.0,
            }
        else:
            comps = _roll_components(prow_sc.iloc[0])

        venue = "H" if pteam == home_n else "A"
        opp = away_n if venue == "H" else home_n
        src = str(sc.get("source") or ("priced" if int(gw) == int(priced_gw) else "opta"))
        rows.append(
            {
                "gw": int(gw),
                "player": pname,
                "player_id": pid,
                "team": pteam,
                "source": src,
                "opponent": opp,
                "venue": venue,
                "hg": int(sc["hg"]),
                "ag": int(sc["ag"]),
                "lam_h": float(sc["lam_h"]),
                "lam_a": float(sc["lam_a"]),
                **comps,
            }
        )

    frame = pd.DataFrame(rows)
    meta_out = {
        "player": pname,
        "player_id": pid,
        "team": pteam,
        "decision_gw": int(decision_gw),
        "horizon_end": int(horizon_end),
        "priced_gw": int(priced_gw),
        "seed": int(seed),
        "n_priced_fits": int(sum(len(v) for v in priced_by_gw.values())),
        "league_gf": league_gf,
        "g_pm": float(league_state["g_pm"]),
    }
    return frame, meta_out


def average_strands(
    *,
    player_name: str,
    decision_gw: int,
    horizon_end: int,
    priced_gw: int,
    seed0: int,
    n_strands: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mean (± sample SD) player xP by GW across ``n_strands`` independent runs."""
    frames: list[pd.DataFrame] = []
    meta0: dict[str, Any] | None = None
    for i in range(int(n_strands)):
        fr, meta = run_one_strand_player_xp(
            player_name=player_name,
            decision_gw=decision_gw,
            horizon_end=horizon_end,
            priced_gw=priced_gw,
            seed=int(seed0) + i,
        )
        fr = fr.copy()
        fr["strand"] = i
        frames.append(fr)
        if meta0 is None:
            meta0 = meta
    assert meta0 is not None
    all_f = pd.concat(frames, ignore_index=True)
    g = all_f.groupby("gw", sort=True)
    out = g.agg(
        player=("player", "first"),
        player_id=("player_id", "first"),
        team=("team", "first"),
        opponent=("opponent", "first"),
        venue=("venue", "first"),
        source=("source", "first"),
        xp_roll=("xp_roll", "mean"),
        xp_appear=("xp_appear", "mean"),
        xp_soft=("xp_soft", "mean"),
        xp_total=("xp_total", "mean"),
        xp_goals=("xp_goals", "mean"),
        xp_assists=("xp_assists", "mean"),
        xp_cs=("xp_cs", "mean"),
        xp_gc=("xp_gc", "mean"),
        sd_xp_total=("xp_total", "std"),
        sd_xp_roll=("xp_roll", "std"),
        n=("xp_total", "count"),
    ).reset_index()
    out["sd_xp_total"] = out["sd_xp_total"].fillna(0.0)
    out["sd_xp_roll"] = out["sd_xp_roll"].fillna(0.0)
    meta0 = {
        **meta0,
        "n_strands": int(n_strands),
        "seed0": int(seed0),
        "seeds": f"{int(seed0)}..{int(seed0) + int(n_strands) - 1}",
    }
    return out, meta0


def plot_strand(frame: pd.DataFrame, meta: dict[str, Any], out: Path) -> Path:
    gws = frame["gw"].to_numpy()
    roll = frame["xp_roll"].to_numpy(dtype=float)
    appear = frame["xp_appear"].to_numpy(dtype=float)
    soft = frame["xp_soft"].to_numpy(dtype=float)
    total = frame["xp_total"].to_numpy(dtype=float)
    n_strands = int(meta.get("n_strands") or 1)
    sd_total = (
        frame["sd_xp_total"].to_numpy(dtype=float)
        if "sd_xp_total" in frame.columns
        else np.zeros_like(total)
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(gws, roll, width=0.65, label="roll (G+A+CS+GC)", color="#1f4e79")
    ax.bar(gws, appear, width=0.65, bottom=roll, label="appearance (fixed)", color="#7eb6d9")
    ax.bar(
        gws,
        soft,
        width=0.65,
        bottom=roll + appear,
        label="soft priors (fixed)",
        color="#c8d6e5",
    )
    ax.errorbar(
        gws,
        total,
        yerr=sd_total if n_strands > 1 else None,
        fmt="o-",
        color="#111111",
        lw=1.4,
        ms=5,
        capsize=3,
        label="total xP" + (" ± SD" if n_strands > 1 else ""),
    )

    for _, r in frame.iterrows():
        if r.get("source") == "blank" or (pd.isna(r.get("opponent")) and r.get("source") == "blank"):
            tag = "blank"
        elif pd.isna(r.get("opponent")):
            tag = "blank"
        else:
            tag = f"{r['venue']} {r['opponent']}"
            if n_strands == 1 and r.get("hg") is not None and not pd.isna(r.get("hg")):
                tag = f"{tag}\n{int(r['hg'])}-{int(r['ag'])}"
            if "short" in str(r.get("source", "")).lower() or r["gw"] == meta["priced_gw"]:
                tag = "SHORT\n" + tag
            else:
                tag = "LONG\n" + tag
        y = float(r["xp_total"]) + (float(r["sd_xp_total"]) if n_strands > 1 else 0.0) + 0.25
        ax.text(
            r["gw"],
            y,
            tag,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    ax.axvline(meta["priced_gw"] + 0.5, color="#888888", ls="--", lw=0.9, alpha=0.8)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("xP" + (f" (mean of {n_strands} strands)" if n_strands > 1 else " (one strand)"))
    ax.set_xticks(list(gws))
    ymax = float((total + sd_total).max()) + 2.5
    ax.set_ylim(0, max(ymax, 6.0))
    if n_strands > 1:
        title = (
            f"{meta['player']} — mean short+long xP over {n_strands} strands "
            f"(seeds {meta.get('seeds')}, decision GW{meta['decision_gw']})"
        )
    else:
        title = (
            f"{meta['player']} — one short+long strand "
            f"(seed={meta['seed']}, decision GW{meta['decision_gw']})"
        )
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", default="Saka")
    p.add_argument("--decision-gw", type=int, default=8)
    p.add_argument("--horizon-end", type=int, default=14)
    p.add_argument("--priced-gw", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-strands", type=int, default=1, help="Average this many strands (seeds seed..seed+N-1)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
    )
    args = p.parse_args()
    n = max(1, int(args.n_strands))
    slug = str(args.player).lower().replace(" ", "_")
    if args.out is None:
        args.out = Path(
            f"/opt/cursor/artifacts/strand_player_xp_{slug}"
            + (f"_mean_n{n}.png" if n > 1 else ".png")
        )
    if args.csv is None:
        args.csv = Path(
            f"/opt/cursor/artifacts/strand_player_xp_{slug}"
            + (f"_mean_n{n}.csv" if n > 1 else ".csv")
        )
    if n > 1:
        frame, meta = average_strands(
            player_name=args.player,
            decision_gw=args.decision_gw,
            horizon_end=args.horizon_end,
            priced_gw=args.priced_gw,
            seed0=args.seed,
            n_strands=n,
        )
    else:
        frame, meta = run_one_strand_player_xp(
            player_name=args.player,
            decision_gw=args.decision_gw,
            horizon_end=args.horizon_end,
            priced_gw=args.priced_gw,
            seed=args.seed,
        )
    frame.to_csv(args.csv, index=False)
    path = plot_strand(frame, meta, args.out)
    print(json.dumps({"meta": meta, "csv": str(args.csv), "plot": str(path)}, indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
