"""Dig: leftover OL XI G overprint under full-club shares.

Decomposes Σ(E[G] − G) on the mean xi_ft open-loop XI (GW8–19) into:

1. **Pot** — FD/path ``xG`` vs actual team GF on the player's team-GW
   (``sG · α_G · (xG − GF)``).
2. **Share** — given actual GF, did ``sG`` over-allocate
   (``sG · α_G · GF − G``).
3. **Selection** — is the XI's mean resid in the left/right bootstrap
   tail of size-matched random ≥60′ attackers?

Uses **centreline** ``score_team_side`` with ``share_players`` = full club
(m>0) so the identity ``pot + share = E[G] − G`` holds. Generative strand
mean E[G] is logged as a side check (should be close).

  PYTHONPATH=. python3 -m src.eval.cascade_g_overprint_dig --n 40
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

from src.eval.cascade_mc_vs_mean_hist import (
    SEASON_E0,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
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
REPORT = ROOT / "reports" / "cascade_g_overprint_dig.md"
STATE = ROOT / "REPORT_STATE.json"
OUT_DEFAULT = Path("/opt/cursor/artifacts/cascade_g_overprint_dig.json")
ROLLING = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json")
N_BOOT = 5000
RNG_SEED = 7


def _now() -> float:
    return time.time()


def _pid_key(pid: str) -> str:
    return str(pid).split(":")[-1]


def _xi_by_gw(src: dict[str, Any]) -> dict[int, set[str]]:
    return {
        int(w["gw"]): {_pid_key(x) for x in (w.get("xi_ids") or w.get("player_ids") or [])}
        for w in src["open_loop"]["weeks"]
    }


def _team_gf_ga(merged: pd.DataFrame) -> pd.DataFrame:
    g = (
        merged.groupby(["GW", "fixture", "team", "was_home"], as_index=False)
        .agg(
            team_h_score=("team_h_score", "first"),
            team_a_score=("team_a_score", "first"),
            team_n=("team_n", "first"),
            date_iso=("date_iso", "first"),
        )
    )
    g["was_home"] = g["was_home"].astype(bool)
    g["gf"] = np.where(
        g["was_home"],
        pd.to_numeric(g["team_h_score"], errors="coerce"),
        pd.to_numeric(g["team_a_score"], errors="coerce"),
    ).astype(float)
    g["ga"] = np.where(
        g["was_home"],
        pd.to_numeric(g["team_a_score"], errors="coerce"),
        pd.to_numeric(g["team_h_score"], errors="coerce"),
    ).astype(float)
    g["team_n"] = g["team_n"].fillna(g["team"].map(lambda t: norm_team(str(t))))
    g["team_n"] = g["team_n"].map(lambda t: norm_team(str(t)))
    g["GW"] = pd.to_numeric(g["GW"], errors="coerce").astype(int)
    return g.dropna(subset=["gf", "ga"])


def _player_actual_g(merged: pd.DataFrame) -> pd.DataFrame:
    m = merged.copy()
    m["GW"] = pd.to_numeric(m["GW"], errors="coerce").astype(int)
    m["element"] = pd.to_numeric(m["element"], errors="coerce").astype(int)
    m["goals"] = pd.to_numeric(m["goals_scored"], errors="coerce").fillna(0.0)
    m["minutes"] = pd.to_numeric(m["minutes"], errors="coerce").fillna(0.0)
    m["team_n"] = m["team_n"].map(lambda t: norm_team(str(t)))
    return m[["GW", "element", "name", "position", "team_n", "goals", "minutes", "fixture"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=ROLLING)
    ap.add_argument("--n", type=int, default=40, help="generative strands (side check)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
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

    print(f"[g-dig] season={season} N_gen={args.n}", flush=True)
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
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws)
    priced_index = _priced_index(priced_by_gw)
    team_sides = _team_gf_ga(merged)
    pact = _player_actual_g(merged)

    # FD xG by (GW, team_n) from priced fits
    xg_by = {}
    for gw, fits in priced_by_gw.items():
        for f in fits:
            hn = norm_team(str(f["home"]))
            an = norm_team(str(f["away"]))
            # moments from λ via scoreline would need build_matrix; use lam as xG proxy
            # Better: use FD index joined on date — priced fits already have lam
            from src.markets.cascade_score_v2 import side_moments_from_lams

            mh = side_moments_from_lams(float(f["lam_h"]), float(f["lam_a"]), rho=float(f.get("rho") or 0))
            ma = side_moments_from_lams(float(f["lam_a"]), float(f["lam_h"]), rho=float(f.get("rho") or 0))
            xg_by[(int(gw), hn)] = float(mh["xg"])
            xg_by[(int(gw), an)] = float(ma["xg"])

    # Centreline boards per GW: score full share club, project
    print("[g-dig] centreline full-share boards…", flush=True)
    rows: list[dict[str, Any]] = []
    pool_attack_rows: list[dict[str, Any]] = []  # ≥60′ MID/FWD/DEF for bootstrap
    from src.markets.cascade_score_v2 import side_moments_from_lams

    for gw in plan_gws:
        xi = xi_by_gw.get(int(gw), set())
        act_gw = pact.loc[pact["GW"] == int(gw)]
        act_by = act_gw.set_index("element")
        tg = team_sides.loc[team_sides["GW"] == int(gw)]
        for sc in priced_by_gw.get(int(gw), []):
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
                scored = numagent_xp.score_team_side(
                    club,
                    xg=mom["xg"],
                    xcs=mom["xcs"],
                    xgc=mom["xgc"],
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    share_players=club,
                    share_se_shrink=True,
                )
                gf_hit = tg.loc[tg["team_n"] == team_key]
                gf = float(gf_hit.iloc[0]["gf"]) if not gf_hit.empty else float("nan")
                xg = float(mom["xg"])
                g_team = float(alpha_g) * xg
                for i in range(len(scored)):
                    r = scored.iloc[i]
                    pid = _pid_key(str(club.iloc[i]["player_id"]))
                    try:
                        eid = int(float(pid))
                    except ValueError:
                        continue
                    sG = float(r["sG"])
                    eG = float(r["E_G"])
                    mins = float(club.iloc[i]["m"])
                    if eid in act_by.index:
                        a = act_by.loc[eid]
                        if isinstance(a, pd.DataFrame):
                            a = a.iloc[0]
                        g = float(a["goals"])
                        name = str(a["name"])
                        pos = str(a["position"])
                        amins = float(a["minutes"])
                    else:
                        g = 0.0
                        name = str(club.iloc[i].get("name", pid))
                        pos = str(club.iloc[i].get("position", ""))
                        amins = 0.0
                    pot_piece = (
                        sG * float(alpha_g) * (xg - gf) if np.isfinite(gf) else float("nan")
                    )
                    share_piece = (
                        sG * float(alpha_g) * gf - g if np.isfinite(gf) else float("nan")
                    )
                    rec = {
                        "gw": int(gw),
                        "element": eid,
                        "player_id": pid,
                        "name": name,
                        "position": pos,
                        "team_n": team_key,
                        "m_asof": mins,
                        "minutes": amins,
                        "sG": sG,
                        "E_G": eG,
                        "G": g,
                        "resid": eG - g,
                        "xg": xg,
                        "gf": gf,
                        "G_team": g_team,
                        "pot_piece": pot_piece,
                        "share_piece": share_piece,
                        "on_xi": pid in xi,
                    }
                    if mins >= 60 and pos in ("MID", "FWD", "DEF", "GKP"):
                        pool_attack_rows.append(rec)
                    if pid in xi:
                        rows.append(rec)

    xi_df = pd.DataFrame(rows)
    pool_df = pd.DataFrame(pool_attack_rows)

    # Generative side-check on XI only (full share dice)
    print("[g-dig] generative side-check…", flush=True)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    eg_acc: dict[tuple[int, str], float] = defaultdict(float)
    for s in range(int(args.n)):
        if s == 0 or (s + 1) % 20 == 0 or s + 1 == int(args.n):
            print(f"  [g-dig] strand {s + 1}/{args.n}", flush=True)
        rng = random.Random(seed0 + s)
        paths = {c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()}
        score_rng = random.Random(seed0 + 10_000 + s)
        for gw in plan_gws:
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
                home_xp, away_xp, _ = score_fixture_generative(
                    home,
                    away,
                    lam_h=float(sc["lam_h"]),
                    lam_a=float(sc["lam_a"]),
                    rho=float(sc.get("rho") or 0.0),
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    rng=score_rng,
                    home_hist=team_hist.get(hn),
                    away_hist=team_hist.get(an),
                    lineup_cov=DEFAULT_LINEUP_COV,
                    goals_home=int(sc["hg"]),
                    goals_away=int(sc["ag"]),
                )
                xi = xi_by_gw.get(int(gw), set())
                for scored in (home_xp, away_xp):
                    if scored is None or scored.empty:
                        continue
                    for i in range(len(scored)):
                        pid = _pid_key(str(scored.iloc[i].get("player_id", "")))
                        if pid not in xi:
                            # also try element
                            if "element" in scored.columns:
                                pid = _pid_key(str(int(scored.iloc[i]["element"])))
                        if pid not in xi:
                            continue
                        eg_acc[(int(gw), pid)] += float(scored.iloc[i].get("E_G") or 0.0)
    for k in eg_acc:
        eg_acc[k] /= float(args.n)
    if not xi_df.empty:
        xi_df["E_G_gen"] = [
            eg_acc.get((int(r.gw), str(r.player_id)), float("nan"))
            for r in xi_df.itertuples()
        ]
        xi_df["resid_gen"] = xi_df["E_G_gen"] - xi_df["G"]

    def _sum_finite(s: pd.Series) -> float:
        x = pd.to_numeric(s, errors="coerce")
        return float(x[np.isfinite(x)].sum())

    decomp = {
        "n_xi_player_gw": int(len(xi_df)),
        "sum_E_G_mean": _sum_finite(xi_df["E_G"]) if len(xi_df) else float("nan"),
        "sum_G": _sum_finite(xi_df["G"]) if len(xi_df) else float("nan"),
        "sum_resid_mean": _sum_finite(xi_df["resid"]) if len(xi_df) else float("nan"),
        "sum_pot_piece": _sum_finite(xi_df["pot_piece"]) if len(xi_df) else float("nan"),
        "sum_share_piece": _sum_finite(xi_df["share_piece"]) if len(xi_df) else float("nan"),
        "sum_E_G_gen": _sum_finite(xi_df["E_G_gen"]) if "E_G_gen" in xi_df else float("nan"),
        "sum_resid_gen": _sum_finite(xi_df["resid_gen"]) if "resid_gen" in xi_df else float("nan"),
        "alpha_g": float(alpha_g),
    }
    # identity check
    decomp["pot_plus_share"] = decomp["sum_pot_piece"] + decomp["sum_share_piece"]
    decomp["identity_err"] = decomp["pot_plus_share"] - decomp["sum_resid_mean"]

    # Team-GW pot bias: unique XI clubs vs league
    xi_tg = (
        xi_df.groupby(["gw", "team_n"], as_index=False)
        .agg(xg=("xg", "first"), gf=("gf", "first"))
        .dropna()
    )
    league_tg = team_sides.loc[team_sides["GW"].isin(plan_gws)].copy()
    # attach xg
    league_tg["xg"] = [
        xg_by.get((int(r.GW), str(r.team_n)), float("nan"))
        for r in league_tg.itertuples()
    ]
    league_tg = league_tg.loc[np.isfinite(league_tg["xg"])].copy()
    league_tg["pot_bias"] = league_tg["xg"] - league_tg["gf"]
    xi_tg["pot_bias"] = xi_tg["xg"] - xi_tg["gf"]
    team_pot = {
        "league_n": int(len(league_tg)),
        "league_mean_xg": float(league_tg["xg"].mean()),
        "league_mean_gf": float(league_tg["gf"].mean()),
        "league_bias": float(league_tg["pot_bias"].mean()),
        "xi_clubs_n": int(len(xi_tg)),
        "xi_mean_xg": float(xi_tg["xg"].mean()) if len(xi_tg) else float("nan"),
        "xi_mean_gf": float(xi_tg["gf"].mean()) if len(xi_tg) else float("nan"),
        "xi_bias": float(xi_tg["pot_bias"].mean()) if len(xi_tg) else float("nan"),
    }

    # Bootstrap selection: mean resid of random n XI attacker-GWs from pool ≥60′
    rng = np.random.default_rng(RNG_SEED)
    xi_attack = xi_df.loc[xi_df["position"].isin(["MID", "FWD", "DEF"])].copy()
    n_draw = int(len(xi_attack))
    pool_res = pool_df.loc[pool_df["position"].isin(["MID", "FWD", "DEF"]), "resid"].to_numpy(
        float
    )
    boot = np.empty(int(args.n_boot), float)
    for i in range(int(args.n_boot)):
        idx = rng.integers(0, len(pool_res), size=max(n_draw, 1))
        boot[i] = float(pool_res[idx].mean())
    sel_mean = float(xi_attack["resid"].mean()) if n_draw else float("nan")
    # one-sided: P(null >= selected) if selected is high overprint
    p_right = float((boot >= sel_mean).mean()) if n_draw else float("nan")
    selection = {
        "n_xi_attack_gw": n_draw,
        "n_pool": int(len(pool_res)),
        "xi_mean_resid": sel_mean,
        "boot_mean": float(boot.mean()),
        "boot_p05": float(np.quantile(boot, 0.05)),
        "boot_p95": float(np.quantile(boot, 0.95)),
        "p_right": p_right,
    }

    # Top contributors
    by_player = (
        xi_df.groupby(["name", "position"], as_index=False)
        .agg(
            n=("resid", "size"),
            sum_E=("E_G", "sum"),
            sum_G=("G", "sum"),
            sum_resid=("resid", "sum"),
            sum_pot=("pot_piece", "sum"),
            sum_share=("share_piece", "sum"),
        )
        .sort_values("sum_resid", ascending=False)
    )

    # Verdict
    pot = decomp["sum_pot_piece"]
    sh = decomp["sum_share_piece"]
    tot = decomp["sum_resid_mean"]
    gen = decomp["sum_resid_gen"]
    abs_parts = abs(pot) + abs(sh)
    frac_pot = float(abs(pot) / abs_parts) if abs_parts > 1e-9 else float("nan")
    if pot < 0 and sh > 0:
        read = (
            "**Not pot.** XI clubs' FD xG ≤ GF "
            f"(team-GW bias {team_pot['xi_bias']:+.3f}; pot piece {pot:+.1f}). "
            f"Centreline resid is small ({tot:+.1f}) and **share-driven** "
            f"(share piece {sh:+.1f}): sG mass on XI players who blanked. "
            f"Generative resid stays higher ({gen:+.1f}) — dice/nonlinear gap "
            "vs centreline, not FD λ mis-fit."
        )
    elif pot > 0 and pot > abs(sh):
        read = (
            "Remaining G overprint is mostly **pot**: XI clubs' FD xG > GF "
            f"(xi team-GW bias {team_pot['xi_bias']:+.3f}; pot piece "
            f"{pot:+.1f} of {tot:+.1f})."
        )
    else:
        read = (
            f"Mixed: pot piece {pot:+.1f}, share piece {sh:+.1f} "
            f"(centreline resid {tot:+.1f}; gen {gen:+.1f}). "
            f"XI club pot bias {team_pot['xi_bias']:+.3f} vs league "
            f"{team_pot['league_bias']:+.3f}."
        )
    if selection["p_right"] < 0.1:
        read += (
            f" Selection also elevated (XI mean resid {sel_mean:+.3f}, "
            f"bootstrap p_right={selection['p_right']:.3f})."
        )
    else:
        read += " XI selection resid is not in the bootstrap tail."

    verdict = {
        "decomp": decomp,
        "team_pot": team_pot,
        "selection": selection,
        "frac_pot_abs": float(frac_pot) if np.isfinite(frac_pot) else float("nan"),
        "read": read,
    }

    state = {
        "task": "cascade_g_overprint_dig",
        "season": season,
        "gws": plan_gws,
        "n_strands_gen": int(args.n),
        "n_boot": int(args.n_boot),
        "share_pool": "full_m_gt_0",
        "n_select": int(len(select)),
        "n_share": int(len(share)),
        "elapsed_s": _now() - t0,
        "verdict": verdict,
        "top_players": by_player.head(12).to_dict(orient="records"),
        "bottom_players": by_player.tail(8).to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# G overprint dig — pot vs share vs selection",
        "",
        f"Season **{season}** · OL XI GW{plan_gws[0]}–{plan_gws[-1]} · "
        f"full-club shares · centreline decomp · gen N={args.n} · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## Identity",
        "",
        "On each XI player-GW (centreline, full-club `sG`):",
        "",
        "`resid = E[G] − G = sG·α_G·(xG − GF) + (sG·α_G·GF − G)`",
        "",
        f"- **pot piece** Σ = **{pot:+.2f}**",
        f"- **share piece** Σ = **{sh:+.2f}**",
        f"- **total** Σ = **{tot:+.2f}** (identity err {decomp['identity_err']:+.3f})",
        f"- generative Σ resid (side check) = **{decomp['sum_resid_gen']:+.2f}**",
        "",
        (
            f"|Pot| vs |share| mass: pot {frac_pot:.0%} of "
            f"|pot|+|share|."
            if np.isfinite(frac_pot)
            else ""
        ),
        "",
        "## Team-GW pot (xG − GF)",
        "",
        f"| panel | n | mean xG | mean GF | bias |",
        f"| --- | ---: | ---: | ---: | ---: |",
        f"| League GW{plan_gws[0]}–{plan_gws[-1]} | {team_pot['league_n']} | "
        f"{team_pot['league_mean_xg']:.3f} | {team_pot['league_mean_gf']:.3f} | "
        f"{team_pot['league_bias']:+.3f} |",
        f"| XI clubs | {team_pot['xi_clubs_n']} | "
        f"{team_pot['xi_mean_xg']:.3f} | {team_pot['xi_mean_gf']:.3f} | "
        f"{team_pot['xi_bias']:+.3f} |",
        "",
        "## Selection bootstrap (MID/FWD/DEF resid)",
        "",
        f"- XI mean resid **{sel_mean:+.3f}** · pool n={selection['n_pool']} · "
        f"boot p05/p95 {selection['boot_p05']:+.3f}/{selection['boot_p95']:+.3f} · "
        f"p_right={selection['p_right']:.3f}",
        "",
        "## Top XI resid contributors",
        "",
        "| name | pos | n | ΣE[G] | ΣG | resid | pot | share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in state["top_players"]:
        lines.append(
            f"| {r['name']} | {r['position']} | {int(r['n'])} | "
            f"{r['sum_E']:.2f} | {r['sum_G']:.0f} | {r['sum_resid']:+.2f} | "
            f"{r['sum_pot']:+.2f} | {r['sum_share']:+.2f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        verdict["read"],
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_g_overprint_dig --n 40",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_g_overprint_dig",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Decompose leftover G resid into pot / share / selection",
                "verdict": verdict,
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(verdict, indent=2, default=str))
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
