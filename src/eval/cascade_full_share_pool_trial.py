"""Quick trial: thin selection pool vs full-club share pool for generative π.

Selection XI stays the open-loop plan from ``cascade_mean_xi_ft_rolling``.
Generative attack dice are built on either:

- **thin** — same ≥60′ pool used for selection (current bug / baseline)
- **full** — all as-of players with m>0 (impact subs included); thin gate
  is not applied before shares

Reports Σ E[G] / Σ G / resid on the OL XI (GW8–19).

  PYTHONPATH=. python3 -m src.eval.cascade_full_share_pool_trial --n 40
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
    SHARE_MIN_APPS,
    SHARE_MIN_MINUTES,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets import numagent_xp
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
REPORT = ROOT / "reports" / "cascade_full_share_pool_trial.md"
STATE = ROOT / "REPORT_STATE.json"


def _now() -> float:
    return time.time()


def _xi_by_gw(src: dict[str, Any]) -> dict[int, set[str]]:
    return {
        int(w["gw"]): {str(x) for x in (w.get("xi_ids") or w.get("player_ids"))}
        for w in src["open_loop"]["weeks"]
    }


def _actual_g(
    merged: pd.DataFrame, xi_by_gw: dict[int, set[str]]
) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for gw, xi in xi_by_gw.items():
        g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
        for pid in xi:
            eid = int(str(pid).split(":")[-1])
            hit = g.loc[g["element"] == eid]
            if hit.empty:
                continue
            out[(gw, pid)] = float(hit.iloc[0]["goals_scored"] or 0)
    return out


def _run_arm(
    *,
    label: str,
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
    xi_by_gw: dict[int, set[str]],
    n_strands: int,
    seed0: int,
) -> dict[str, Any]:
    priced_index = _priced_index(priced_by_gw)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    eg: dict[tuple[int, str], float] = defaultdict(float)
    # Club size / Haaland π diagnostics on first strand GW8 first City fixture
    pi_diag: dict[str, Any] = {}
    t0 = _now()
    for s in range(int(n_strands)):
        if s == 0 or (s + 1) % 20 == 0 or s + 1 == int(n_strands):
            print(f"  [{label}] strand {s + 1}/{n_strands}", flush=True)
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
            xi = xi_by_gw[int(gw)]
            for sc in rolled[int(gw)]:
                hn = norm_team(str(sc["home"]))
                an = norm_team(str(sc["away"]))
                home = share_pool.loc[
                    share_pool["team_norm"].map(norm_team) == hn
                ].copy()
                away = share_pool.loc[
                    share_pool["team_norm"].map(norm_team) == an
                ].copy()
                if s == 0 and gw == plan_gws[0] and not pi_diag:
                    for team, club in ((hn, home), (an, away)):
                        if club.empty:
                            continue
                        dice, _ = build_club_dice_from_side(
                            club,
                            alpha_g=alpha_g,
                            alpha_a=alpha_a,
                            G_team=float(alpha_g) * 2.0,
                        )
                        if dice is None:
                            continue
                        top = sorted(
                            zip(dice.names, dice.pi.tolist(), strict=True),
                            key=lambda x: -x[1],
                        )[:5]
                        pi_diag[team] = {
                            "n_on_dice": len(dice.names),
                            "top_pi": [{"name": n, "pi": float(p)} for n, p in top],
                        }
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
                        if pid not in xi:
                            continue
                        eg[(int(gw), pid)] += float(row.get("E_G") or 0.0)

    # Average over strands
    for k in list(eg):
        eg[k] /= float(n_strands)
    return {
        "label": label,
        "n_share_pool": int(len(share_pool)),
        "n_strands": int(n_strands),
        "elapsed_s": _now() - t0,
        "E_G": {f"{gw}:{pid}": v for (gw, pid), v in eg.items()},
        "pi_diag": pi_diag,
        "share_pool_by_team": (
            share_pool.groupby(share_pool["team_norm"].map(norm_team))
            .size()
            .sort_values()
            .to_dict()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json"),
    )
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_full_share_pool_trial.json"),
    )
    args = ap.parse_args(argv)

    src = json.loads(Path(args.artifact).read_text())
    season = str(src["season"])
    e0 = SEASON_E0[season]
    n_strands = int(args.n)
    seed0 = int(src["seed"])
    start_gw = int(src["horizon_start"])
    end_gw = int(src["horizon_end"])
    xi_by_gw = _xi_by_gw(src)

    print(f"[full-share] N={n_strands} season={season}", flush=True)
    t_all = _now()
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
    thin = _build_hist_pool(hist, prices, decision_gw=start_gw, season=season)
    full = _build_hist_share_pool(hist, prices, decision_gw=start_gw, season=season)
    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start_gw, end=end_gw)
    plan_gws = sorted(fixtures_gw)
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    actual = _actual_g(merged, xi_by_gw)

    print(
        f"[full-share] thin pool={len(thin)} full share pool={len(full)} "
        f"(City thin={int((thin.team_norm.map(norm_team)=='man city').sum())} "
        f"full={int((full.team_norm.map(norm_team)=='man city').sum())})",
        flush=True,
    )

    arms = {}
    for label, pool in (("thin_shares", thin), ("full_shares", full)):
        print(f"[full-share] arm={label}…", flush=True)
        arms[label] = _run_arm(
            label=label,
            share_pool=pool,
            fixtures_gw=fixtures_gw,
            priced_by_gw=priced_by_gw,
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            plan_gws=plan_gws,
            xi_by_gw=xi_by_gw,
            n_strands=n_strands,
            seed0=seed0,
        )

    def summarise(arm: dict[str, Any]) -> dict[str, float]:
        eg_map = arm["E_G"]
        sum_e = 0.0
        sum_g = 0.0
        n = 0
        for (gw, pid), g in actual.items():
            key = f"{gw}:{pid}"
            e = float(eg_map.get(key, 0.0))
            sum_e += e
            sum_g += float(g)
            n += 1
        return {
            "n_player_gw": n,
            "sum_E_G": sum_e,
            "sum_G": sum_g,
            "sum_resid": sum_e - sum_g,
            "mean_resid": (sum_e - sum_g) / n if n else float("nan"),
        }

    summaries = {k: summarise(v) for k, v in arms.items()}
    # Haaland slice
    haaland = {}
    for label, arm in arms.items():
        e = sum(v for k, v in arm["E_G"].items() if k.endswith(":430") or "Haaland" in k)
        # find haaland pid from thin pool
        hit = thin.loc[thin["name"].astype(str) == "Erling Haaland"]
        if not hit.empty:
            pid = str(hit.iloc[0]["player_id"])
            e = sum(v for k, v in arm["E_G"].items() if k.endswith(pid) or k.split(":", 1)[-1] == pid)
            # keys are gw:pid
            e = sum(v for k, v in arm["E_G"].items() if k.split(":", 1)[-1] == pid)
            g = sum(
                actual[(gw, p)]
                for (gw, p) in actual
                if p == pid
            )
            haaland[label] = {"sum_E_G": e, "sum_G": g, "resid": e - g}

    state = {
        "task": "cascade_full_share_pool_trial",
        "season": season,
        "n_strands": n_strands,
        "seed": seed0,
        "alpha_g": alpha_g,
        "selection_pool": "minutes>60, n_apps≥2 (unchanged; XI from artifact)",
        "share_pools": {
            "thin_shares": "same as selection (≥60′)",
            "full_shares": f"m>0, min_minutes={SHARE_MIN_MINUTES}, min_apps={SHARE_MIN_APPS}",
        },
        "n_thin": int(len(thin)),
        "n_full": int(len(full)),
        "summaries": summaries,
        "haaland": haaland,
        "pi_diag": {k: v["pi_diag"] for k, v in arms.items()},
        "share_pool_by_team": {
            k: v["share_pool_by_team"] for k, v in arms.items()
        },
        "elapsed_s": _now() - t_all,
        "source_artifact": str(args.artifact),
    }

    delta = summaries["full_shares"]["sum_resid"] - summaries["thin_shares"]["sum_resid"]
    state["delta_sum_resid_full_minus_thin"] = delta

    lines = [
        "# Full-club share pool trial (selection stays thin)",
        "",
        f"Season **{season}** · N={n_strands} · seed={seed0} · "
        f"α_G={alpha_g:.4f} · wall {state['elapsed_s']:.1f}s",
        "",
        "Open-loop XI from the mean `xi_ft` rolling artifact (selection ≥60′). "
        "Generative dice built on **thin** (≥60′) vs **full** (m>0, impact subs).",
        "",
        f"- Thin share pool size: **{len(thin)}** (City "
        f"{int((thin.team_norm.map(norm_team)=='man city').sum())})",
        f"- Full share pool size: **{len(full)}** (City "
        f"{int((full.team_norm.map(norm_team)=='man city').sum())})",
        "",
        "## OL XI GW8–19 count residual",
        "",
        "| arm | Σ E[G] | Σ G | Σ resid | mean resid / match |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for lab in ("thin_shares", "full_shares"):
        s = summaries[lab]
        lines.append(
            f"| {lab} | {s['sum_E_G']:.2f} | {s['sum_G']:.2f} | "
            f"{s['sum_resid']:+.2f} | {s['mean_resid']:+.3f} |"
        )
    lines += [
        "",
        f"Δ Σ resid (full − thin) = **{delta:+.2f}**",
        "",
        "## Haaland",
        "",
        "| arm | Σ E[G] | Σ G | resid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for lab in ("thin_shares", "full_shares"):
        h = haaland.get(lab, {})
        lines.append(
            f"| {lab} | {h.get('sum_E_G', float('nan')):.2f} | "
            f"{h.get('sum_G', float('nan')):.2f} | {h.get('resid', float('nan')):+.2f} |"
        )
    lines += ["", "## Sample π (strand 0, GW8 first scored sides)", ""]
    for lab in ("thin_shares", "full_shares"):
        lines.append(f"### {lab}")
        lines.append("")
        for team, d in (arms[lab]["pi_diag"] or {}).items():
            tops = ", ".join(f"{t['name']} {t['pi']:.2f}" for t in d["top_pi"])
            lines.append(f"- **{team}** (n={d['n_on_dice']}): {tops}")
        lines.append("")
    lines += [
        "## Verdict",
        "",
        "Full-club shares should cut XI E[G] by spreading π onto impact subs "
        "before the thin selection gate. Selection pool unchanged.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_full_share_pool_trial --n {n_strands}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_full_share_pool_trial",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Trial full-club share pool vs thin ≥60′ shares; selection unchanged",
                "verdict": {
                    "thin": summaries["thin_shares"],
                    "full": summaries["full_shares"],
                    "delta_sum_resid": delta,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summaries, indent=2))
    print("delta_sum_resid", delta)
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
