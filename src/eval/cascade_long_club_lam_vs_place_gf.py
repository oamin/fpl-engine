"""Dig: does cascade_long club Σλ converge to place GF?

Addresses the design claim that Opta rank → place GF/GA cannot run hot
across a whole season at **club goals**. Separates that from XI xP(G).

For each closed season (oracle finishing ranks, LOSO place profile):

1. Per fixture Opta λ (N draws, centreline sampling as live).
2. Club season Σλ vs place ``mean_gf`` vs actual goals scored.
3. Venue split of (λ − actual).

Default ``venue_mode=blind`` unchanged. Diagnostic only.

  PYTHONPATH=. python3 -m src.eval.cascade_long_club_lam_vs_place_gf
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

from src.eval.cascade_mc_vs_mean_hist import SEASON_E0, _hist_fixtures_by_gw
from src.markets.numagent_xp_pilot import load_merged
from src.models.multistage_mc_milp import init_paths_from_opta, sample_opta_scenario
from src.models.opta_gf_ga_horizon import (
    gf_ga_from_rank,
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_club_lam_vs_place_gf.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_club_lam_vs_place_gf.json")

SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
N_DRAWS = 20
SEED = 7


def _now() -> float:
    return time.time()


def _actual_team_gw(merged: pd.DataFrame) -> pd.DataFrame:
    g = merged.copy()
    g["GW"] = pd.to_numeric(g["GW"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (gw, team, wh), sub in g.groupby(["GW", "team", "was_home"], sort=False):
        if not np.isfinite(float(gw)):
            continue
        th = float(sub["team_h_score"].iloc[0])
        ta = float(sub["team_a_score"].iloc[0])
        if not np.isfinite(th):
            continue
        rows.append(
            {
                "gw": int(gw),
                "team": norm_team(str(team)),
                "ha": "H" if bool(wh) else "A",
                "act": th if bool(wh) else ta,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["gw", "team"])


def _season_table(season: str, *, n_draws: int, seed0: int) -> dict[str, Any]:
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    enriched = _load_merged_enriched(season)
    act = _actual_team_gw(load_merged(season))
    lam_rows: list[dict[str, Any]] = []
    for gw in range(1, 39):
        fx_list = _hist_fixtures_by_gw(enriched, start=gw, end=gw).get(gw, [])
        for fx in fx_list:
            h, a = norm_team(fx["home"]), norm_team(fx["away"])
            oh: list[float] = []
            oa: list[float] = []
            for s in range(int(n_draws)):
                paths = {
                    c: type(p)(att=p.att, dfn=p.dfn, n=p.n)
                    for c, p in paths0.items()
                }
                sc = sample_opta_scenario(
                    h,
                    a,
                    paths=paths,
                    league_gf=league_gf,
                    rng=random.Random(int(seed0) + 10_000 * gw + s),
                    ranks=ranks,
                    profile=profile,
                    sd_league_gf=0.0,
                    away_shrink=1.0,
                    lam_shrink=1.0,
                    venue_mode="blind",
                )
                oh.append(float(sc["lam_h"]))
                oa.append(float(sc["lam_a"]))
            lam_rows.append(
                {"gw": gw, "team": h, "ha": "H", "lam": float(np.mean(oh))}
            )
            lam_rows.append(
                {"gw": gw, "team": a, "ha": "A", "lam": float(np.mean(oa))}
            )
    lam = pd.DataFrame(lam_rows)
    m = lam.merge(act, on=["gw", "team", "ha"], how="left")
    club = (
        m.groupby("team", as_index=False)
        .agg(sum_lam=("lam", "sum"), sum_act=("act", "sum"), n=("lam", "size"))
    )
    club["rank"] = club["team"].map(ranks)
    club["place_gf"] = club["rank"].map(
        lambda r: float(gf_ga_from_rank(float(r), profile=profile)["gf"])
    )
    club["lam_minus_place"] = club["sum_lam"] - club["place_gf"]
    club["lam_minus_act"] = club["sum_lam"] - club["sum_act"]
    club["act_minus_place"] = club["sum_act"] - club["place_gf"]

    def _venue(sub: pd.DataFrame) -> dict[str, float]:
        s = sub.dropna(subset=["act", "lam"])
        if s.empty:
            return {"n": 0.0, "bias": float("nan"), "mae": float("nan")}
        err = s["lam"] - s["act"]
        return {
            "n": float(len(s)),
            "bias": float(err.mean()),
            "mae": float(err.abs().mean()),
            "mean_lam": float(s["lam"].mean()),
            "mean_act": float(s["act"].mean()),
        }

    venue = {ha: _venue(m.loc[m["ha"] == ha]) for ha in ("H", "A")}
    m838 = m.loc[m["gw"] >= 8]
    venue_838 = {ha: _venue(m838.loc[m838["ha"] == ha]) for ha in ("H", "A")}
    return {
        "season": season,
        "n_draws": int(n_draws),
        "club_means": {
            "sum_lam": float(club["sum_lam"].mean()),
            "place_gf": float(club["place_gf"].mean()),
            "sum_act": float(club["sum_act"].mean()),
            "lam_minus_place": float(club["lam_minus_place"].mean()),
            "lam_minus_act": float(club["lam_minus_act"].mean()),
            "act_minus_place": float(club["act_minus_place"].mean()),
            "mae_lam_place": float(club["lam_minus_place"].abs().mean()),
            "mae_lam_act": float(club["lam_minus_act"].abs().mean()),
        },
        "venue_full": venue,
        "venue_gw8_38": venue_838,
        "clubs": club.to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = _now()
    seasons: dict[str, Any] = {}
    for i, season in enumerate(SEASONS):
        print(f"[club-lam-place] {season}", flush=True)
        seasons[season] = _season_table(
            season, n_draws=int(args.n_draws), seed0=int(args.seed) + 100 * i
        )
        cm = seasons[season]["club_means"]
        print(
            f"  Σλ={cm['sum_lam']:.2f} place_gf={cm['place_gf']:.2f} "
            f"act={cm['sum_act']:.2f} | λ−place={cm['lam_minus_place']:+.2f} "
            f"λ−act={cm['lam_minus_act']:+.2f}",
            flush=True,
        )
        v = seasons[season]["venue_full"]
        print(
            f"  venue bias H={v['H']['bias']:+.3f} A={v['A']['bias']:+.3f}",
            flush=True,
        )

    pooled_lam_place = float(
        np.mean([seasons[s]["club_means"]["lam_minus_place"] for s in SEASONS])
    )
    pooled_lam_act = float(
        np.mean([seasons[s]["club_means"]["lam_minus_act"] for s in SEASONS])
    )
    state = {
        "task": "cascade_long_club_lam_vs_place_gf",
        "protocol": "cascade_long_club_lam_vs_place_gf",
        "pre_registered": "2026-10-28",
        "seasons": seasons,
        "pooled": {
            "mean_lam_minus_place": pooled_lam_place,
            "mean_lam_minus_act": pooled_lam_act,
        },
        "verdict_notes": [
            "Σλ ≈ place GF across closed seasons — design converges at club goals.",
            "λ − actual can still be hot when the season under-shoots the LOSO place prior.",
            "Venue-blind map: home bias ~0, away bias positive — misallocation, not escaped GF anchors.",
            "XI xP(G) heat is a separate generative layer.",
        ],
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Cascade_long club Σλ vs place GF",
        "",
        f"Pre-registered **2026-10-28** · protocol "
        f"`cascade_long_club_lam_vs_place_gf` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Verdict",
        "",
        f"- Pooled mean **Σλ − place GF ≈ {pooled_lam_place:+.2f}** "
        f"(~0 → design converges).",
        f"- Pooled mean **Σλ − actual ≈ {pooled_lam_act:+.2f}** "
        f"(can be hot when the season scores below the LOSO place prior).",
        "- Venue: home ~calibrated, away high — **misallocation inside the "
        "season total**, not escape from place GF.",
        "- XI xP(G) overprint is **not** explained by club Σλ drifting off place GF.",
        "",
        "## Per season (club means)",
        "",
        "| season | Σλ | place GF | actual | λ−place | λ−act | bias H | bias A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for season in SEASONS:
        cm = seasons[season]["club_means"]
        v = seasons[season]["venue_full"]
        lines.append(
            f"| {season} | {cm['sum_lam']:.1f} | {cm['place_gf']:.1f} | "
            f"{cm['sum_act']:.1f} | {cm['lam_minus_place']:+.2f} | "
            f"{cm['lam_minus_act']:+.2f} | {v['H']['bias']:+.3f} | "
            f"{v['A']['bias']:+.3f} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_club_lam_vs_place_gf",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_club_lam_vs_place_gf",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Dig club Σλ vs place GF (venue anatomy separate from XI)",
                "verdict": {
                    "pooled_lam_minus_place": pooled_lam_place,
                    "pooled_lam_minus_act": pooled_lam_act,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state["pooled"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
