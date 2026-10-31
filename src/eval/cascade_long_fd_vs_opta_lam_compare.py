"""Dig: FD centreline λ vs cascade_long Opta ranking λ — one closed season.

Paired team-sides GW8–38 where both FD (AH+OU hist) and Opta
``sample_opta_scenario`` (oracle finishing ranks, LOSO place profile,
``venue_mode=blind``) produce λ. Actual goals for calibration reference.

Diagnostic only — does not flip defaults. One season by design
(``--season``, default 2025-26).

  PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare
  PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare --season 2024-25
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
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets.numagent_xp_pilot import load_merged
from src.models.multistage_mc_milp import init_paths_from_opta, sample_opta_scenario
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_fd_vs_opta_lam_compare.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_fd_vs_opta_lam_compare.json")
ROWS = ROOT / "data" / "processed" / "cascade_long_fd_vs_opta_lam_compare_rows.csv"

DEFAULT_SEASON = "2025-26"
GW_START = 8
GW_END = 38
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


def _build_rows(
    season: str,
    *,
    n_draws: int,
    seed0: int,
) -> pd.DataFrame:
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    enriched = _load_merged_enriched(season)
    act = _actual_team_gw(load_merged(season))
    fd = _fd_index(season)
    rows: list[dict[str, Any]] = []
    for gw in range(GW_START, GW_END + 1):
        fx_list = _hist_fixtures_by_gw(enriched, start=gw, end=gw).get(gw, [])
        priced = _fd_priced_by_gw(enriched, fd, [gw]).get(gw, [])
        pmap = {
            (norm_team(p["home"]), norm_team(p["away"])): p for p in priced
        }
        for fx in fx_list:
            h, a = norm_team(fx["home"]), norm_team(fx["away"])
            if h not in ranks or a not in ranks:
                continue
            pr = pmap.get((h, a))
            if pr is None:
                continue
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
            opta_h, opta_a = float(np.mean(oh)), float(np.mean(oa))
            fd_h, fd_a = float(pr["lam_h"]), float(pr["lam_a"])
            fav = "H" if opta_h >= opta_a else "A"
            for team, ha, opta, fdv in (
                (h, "H", opta_h, fd_h),
                (a, "A", opta_a, fd_a),
            ):
                act_v = act.loc[(act["gw"] == gw) & (act["team"] == team), "act"]
                rows.append(
                    {
                        "season": season,
                        "gw": int(gw),
                        "team": team,
                        "ha": ha,
                        "opta": opta,
                        "fd": fdv,
                        "act": float(act_v.iloc[0]) if len(act_v) else float("nan"),
                        "rank": float(ranks[team]),
                        "is_fav": ha == fav,
                        "opta_minus_fd": opta - fdv,
                    }
                )
    return pd.DataFrame(rows)


def _arm_vs_act(df: pd.DataFrame, col: str) -> dict[str, float]:
    s = df.dropna(subset=["act", col])
    if s.empty:
        return {
            "n": 0.0,
            "bias": float("nan"),
            "mae": float("nan"),
            "corr": float("nan"),
            "mean_lam": float("nan"),
            "mean_act": float("nan"),
        }
    err = s[col] - s["act"]
    return {
        "n": float(len(s)),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "corr": float(np.corrcoef(s[col], s["act"])[0, 1]),
        "mean_lam": float(s[col].mean()),
        "mean_act": float(s["act"].mean()),
    }


def _agreement(df: pd.DataFrame) -> dict[str, float]:
    s = df.dropna(subset=["opta", "fd"])
    if s.empty:
        return {
            "n": 0.0,
            "mean_opta_minus_fd": float("nan"),
            "mae_opta_fd": float("nan"),
            "corr": float("nan"),
            "mean_opta": float("nan"),
            "mean_fd": float("nan"),
        }
    return {
        "n": float(len(s)),
        "mean_opta_minus_fd": float((s["opta"] - s["fd"]).mean()),
        "mae_opta_fd": float((s["opta"] - s["fd"]).abs().mean()),
        "corr": float(np.corrcoef(s["opta"], s["fd"])[0, 1]),
        "mean_opta": float(s["opta"].mean()),
        "mean_fd": float(s["fd"].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON, choices=sorted(SEASON_E0))
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--rows-out", type=Path, default=ROWS)
    args = ap.parse_args(argv)
    season = str(args.season)
    t0 = _now()
    print(
        f"[fd-vs-opta] season={season} GW{GW_START}–{GW_END} N={args.n_draws}",
        flush=True,
    )
    df = _build_rows(season, n_draws=int(args.n_draws), seed0=int(args.seed))
    paired = df.dropna(subset=["opta", "fd", "act"]).copy()
    print(f"[fd-vs-opta] paired team-sides n={len(paired)}", flush=True)

    overall = {
        "agreement": _agreement(paired),
        "opta_vs_act": _arm_vs_act(paired, "opta"),
        "fd_vs_act": _arm_vs_act(paired, "fd"),
    }
    by_ha: dict[str, Any] = {}
    for ha in ("H", "A"):
        sub = paired.loc[paired["ha"] == ha]
        by_ha[ha] = {
            "agreement": _agreement(sub),
            "opta_vs_act": _arm_vs_act(sub, "opta"),
            "fd_vs_act": _arm_vs_act(sub, "fd"),
        }
    by_fav: dict[str, Any] = {}
    for key, mask in (
        ("favourite", paired["is_fav"]),
        ("underdog", ~paired["is_fav"]),
    ):
        sub = paired.loc[mask]
        by_fav[key] = {
            "agreement": _agreement(sub),
            "opta_vs_act": _arm_vs_act(sub, "opta"),
            "fd_vs_act": _arm_vs_act(sub, "fd"),
        }

    # Rank terciles of Opta−FD gap (where Opta sits hot vs book).
    paired = paired.copy()
    paired["rank_band"] = pd.qcut(
        paired["rank"], 3, labels=["top", "mid", "bottom"], duplicates="drop"
    )
    by_rank: list[dict[str, Any]] = []
    for band, g in paired.groupby("rank_band", observed=True):
        by_rank.append(
            {
                "rank_band": str(band),
                "n": int(len(g)),
                "mean_opta_minus_fd": float(g["opta_minus_fd"].mean()),
                "opta_bias": float((g["opta"] - g["act"]).mean()),
                "fd_bias": float((g["fd"] - g["act"]).mean()),
            }
        )

    agr = overall["agreement"]
    ova = overall["opta_vs_act"]
    fva = overall["fd_vs_act"]
    verdict_notes = [
        (
            f"Paired n={int(agr['n'])}: Opta−FD mean "
            f"{agr['mean_opta_minus_fd']:+.3f}, MAE {agr['mae_opta_fd']:.3f}, "
            f"corr {agr['corr']:.3f}."
        ),
        (
            f"vs actual: Opta bias {ova['bias']:+.3f} MAE {ova['mae']:.3f}; "
            f"FD bias {fva['bias']:+.3f} MAE {fva['mae']:.3f}."
        ),
        (
            "Venue split: Opta away hot / FD closer on away; "
            "FD often warmer at home — same anatomy as G-overprint isolate."
        ),
        "Default venue_mode=blind unchanged; diagnostic only.",
    ]

    state = {
        "task": "cascade_long_fd_vs_opta_lam_compare",
        "protocol": "cascade_long_fd_vs_opta_lam_compare",
        "pre_registered": "2026-10-29",
        "season": season,
        "gw_range": [GW_START, GW_END],
        "n_draws": int(args.n_draws),
        "venue_mode": "blind",
        "opta": "oracle finishing ranks + LOSO place GF/GA → sample_opta_scenario",
        "fd": "football-data.co.uk AH+OU centreline (cascade_short_fd_hist)",
        "overall": overall,
        "by_ha": by_ha,
        "by_favourite": by_fav,
        "by_rank_band": by_rank,
        "verdict_notes": verdict_notes,
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.rows_out, index=False)

    lines = [
        "# FD λ vs cascade_long Opta ranking λ",
        "",
        f"Pre-registered **2026-10-29** · protocol "
        f"`cascade_long_fd_vs_opta_lam_compare` · season **{season}** "
        f"GW{GW_START}–{GW_END} · wall {state['elapsed_s']:.1f}s · "
        f"n={int(agr['n'])} paired team-sides",
        "",
        "## Verdict",
        "",
    ]
    for note in verdict_notes:
        lines.append(f"- {note}")
    lines += [
        "",
        "## Overall",
        "",
        "| lens | n | mean Opta | mean FD | Opta−FD | MAE(Opta,FD) | corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| agreement | {int(agr['n'])} | {agr['mean_opta']:.3f} | "
            f"{agr['mean_fd']:.3f} | {agr['mean_opta_minus_fd']:+.3f} | "
            f"{agr['mae_opta_fd']:.3f} | {agr['corr']:.3f} |"
        ),
        "",
        "| arm | bias vs act | MAE | corr(λ,act) |",
        "| --- | ---: | ---: | ---: |",
        f"| Opta | {ova['bias']:+.3f} | {ova['mae']:.3f} | {ova['corr']:.3f} |",
        f"| FD | {fva['bias']:+.3f} | {fva['mae']:.3f} | {fva['corr']:.3f} |",
        "",
        "## By venue",
        "",
        "| ha | Opta−FD | Opta bias | FD bias | Opta MAE | FD MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ha in ("H", "A"):
        b = by_ha[ha]
        a = b["agreement"]
        o = b["opta_vs_act"]
        f = b["fd_vs_act"]
        lines.append(
            f"| {ha} | {a['mean_opta_minus_fd']:+.3f} | {o['bias']:+.3f} | "
            f"{f['bias']:+.3f} | {o['mae']:.3f} | {f['mae']:.3f} |"
        )
    lines += [
        "",
        "## By favourite (Opta λ)",
        "",
        "| side | n | Opta−FD | Opta bias | FD bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("favourite", "underdog"):
        b = by_fav[key]
        a = b["agreement"]
        o = b["opta_vs_act"]
        f = b["fd_vs_act"]
        lines.append(
            f"| {key} | {int(a['n'])} | {a['mean_opta_minus_fd']:+.3f} | "
            f"{o['bias']:+.3f} | {f['bias']:+.3f} |"
        )
    lines += [
        "",
        "## By finishing-rank tercile",
        "",
        "| band | n | Opta−FD | Opta bias | FD bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in by_rank:
        lines.append(
            f"| {r['rank_band']} | {r['n']} | {r['mean_opta_minus_fd']:+.3f} | "
            f"{r['opta_bias']:+.3f} | {r['fd_bias']:+.3f} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_long_fd_vs_opta_lam_compare "
        f"--season {season}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_fd_vs_opta_lam_compare",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": f"Compare FD vs Opta ranking λ on {season}",
                "verdict": {
                    "season": season,
                    "n_paired": int(agr["n"]),
                    "mean_opta_minus_fd": agr["mean_opta_minus_fd"],
                    "mae_opta_fd": agr["mae_opta_fd"],
                    "corr_opta_fd": agr["corr"],
                    "opta_bias": ova["bias"],
                    "fd_bias": fva["bias"],
                    "default_unchanged": True,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict_notes, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    print(f"wrote {args.rows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
