"""Pre-registered combined λ shrink trial (scale both λ_H and λ_A).

Protocol: ``cascade_long_lam_shrink_screen`` (diagnostic — does **not**
flip ``LAM_SHRINK=1.0``).

Formula: with ``away_shrink=1.0`` fixed,
``λ_H ← λ_H · s`` and ``λ_A ← λ_A · s`` (``s = lam_shrink``). No λ clip.

Arms locked before 2025-26 holdout totals:

- **baseline** ``s=1.0``
- **fit** argmin train MAE on 2022-23…2024-25, grid 0.80…1.00 step 0.02

Promote (holdout vs baseline): overall ``|bias|`` ↓ ≥0.05 **and** neither
home nor away ``|bias|`` rises by >0.05 **and** MAE ↑ ≤0.02.

Also report the same promote per prior season (OOS; those seasons did not
select ``s`` from XI).

  PYTHONPATH=. python3 -m src.eval.cascade_long_lam_shrink_screen
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
    LAM_SHRINK,
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_lam_shrink_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_lam_shrink_screen.json")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
HOLDOUT_SEASON = "2025-26"
OOS_SEASONS = TRAIN_SEASONS
GW_START = 8
GW_END = 38
N_DRAWS = 40
SEED = 7
GRID = [round(x, 2) for x in np.arange(0.80, 1.01, 0.02)]
OVERALL_IMPROVE_MIN = 0.05
SIDE_WORSEN_MAX = 0.05
MAE_WORSEN_MAX = 0.02


def _now() -> float:
    return time.time()


def _actual_team_goals(merged: pd.DataFrame, gw: int) -> dict[str, float]:
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
    act: dict[str, float] = {}
    if g.empty:
        return act
    for r in g.itertuples(index=False):
        th = float(r.team_h_score) if pd.notna(r.team_h_score) else float("nan")
        if not np.isfinite(th):
            continue
        tn = norm_team(str(r.team))
        ta = float(r.team_a_score) if pd.notna(r.team_a_score) else float("nan")
        act[tn] = th if bool(r.was_home) else ta
    return act


def _season_team_sides(
    season: str,
    *,
    lam_shrink: float,
    n_draws: int,
    seed0: int,
    with_fd: bool,
) -> pd.DataFrame:
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    enriched = _load_merged_enriched(season)
    merged = load_merged(season)
    fd = _fd_index(season) if with_fd else None
    paths0 = init_paths_from_opta(ranks, profile=profile)
    rows: list[dict[str, Any]] = []
    for gw in range(GW_START, GW_END + 1):
        fx_list = _hist_fixtures_by_gw(enriched, start=gw, end=gw).get(gw, [])
        if not fx_list:
            continue
        act = _actual_team_goals(merged, gw)
        pmap: dict[tuple[str, str], dict[str, Any]] = {}
        if with_fd and fd is not None:
            priced = _fd_priced_by_gw(enriched, fd, [gw]).get(gw, [])
            pmap = {
                (norm_team(p["home"]), norm_team(p["away"])): p for p in priced
            }
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
                    lam_shrink=float(lam_shrink),
                )
                oh.append(float(sc["lam_h"]))
                oa.append(float(sc["lam_a"]))
            pr = pmap.get((h, a))
            rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "team": h,
                    "ha": "H",
                    "opta": float(np.mean(oh)),
                    "fd": float(pr["lam_h"]) if pr else float("nan"),
                    "act": float(act[h]) if h in act else float("nan"),
                    "lam_shrink": float(lam_shrink),
                }
            )
            rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "team": a,
                    "ha": "A",
                    "opta": float(np.mean(oa)),
                    "fd": float(pr["lam_a"]) if pr else float("nan"),
                    "act": float(act[a]) if a in act else float("nan"),
                    "lam_shrink": float(lam_shrink),
                }
            )
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, *, lam_col: str = "opta") -> dict[str, float]:
    s = df.dropna(subset=["act", lam_col]).copy()
    if s.empty:
        return {
            "n": 0.0,
            "bias": float("nan"),
            "bias_H": float("nan"),
            "bias_A": float("nan"),
            "mae": float("nan"),
        }
    err = s[lam_col] - s["act"]
    out = {
        "n": float(len(s)),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "bias_H": float("nan"),
        "bias_A": float("nan"),
    }
    for ha, key in (("H", "bias_H"), ("A", "bias_A")):
        sub = err.loc[s["ha"] == ha]
        if len(sub):
            out[key] = float(sub.mean())
    return out


def _promote(base: dict[str, float], cand: dict[str, float]) -> dict[str, Any]:
    overall_improve = abs(float(base["bias"])) - abs(float(cand["bias"]))
    home_worsen = abs(float(cand["bias_H"])) - abs(float(base["bias_H"]))
    away_worsen = abs(float(cand["bias_A"])) - abs(float(base["bias_A"]))
    mae_worsen = float(cand["mae"]) - float(base["mae"])
    ok = (
        overall_improve >= OVERALL_IMPROVE_MIN
        and home_worsen <= SIDE_WORSEN_MAX
        and away_worsen <= SIDE_WORSEN_MAX
        and mae_worsen <= MAE_WORSEN_MAX
    )
    return {
        "pass": bool(ok),
        "overall_improve": float(overall_improve),
        "home_worsen": float(home_worsen),
        "away_worsen": float(away_worsen),
        "mae_worsen": float(mae_worsen),
        "thresholds": {
            "overall_improve_min": OVERALL_IMPROVE_MIN,
            "side_worsen_max": SIDE_WORSEN_MAX,
            "mae_worsen_max": MAE_WORSEN_MAX,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    n_draws = int(args.n_draws)
    seed0 = int(args.seed)
    t0 = _now()

    print(
        f"[lam-shrink] train={TRAIN_SEASONS} holdout={HOLDOUT_SEASON} "
        f"GW{GW_START}–{GW_END} N={n_draws} grid={GRID[0]}…{GRID[-1]}",
        flush=True,
    )

    fit_rows: list[dict[str, Any]] = []
    for s in GRID:
        print(f"[lam-shrink] train scan s={s:.2f}", flush=True)
        parts = [
            _season_team_sides(
                season,
                lam_shrink=s,
                n_draws=n_draws,
                seed0=seed0 + 100 * i,
                with_fd=False,
            )
            for i, season in enumerate(TRAIN_SEASONS)
        ]
        train = pd.concat(parts, ignore_index=True)
        summ = _summarise(train)
        fit_rows.append({"lam_shrink": float(s), **summ})
        print(
            f"  bias={summ['bias']:+.3f} bias_A={summ['bias_A']:+.3f} "
            f"bias_H={summ['bias_H']:+.3f} MAE={summ['mae']:.3f}",
            flush=True,
        )
    fit_df = pd.DataFrame(fit_rows)
    fit_s = float(fit_df.loc[fit_df["mae"].idxmin(), "lam_shrink"])
    print(f"[lam-shrink] fit s={fit_s:.2f} (min train MAE)", flush=True)

    arms = {"baseline_1_00": float(LAM_SHRINK), "fit_train": fit_s}
    holdout: dict[str, Any] = {}
    for name, s in arms.items():
        print(f"[lam-shrink] holdout {HOLDOUT_SEASON} {name} s={s:.2f}", flush=True)
        df = _season_team_sides(
            HOLDOUT_SEASON,
            lam_shrink=s,
            n_draws=n_draws,
            seed0=seed0 + 999,
            with_fd=True,
        )
        opta = _summarise(df, lam_col="opta")
        fd = _summarise(df, lam_col="fd")
        holdout[name] = {
            "lam_shrink": float(s),
            "opta": opta,
            "fd": fd,
            "n_team_sides": int(opta["n"]),
        }
        print(
            f"  opta bias={opta['bias']:+.3f} A={opta['bias_A']:+.3f} "
            f"H={opta['bias_H']:+.3f} MAE={opta['mae']:.3f}",
            flush=True,
        )

    base = holdout["baseline_1_00"]["opta"]
    holdout_dec = {
        name: _promote(base, holdout[name]["opta"])
        for name in holdout
        if name != "baseline_1_00"
    }

    # OOS prior seasons at fit s vs s=1.0
    oos: dict[str, Any] = {}
    for i, season in enumerate(OOS_SEASONS):
        print(f"[lam-shrink] OOS {season}", flush=True)
        bdf = _season_team_sides(
            season,
            lam_shrink=1.0,
            n_draws=n_draws,
            seed0=seed0 + 2000 + i,
            with_fd=False,
        )
        cdf = _season_team_sides(
            season,
            lam_shrink=fit_s,
            n_draws=n_draws,
            seed0=seed0 + 3000 + i,
            with_fd=False,
        )
        bo, co = _summarise(bdf), _summarise(cdf)
        dec = _promote(bo, co)
        oos[season] = {
            "baseline": bo,
            "candidate": co,
            "decision": dec,
            "lam_shrink": fit_s,
        }
        print(
            f"  promote={'PASS' if dec['pass'] else 'FAIL'} "
            f"bias {bo['bias']:+.3f}→{co['bias']:+.3f}",
            flush=True,
        )
    n_pass = sum(1 for b in oos.values() if b["decision"]["pass"])

    state = {
        "task": "cascade_long_lam_shrink_screen",
        "protocol": "cascade_long_lam_shrink_screen",
        "pre_registered": "2026-10-27",
        "formula": "lam_H *= s; lam_A *= s; away_shrink=1.0; no lam clip",
        "train_seasons": list(TRAIN_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "gw_range": [GW_START, GW_END],
        "n_draws": n_draws,
        "seed": seed0,
        "grid": GRID,
        "fit_train": {"lam_shrink": fit_s, "table": fit_rows},
        "holdout": holdout,
        "holdout_decisions": holdout_dec,
        "oos": oos,
        "oos_n_pass": n_pass,
        "default_unchanged": True,
        "default_lam_shrink": float(LAM_SHRINK),
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Combined λ shrink screen (λ_H and λ_A × s)",
        "",
        f"Pre-registered **2026-10-27** · wall {state['elapsed_s']:.1f}s",
        "",
        "## Formula",
        "",
        "- `λ_H ← λ_H · s`, `λ_A ← λ_A · s` (`away_shrink=1.0`).",
        "- No 0.20–3.50 λ clip.",
        f"- Default `LAM_SHRINK={LAM_SHRINK}` **not** flipped here.",
        "",
        f"Train fit → **s={fit_s:.2f}**. Holdout **{HOLDOUT_SEASON}** GW"
        f"{GW_START}–{GW_END}.",
        f"Promote: overall |bias| ↓ ≥{OVERALL_IMPROVE_MIN}, side |bias| ↑ ≤"
        f"{SIDE_WORSEN_MAX}, MAE ↑ ≤{MAE_WORSEN_MAX}.",
        "",
        "## Holdout 2025-26",
        "",
        "| arm | s | bias | bias H | bias A | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in holdout.items():
        o = block["opta"]
        lines.append(
            f"| {name} | {block['lam_shrink']:.2f} | {o['bias']:+.3f} | "
            f"{o['bias_H']:+.3f} | {o['bias_A']:+.3f} | {o['mae']:.3f} |"
        )
    lines += ["", "## Holdout promote", ""]
    for name, dec in holdout_dec.items():
        lines.append(
            f"- **{name}**: {'PASS' if dec['pass'] else 'FAIL'} "
            f"(overall_improve={dec['overall_improve']:+.3f}, "
            f"home_worsen={dec['home_worsen']:+.3f}, "
            f"away_worsen={dec['away_worsen']:+.3f}, "
            f"mae_worsen={dec['mae_worsen']:+.3f})"
        )
    lines += [
        "",
        f"## OOS prior seasons at fit s={fit_s:.2f}",
        "",
        "| season | s=1 bias | s=fit bias | bias H Δ|bias| | bias A Δ|bias| | promote |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for season, block in oos.items():
        b, c, d = block["baseline"], block["candidate"], block["decision"]
        lines.append(
            f"| {season} | {b['bias']:+.3f} | {c['bias']:+.3f} | "
            f"{d['home_worsen']:+.3f} | {d['away_worsen']:+.3f} | "
            f"{'PASS' if d['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**OOS PASS count:** {n_pass}/{len(OOS_SEASONS)}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_lam_shrink_screen",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    winners = [n for n, d in holdout_dec.items() if d["pass"]]
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_lam_shrink_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Trial combined lam_shrink on both λ_H and λ_A",
                "verdict": {
                    "fit_lam_shrink": fit_s,
                    "holdout_winners": winners,
                    "oos_n_pass": n_pass,
                    "default_unchanged": True,
                    "default_lam_shrink": float(LAM_SHRINK),
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fit": fit_s,
                "holdout_winners": winners,
                "oos_n_pass": n_pass,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
