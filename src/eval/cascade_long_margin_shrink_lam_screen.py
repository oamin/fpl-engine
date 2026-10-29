"""Pre-registered club-λ screen: Opta fixture-margin shrink vs blind.

Protocol: ``cascade_long_margin_shrink_lam_screen``.

Trials the strength-mismatch lever from the Opta−FD residual driver dig:
favourites / place rank_gap drive Opta−FD, not short-run form. Compress
each fixture's Opta λ spread toward the match mean (preserve total λ):

  m = (λ_H + λ_A) / 2
  λ'_H = m + α · (λ_H − m)
  λ'_A = m + α · (λ_A − m)

α=1 is identity (baseline blind). α<1 shrinks margins.

Arms (locked before reading holdout totals):

- **baseline** α=1.0 (blind Opta)
- **margin_shrink_fit** train-fit α on MAE(λ,act), grid 0.50..1.00 step 0.05
  on 2022-23…2024-25
- **margin_shrink_075** fixed α=0.75 diagnostic (halfway compression)

Promote rule (same as place×venue): vs blind, |away bias| ↓ ≥0.10 AND
home |bias| ↑ ≤0.05 AND MAE ↑ ≤0.02. OOS on prior closed seasons required.
Default unchanged without PI.

  PYTHONPATH=. python3 -m src.eval.cascade_long_margin_shrink_lam_screen
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
REPORT = ROOT / "reports" / "cascade_long_margin_shrink_lam_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_margin_shrink_lam_screen.json")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
HOLDOUT_SEASON = "2025-26"
OOS_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
GW_START = 8
GW_END = 38
N_DRAWS = 20
SEED = 7
ALPHA_GRID = [round(x, 2) for x in np.arange(0.50, 1.01, 0.05)]
FIXED_ALPHA = 0.75
AWAY_IMPROVE_MIN = 0.10
HOME_WORSEN_MAX = 0.05
MAE_WORSEN_MAX = 0.02


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


def _apply_margin_shrink(lam_h: float, lam_a: float, alpha: float) -> tuple[float, float]:
    a = float(alpha)
    m = 0.5 * (float(lam_h) + float(lam_a))
    return m + a * (float(lam_h) - m), m + a * (float(lam_a) - m)


def _season_fixtures(
    season: str,
    *,
    n_draws: int,
    seed0: int,
    with_fd: bool,
) -> pd.DataFrame:
    """One row per fixture with blind Opta λ_H/λ_A (+ optional FD)."""
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    enriched = _load_merged_enriched(season)
    act = _actual_team_gw(load_merged(season))
    fd = _fd_index(season) if with_fd else None
    rows: list[dict[str, Any]] = []
    for gw in range(GW_START, GW_END + 1):
        fx_list = _hist_fixtures_by_gw(enriched, start=gw, end=gw).get(gw, [])
        pmap: dict[tuple[str, str], dict[str, Any]] = {}
        if with_fd and fd is not None:
            priced = _fd_priced_by_gw(enriched, fd, [gw]).get(gw, [])
            pmap = {
                (norm_team(p["home"]), norm_team(p["away"])): p for p in priced
            }
        for fx in fx_list:
            h, a = norm_team(fx["home"]), norm_team(fx["away"])
            if h not in ranks or a not in ranks:
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
            blind_h, blind_a = float(np.mean(oh)), float(np.mean(oa))
            pr = pmap.get((h, a))
            act_h = act.loc[(act["gw"] == gw) & (act["team"] == h), "act"]
            act_a = act.loc[(act["gw"] == gw) & (act["team"] == a), "act"]
            rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "home": h,
                    "away": a,
                    "blind_h": blind_h,
                    "blind_a": blind_a,
                    "fd_h": float(pr["lam_h"]) if pr else float("nan"),
                    "fd_a": float(pr["lam_a"]) if pr else float("nan"),
                    "act_h": float(act_h.iloc[0]) if len(act_h) else float("nan"),
                    "act_a": float(act_a.iloc[0]) if len(act_a) else float("nan"),
                    "rank_h": float(ranks[h]),
                    "rank_a": float(ranks[a]),
                }
            )
    return pd.DataFrame(rows)


def _fixtures_to_sides(fx: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in fx.itertuples(index=False):
        lh, la = _apply_margin_shrink(r.blind_h, r.blind_a, alpha)
        rows.append(
            {
                "season": r.season,
                "gw": r.gw,
                "team": r.home,
                "ha": "H",
                "lam": lh,
                "fd": r.fd_h,
                "act": r.act_h,
                "blind": r.blind_h,
                "alpha": float(alpha),
            }
        )
        rows.append(
            {
                "season": r.season,
                "gw": r.gw,
                "team": r.away,
                "ha": "A",
                "lam": la,
                "fd": r.fd_a,
                "act": r.act_a,
                "blind": r.blind_a,
                "alpha": float(alpha),
            }
        )
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, *, lam_col: str = "lam") -> dict[str, float]:
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
    away_improve = abs(float(base["bias_A"])) - abs(float(cand["bias_A"]))
    home_worsen = abs(float(cand["bias_H"])) - abs(float(base["bias_H"]))
    mae_worsen = float(cand["mae"]) - float(base["mae"])
    ok = (
        away_improve >= AWAY_IMPROVE_MIN
        and home_worsen <= HOME_WORSEN_MAX
        and mae_worsen <= MAE_WORSEN_MAX
    )
    return {
        "pass": bool(ok),
        "away_improve": float(away_improve),
        "home_worsen": float(home_worsen),
        "mae_worsen": float(mae_worsen),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = _now()
    print(
        f"[margin-shrink] train={TRAIN_SEASONS} holdout={HOLDOUT_SEASON} "
        f"GW{GW_START}–{GW_END} N={args.n_draws}",
        flush=True,
    )

    tables: dict[str, pd.DataFrame] = {}
    for i, season in enumerate(OOS_SEASONS):
        print(f"[margin-shrink] build {season}", flush=True)
        tables[season] = _season_fixtures(
            season,
            n_draws=int(args.n_draws),
            seed0=int(args.seed) + 100 * i,
            with_fd=(season == HOLDOUT_SEASON),
        )

    fit_rows: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        parts = [
            _fixtures_to_sides(tables[s], alpha) for s in TRAIN_SEASONS
        ]
        train = pd.concat(parts, ignore_index=True)
        summ = _summarise(train)
        fit_rows.append({"alpha": float(alpha), **summ})
        print(
            f"  train α={alpha:.2f} bias_A={summ['bias_A']:+.3f} "
            f"bias_H={summ['bias_H']:+.3f} MAE={summ['mae']:.3f}",
            flush=True,
        )
    fit_df = pd.DataFrame(fit_rows)
    fit_alpha = float(fit_df.loc[fit_df["mae"].idxmin(), "alpha"])
    print(f"[margin-shrink] fit α={fit_alpha:.2f}", flush=True)

    arm_alphas = {
        "baseline_blind": 1.0,
        "margin_shrink_fit": fit_alpha,
        "margin_shrink_075": FIXED_ALPHA,
    }

    holdout: dict[str, Any] = {}
    hdf = tables[HOLDOUT_SEASON]
    for name, alpha in arm_alphas.items():
        sides = _fixtures_to_sides(hdf, alpha)
        opta = _summarise(sides, lam_col="lam")
        fd = _summarise(sides, lam_col="fd")
        holdout[name] = {"alpha": float(alpha), "opta": opta, "fd": fd}
        print(
            f"[margin-shrink] holdout {name} α={alpha:.2f} "
            f"H={opta['bias_H']:+.3f} A={opta['bias_A']:+.3f} MAE={opta['mae']:.3f}",
            flush=True,
        )

    base = holdout["baseline_blind"]["opta"]
    decisions = {
        name: _promote(base, holdout[name]["opta"])
        for name in holdout
        if name != "baseline_blind"
    }
    winners = [n for n, d in decisions.items() if d["pass"]]

    oos: dict[str, Any] = {}
    for season in OOS_SEASONS:
        sdf = tables[season]
        arms = {
            name: _summarise(_fixtures_to_sides(sdf, alpha))
            for name, alpha in arm_alphas.items()
        }
        base_oos = arms["baseline_blind"]
        dec = {
            name: _promote(base_oos, arms[name])
            for name in arms
            if name != "baseline_blind"
        }
        oos[season] = {
            "arms": arms,
            "decisions": dec,
            "winners": [n for n, d in dec.items() if d["pass"]],
        }
        print(
            f"[margin-shrink] OOS {season} winners={oos[season]['winners'] or ['none']}",
            flush=True,
        )

    oos_pass_counts = {
        name: sum(1 for s in oos.values() if name in s["winners"])
        for name in decisions
    }
    recommended = None
    for pref in ("margin_shrink_fit", "margin_shrink_075"):
        if pref in winners and oos_pass_counts.get(pref, 0) >= 3:
            recommended = pref
            break

    state = {
        "task": "cascade_long_margin_shrink_lam_screen",
        "protocol": "cascade_long_margin_shrink_lam_screen",
        "pre_registered": "2026-10-29",
        "formula": {
            "m": "(λ_H + λ_A) / 2",
            "lam_prime": "m + α·(λ − m) per side; match total preserved",
        },
        "lever": "strength-mismatch / Opta margin compression (not venue spice)",
        "train_seasons": list(TRAIN_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "oos_seasons": list(OOS_SEASONS),
        "gw_range": [GW_START, GW_END],
        "n_draws": int(args.n_draws),
        "fit_train": {"alpha": fit_alpha, "table": fit_rows},
        "holdout": holdout,
        "decisions": decisions,
        "winners": winners,
        "oos": oos,
        "oos_pass_counts": oos_pass_counts,
        "recommended_arm": recommended,
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# cascade_long Opta margin-shrink λ screen",
        "",
        f"Pre-registered **2026-10-29** · protocol "
        f"`cascade_long_margin_shrink_lam_screen` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Lever",
        "",
        "Strength-mismatch from Opta−FD residual drivers (not form, not "
        "venue spice). Compress fixture Opta margins toward the match mean:",
        "",
        "```",
        "m = (λ_H + λ_A) / 2",
        "λ'_H = m + α · (λ_H − m)",
        "λ'_A = m + α · (λ_A − m)",
        "```",
        "",
        f"Train-fit α (MAE on {TRAIN_SEASONS}): **{fit_alpha:.2f}**.",
        "",
        "## Holdout 2025-26",
        "",
        "| arm | α | bias | bias H | bias A | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in holdout.items():
        o = block["opta"]
        lines.append(
            f"| {name} | {block['alpha']:.2f} | {o['bias']:+.3f} | "
            f"{o['bias_H']:+.3f} | {o['bias_A']:+.3f} | {o['mae']:.3f} |"
        )
    lines += ["", "## Promote vs baseline", ""]
    for name, dec in decisions.items():
        mark = "PASS" if dec["pass"] else "FAIL"
        lines.append(
            f"- **{name}**: {mark} "
            f"(away_improve={dec['away_improve']:+.3f}, "
            f"home_worsen={dec['home_worsen']:+.3f}, "
            f"mae_worsen={dec['mae_worsen']:+.3f})"
        )
    lines += ["", "## OOS", ""]
    for season, block in oos.items():
        lines.append(f"- **{season}**: winners={block['winners'] or ['none']}")
    lines += ["", "### OOS pass counts", ""]
    for name, npass in oos_pass_counts.items():
        lines.append(f"- {name}: {npass}/{len(OOS_SEASONS)}")
    lines += [
        "",
        f"**Recommended:** {recommended or 'none — keep blind α=1'}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_margin_shrink_lam_screen",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_margin_shrink_lam_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Screen Opta margin-shrink λ vs blind (strength-mismatch lever)",
                "verdict": {
                    "fit_alpha": fit_alpha,
                    "winners": winners,
                    "oos_pass_counts": oos_pass_counts,
                    "recommended_arm": recommended,
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
    print(
        json.dumps(
            {
                "fit_alpha": fit_alpha,
                "winners": winners,
                "oos_pass_counts": oos_pass_counts,
                "recommended": recommended,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
