"""Pre-registered club-λ place×venue screen for cascade_long.

Protocol: ``cascade_long_place_venue_lam_screen`` (diagnostic — does **not**
flip ``DEFAULT_VENUE_MODE=blind``).

Arms (locked before reading 2025-26 holdout totals):

- **baseline** ``venue_mode=blind`` (venue-blind season-total place GF/GA)
- **venue_away** λ_H blind; λ_A = place×venue away attack × blind home defence
- **venue** EV-matched full place×venue (both sides)
- **naive** isolate diagnostic (blended ḡ denom; overshoots home)
- **blend_fit** train-fit weight on ``venue_away`` λ_A
  (grid 0.00…1.00 step 0.10 on 2022-23…2024-25)

Holdout: 2025-26 GW8–38 team-sides. OOS: same promote rule on
2022-23…2024-25 (required before any default flip).

Promote rule (vs blind): ``|away bias|`` ↓ ≥0.10 **and** home ``|bias|``
does not rise by >0.05 **and** MAE does not rise by >0.02.
PI re-approval still required before changing
``protocol.cascade_long_match_outcome``.

  PYTHONPATH=. python3 -m src.eval.cascade_long_place_venue_lam_screen
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
    DEFAULT_VENUE_MODE,
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    rank_venue_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_place_venue_lam_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_place_venue_lam_screen.json")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
HOLDOUT_SEASON = "2025-26"
OOS_SEASONS = ("2022-23", "2023-24", "2024-25")
GW_START = 8
GW_END = 38
N_DRAWS = 40
SEED = 7
BLEND_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.10)]
AWAY_IMPROVE_MIN = 0.10
HOME_WORSEN_MAX = 0.05
MAE_WORSEN_MAX = 0.02


def _now() -> float:
    return time.time()


def _actual_team_goals(merged: pd.DataFrame, gw: int) -> dict[str, float]:
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
    act: dict[str, float] = {}
    for r in g.itertuples(index=False):
        th = float(r.team_h_score) if pd.notna(r.team_h_score) else float("nan")
        ta = float(r.team_a_score) if pd.notna(r.team_a_score) else float("nan")
        if not np.isfinite(th):
            continue
        tn = norm_team(str(r.team))
        act[tn] = th if bool(r.was_home) else ta
    return act


def _season_team_sides(
    season: str,
    *,
    venue_mode: str,
    venue_blend: float,
    n_draws: int,
    seed0: int,
    with_fd: bool,
) -> pd.DataFrame:
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    venue_profile = rank_venue_gf_ga_profile_excluding(e0)
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
                    lam_shrink=1.0,
                    venue_mode=str(venue_mode),
                    venue_blend=float(venue_blend),
                    venue_profile=venue_profile,
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
                    "venue_mode": str(venue_mode),
                    "venue_blend": float(venue_blend),
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
                    "venue_mode": str(venue_mode),
                    "venue_blend": float(venue_blend),
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
        "thresholds": {
            "away_improve_min": AWAY_IMPROVE_MIN,
            "home_worsen_max": HOME_WORSEN_MAX,
            "mae_worsen_max": MAE_WORSEN_MAX,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--skip-oos",
        action="store_true",
        help="Skip OOS seasons (faster smoke).",
    )
    args = ap.parse_args(argv)
    n_draws = int(args.n_draws)
    seed0 = int(args.seed)
    t0 = _now()

    print(
        f"[place-venue-lam] train={TRAIN_SEASONS} holdout={HOLDOUT_SEASON} "
        f"GW{GW_START}–{GW_END} N={n_draws} blend_grid={BLEND_GRID[0]}…{BLEND_GRID[-1]}",
        flush=True,
    )

    # Train-fit blend weight on venue_away.
    fit_rows: list[dict[str, Any]] = []
    for w in BLEND_GRID:
        print(f"[place-venue-lam] train scan blend w={w:.2f}", flush=True)
        parts = [
            _season_team_sides(
                season,
                venue_mode="blend",
                venue_blend=float(w),
                n_draws=n_draws,
                seed0=seed0 + 100 * i,
                with_fd=False,
            )
            for i, season in enumerate(TRAIN_SEASONS)
        ]
        train = pd.concat(parts, ignore_index=True)
        summ = _summarise(train)
        fit_rows.append({"venue_blend": float(w), **summ})
        print(
            f"  bias_A={summ['bias_A']:+.3f} bias_H={summ['bias_H']:+.3f} "
            f"MAE={summ['mae']:.3f}",
            flush=True,
        )
    fit_df = pd.DataFrame(fit_rows)
    fit_w = float(fit_df.loc[fit_df["mae"].idxmin(), "venue_blend"])
    print(f"[place-venue-lam] fit blend w={fit_w:.2f} (min train MAE)", flush=True)

    arms: dict[str, dict[str, Any]] = {
        "baseline_blind": {"venue_mode": "blind", "venue_blend": 0.0},
        "venue_away": {"venue_mode": "venue_away", "venue_blend": 1.0},
        "venue_ev": {"venue_mode": "venue", "venue_blend": 1.0},
        "naive": {"venue_mode": "naive", "venue_blend": 1.0},
        "blend_fit": {"venue_mode": "blend", "venue_blend": fit_w},
    }

    holdout: dict[str, Any] = {}
    for name, cfg in arms.items():
        print(
            f"[place-venue-lam] holdout {HOLDOUT_SEASON} {name} "
            f"mode={cfg['venue_mode']} w={cfg['venue_blend']:.2f}",
            flush=True,
        )
        df = _season_team_sides(
            HOLDOUT_SEASON,
            venue_mode=str(cfg["venue_mode"]),
            venue_blend=float(cfg["venue_blend"]),
            n_draws=n_draws,
            seed0=seed0 + 999,
            with_fd=True,
        )
        opta = _summarise(df, lam_col="opta")
        fd = _summarise(df, lam_col="fd")
        holdout[name] = {
            **cfg,
            "opta": opta,
            "fd": fd,
            "n_team_sides": int(opta["n"]),
        }
        print(
            f"  opta bias={opta['bias']:+.3f} H={opta['bias_H']:+.3f} "
            f"A={opta['bias_A']:+.3f} MAE={opta['mae']:.3f}",
            flush=True,
        )

    base = holdout["baseline_blind"]["opta"]
    decisions = {
        name: _promote(base, holdout[name]["opta"])
        for name in holdout
        if name != "baseline_blind"
    }
    winners = [n for n, d in decisions.items() if d["pass"]]
    recommended = None
    for pref in ("venue_away", "blend_fit", "venue_ev"):
        if pref in winners:
            recommended = pref
            break

    oos: dict[str, Any] = {}
    if not args.skip_oos:
        for season in OOS_SEASONS:
            print(f"[place-venue-lam] OOS {season}", flush=True)
            season_arms: dict[str, Any] = {}
            for name, cfg in arms.items():
                df = _season_team_sides(
                    season,
                    venue_mode=str(cfg["venue_mode"]),
                    venue_blend=float(cfg["venue_blend"]),
                    n_draws=n_draws,
                    seed0=seed0 + hash(season) % 10_000,
                    with_fd=False,
                )
                season_arms[name] = {
                    **cfg,
                    "opta": _summarise(df),
                }
            base_oos = season_arms["baseline_blind"]["opta"]
            dec = {
                name: _promote(base_oos, season_arms[name]["opta"])
                for name in season_arms
                if name != "baseline_blind"
            }
            oos[season] = {
                "arms": {
                    n: {"opta": season_arms[n]["opta"], **{k: season_arms[n][k] for k in ("venue_mode", "venue_blend")}}
                    for n in season_arms
                },
                "decisions": dec,
                "winners": [n for n, d in dec.items() if d["pass"]],
            }
            print(
                f"  winners={oos[season]['winners'] or ['none']}",
                flush=True,
            )

    oos_pass_counts = {
        name: sum(1 for s in oos.values() if name in s["winners"])
        for name in decisions
    }

    state = {
        "task": "cascade_long_place_venue_lam_screen",
        "protocol": "cascade_long_place_venue_lam_screen",
        "pre_registered": "2026-10-27",
        "formula": {
            "blind": "season-total place GF/GA (default)",
            "venue_away": "λ_H blind; λ_A = gf_a_pm(ra)·((GA_H_blind/38)/ḡ)",
            "venue": "EV-matched λ_H=gf_h·(ga_a/league_ga_a); λ_A=gf_a·(ga_h/league_ga_h)",
            "naive": "isolate diagnostic / blended ḡ denom",
            "blend": "λ_A ← (1−w)·blind + w·venue_away; λ_H blind",
        },
        "train_seasons": list(TRAIN_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "oos_seasons": list(OOS_SEASONS) if oos else [],
        "gw_range": [GW_START, GW_END],
        "n_draws": n_draws,
        "seed": seed0,
        "blend_grid": BLEND_GRID,
        "fit_train": {"venue_blend": fit_w, "table": fit_rows},
        "holdout": holdout,
        "decisions": decisions,
        "winners": winners,
        "recommended_arm": recommended,
        "oos": oos,
        "oos_pass_counts": oos_pass_counts,
        "default_unchanged": True,
        "default_venue_mode": DEFAULT_VENUE_MODE,
        "pi_reapproval_required": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# cascade_long place×venue club-λ screen",
        "",
        f"Pre-registered **2026-10-27** · protocol "
        f"`cascade_long_place_venue_lam_screen` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Formula",
        "",
        "- **blind** (default): season-total place GF/GA.",
        "- **venue_away**: `λ_H` blind; "
        "`λ_A = gf_a_pm(ra)·((GA_H_blind/38)/ḡ)`.",
        "- **venue**: EV-matched full place×venue.",
        "- **naive**: isolate diagnostic (blended ḡ; overshoots home).",
        "- **blend**: `λ_A ← (1−w)·blind + w·venue_away`.",
        "",
        "## Locked before holdout",
        "",
        f"- Baseline: `venue_mode={DEFAULT_VENUE_MODE}`.",
        f"- Fit blend w on {', '.join(TRAIN_SEASONS)} → **`w={fit_w:.2f}`**.",
        f"- Holdout: **{HOLDOUT_SEASON}** GW{GW_START}–{GW_END}.",
        f"- OOS: {', '.join(OOS_SEASONS)}.",
        f"- Promote if vs baseline: away |bias| ↓ ≥{AWAY_IMPROVE_MIN}, "
        f"home |bias| ↑ ≤{HOME_WORSEN_MAX}, MAE ↑ ≤{MAE_WORSEN_MAX}.",
        "- Default `DEFAULT_VENUE_MODE` **not** flipped here "
        "(needs OOS + PI re-approval).",
        "",
        "## Holdout 2025-26 (Opta λ − actual)",
        "",
        "| arm | mode | w | bias | bias H | bias A | MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in holdout.items():
        o = block["opta"]
        lines.append(
            f"| {name} | {block['venue_mode']} | {block['venue_blend']:.2f} | "
            f"{o['bias']:+.3f} | {o['bias_H']:+.3f} | {o['bias_A']:+.3f} | "
            f"{o['mae']:.3f} |"
        )
    lines += ["", "## Promote decisions vs baseline", ""]
    for name, dec in decisions.items():
        mark = "PASS" if dec["pass"] else "FAIL"
        lines.append(
            f"- **{name}**: {mark} "
            f"(away_improve={dec['away_improve']:+.3f}, "
            f"home_worsen={dec['home_worsen']:+.3f}, "
            f"mae_worsen={dec['mae_worsen']:+.3f})"
        )
    if oos:
        lines += ["", "## OOS prior seasons", ""]
        for season, block in oos.items():
            lines.append(
                f"- **{season}**: winners="
                f"{block['winners'] or ['none']}"
            )
        lines += ["", "### OOS pass counts (of 3)", ""]
        for name, npass in oos_pass_counts.items():
            lines.append(f"- {name}: {npass}/3")
    lines += [
        "",
        f"**Recommended (counterfactual only):** "
        f"{recommended or 'none — keep venue_mode=blind'}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_place_venue_lam_screen",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_place_venue_lam_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Pre-register + screen place×venue Opta λ map (default unchanged)",
                "verdict": {
                    "fit_blend_w": fit_w,
                    "winners": winners,
                    "recommended_arm": recommended,
                    "oos_pass_counts": oos_pass_counts,
                    "default_unchanged": True,
                    "holdout_baseline_bias_A": base["bias_A"],
                    "holdout_baseline_bias_H": base["bias_H"],
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state["decisions"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
