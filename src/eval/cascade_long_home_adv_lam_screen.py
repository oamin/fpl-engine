"""Pre-registered club-λ venue screen for cascade_long ``HOME_ADV``.

Protocol: ``cascade_long_home_adv_lam_screen`` (diagnostic — does **not**
flip the PI-locked default ``HOME_ADV=1.0``).

Arms (locked before reading 2025-26 holdout totals):

- **baseline** ``home_adv=1.0``
- **candidate** ``home_adv=1.08`` (retired spice, re-opened as candidate)
- **fit** ``home_adv`` = argmin MAE(λ, actual) on train seasons
  2022-23…2024-25, grid 1.00…1.20 step 0.02

Holdout: 2025-26 GW8–38 team-sides. Also report FD centreline λ as a
book reference (not an Opta arm).

Promote rule (holdout vs baseline): ``|away bias|`` falls by ≥0.10 **and**
home ``|bias|`` does not rise by >0.05 **and** MAE does not rise by >0.02.
Even if the rule fires, PI re-approval is required before changing
``protocol.cascade_long_match_outcome``.

  PYTHONPATH=. python3 -m src.eval.cascade_long_home_adv_lam_screen
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
    HOME_ADV,
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_home_adv_lam_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_home_adv_lam_screen.json")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
HOLDOUT_SEASON = "2025-26"
GW_START = 8
GW_END = 38
N_DRAWS = 40
SEED = 7
GRID = [round(x, 2) for x in np.arange(1.00, 1.21, 0.02)]
CANDIDATE_108 = 1.08
# Locked promote thresholds (protocol).
AWAY_IMPROVE_MIN = 0.10
HOME_WORSEN_MAX = 0.05
MAE_WORSEN_MAX = 0.02


def _now() -> float:
    return time.time()


def _actual_team_goals(merged: pd.DataFrame, gw: int) -> dict[str, float]:
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
    act: dict[str, float] = {}
    if g.empty:
        return act
    # One row per player; scores repeat — unique on team×home flag.
    for r in g.itertuples(index=False):
        th = float(r.team_h_score) if pd.notna(r.team_h_score) else float("nan")
        ta = float(r.team_a_score) if pd.notna(r.team_a_score) else float("nan")
        if not np.isfinite(th):
            continue
        tn = norm_team(str(r.team))
        if bool(r.was_home):
            act[tn] = th
        else:
            act[tn] = ta
    return act


def _season_team_sides(
    season: str,
    *,
    home_adv: float,
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
                    home_adv=float(home_adv),
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
                    "home_adv": float(home_adv),
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
                    "home_adv": float(home_adv),
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


def _promote(
    base: dict[str, float], cand: dict[str, float]
) -> dict[str, Any]:
    away_base = abs(float(base["bias_A"]))
    away_cand = abs(float(cand["bias_A"]))
    home_base = abs(float(base["bias_H"]))
    home_cand = abs(float(cand["bias_H"]))
    mae_base = float(base["mae"])
    mae_cand = float(cand["mae"])
    away_improve = away_base - away_cand
    home_worsen = home_cand - home_base
    mae_worsen = mae_cand - mae_base
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
    args = ap.parse_args(argv)
    n_draws = int(args.n_draws)
    seed0 = int(args.seed)
    t0 = _now()

    print(
        f"[home-adv-lam] train={TRAIN_SEASONS} holdout={HOLDOUT_SEASON} "
        f"GW{GW_START}–{GW_END} N_draws={n_draws} grid={GRID[0]}…{GRID[-1]}",
        flush=True,
    )

    # --- Fit home_adv on train seasons (MAE) ---
    fit_rows: list[dict[str, Any]] = []
    for h_adv in GRID:
        print(f"[home-adv-lam] train scan home_adv={h_adv:.2f}", flush=True)
        parts = [
            _season_team_sides(
                season,
                home_adv=h_adv,
                n_draws=n_draws,
                seed0=seed0 + 100 * i,
                with_fd=False,
            )
            for i, season in enumerate(TRAIN_SEASONS)
        ]
        train = pd.concat(parts, ignore_index=True)
        summ = _summarise(train)
        fit_rows.append({"home_adv": float(h_adv), **summ})
        print(
            f"  train bias_A={summ['bias_A']:+.3f} bias_H={summ['bias_H']:+.3f} "
            f"MAE={summ['mae']:.3f}",
            flush=True,
        )
    fit_df = pd.DataFrame(fit_rows)
    fit_adv = float(fit_df.loc[fit_df["mae"].idxmin(), "home_adv"])
    print(f"[home-adv-lam] fit home_adv={fit_adv:.2f} (min train MAE)", flush=True)

    arms = {
        "baseline_1_00": float(HOME_ADV),
        "candidate_1_08": float(CANDIDATE_108),
        "fit_train": fit_adv,
    }
    # Deduplicate if fit lands on 1.00 or 1.08
    holdout: dict[str, Any] = {}
    for name, h_adv in arms.items():
        print(f"[home-adv-lam] holdout {HOLDOUT_SEASON} arm={name} h={h_adv:.2f}", flush=True)
        df = _season_team_sides(
            HOLDOUT_SEASON,
            home_adv=h_adv,
            n_draws=n_draws,
            seed0=seed0 + 999,
            with_fd=True,
        )
        opta = _summarise(df, lam_col="opta")
        fd = _summarise(df, lam_col="fd")
        holdout[name] = {
            "home_adv": float(h_adv),
            "opta": opta,
            "fd": fd,
            "n_team_sides": int(opta["n"]),
        }
        print(
            f"  opta bias_A={opta['bias_A']:+.3f} bias_H={opta['bias_H']:+.3f} "
            f"MAE={opta['mae']:.3f} | fd bias_A={fd['bias_A']:+.3f} "
            f"bias_H={fd['bias_H']:+.3f} MAE={fd['mae']:.3f}",
            flush=True,
        )

    base = holdout["baseline_1_00"]["opta"]
    decisions = {
        name: _promote(base, holdout[name]["opta"])
        for name in holdout
        if name != "baseline_1_00"
    }
    winners = [n for n, d in decisions.items() if d["pass"]]
    # Prefer fit if both pass and fit≠1.08; else any pass.
    recommended = None
    if "fit_train" in winners:
        recommended = "fit_train"
    elif "candidate_1_08" in winners:
        recommended = "candidate_1_08"

    state = {
        "task": "cascade_long_home_adv_lam_screen",
        "protocol": "cascade_long_home_adv_lam_screen",
        "pre_registered": "2026-10-27",
        "train_seasons": list(TRAIN_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "gw_range": [GW_START, GW_END],
        "n_draws": n_draws,
        "seed": seed0,
        "grid": GRID,
        "fit_train": {
            "home_adv": fit_adv,
            "table": fit_rows,
        },
        "holdout": holdout,
        "decisions": decisions,
        "winners": winners,
        "recommended_arm": recommended,
        "default_unchanged": True,
        "default_home_adv": float(HOME_ADV),
        "pi_reapproval_required": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# cascade_long HOME_ADV club-λ venue screen",
        "",
        f"Pre-registered **2026-10-27** · protocol "
        f"`cascade_long_home_adv_lam_screen` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Locked before holdout",
        "",
        f"- Baseline: `home_adv={HOME_ADV}` (PI lock).",
        f"- Candidate: `home_adv={CANDIDATE_108}`.",
        f"- Fit: argmin train MAE on {', '.join(TRAIN_SEASONS)} → "
        f"**`home_adv={fit_adv:.2f}`**.",
        f"- Holdout: **{HOLDOUT_SEASON}** GW{GW_START}–{GW_END}.",
        f"- Promote if vs baseline: away |bias| ↓ ≥{AWAY_IMPROVE_MIN}, "
        f"home |bias| ↑ ≤{HOME_WORSEN_MAX}, MAE ↑ ≤{MAE_WORSEN_MAX}.",
        "- Default `HOME_ADV` **not** flipped here (PI re-approval required).",
        "",
        "## Holdout 2025-26 (Opta λ − actual)",
        "",
        "| arm | home_adv | bias | bias H | bias A | MAE | FD bias A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in holdout.items():
        o, f = block["opta"], block["fd"]
        lines.append(
            f"| {name} | {block['home_adv']:.2f} | {o['bias']:+.3f} | "
            f"{o['bias_H']:+.3f} | {o['bias_A']:+.3f} | {o['mae']:.3f} | "
            f"{f['bias_A']:+.3f} |"
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
    lines += [
        "",
        f"**Recommended (counterfactual only):** "
        f"{recommended or 'none — keep HOME_ADV=1.0'}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_home_adv_lam_screen",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_home_adv_lam_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Pre-registered Opta HOME_ADV club-λ venue screen",
                "verdict": {
                    "fit_home_adv": fit_adv,
                    "winners": winners,
                    "recommended_arm": recommended,
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
    print(json.dumps({"winners": winners, "recommended": recommended, "fit": fit_adv}, indent=2))
    print(f"wrote {args.out} {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
