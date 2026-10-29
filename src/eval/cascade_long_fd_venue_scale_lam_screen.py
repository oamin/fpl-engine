"""Pre-registered club-λ screen: venue mean-ratio scale Opta → FD levels.

Protocol: ``cascade_long_fd_venue_scale_lam_screen``.

Follows the Opta−FD residual driver dig (venue dominates) and the
margin-shrink FAIL (margin compression cannot move venue means when
E[λ_H]≈E[λ_A] under blind place totals).

Scale blind Opta λ by venue toward FD venue means:

  λ'_H = r_H · λ_H
  λ'_A = r_A · λ_A
  r_ha = mean(FD_ha) / mean(Opta_ha)   (fit on train seasons)

Arms (locked before reading holdout totals):

- **baseline** r_H=r_A=1 (blind)
- **fd_venue_fit** train-fit (r_H, r_A) from pooled FD/Opta venue means
  on 2022-23…2024-25 (FD available on all closed E0 seasons)
- **fd_venue_away_only** r_H=1; r_A = train away FD/Opta
- **pooled_const** fixed r_H=1.115, r_A=0.893 from 2024-25+2025-26 dig
  (diagnostic; not a fit on holdout alone)

Promote rule (same as place×venue): vs blind, |away bias| ↓ ≥0.10 AND
home |bias| ↑ ≤0.05 AND MAE ↑ ≤0.02. OOS on closed seasons. Default
unchanged without PI.

  PYTHONPATH=. python3 -m src.eval.cascade_long_fd_venue_scale_lam_screen
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
REPORT = ROOT / "reports" / "cascade_long_fd_venue_scale_lam_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_fd_venue_scale_lam_screen.json")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
HOLDOUT_SEASON = "2025-26"
OOS_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
GW_START = 8
GW_END = 38
N_DRAWS = 20
SEED = 7
# From cascade_long_fd_opta_lam_scale_dig (2024-25+2025-26 pooled) — diagnostic arm.
POOLED_RH = 1.1146
POOLED_RA = 0.8928
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


def _season_sides(
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
            blind_h, blind_a = float(np.mean(oh)), float(np.mean(oa))
            for team, ha, blind, fdv in (
                (h, "H", blind_h, float(pr["lam_h"])),
                (a, "A", blind_a, float(pr["lam_a"])),
            ):
                act_v = act.loc[(act["gw"] == gw) & (act["team"] == team), "act"]
                rows.append(
                    {
                        "season": season,
                        "gw": int(gw),
                        "team": team,
                        "ha": ha,
                        "blind": blind,
                        "fd": fdv,
                        "act": float(act_v.iloc[0]) if len(act_v) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _venue_ratios(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for ha in ("H", "A"):
        sub = df.loc[df["ha"] == ha].dropna(subset=["blind", "fd"])
        out[ha] = float(sub["fd"].mean() / sub["blind"].mean())
    return out


def _apply_scale(df: pd.DataFrame, r_h: float, r_a: float) -> pd.DataFrame:
    out = df.copy()
    out["lam"] = out["blind"]
    out.loc[out["ha"] == "H", "lam"] = out.loc[out["ha"] == "H", "blind"] * float(r_h)
    out.loc[out["ha"] == "A", "lam"] = out.loc[out["ha"] == "A", "blind"] * float(r_a)
    return out


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
        f"[fd-venue-scale] train={TRAIN_SEASONS} holdout={HOLDOUT_SEASON} "
        f"GW{GW_START}–{GW_END} N={args.n_draws}",
        flush=True,
    )

    tables: dict[str, pd.DataFrame] = {}
    for i, season in enumerate(OOS_SEASONS):
        print(f"[fd-venue-scale] build {season}", flush=True)
        tables[season] = _season_sides(
            season, n_draws=int(args.n_draws), seed0=int(args.seed) + 100 * i
        )

    train = pd.concat([tables[s] for s in TRAIN_SEASONS], ignore_index=True)
    fit = _venue_ratios(train)
    r_h_fit, r_a_fit = fit["H"], fit["A"]
    print(
        f"[fd-venue-scale] train ratios r_H={r_h_fit:.4f} r_A={r_a_fit:.4f}",
        flush=True,
    )

    arm_scales = {
        "baseline_blind": (1.0, 1.0),
        "fd_venue_fit": (r_h_fit, r_a_fit),
        "fd_venue_away_only": (1.0, r_a_fit),
        "pooled_const": (POOLED_RH, POOLED_RA),
    }

    holdout: dict[str, Any] = {}
    hdf = tables[HOLDOUT_SEASON]
    for name, (rh, ra) in arm_scales.items():
        frame = _apply_scale(hdf, rh, ra)
        opta = _summarise(frame, lam_col="lam")
        fd = _summarise(frame, lam_col="fd")
        holdout[name] = {"r_H": float(rh), "r_A": float(ra), "opta": opta, "fd": fd}
        print(
            f"[fd-venue-scale] holdout {name} H={opta['bias_H']:+.3f} "
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

    oos: dict[str, Any] = {}
    for season in OOS_SEASONS:
        sdf = tables[season]
        arms = {
            name: _summarise(_apply_scale(sdf, rh, ra))
            for name, (rh, ra) in arm_scales.items()
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
            "oracle_ratios": _venue_ratios(sdf),
        }
        print(
            f"[fd-venue-scale] OOS {season} winners={oos[season]['winners'] or ['none']}",
            flush=True,
        )

    oos_pass_counts = {
        name: sum(1 for s in oos.values() if name in s["winners"])
        for name in decisions
    }
    recommended = None
    for pref in ("fd_venue_away_only", "fd_venue_fit", "pooled_const"):
        if pref in winners and oos_pass_counts.get(pref, 0) >= 3:
            recommended = pref
            break

    state = {
        "task": "cascade_long_fd_venue_scale_lam_screen",
        "protocol": "cascade_long_fd_venue_scale_lam_screen",
        "pre_registered": "2026-10-29",
        "formula": "λ'_H = r_H·λ_H ; λ'_A = r_A·λ_A ; r=mean(FD)/mean(Opta) by venue",
        "lever": "venue mean-ratio toward FD (dominant Opta−FD residual driver)",
        "note_prior": "margin_shrink screen FAIL — inert when E[λ_H]≈E[λ_A]",
        "train_ratios": {"H": r_h_fit, "A": r_a_fit},
        "pooled_const": {"H": POOLED_RH, "A": POOLED_RA},
        "train_seasons": list(TRAIN_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "oos_seasons": list(OOS_SEASONS),
        "gw_range": [GW_START, GW_END],
        "n_draws": int(args.n_draws),
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
        "# cascade_long FD venue mean-ratio λ screen",
        "",
        f"Pre-registered **2026-10-29** · protocol "
        f"`cascade_long_fd_venue_scale_lam_screen` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Lever",
        "",
        "Venue dominates the Opta−FD residual. Scale blind Opta by venue:",
        "",
        "```",
        "λ'_H = r_H · λ_H",
        "λ'_A = r_A · λ_A",
        "r_ha = mean(FD_ha) / mean(Opta_ha)   # train fit",
        "```",
        "",
        f"Train-fit: **r_H={r_h_fit:.4f}**, **r_A={r_a_fit:.4f}**.",
        "",
        "Prior: margin-shrink FAIL (does not move venue means when "
        "E[λ_H]≈E[λ_A] under blind).",
        "",
        "## Holdout 2025-26",
        "",
        "| arm | r_H | r_A | bias | bias H | bias A | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in holdout.items():
        o = block["opta"]
        lines.append(
            f"| {name} | {block['r_H']:.3f} | {block['r_A']:.3f} | "
            f"{o['bias']:+.3f} | {o['bias_H']:+.3f} | {o['bias_A']:+.3f} | "
            f"{o['mae']:.3f} |"
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
        f"**Recommended:** {recommended or 'none — keep r_H=r_A=1 (blind)'}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_fd_venue_scale_lam_screen",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_fd_venue_scale_lam_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Screen FD venue mean-ratio scale on Opta λ",
                "verdict": {
                    "train_r_H": r_h_fit,
                    "train_r_A": r_a_fit,
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
                "train_ratios": fit,
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
