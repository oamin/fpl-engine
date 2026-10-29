"""Pre-registered club-λ screen: as-of expanding club H/A rates vs Opta blind.

Protocol: ``cascade_long_expanding_ha_lam_screen``.

Reopens the parked place×venue venue fix with **as-of** club home/away
scoring rates (prior closed seasons + current season matches before GW),
not final-place×venue oracles.

Arms (locked before reading totals):

- **baseline** Opta ``venue_mode=blind`` place season-total λ
- **expanding_ha** full as-of club H/A Dixon–Coles λ
- **expanding_away** λ_H blind; λ_A from expanding club rates
- **blend_fit** train-fit weight on expanding_ha (grid 0..1 step 0.1)

Promote rule (same as place×venue): vs blind, |away bias| ↓ ≥0.10 AND
home |bias| ↑ ≤0.05 AND MAE ↑ ≤0.02. OOS required on all closed seasons
before any default flip. PI re-approval still required.

  PYTHONPATH=. python3 -m src.eval.cascade_long_expanding_ha_lam_screen
"""

from __future__ import annotations

import argparse
import csv
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
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets.numagent_xp_pilot import load_merged
from src.models.multistage_mc_milp import init_paths_from_opta, sample_opta_scenario
from src.models.opta_gf_ga_horizon import (
    CACHE,
    CLOSED,
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_expanding_ha_lam_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_expanding_ha_lam_screen.json")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
HOLDOUT_SEASON = "2025-26"
OOS_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
GW_START = 8
GW_END = 38
N_DRAWS = 20
SEED = 7
PRIOR_N = 10.0
BLEND_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.10)]
AWAY_IMPROVE_MIN = 0.10
HOME_WORSEN_MAX = 0.05
MAE_WORSEN_MAX = 0.02


def _now() -> float:
    return time.time()


def _read_e0(fname: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (CACHE / fname).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("FTHG"):
                continue
            try:
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "h": norm_team(str(row.get("HomeTeam") or "")),
                    "a": norm_team(str(row.get("AwayTeam") or "")),
                    "hg": hg,
                    "ag": ag,
                }
            )
    return rows


def _league_and_club_prior(exclude_e0: str) -> tuple[dict[str, float], dict[str, dict[str, list[float]]]]:
    pri: dict[str, list[float]] = {
        "gf_h": [],
        "ga_h": [],
        "gf_a": [],
        "ga_a": [],
    }
    club: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gf_h": [], "ga_h": [], "gf_a": [], "ga_a": []}
    )
    for _, fname in CLOSED:
        if fname == exclude_e0:
            continue
        for r in _read_e0(fname):
            club[r["h"]]["gf_h"].append(float(r["hg"]))
            club[r["h"]]["ga_h"].append(float(r["ag"]))
            club[r["a"]]["gf_a"].append(float(r["ag"]))
            club[r["a"]]["ga_a"].append(float(r["hg"]))
            pri["gf_h"].append(float(r["hg"]))
            pri["ga_h"].append(float(r["ag"]))
            pri["gf_a"].append(float(r["ag"]))
            pri["ga_a"].append(float(r["hg"]))
    lg = {k: float(np.mean(v)) if v else 1.0 for k, v in pri.items()}
    return lg, club


def _rates(
    club: str,
    cur: dict[str, list[float]],
    *,
    club_pri: dict[str, dict[str, list[float]]],
    lg: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ("gf_h", "ga_h", "gf_a", "ga_a"):
        prior_xs = club_pri.get(club, {}).get(k) or []
        pmean = float(np.mean(prior_xs)) if prior_xs else float(lg[k])
        xs = cur.get(k) or []
        out[k] = (PRIOR_N * pmean + float(sum(xs))) / (PRIOR_N + len(xs))
    return out


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


def _season_rows(
    season: str,
    *,
    n_draws: int,
    seed0: int,
    with_fd: bool,
) -> pd.DataFrame:
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    lg, club_pri = _league_and_club_prior(e0)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    enriched = _load_merged_enriched(season)
    act = _actual_team_gw(load_merged(season))
    fd = _fd_index(season) if with_fd else None
    played: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gf_h": [], "ga_h": [], "gf_a": [], "ga_a": []}
    )
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
            rh = _rates(h, played[h], club_pri=club_pri, lg=lg)
            ra = _rates(a, played[a], club_pri=club_pri, lg=lg)
            den_a = max(float(lg["ga_a"]), 1e-9)
            den_h = max(float(lg["ga_h"]), 1e-9)
            exp_h = float(rh["gf_h"]) * (float(ra["ga_a"]) / den_a)
            exp_a = float(ra["gf_a"]) * (float(rh["ga_h"]) / den_h)
            pr = pmap.get((h, a))
            for team, ha, blind, exp, ao, fdv in (
                (
                    h,
                    "H",
                    blind_h,
                    exp_h,
                    blind_h,
                    float(pr["lam_h"]) if pr else float("nan"),
                ),
                (
                    a,
                    "A",
                    blind_a,
                    exp_a,
                    exp_a,
                    float(pr["lam_a"]) if pr else float("nan"),
                ),
            ):
                act_v = act.loc[(act["gw"] == gw) & (act["team"] == team), "act"]
                rows.append(
                    {
                        "season": season,
                        "gw": int(gw),
                        "team": team,
                        "ha": ha,
                        "blind": float(blind),
                        "expanding_ha": float(exp),
                        "expanding_away": float(ao),
                        "fd": fdv,
                        "act": float(act_v.iloc[0]) if len(act_v) else float("nan"),
                    }
                )
        # Update expanding history after the GW's fixtures (as-of = before GW).
        for fx in fx_list:
            h, a = norm_team(fx["home"]), norm_team(fx["away"])
            ah = act.loc[(act["gw"] == gw) & (act["team"] == h), "act"]
            aa = act.loc[(act["gw"] == gw) & (act["team"] == a), "act"]
            if not len(ah) or not len(aa):
                continue
            hg, ag = float(ah.iloc[0]), float(aa.iloc[0])
            played[h]["gf_h"].append(hg)
            played[h]["ga_h"].append(ag)
            played[a]["gf_a"].append(ag)
            played[a]["ga_a"].append(hg)
    return pd.DataFrame(rows)


def _summarise(df: pd.DataFrame, *, lam_col: str) -> dict[str, float]:
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


def _blend_col(df: pd.DataFrame, w: float) -> pd.Series:
    return (1.0 - float(w)) * df["blind"] + float(w) * df["expanding_ha"]


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
    args = ap.parse_args(argv)
    n_draws = int(args.n_draws)
    seed0 = int(args.seed)
    t0 = _now()

    print(
        f"[expanding-ha] train={TRAIN_SEASONS} holdout={HOLDOUT_SEASON} "
        f"GW{GW_START}–{GW_END} N={n_draws}",
        flush=True,
    )

    # Cache season tables once.
    tables: dict[str, pd.DataFrame] = {}
    for i, season in enumerate(OOS_SEASONS):
        print(f"[expanding-ha] build {season}", flush=True)
        tables[season] = _season_rows(
            season,
            n_draws=n_draws,
            seed0=seed0 + 100 * i,
            with_fd=(season == HOLDOUT_SEASON),
        )

    fit_rows: list[dict[str, Any]] = []
    for w in BLEND_GRID:
        parts = [tables[s].assign(lam=_blend_col(tables[s], w)) for s in TRAIN_SEASONS]
        train = pd.concat(parts, ignore_index=True)
        summ = _summarise(train, lam_col="lam")
        fit_rows.append({"venue_blend": float(w), **summ})
        print(
            f"  train blend w={w:.2f} bias_A={summ['bias_A']:+.3f} "
            f"bias_H={summ['bias_H']:+.3f} MAE={summ['mae']:.3f}",
            flush=True,
        )
    fit_df = pd.DataFrame(fit_rows)
    fit_w = float(fit_df.loc[fit_df["mae"].idxmin(), "venue_blend"])
    print(f"[expanding-ha] fit blend w={fit_w:.2f}", flush=True)

    def _arm_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
        out = df.copy()
        if name == "baseline_blind":
            out["lam"] = out["blind"]
        elif name == "expanding_ha":
            out["lam"] = out["expanding_ha"]
        elif name == "expanding_away":
            out["lam"] = out["expanding_away"]
        elif name == "blend_fit":
            out["lam"] = _blend_col(out, fit_w)
        else:
            raise KeyError(name)
        return out

    arm_names = ("baseline_blind", "expanding_ha", "expanding_away", "blend_fit")
    holdout: dict[str, Any] = {}
    hdf = tables[HOLDOUT_SEASON]
    for name in arm_names:
        frame = _arm_frame(hdf, name)
        opta = _summarise(frame, lam_col="lam")
        fd = _summarise(frame, lam_col="fd")
        holdout[name] = {
            "opta": opta,
            "fd": fd,
            "venue_blend": float(fit_w if name == "blend_fit" else (1.0 if name == "expanding_ha" else 0.0)),
        }
        print(
            f"[expanding-ha] holdout {name} H={opta['bias_H']:+.3f} "
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
            name: _summarise(_arm_frame(sdf, name), lam_col="lam") for name in arm_names
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
        print(f"[expanding-ha] OOS {season} winners={oos[season]['winners'] or ['none']}", flush=True)

    oos_pass_counts = {
        name: sum(1 for s in oos.values() if name in s["winners"])
        for name in decisions
    }
    recommended = None
    for pref in ("expanding_away", "blend_fit", "expanding_ha"):
        if pref in winners and oos_pass_counts.get(pref, 0) >= 3:
            recommended = pref
            break

    state = {
        "task": "cascade_long_expanding_ha_lam_screen",
        "protocol": "cascade_long_expanding_ha_lam_screen",
        "pre_registered": "2026-10-29",
        "formula": {
            "blind": "Opta place season-total GF/GA",
            "expanding_ha": "as-of club H/A rates × opp venue def / league venue mean",
            "expanding_away": "λ_H blind; λ_A from expanding club rates",
            "blend": "λ ← (1−w)·blind + w·expanding_ha",
        },
        "prior_pseudo_matches": PRIOR_N,
        "train_seasons": list(TRAIN_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "oos_seasons": list(OOS_SEASONS),
        "gw_range": [GW_START, GW_END],
        "n_draws": n_draws,
        "fit_train": {"venue_blend": fit_w, "table": fit_rows},
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
        "# cascade_long as-of expanding club H/A λ screen",
        "",
        f"Pre-registered **2026-10-29** · protocol "
        f"`cascade_long_expanding_ha_lam_screen` · wall {state['elapsed_s']:.1f}s",
        "",
        "## Formula",
        "",
        "- **blind**: Opta place season-total GF/GA (default).",
        "- **expanding_ha**: as-of club home/away GF/GA (prior seasons + "
        f"current before GW; {PRIOR_N:g} pseudo prior) × opp venue defence / "
        "league venue mean.",
        "- **expanding_away**: `λ_H` blind; `λ_A` from expanding rates.",
        "- **blend**: `(1−w)·blind + w·expanding_ha`.",
        "",
        "## Holdout 2025-26",
        "",
        "| arm | bias | bias H | bias A | MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, block in holdout.items():
        o = block["opta"]
        lines.append(
            f"| {name} | {o['bias']:+.3f} | {o['bias_H']:+.3f} | "
            f"{o['bias_A']:+.3f} | {o['mae']:.3f} |"
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
        f"**Recommended:** {recommended or 'none — keep venue_mode=blind'}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_expanding_ha_lam_screen",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_expanding_ha_lam_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Screen as-of expanding club H/A λ vs Opta blind",
                "verdict": {
                    "fit_blend_w": fit_w,
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
    print(json.dumps({"winners": winners, "oos_pass_counts": oos_pass_counts, "recommended": recommended}, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
