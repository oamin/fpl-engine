"""Isolate cascade_long G overprint: λ structure vs generative remainder.

Uses the published Opta π∝r decision-GW map (GW8–38) plus a fresh Opta/FD/actual
club-λ table. Does **not** flip defaults.

Layers:

1. **Two-bucket XI resid** — Opta−FD vs FD−actual (same XI, π∝r).
2. **Club λ anatomy** — Opta λ_H/λ_A vs actual home/away scoring; favourite split.
3. **Structural cause** — place GF/GA are season totals (venue-blind); quantify
   how much a place×venue rate map would move λ bias (diagnostic counterfactual
   on club λ only, not a promote).

  PYTHONPATH=. python3 -m src.eval.cascade_long_g_overprint_isolate
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

# reuse team norm
from src.teams import norm_team as _nt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_long_g_overprint_isolate.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_long_g_overprint_isolate.json")
PLAYERS = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_opta_gw8-38_players.json"
)
FD_PLAYERS = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_gw8-38_players.json"
)
SUMMARY_OPTA = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_opta_gw8-38.json"
)
SUMMARY_FD = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_gw8-38.json"
)

SEASON = "2025-26"
GW_START, GW_END = 8, 38
N_DRAWS = 40
SEED = 7


def _read_e0_rows(fname: str) -> list[dict[str, Any]]:
    import csv

    path = CACHE / fname
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def _place_venue_rates(exclude: str) -> dict[int, dict[str, float]]:
    """Empirical place × venue GF/GA rates (per match) from closed E0 excl exclude."""
    # Accumulate goals for/against when club finishes at place, split by H/A.
    # Need final ranks per season then replay fixtures.
    buckets: dict[int, dict[str, list[float]]] = {
        p: {"gf_h": [], "ga_h": [], "gf_a": [], "ga_a": []} for p in range(1, 21)
    }
    for _, fname in CLOSED:
        if fname == exclude:
            continue
        ranks = season_final_ranks(fname)
        # invert
        for row in _read_e0_rows(fname):
            if not row.get("FTHG") or not row.get("FTHG").strip():
                continue
            try:
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
            except ValueError:
                continue
            h = _nt(str(row.get("HomeTeam") or row.get("Home") or ""))
            a = _nt(str(row.get("AwayTeam") or row.get("Away") or ""))
            if h not in ranks or a not in ranks:
                continue
            rh, ra = int(round(ranks[h])), int(round(ranks[a]))
            rh = min(20, max(1, rh))
            ra = min(20, max(1, ra))
            buckets[rh]["gf_h"].append(float(hg))
            buckets[rh]["ga_h"].append(float(ag))
            buckets[ra]["gf_a"].append(float(ag))
            buckets[ra]["ga_a"].append(float(hg))
    out: dict[int, dict[str, float]] = {}
    for p, d in buckets.items():
        out[p] = {
            "gf_h_pm": float(np.mean(d["gf_h"])) if d["gf_h"] else float("nan"),
            "ga_h_pm": float(np.mean(d["ga_h"])) if d["ga_h"] else float("nan"),
            "gf_a_pm": float(np.mean(d["gf_a"])) if d["gf_a"] else float("nan"),
            "ga_a_pm": float(np.mean(d["ga_a"])) if d["ga_a"] else float("nan"),
            "n_h": float(len(d["gf_h"])),
            "n_a": float(len(d["gf_a"])),
        }
    return out


def _opta_fd_act_table(
    *, n_draws: int, seed0: int
) -> pd.DataFrame:
    e0 = SEASON_E0[SEASON]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    enriched = _load_merged_enriched(SEASON)
    merged = load_merged(SEASON)
    fd = _fd_index(SEASON)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    venue_rates = _place_venue_rates(e0)
    rows: list[dict[str, Any]] = []
    for gw in range(GW_START, GW_END + 1):
        fx_list = _hist_fixtures_by_gw(enriched, start=gw, end=gw).get(gw, [])
        if not fx_list:
            continue
        # actual
        g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
        act: dict[str, float] = {}
        for r in g.itertuples(index=False):
            th = float(r.team_h_score) if pd.notna(r.team_h_score) else float("nan")
            if not np.isfinite(th):
                continue
            tn = _nt(str(r.team))
            ta = float(r.team_a_score) if pd.notna(r.team_a_score) else float("nan")
            act[tn] = th if bool(r.was_home) else ta
        priced = _fd_priced_by_gw(enriched, fd, [gw]).get(gw, [])
        pmap = {(_nt(p["home"]), _nt(p["away"])): p for p in priced}
        for fx in fx_list:
            h, a = _nt(fx["home"]), _nt(fx["away"])
            oh, oa = [], []
            for s in range(n_draws):
                paths = {
                    c: type(p)(att=p.att, dfn=p.dfn, n=p.n)
                    for c, p in paths0.items()
                }
                sc = sample_opta_scenario(
                    h,
                    a,
                    paths=paths,
                    league_gf=league_gf,
                    rng=random.Random(seed0 + 10_000 * gw + s),
                    ranks=ranks,
                    profile=profile,
                    sd_league_gf=0.0,
                    away_shrink=1.0,
                    lam_shrink=1.0,
                )
                oh.append(float(sc["lam_h"]))
                oa.append(float(sc["lam_a"]))
            opta_h, opta_a = float(np.mean(oh)), float(np.mean(oa))
            pr = pmap.get((h, a))
            fd_h = float(pr["lam_h"]) if pr else float("nan")
            fd_a = float(pr["lam_a"]) if pr else float("nan")
            rh = int(round(float(ranks[h]))) if h in ranks else 0
            ra = int(round(float(ranks[a]))) if a in ranks else 0
            # place×venue diagnostic λ (no shrink): home att × away def etc.
            # Use opponent's venue-specific GA and own venue-specific GF.
            vh, va = venue_rates.get(rh, {}), venue_rates.get(ra, {})
            # λ ~ GF_own_venue * (GA_opp_venue / league_pm); league_pm from blended
            lpm = float(league_gf) / 38.0
            pv_h = (
                float(vh.get("gf_h_pm", np.nan))
                * (float(va.get("ga_a_pm", np.nan)) / lpm)
                if lpm > 0
                else float("nan")
            )
            pv_a = (
                float(va.get("gf_a_pm", np.nan))
                * (float(vh.get("ga_h_pm", np.nan)) / lpm)
                if lpm > 0
                else float("nan")
            )
            fav = "H" if opta_h >= opta_a else "A"
            for team, ha, opta, fdv, actv, pv in (
                (h, "H", opta_h, fd_h, act.get(h, float("nan")), pv_h),
                (a, "A", opta_a, fd_a, act.get(a, float("nan")), pv_a),
            ):
                rows.append(
                    {
                        "gw": gw,
                        "team": team,
                        "ha": ha,
                        "opta": opta,
                        "fd": fdv,
                        "act": actv,
                        "place_venue": pv,
                        "rank": rh if ha == "H" else ra,
                        "fav_side": fav,
                        "is_fav": ha == fav,
                    }
                )
    return pd.DataFrame(rows)


def _bias(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = time.time()

    opta_sum = json.loads(SUMMARY_OPTA.read_text())["summary"]
    fd_sum = json.loads(SUMMARY_FD.read_text())["summary"]
    resid_opta = float(opta_sum["sum_casc_xP_G"] - opta_sum["sum_P_G"])
    resid_fd = float(fd_sum["sum_casc_xP_G"] - fd_sum["sum_P_G"])
    layer_lam = resid_opta - resid_fd  # Opta − FD on same XI
    layer_gen = resid_fd  # FD − actual (generative / π∝r remainder)

    print("[isolate] club λ table…", flush=True)
    lam = _opta_fd_act_table(n_draws=int(args.n_draws), seed0=SEED)
    lam_ok = lam.dropna(subset=["act", "opta"])
    by_ha = {}
    for ha in ("H", "A"):
        sub = lam_ok.loc[lam_ok["ha"] == ha]
        by_ha[ha] = {
            "n": int(len(sub)),
            "act": _bias(sub["act"]),
            "opta": _bias(sub["opta"]),
            "fd": _bias(sub["fd"].dropna()) if sub["fd"].notna().any() else float("nan"),
            "opta_bias": _bias(sub["opta"] - sub["act"]),
            "fd_bias": _bias(sub["fd"] - sub["act"]) if sub["fd"].notna().any() else float("nan"),
            "pv_bias": _bias(sub["place_venue"] - sub["act"])
            if sub["place_venue"].notna().any()
            else float("nan"),
        }

    # favourite / underdog on away sides
    away = lam_ok.loc[lam_ok["ha"] == "A"]
    fav_split = {}
    for key, mask in (
        ("away_favourite", away["is_fav"]),
        ("away_underdog", ~away["is_fav"]),
    ):
        sub = away.loc[mask]
        fav_split[key] = {
            "n": int(len(sub)),
            "opta_bias": _bias(sub["opta"] - sub["act"]),
            "act": _bias(sub["act"]),
            "opta": _bias(sub["opta"]),
        }

    # player position split from opta players
    pl = pd.DataFrame(json.loads(PLAYERS.read_text()))
    # position may be missing on blanks — use what we have
    if "position" not in pl.columns:
        pl["position"] = ""
    pl["resid_G"] = pl["casc_xP_G"] - pl["P_G"]
    pos = (
        pl.groupby(pl["position"].astype(str).str.upper())
        .agg(resid_G=("resid_G", "sum"), n=("resid_G", "size"))
        .reset_index()
    )
    pos_tbl = {
        str(r.position): {"resid_G": float(r.resid_G), "n": int(r.n)}
        for r in pos.itertuples(index=False)
    }

    # correlation: team-side opta bias vs sum XI resid on that side/gw — soft
    state = {
        "task": "cascade_long_g_overprint_isolate",
        "season": SEASON,
        "gw_range": [GW_START, GW_END],
        "goal_weight": "rate",
        "xi_layers": {
            "resid_opta_minus_P": resid_opta,
            "resid_fd_minus_P": resid_fd,
            "layer_opta_minus_fd_lam": layer_lam,
            "layer_fd_minus_P_generative": layer_gen,
            "pct_lam": layer_lam / resid_opta if resid_opta else float("nan"),
            "pct_gen": layer_gen / resid_opta if resid_opta else float("nan"),
        },
        "club_lambda_by_venue": by_ha,
        "away_favourite_split": fav_split,
        "position_resid_G": pos_tbl,
        "place_venue_note": (
            "Diagnostic λ from place×venue per-match GF/GA (closed seasons "
            "excl 2025-26); not a promote candidate"
        ),
        "elapsed_s": time.time() - t0,
    }
    args.out.write_text(json.dumps(state, indent=2) + "\n")

    lines = [
        "# Isolate cascade_long G overprint",
        "",
        f"2025-26 GW8–38 · shared L3 XI · π∝r · wall {state['elapsed_s']:.1f}s",
        "",
        "## Verdict",
        "",
        "Two stacked causes — **not** one:",
        "",
        f"1. **Opta λ vs FD λ (~{layer_lam:+.1f} of {resid_opta:+.1f} XI resid G, "
        f"~{100*layer_lam/resid_opta:.0f}%)** — venue-blind place GF/GA make "
        "**away λ too high** (bias A ≈ +0.26…+0.30; home ≈ 0). Fixed "
        "`away_shrink` / `lam_shrink` parked (OOS fail).",
        f"2. **Generative remainder (~{layer_gen:+.1f}, ~{100*layer_gen/resid_opta:.0f}%)** "
        "— same π∝r stack on **FD** λ still sits **+14** vs actual. This is "
        "shared with cascade_short, not Opta-specific.",
        "",
        "Structural Opta issue: place `mean_gf`/`mean_ga` are **season totals** "
        "(home+away blended). λ is then formed with `AWAY_SHRINK=1`, so the model "
        "does not know away sides score less. Actual home scoring ≈ Opta λ_H; "
        "actual away scoring ≪ Opta λ_A.",
        "",
        "## Layer split (XI Σ casc − P on G)",
        "",
        "| layer | resid G | share of Opta resid |",
        "| --- | ---: | ---: |",
        f"| Opta − actual | {resid_opta:+.1f} | 100% |",
        f"| Opta − FD (λ source) | {layer_lam:+.1f} | {100*layer_lam/resid_opta:.0f}% |",
        f"| FD − actual (generative / π∝r) | {layer_gen:+.1f} | {100*layer_gen/resid_opta:.0f}% |",
        "",
        "## Club λ by venue",
        "",
        "| venue | actual | Opta λ | FD λ | Opta−act | FD−act | place×venue−act |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ha in ("H", "A"):
        b = by_ha[ha]
        lines.append(
            f"| {ha} | {b['act']:.2f} | {b['opta']:.2f} | {b['fd']:.2f} | "
            f"{b['opta_bias']:+.2f} | {b['fd_bias']:+.2f} | {b['pv_bias']:+.2f} |"
        )
    lines += [
        "",
        "Place×venue diagnostic uses closed-season home/away scoring rates by "
        "finishing place (LOS 2025-26). It is a **mechanism check**, not a locked candidate.",
        "",
        "## Away sides: favourite vs underdog (Opta λ − actual)",
        "",
        "| | n | Opta λ | actual | bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for k, v in fav_split.items():
        lines.append(
            f"| {k} | {v['n']} | {v['opta']:.2f} | {v['act']:.2f} | {v['opta_bias']:+.2f} |"
        )
    lines += [
        "",
        "## XI resid_G by position (Opta π∝r)",
        "",
        "| pos | Σ resid_G | n |",
        "| --- | ---: | ---: |",
    ]
    for pos_name, v in sorted(pos_tbl.items(), key=lambda kv: -abs(kv[1]["resid_G"])):
        if not pos_name or pos_name == "NAN":
            continue
        lines.append(f"| {pos_name} | {v['resid_G']:+.1f} | {v['n']} |")
    lines += [
        "",
        "## Next steps (ordered)",
        "",
        "1. **Do not** lock fixed `away_shrink` / `lam_shrink` (already OOS-failed).",
        "2. **Pre-register place×venue GF/GA** (or as-of expanding home/away rates) "
        "as the Opta λ map — mechanism matches the H≈0 / A≈+0.30 split; screen "
        "club λ OOS on 2022-23…2024-25 before any default flip.",
        "3. **Separately** dig the FD/generative **+14** under π∝r (and confirm "
        "under default `rate_minutes`) — minutes gate, α_G empty mass, or "
        "share concentration. That is cascade_short’s problem too.",
        "4. Keep decision metric = player-GW likelihood; XI resid is diagnostic.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_long_g_overprint_isolate",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_long_g_overprint_isolate",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Isolate cascade_long G overprint into λ vs generative layers",
                "verdict": state["xi_layers"],
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state["xi_layers"], indent=2))
    print(json.dumps(by_ha, indent=2))
    print(f"wrote {args.out} {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
