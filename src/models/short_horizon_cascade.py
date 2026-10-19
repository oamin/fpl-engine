"""Short-horizon scenario layer on priced GW t and t+1.

Architecture (does **not** replace published ``score_xp``):

1. **Mean layer** — centreline opening/Shin pots → ``xp_on_pot`` with frozen
   deadline shares (same as ``forecast_xp``).
2. **Scenario layer** — sample scorelines from those pots (±15% λ band),
   re-score the pool on sampled λ, run path FT between t and t+1
   (``xi_plus_captain`` / ``xi_plus_captain_ft``). Shares stay frozen.

Diagnostic only until a path-V screen clears. No Odds API; historical E0
opening lines only. Long-horizon Opta cascade stays at t+2….
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.ingest.fpl_odds import load_player_logs
from src.models.forecast_xp import (
    SEASON_CODE,
    _matched_sheet,
    opening_pots_for_sheet,
    xp_on_pot,
)
from src.models.multistage_mc_milp import (
    _pick_squad_xi_captain,
    _pick_squad_xi_captain_ft,
    deadline_share_frame,
    pot_from_sampled_lam,
    sample_scoreline,
)
from src.rules.fpl_2026 import advance_ft
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

MARKET_BAND = 0.15
DEFAULT_N = 8
DEFAULT_SEED = 7
DEFAULT_GAMMA = 0.9
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def load_season_inputs(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (player_logs, E0 odds, merged sheet)."""
    code = SEASON_CODE[season]
    stem = season.replace("-", "_")
    logs = load_player_logs(season=season)
    odds = pd.read_csv(CACHE / f"E0_{code}.csv")
    sheet = pd.read_csv(CACHE / f"merged_gw_{stem}.csv")
    return logs, odds, sheet


def fixtures_for_gw(sheet: pd.DataFrame, gw: int) -> list[tuple[str, str]]:
    matched = _matched_sheet(sheet)
    rows = matched.loc[pd.to_numeric(matched["gw"], errors="coerce") == int(gw)]
    out = []
    for row in rows.itertuples(index=False):
        out.append((norm_team(str(row.home)), norm_team(str(row.away))))
    return out


def historical_pool(logs: pd.DataFrame, *, before_gw: int) -> pd.DataFrame:
    """Selectable-ish pool: frozen shares + latest value/position before t."""
    shares = deadline_share_frame(logs, before_gw=int(before_gw))
    hist = logs.loc[pd.to_numeric(logs["gw"], errors="coerce") < int(before_gw)].copy()
    hist["player_id"] = hist["player_id"].astype(str)
    last = (
        hist.sort_values(["player_id", "gw", "date"], kind="mergesort")
        .groupby("player_id", as_index=False)
        .tail(1)
    )
    meta = last.set_index("player_id")[["position", "value", "team_norm", "player_name"]]
    out = shares.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(
        meta.reset_index().rename(columns={"player_id": "player_id"}),
        on="player_id",
        how="left",
        suffixes=("", "_meta"),
    )
    if "position_meta" in out.columns:
        out["position"] = out["position"].fillna(out["position_meta"])
    if "team_norm_meta" in out.columns:
        out["team_norm"] = out["team_norm"].fillna(out["team_norm_meta"])
    if "value_meta" in out.columns:
        out["value"] = out["value"].fillna(out["value_meta"])
    out["team_norm"] = out["team_norm"].map(lambda n: norm_team(str(n)))
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(50.0)
    out["fwd_goal_scale"] = 1.0
    n_prior = pd.to_numeric(out["n_prior"], errors="coerce") if "n_prior" in out.columns else 3
    out["eligible"] = (n_prior >= 3) & (pd.to_numeric(out["xmi"], errors="coerce").fillna(0) >= 45)
    elig = out.loc[out["eligible"]].copy()
    # Cap MILP size: keep highest xmi within each position.
    if len(elig) > 100:
        parts = []
        for _, block in elig.groupby("position", sort=False):
            parts.append(block.nlargest(min(len(block), 30), "xmi"))
        elig = pd.concat(parts, ignore_index=True)
    if len(elig) >= 40:
        return elig.reset_index(drop=True)
    return out.reset_index(drop=True)


def centreline_club_pots(
    pots: Mapping[tuple[int, str], list[dict[str, float]]],
    gw: int,
) -> dict[str, dict[str, float]]:
    """First listed pot per club in ``gw`` (single fixtures dominate)."""
    out: dict[str, dict[str, float]] = {}
    for (g, club), lst in pots.items():
        if int(g) != int(gw) or not lst:
            continue
        out[str(club)] = dict(lst[0])
    return out


def score_pool_on_club_pots(pool: pd.DataFrame, club_pots: Mapping[str, dict[str, float]]) -> pd.DataFrame:
    out = pool.copy()
    scores = []
    for row in out.itertuples(index=False):
        pot = club_pots.get(str(row.team_norm))
        if pot is None:
            scores.append(0.0)
            continue
        scores.append(
            float(
                xp_on_pot(
                    position=str(row.position),
                    xmi=float(row.xmi),
                    share_xg=float(row.share_xG),
                    share_xa=float(row.share_xA),
                    exp_defcon_hit=float(row.exp_defcon_hit),
                    fwd_goal_scale=float(getattr(row, "fwd_goal_scale", 1.0) or 1.0),
                    pot=pot,
                )
            )
        )
    out["score"] = scores
    return out


def sample_week_club_pots(
    fixtures: Sequence[tuple[str, str]],
    centre: Mapping[str, dict[str, float]],
    rng: random.Random,
    *,
    band: float = MARKET_BAND,
) -> dict[str, dict[str, float]]:
    """Sample λ inside ±band of centreline for each fixture; map to club pots."""
    out: dict[str, dict[str, float]] = {}
    for home, away in fixtures:
        ph = centre.get(home)
        pa = centre.get(away)
        if ph is None or pa is None:
            continue
        lh = float(ph["lam_scored"])
        la = float(pa["lam_scored"])
        lo_h, hi_h = lh * (1.0 - band), lh * (1.0 + band)
        lo_a, hi_a = la * (1.0 - band), la * (1.0 + band)
        _gh, _ga, sh, sa = sample_scoreline(
            lh,
            la,
            lam_h_lo=lo_h,
            lam_h_hi=hi_h,
            lam_a_lo=lo_a,
            lam_a_hi=hi_a,
            rng=rng,
            centre_blend=0.5,
        )
        out[home] = pot_from_sampled_lam(sh, sa)
        out[away] = pot_from_sampled_lam(sa, sh)
    # Clubs with no fixture this GW keep centreline (blank / bye).
    for club, pot in centre.items():
        out.setdefault(club, dict(pot))
    return out


def mean_player_xp_delta(
    pool: pd.DataFrame,
    centre: Mapping[str, dict[str, float]],
    samples: Sequence[Mapping[str, dict[str, float]]],
) -> dict[str, float]:
    """Mean |sampled xP − centreline xP| across players with a pot."""
    base = score_pool_on_club_pots(pool, centre)
    deltas = []
    for samp in samples:
        scored = score_pool_on_club_pots(pool, samp)
        d = (scored["score"] - base["score"]).abs()
        deltas.append(float(d.mean()))
    return {
        "mean_abs_player_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "max_mean_abs_player_delta": float(np.max(deltas)) if deltas else float("nan"),
    }


def run_decision(
    season: str,
    decision_gw: int,
    *,
    n_strands: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    gamma: float = DEFAULT_GAMMA,
    band: float = MARKET_BAND,
    time_limit: float = 6.0,
) -> dict[str, Any]:
    """One deadline: independent E[V] vs scenario path V on t and t+1."""
    t = int(decision_gw)
    t1 = t + 1
    logs, odds, sheet = load_season_inputs(season)
    pots = opening_pots_for_sheet(odds, sheet)
    fx_t = fixtures_for_gw(sheet, t)
    fx_t1 = fixtures_for_gw(sheet, t1)
    if not fx_t or not fx_t1:
        raise RuntimeError(f"missing fixtures for GW{t}/{t1} in {season}")

    pool = historical_pool(logs, before_gw=t)
    centre_t = centreline_club_pots(pots, t)
    centre_t1 = centreline_club_pots(pots, t1)
    if len(centre_t) < 10 or len(centre_t1) < 10:
        raise RuntimeError(f"thin market pots GW{t}/{t1}: {len(centre_t)}/{len(centre_t1)}")

    # Independent mean layer: free XI+captain each week on centreline pots.
    scored_t = score_pool_on_club_pots(pool, centre_t)
    scored_t1 = score_pool_on_club_pots(pool, centre_t1)
    indep_t = _pick_squad_xi_captain(scored_t, "score", time_limit=time_limit)
    indep_t1 = _pick_squad_xi_captain(scored_t1, "score", time_limit=time_limit)
    v_indep = float(indep_t["objective"] + gamma * indep_t1["objective"])
    indep_ids_t = set(str(x) for x in indep_t["squad"]["player_id"])
    indep_ids_t1 = set(str(x) for x in indep_t1["squad"]["player_id"])
    indep_overlap = len(indep_ids_t & indep_ids_t1) / 15.0

    rng = random.Random(int(seed))
    strand_rows: list[dict[str, Any]] = []
    sample_pots_t: list[dict[str, dict[str, float]]] = []
    sample_pots_t1: list[dict[str, dict[str, float]]] = []

    for s_i in range(int(n_strands)):
        samp_t = sample_week_club_pots(fx_t, centre_t, rng, band=band)
        samp_t1 = sample_week_club_pots(fx_t1, centre_t1, rng, band=band)
        sample_pots_t.append(samp_t)
        sample_pots_t1.append(samp_t1)

        st = score_pool_on_club_pots(pool, samp_t)
        st1 = score_pool_on_club_pots(pool, samp_t1)

        # Path: free rebuild at t, FT bank → 1, then FT/hits at t+1.
        pick_t = _pick_squad_xi_captain(st, "score", time_limit=time_limit)
        owned = set(str(x) for x in pick_t["squad"]["player_id"])
        ft_after = advance_ft(1, 0)  # free rebuild consumes the open; bank → 1
        # After wildcard-like open, FT bank is 1 entering next GW (same as multistage).
        ft_before_t1 = 1
        pick_t1 = _pick_squad_xi_captain_ft(
            st1,
            "score",
            owned_ids=owned,
            ft=ft_before_t1,
            time_limit=time_limit,
        )
        owned_t1 = set(str(x) for x in pick_t1["squad"]["player_id"])
        stick = len(owned & owned_t1) / 15.0
        v_path = float(pick_t["objective"] + gamma * pick_t1["net_objective"])
        strand_rows.append(
            {
                "strand": s_i,
                "v_path": v_path,
                "obj_t": float(pick_t["objective"]),
                "obj_t1_net": float(pick_t1["net_objective"]),
                "transfers_t1": int(pick_t1["n_transfers"]),
                "hits_t1": int(pick_t1["hits"]),
                "stick_rate": stick,
                "captain_t": pick_t["captain_id"],
                "captain_t1": pick_t1["captain_id"],
            }
        )

    frame = pd.DataFrame(strand_rows)
    xp_delta = mean_player_xp_delta(pool, centre_t, sample_pots_t)
    xp_delta_t1 = mean_player_xp_delta(pool, centre_t1, sample_pots_t1)

    return {
        "season": season,
        "decision_gw": t,
        "next_gw": t1,
        "n_strands": int(n_strands),
        "seed": int(seed),
        "gamma": float(gamma),
        "band": float(band),
        "n_pool": int(len(pool)),
        "n_fixtures_t": len(fx_t),
        "n_fixtures_t1": len(fx_t1),
        "v_indep": v_indep,
        "indep_overlap_t_t1": indep_overlap,
        "mean_v_path": float(frame["v_path"].mean()),
        "std_v_path": float(frame["v_path"].std(ddof=0)),
        "delta_v_path_minus_indep": float(frame["v_path"].mean() - v_indep),
        "mean_transfers_t1": float(frame["transfers_t1"].mean()),
        "mean_hits_t1": float(frame["hits_t1"].mean()),
        "mean_stick_rate": float(frame["stick_rate"].mean()),
        "player_xp_delta_t": xp_delta,
        "player_xp_delta_t1": xp_delta_t1,
        "strands": strand_rows,
        "notes": (
            "Shares frozen at deadline before t. Market λ ± band sampled; "
            "no share roll from scorelines. score_xp unpublished."
        ),
    }


def run_screen(
    *,
    season: str = "2024-25",
    decision_gws: Sequence[int] = (10, 20, 28),
    n_strands: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Multi-deadline diagnostic on one closed season."""
    rows = []
    for gw in decision_gws:
        print(f"{season} GW{gw}…", flush=True)
        try:
            rows.append(
                run_decision(season, int(gw), n_strands=n_strands, seed=seed)
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic continues
            rows.append({"season": season, "decision_gw": int(gw), "error": str(exc)})

    ok = [r for r in rows if "error" not in r]
    summary = {
        "season": season,
        "decision_gws": list(decision_gws),
        "n_ok": len(ok),
        "mean_delta_v": float(np.mean([r["delta_v_path_minus_indep"] for r in ok])) if ok else float("nan"),
        "mean_stick": float(np.mean([r["mean_stick_rate"] for r in ok])) if ok else float("nan"),
        "mean_hits": float(np.mean([r["mean_hits_t1"] for r in ok])) if ok else float("nan"),
        "mean_abs_player_xp_delta_t": float(
            np.mean([r["player_xp_delta_t"]["mean_abs_player_delta"] for r in ok])
        )
        if ok
        else float("nan"),
    }
    # Advisor keep: path stickier than independent week-rebuilds by ≥0.05.
    # Mean |ΔxP| under sampling is diagnostic only — published score_xp stays
    # the centreline mean layer regardless (scenario is not a μ replacement).
    indep_overlap = float(np.mean([r["indep_overlap_t_t1"] for r in ok])) if ok else float("nan")
    path_signal = bool(ok and summary["mean_stick"] >= indep_overlap + 0.05)
    mean_layer_quiet = bool(
        ok and summary["mean_abs_player_xp_delta_t"] < 0.25
    )
    decision = "keep_scenario_advisor" if path_signal else "park_scenario_layer"
    payload = {
        "task": "short_horizon_scenario",
        "decision": decision,
        "path_signal": path_signal,
        "mean_layer_quiet": mean_layer_quiet,
        "summary": summary,
        "indep_overlap_mean": indep_overlap,
        "per_deadline": [
            {k: v for k, v in r.items() if k != "strands"} for r in rows
        ],
        "strands_by_deadline": {
            str(r.get("decision_gw")): r.get("strands") for r in ok
        },
    }
    return payload


def write_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (REPORTS / "short_horizon_scenario.md")
    s = payload["summary"]
    lines = [
        "# Short-horizon scenario layer (priced t : t+1)",
        "",
        "Advisor layer on market-priced weeks. Does **not** replace published "
        "`score_xp`. Shares frozen at the deadline; λ sampled inside ±15% of "
        "opening Shin pots; path FT between t and t+1.",
        "",
        "## Decision",
        "",
        f"**{payload['decision']}**",
        "",
        "- Keep advisor if path stick ≥ independent t∩t+1 overlap + 0.05. "
        "Published `score_xp` stays the centreline mean layer either way.",
        f"- path_signal={payload['path_signal']} · "
        f"mean_layer_quiet={payload.get('mean_layer_quiet')} "
        f"(|ΔxP|<0.25)",
        "",
        "## Summary",
        "",
        f"| season | {s['season']} |",
        f"| deadlines ok | {s['n_ok']} / {len(payload.get('per_deadline') or [])} |",
        f"| mean ΔV (path − indep) | {s['mean_delta_v']:+.3f} |",
        f"| mean stick (path) | {s['mean_stick']:.3f} |",
        f"| mean indep overlap t∩t+1 | {payload['indep_overlap_mean']:.3f} |",
        f"| mean hits t+1 | {s['mean_hits']:.2f} |",
        f"| mean |ΔxP| vs centreline (t) | {s['mean_abs_player_xp_delta_t']:.4f} |",
        "",
        "## Per deadline",
        "",
        "| GW | V_indep | mean V_path | ΔV | stick | hits | |ΔxP| |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["per_deadline"]:
        if "error" in r:
            lines.append(f"| {r['decision_gw']} | error: {r['error']} | | | | | |")
            continue
        lines.append(
            f"| {r['decision_gw']} | {r['v_indep']:.2f} | {r['mean_v_path']:.2f} | "
            f"{r['delta_v_path_minus_indep']:+.2f} | {r['mean_stick_rate']:.2f} | "
            f"{r['mean_hits_t1']:.2f} | {r['player_xp_delta_t']['mean_abs_player_delta']:.4f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "1. Small mean |ΔxP| under sampling → centreline mean layer stays the "
        "published score.",
        "2. Higher path stick vs independent week rebuilds → scenario+FT layer "
        "is the short-horizon analogue of multistage item 1b.",
        "3. Negative ΔV is expected (hit cost); not a kill by itself.",
        "4. Hand-off of strand rates into t+2 Opta cascade is deferred.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.models.short_horizon_cascade",
        "```",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    payload = run_screen()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # Drop bulky strand detail from the small json summary file.
    slim = {k: v for k, v in payload.items() if k != "strands_by_deadline"}
    (PROCESSED / "short_horizon_scenario.json").write_text(
        json.dumps(slim, indent=2, default=str) + "\n"
    )
    report = write_report(payload)
    print(json.dumps({"decision": payload["decision"], "summary": payload["summary"]}, indent=2))
    print(f"report={report}")


if __name__ == "__main__":
    main()
