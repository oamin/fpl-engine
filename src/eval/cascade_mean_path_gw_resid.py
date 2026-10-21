"""Per-GW residual: mean MILP XI vs stable cascade path XI (one closed season).

Protocol: ``cascade_mean_path_gw_resid`` (diagnostic).

For each horizon GW after a single decision:
  - **Mean:** locked ``xi_plus_captain`` from priced t:t+1 mean boards (held).
  - **Path (stable):** strand-mean of held path XIs from ``xi_plus_captain_ft``
    (N strands; squad updates only at snapshots, held between).

At each GW report predicted XI points (FD centreline ``score_xp_v2`` on that
week's board), actual FPL points (planned XI + captain, no autosub), and
residual = pred − actual.

  PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid
  PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid --season 2025-26 --n 12
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
    DECISION_GW,
    DEFAULT_SEASON,
    HORIZON_END,
    N_STRANDS,
    SEASON_E0,
    SEED,
    SNAPSHOT_EVERY,
    START_OFFSET,
    _build_hist_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
    _mean_board,
    run_mean_arm,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_GAMMA,
    DEFAULT_LINEUP_COV,
    OBJECTIVE_XI_CAPTAIN_FT,
    init_paths_from_opta,
    run_strand,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import build_frozen_side_table, load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mean_path_gw_resid.md"
STATE_JSON = ROOT / "data/processed/cascade_mean_path_gw_resid" / "state.json"


def _out_dir(season: str) -> Path:
    return ROOT / "data" / "predictions" / str(season) / "cascade_mean_path_gw_resid"


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def _element_from_pid(pid: str) -> int | None:
    text = str(pid)
    if ":" in text:
        text = text.split(":")[-1]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _points_by_element(merged_gw: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    if merged_gw is None or merged_gw.empty:
        return out
    for row in merged_gw.itertuples(index=False):
        try:
            eid = int(row.element)
        except (TypeError, ValueError):
            continue
        out[eid] = float(getattr(row, "total_points", 0.0) or 0.0)
    return out


def _xi_actual(pts: dict[int, float], xi_ids: list[str], captain_id: str | None) -> float:
    total = 0.0
    for pid in xi_ids:
        eid = _element_from_pid(pid)
        if eid is not None:
            total += float(pts.get(eid, 0.0))
    if captain_id:
        ce = _element_from_pid(captain_id)
        if ce is not None:
            total += float(pts.get(ce, 0.0))
    return float(total)


def _mu_by_element(board: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    if board is None or board.empty:
        return out
    col = "score_xp_v2" if "score_xp_v2" in board.columns else "mu_post"
    for row in board.itertuples(index=False):
        eid = None
        if hasattr(row, "element") and pd.notna(row.element):
            try:
                eid = int(row.element)
            except (TypeError, ValueError):
                eid = None
        if eid is None and hasattr(row, "player_id"):
            eid = _element_from_pid(str(row.player_id))
        if eid is None:
            continue
        out[eid] = float(getattr(row, col, 0.0) or 0.0)
    return out


def _xi_pred(mu: dict[int, float], xi_ids: list[str], captain_id: str | None) -> float:
    total = 0.0
    for pid in xi_ids:
        eid = _element_from_pid(pid)
        if eid is not None:
            total += float(mu.get(eid, 0.0))
    if captain_id:
        ce = _element_from_pid(captain_id)
        if ce is not None:
            total += float(mu.get(ce, 0.0))
    return float(total)


def _held_lineups(
    strand_squads: list[dict[str, Any]], horizon_gws: list[int]
) -> dict[int, dict[str, Any]]:
    """Map each horizon GW → held XI/captain from the latest snapshot ≤ GW."""
    by_snap = {int(s["gw"]): s for s in strand_squads if s.get("xi_ids")}
    snaps = sorted(by_snap)
    out: dict[int, dict[str, Any]] = {}
    if not snaps:
        return out
    si = 0
    cur = None
    for gw in horizon_gws:
        while si < len(snaps) and snaps[si] <= int(gw):
            cur = by_snap[snaps[si]]
            si += 1
        if cur is None:
            continue
        out[int(gw)] = {
            "xi_ids": list(cur["xi_ids"]),
            "captain_id": cur.get("captain_id"),
            "snapshot_gw": int(cur["gw"]),
        }
    return out


def run_path_strands(
    *,
    pool: pd.DataFrame,
    fixtures_gw: dict[int, list[dict[str, str]]],
    priced_by_gw: dict[int, list[dict[str, Any]]],
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    hist_by_team: dict[str, pd.DataFrame],
    alpha_g: float,
    alpha_a: float,
    horizon_gws: list[int],
    snapshot_gws: list[int],
    n_strands: int,
    seed: int,
) -> dict[str, Any]:
    paths0 = init_paths_from_opta(ranks, profile=profile)
    rng = random.Random(int(seed))
    strands: list[dict[str, Any]] = []
    t0 = _now()
    for i in range(int(n_strands)):
        print(f"  [path] strand {i + 1}/{n_strands}", flush=True)
        strand_rng = random.Random(rng.randint(0, 2**31 - 1))
        res = run_strand(
            strand_id=i,
            fixtures_gw=fixtures_gw,
            snapshot_gws=snapshot_gws,
            horizon_gws=horizon_gws,
            pool=pool,
            paths0=paths0,
            league_gf=league_gf,
            rng=strand_rng,
            gamma=DEFAULT_GAMMA,
            objective=OBJECTIVE_XI_CAPTAIN_FT,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            hist_by_team=hist_by_team,
            generative_attack=True,
            lineup_cov=DEFAULT_LINEUP_COV,
            priced_by_gw=priced_by_gw,
            ranks=ranks,
            profile=profile,
        )
        strands.append(
            {
                "strand_id": i,
                "squads": list(res.squads),
                "held": _held_lineups(list(res.squads), horizon_gws),
            }
        )
    return {
        "elapsed_s": _now() - t0,
        "n_strands": int(n_strands),
        "strands_with_squads": sum(1 for s in strands if s["squads"]),
        "strands": strands,
    }


def per_gw_residuals(
    *,
    mean: dict[str, Any],
    path: dict[str, Any],
    side: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    merged: pd.DataFrame,
    season: str,
    decision_gw: int,
    horizon_gws: list[int],
    alpha_g: float,
    alpha_a: float,
    pool_ids: set[str],
) -> list[dict[str, Any]]:
    mean_xi = list(mean.get("xi_ids") or [])
    mean_cap = mean.get("captain_id")
    rows: list[dict[str, Any]] = []
    for gw in horizon_gws:
        board = _mean_board(
            side,
            enriched,
            fd,
            prices,
            season=season,
            decision_gw=decision_gw,
            gw=int(gw),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            pool_ids=pool_ids,
        )
        mu = _mu_by_element(board)
        gw_rows = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
        pts = _points_by_element(gw_rows)

        mean_pred = _xi_pred(mu, mean_xi, mean_cap)
        mean_act = _xi_actual(pts, mean_xi, mean_cap)

        path_preds: list[float] = []
        path_acts: list[float] = []
        for st in path.get("strands") or []:
            held = (st.get("held") or {}).get(int(gw))
            if not held:
                continue
            xi = list(held["xi_ids"])
            cap = held.get("captain_id")
            path_preds.append(_xi_pred(mu, xi, cap))
            path_acts.append(_xi_actual(pts, xi, cap))
        if not path_preds:
            continue
        path_pred = float(np.mean(path_preds))
        path_act = float(np.mean(path_acts))
        rows.append(
            {
                "gw": int(gw),
                "mean_pred": mean_pred,
                "mean_actual": mean_act,
                "mean_resid": mean_pred - mean_act,
                "path_pred": path_pred,
                "path_actual": path_act,
                "path_resid": path_pred - path_act,
                "delta_actual_path_minus_mean": path_act - mean_act,
                "delta_resid_path_minus_mean": (path_pred - path_act)
                - (mean_pred - mean_act),
                "n_path_strands": len(path_preds),
                "n_board": int(len(board)),
            }
        )
    return rows


def write_report(state: dict[str, Any], out_dir: Path) -> None:
    rows = state.get("per_gw") or []
    lines = [
        "# Mean vs stable cascade path — per-GW residual (actual)",
        "",
        f"Protocol: `cascade_mean_path_gw_resid` · season **{state['season']}** · "
        f"decision GW{state['decision_gw']} → horizon "
        f"{state['horizon_start']}–{state['horizon_end']} · "
        f"N={state['n_strands']} · seed={state['seed']}",
        "",
        "**Diagnostic.** Mean XI is locked from priced t:t+1 MILP and held. "
        "Path actual/pred are **strand means** of held `xi_plus_captain_ft` "
        "lineups (stable cascade path). Pred = FD centreline `score_xp_v2` on "
        "that GW's board for the planned XI (+ captain). Actual = Vaastav "
        "`total_points` on the planned XI (+ captain), no autosub. "
        "Residual = pred − actual.",
        "",
        f"- As-of freeze: GW `< {state['decision_gw']}` (rates use historical baseline)",
        f"- Snapshots: {state['snapshot_gws']}",
        f"- Priced mean weeks: {state['priced_gws']}",
        f"- Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Per GW",
        "",
        "| GW | mean pred | mean actual | mean resid | path pred | path actual | path resid | path−mean actual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['gw']} | {r['mean_pred']:.1f} | {r['mean_actual']:.1f} | "
            f"{r['mean_resid']:.1f} | {r['path_pred']:.1f} | {r['path_actual']:.1f} | "
            f"{r['path_resid']:.1f} | {r['delta_actual_path_minus_mean']:.1f} |"
        )
    if rows:
        m_resid = float(np.mean([r["mean_resid"] for r in rows]))
        p_resid = float(np.mean([r["path_resid"] for r in rows]))
        m_mae = float(np.mean([abs(r["mean_resid"]) for r in rows]))
        p_mae = float(np.mean([abs(r["path_resid"]) for r in rows]))
        m_act = float(np.sum([r["mean_actual"] for r in rows]))
        p_act = float(np.sum([r["path_actual"] for r in rows]))
        lines += [
            "",
            "## Horizon summary",
            "",
            f"- Mean residual (bias): **{m_resid:.2f}** · MAE **{m_mae:.2f}** · "
            f"sum actual **{m_act:.1f}**",
            f"- Path residual (bias): **{p_resid:.2f}** · MAE **{p_mae:.2f}** · "
            f"sum actual **{p_act:.1f}**",
            f"- Path − mean sum actual: **{p_act - m_act:.1f}**",
            "",
        ]
    lines += [
        "## Artefacts",
        "",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{out_dir.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_path_gw_resid "
        f"--season {state['season']} --n {state['n_strands']}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--decision-gw", type=int, default=DECISION_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}; choose from {sorted(SEASON_E0)}")
    e0 = SEASON_E0[season]
    decision_gw = int(args.decision_gw)
    horizon_end = int(args.horizon_end)
    start = decision_gw + START_OFFSET

    print(
        f"[resid] season={season} decision={decision_gw} "
        f"horizon={start}-{horizon_end} N={args.n}",
        flush=True,
    )
    t_all = _now()

    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)

    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    hist = numagent_xp.asof_share_history(
        merged, before_gw=decision_gw, season=season
    )
    if hist.empty:
        raise RuntimeError(f"no history before GW{decision_gw}")
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)

    pool = _build_hist_pool(hist, prices, decision_gw=decision_gw, season=season)
    pool_ids = set(pool["player_id"].astype(str))
    print(f"[resid] pool={len(pool)}", flush=True)

    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start, end=horizon_end)
    horizon_gws = sorted(fixtures_gw)
    snapshot_gws = [
        g for i, g in enumerate(horizon_gws) if i % max(1, int(args.snapshot_every)) == 0
    ]
    priced_candidates = horizon_gws[:2]
    priced_by_gw = _fd_priced_by_gw(enriched, fd, priced_candidates)
    priced_gws = sorted(priced_by_gw)
    if len(priced_gws) < 1:
        raise RuntimeError("no FD priced fits in horizon head")

    side, _, _ = build_frozen_side_table(
        hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        season=season,
        historical_rate_baseline=True,
    )

    print("[resid] mean arm…", flush=True)
    mean = run_mean_arm(
        side=side,
        enriched=enriched,
        fd=fd,
        prices=prices,
        season=season,
        decision_gw=decision_gw,
        priced_gws=priced_gws,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        pool_ids=pool_ids,
    )
    print(
        f"[resid] mean done cap={mean['captain_id']} ({mean['elapsed_s']:.1f}s)",
        flush=True,
    )

    print("[resid] path arm…", flush=True)
    path = run_path_strands(
        pool=pool,
        fixtures_gw=fixtures_gw,
        priced_by_gw=priced_by_gw,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        hist_by_team=team_hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        horizon_gws=horizon_gws,
        snapshot_gws=snapshot_gws,
        n_strands=int(args.n),
        seed=int(args.seed),
    )
    print(
        f"[resid] path done strands_with_squads={path['strands_with_squads']} "
        f"({path['elapsed_s']:.1f}s)",
        flush=True,
    )

    print("[resid] scoring per-GW residuals…", flush=True)
    per_gw = per_gw_residuals(
        mean=mean,
        path=path,
        side=side,
        enriched=enriched,
        fd=fd,
        prices=prices,
        merged=merged,
        season=season,
        decision_gw=decision_gw,
        horizon_gws=horizon_gws,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        pool_ids=pool_ids,
    )

    # Slim path for state (drop full held maps duplication — keep squads).
    path_slim = {
        "elapsed_s": path["elapsed_s"],
        "n_strands": path["n_strands"],
        "strands_with_squads": path["strands_with_squads"],
        "strands": [
            {"strand_id": s["strand_id"], "squads": s["squads"]}
            for s in path["strands"]
        ],
    }

    summary = {}
    if per_gw:
        summary = {
            "n_gws": len(per_gw),
            "mean_resid_bias": float(np.mean([r["mean_resid"] for r in per_gw])),
            "path_resid_bias": float(np.mean([r["path_resid"] for r in per_gw])),
            "mean_resid_mae": float(np.mean([abs(r["mean_resid"]) for r in per_gw])),
            "path_resid_mae": float(np.mean([abs(r["path_resid"]) for r in per_gw])),
            "mean_actual_sum": float(np.sum([r["mean_actual"] for r in per_gw])),
            "path_actual_sum": float(np.sum([r["path_actual"] for r in per_gw])),
            "path_minus_mean_actual_sum": float(
                np.sum([r["delta_actual_path_minus_mean"] for r in per_gw])
            ),
        }

    state = {
        "task": "cascade_mean_path_gw_resid",
        "season": season,
        "e0": e0,
        "decision_gw": decision_gw,
        "horizon_start": start,
        "horizon_end": horizon_end,
        "n_strands": int(args.n),
        "seed": int(args.seed),
        "snapshot_gws": snapshot_gws,
        "priced_gws": priced_gws,
        "n_pool": int(len(pool)),
        "ranks": "oracle_post_gw38",
        "gf_ga_profile": "loso_prior_seasons",
        "mean": {
            "elapsed_s": mean.get("elapsed_s"),
            "captain_id": mean.get("captain_id"),
            "xi_ids": mean.get("xi_ids"),
            "objective_value": mean.get("objective_value"),
        },
        "path": path_slim,
        "per_gw": per_gw,
        "summary": summary,
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    out_dir = _out_dir(season)
    _save(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_gw).to_csv(out_dir / "per_gw.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "mean_squad.json").write_text(
        json.dumps(mean, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(state, out_dir)
    print(json.dumps({"elapsed_s": state["elapsed_s"], "summary": summary}, indent=2))
    print(f"report={REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
