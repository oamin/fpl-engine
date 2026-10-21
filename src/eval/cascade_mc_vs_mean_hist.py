"""One closed-season diagnostic: path-frequency MILP vs priced mean MILP.

Protocol: ``cascade_mc_vs_mean_hist`` (diagnostic — strategy emergence, not gated).

Path arm (N strands, ``xi_plus_captain_ft``):
  - Short weeks with FD openings: centreline AH+OU → ``M`` → ``k∼M``
  - Longer weeks: **oracle post-GW38 ranks** + **LOSO prior-season GF/GA ± bands**
    (not live Opta JSON)
  - Generative player stack; aggregate pick / XI / captain frequencies

Mean arm (single MILP):
  - Priced ``t:t+1`` mean boards only (FD centreline ``score_team_side``)
  - One ``xi_plus_captain`` solve on discounted GW(t)+γ·GW(t+1)

Not comparable to live Opta path — assesses whether a stable strategy
emerges vs the raw mean MILP.

  PYTHONPATH=. python3 -m src.eval.cascade_mc_vs_mean_hist
  PYTHONPATH=. python3 -m src.eval.cascade_mc_vs_mean_hist --season 2025-26 --n 12
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

from src.markets import numagent_xp
from src.markets.cascade_score_v2 import rolling_m_sigma
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_GAMMA,
    DEFAULT_LINEUP_COV,
    OBJECTIVE_XI_CAPTAIN_FT,
    _pick_squad_xi_captain,
    aggregate_frequencies,
    init_paths_from_opta,
    run_strand,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.selectable_pool import expanding_mean_minutes_from_history
from src.models.short_horizon_cascade import (
    attach_prices_and_cap,
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    load_prices,
    score_week_on_fixture_moments,
)
from src.rules.fpl_2026 import normalize_position
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mc_vs_mean_hist.md"
OUT_DIR = ROOT / "data/predictions/2025-26/cascade_mc_vs_mean_hist"
STATE_JSON = ROOT / "data/processed/cascade_mc_vs_mean_hist" / "state.json"

SEASON_E0 = {
    "2022-23": "E0_2223.csv",
    "2023-24": "E0_2324.csv",
    "2024-25": "E0_2425.csv",
    "2025-26": "E0_2526.csv",
}

DEFAULT_SEASON = "2025-26"
DECISION_GW = 6
HORIZON_END = 19
START_OFFSET = 2
SNAPSHOT_EVERY = 2
N_STRANDS = 12
SEED = 7
MIN_MINUTES = 60.0
MIN_APPS = 2.0
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def _hist_fixtures_by_gw(
    enriched: pd.DataFrame, *, start: int, end: int
) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for gw in range(int(start), int(end) + 1):
        rows: list[dict[str, str]] = []
        for fx in fixtures_for_gw(enriched, gw):
            rows.append(
                {
                    "home": norm_team(str(fx["home_n"])),
                    "away": norm_team(str(fx["away_n"])),
                }
            )
        if rows:
            out[int(gw)] = rows
    return out


def _fd_priced_by_gw(
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    gws: list[int],
) -> dict[int, list[dict[str, Any]]]:
    priced: dict[int, list[dict[str, Any]]] = {}
    for gw in gws:
        fits = centreline_fixture_fits(fixtures_for_gw(enriched, gw), fd)
        if not fits:
            continue
        priced[int(gw)] = [
            {
                "gw": int(gw),
                "home": norm_team(str(f["home_n"])),
                "away": norm_team(str(f["away_n"])),
                "lam_h": float(f["lam_h"]),
                "lam_a": float(f["lam_a"]),
                "rho": float(f.get("rho") or 0.0),
                "home_props": {},
                "away_props": {},
                "n_props_matched": 0,
                "source": "cascade_short_fd_hist",
            }
            for f in fits
        ]
    return priced


def _build_hist_pool(
    hist: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    decision_gw: int,
    season: str,
) -> pd.DataFrame:
    """Minutes>60 as-of pool with frozen rates/priors + decision-GW prices."""
    rows = hist.to_dict(orient="records")
    mins = expanding_mean_minutes_from_history(rows, before_gw=int(decision_gw))
    keep = {
        pid
        for pid, m in mins.items()
        if float(m["mean_minutes"]) > MIN_MINUTES and float(m["n_apps"]) >= MIN_APPS
    }
    if len(keep) < 15:
        raise RuntimeError(f"hist pool too small: {len(keep)}")

    side, _, _ = build_frozen_side_table(
        hist, alpha_g=1.0, alpha_a=1.0, season=season, historical_rate_baseline=True
    )
    if side.empty:
        raise RuntimeError("empty frozen side table")
    side = side.loc[side["element"].astype(str).isin(keep)].copy()

    pr = prices.loc[prices["gw"] == int(decision_gw)].drop_duplicates("element")
    meta = (
        hist.groupby("element", as_index=False)
        .agg(
            name=("name", "first"),
            position=("position", "first"),
            team=("team", "first"),
        )
        if "name" in hist.columns
        else hist.groupby("element", as_index=False).agg(
            position=("position", "first"),
            team=("team", "first"),
        )
    )
    if "name" not in meta.columns:
        meta["name"] = meta["element"].astype(str)

    frame = meta.merge(pr[["element", "value", "player_id", "team_norm"]], on="element", how="inner")
    frame = frame.loc[frame["element"].astype(str).isin(keep)].copy()
    frame["player_id"] = frame["player_id"].fillna(frame["element"]).astype(str)
    frame["team_norm"] = frame["team_norm"].fillna(frame["team"].map(norm_team)).map(
        lambda t: norm_team(str(t))
    )
    frame["club"] = frame["team"]
    frame["team"] = frame["team"].astype(str)

    # Rates / priors from side table
    rate_cols = [
        c
        for c in side.columns
        if c
        in {
            "element",
            "rG",
            "drG",
            "rA",
            "drA",
            "rate_source",
            "rate_minutes",
            "bonus_mu",
            "bonus_se",
            "card_mu",
            "card_se",
            "saves_mu",
            "saves_se",
            "defcon_mu",
            "defcon_sig",
            "defcon_n",
        }
    ]
    frame = frame.merge(side[rate_cols], on="element", how="left")
    ms = []
    for eid in frame["element"].astype(int):
        m, sig = rolling_m_sigma(hist, int(eid))
        ms.append((m, sig))
    frame["m"] = [min(90.0, max(0.0, m)) for m, _ in ms]
    frame["sigma_m"] = [sig for _, sig in ms]
    frame["xmi"] = frame["m"]
    frame["fwd_goal_scale"] = 1.0
    frame["eligible"] = True
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.2),
        ("bonus_se", 0.1),
        ("card_mu", -0.1),
        ("card_se", 0.05),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", numagent_xp.SIG_FLOOR),
        ("defcon_n", 5),
    ):
        if c not in frame.columns:
            frame[c] = default
        else:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(default)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").fillna(50).astype(int)
    frame["position"] = frame["position"].map(
        lambda p: normalize_position(str(p)) or str(p)
    )
    # MILP quota keys are GKP/DEF/MID/FWD
    frame.loc[frame["position"] == "GK", "position"] = "GKP"
    return frame.reset_index(drop=True)


def _mean_board(
    side: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    season: str,
    decision_gw: int,
    gw: int,
    alpha_g: float,
    alpha_a: float,
    pool_ids: set[str],
) -> pd.DataFrame:
    fits = centreline_fixture_fits(fixtures_for_gw(enriched, gw), fd)
    if not fits:
        return pd.DataFrame()
    board = score_week_on_fixture_moments(
        side, fits, alpha_g=alpha_g, alpha_a=alpha_a
    )
    if board.empty:
        return board
    return attach_prices_and_cap(
        board,
        prices,
        season=season,
        decision_gw=decision_gw,
        selectable_ids=pool_ids,
        apply_player_pool=False,
    )


def run_path_arm(
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
    strands = []
    t0 = _now()
    for i in range(int(n_strands)):
        print(f"  [path] strand {i + 1}/{n_strands}", flush=True)
        strand_rng = random.Random(rng.randint(0, 2**31 - 1))
        strands.append(
            run_strand(
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
        )
    freq = aggregate_frequencies(strands, pool)
    sources = sorted(
        {
            str(s.get("source"))
            for st in strands
            for s in st.scorelines
            if s.get("source")
        }
    )
    return {
        "elapsed_s": _now() - t0,
        "n_strands": int(n_strands),
        "n_squads_solved": int(freq.attrs.get("n_squads") or 0),
        "strands_with_squads": sum(1 for s in strands if s.squads),
        "mean_transfers": freq.attrs.get("mean_transfers"),
        "mean_hits": freq.attrs.get("mean_hits"),
        "frequencies": freq.to_dict("records"),
        "sources": sources,
        "sample_scorelines": strands[0].scorelines[:16] if strands else [],
        "sample_squads": strands[0].squads[:3] if strands and strands[0].squads else [],
    }


def run_mean_arm(
    *,
    side: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    season: str,
    decision_gw: int,
    priced_gws: list[int],
    alpha_g: float,
    alpha_a: float,
    pool_ids: set[str],
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, Any]:
    """Single xi+captain MILP on discounted mean boards over priced t:t+1."""
    t0 = _now()
    if len(priced_gws) < 1:
        raise RuntimeError("no priced GWs for mean arm")
    boards: list[pd.DataFrame] = []
    for gw in priced_gws[:2]:
        b = _mean_board(
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
        if b.empty:
            raise RuntimeError(f"empty mean board GW{gw}")
        boards.append(b)

    # Discounted sum on decision-week universe (GW0 + γ·GW1 when present).
    base = boards[0].copy()
    base["mc_score"] = pd.to_numeric(base["score_xp_v2"], errors="coerce").fillna(0.0)
    if len(boards) > 1:
        b1 = boards[1].set_index(
            pd.to_numeric(boards[1]["element"], errors="coerce")
        )
        add = []
        for eid in pd.to_numeric(base["element"], errors="coerce"):
            if eid in b1.index:
                row = b1.loc[eid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                add.append(float(row["score_xp_v2"]) * float(gamma))
            else:
                add.append(0.0)
        base["mc_score"] = base["mc_score"] + np.asarray(add, float)

    # Align team_norm for MILP club cap (no NaNs — breaks club-cap sort).
    def _club_key(row: pd.Series) -> str:
        for key in ("team_norm", "club", "team"):
            if key in row.index and pd.notna(row[key]):
                s = str(row[key]).strip()
                if s and s.lower() != "nan":
                    return norm_team(s)
        return "unk"

    base["team_norm"] = [_club_key(row) for _, row in base.iterrows()]
    if "name" not in base.columns:
        base["name"] = base["player_id"].astype(str)
    if "position" in base.columns:
        base["position"] = base["position"].map(
            lambda p: normalize_position(str(p)) if pd.notna(p) else "MID"
        )
    picked = _pick_squad_xi_captain(base, "mc_score", time_limit=20.0)
    squad = picked["squad"]
    return {
        "elapsed_s": _now() - t0,
        "priced_gws": list(priced_gws[:2]),
        "objective": "xi_plus_captain",
        "objective_value": float(picked["objective"]),
        "captain_id": picked["captain_id"],
        "xi_ids": list(picked["xi_ids"]),
        "squad": [
            {
                "player_id": str(r.player_id),
                "name": str(r.name),
                "position": str(r.position),
                "team_norm": str(getattr(r, "team_norm", "")),
                "mc_score": float(r.mc_score),
                "in_xi": str(r.player_id) in set(picked["xi_ids"]),
                "captain": str(r.player_id) == str(picked["captain_id"]),
            }
            for r in squad.itertuples(index=False)
        ],
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return float(len(a & b) / len(u)) if u else 0.0


def compare(path: dict[str, Any], mean: dict[str, Any], *, top_n: int = 15) -> dict[str, Any]:
    freqs = path.get("frequencies") or []
    top = freqs[: int(top_n)]
    path_ids = {str(r["player_id"]) for r in top}
    mean_ids = {str(r["player_id"]) for r in mean.get("squad") or []}
    mean_xi = {str(x) for x in mean.get("xi_ids") or []}
    path_cap = None
    if freqs:
        by_cap = sorted(freqs, key=lambda r: float(r.get("freq_captain") or 0), reverse=True)
        if by_cap and float(by_cap[0].get("freq_captain") or 0) > 0:
            path_cap = {
                "player_id": str(by_cap[0]["player_id"]),
                "name": by_cap[0].get("name"),
                "freq_captain": float(by_cap[0]["freq_captain"]),
            }
    mean_in_top = sorted(path_ids & mean_ids)
    mean_names = {
        str(r["player_id"]): str(r.get("name") or r["player_id"])
        for r in mean.get("squad") or []
    }
    return {
        "top_n": int(top_n),
        "jaccard_top_vs_mean15": _jaccard(path_ids, mean_ids),
        "n_mean_in_path_top": len(mean_in_top),
        "mean_in_path_top": [
            {"player_id": pid, "name": mean_names.get(pid, pid)} for pid in mean_in_top
        ],
        "path_top_captain": path_cap,
        "mean_captain_id": mean.get("captain_id"),
        "mean_captain_in_path_top": str(mean.get("captain_id")) in path_ids,
        "mean_xi_in_path_top": sorted(mean_xi & path_ids),
    }


def write_report(state: dict[str, Any]) -> None:
    path = state["path"]
    mean = state["mean"]
    cmp = state["compare"]
    freqs = path.get("frequencies") or []
    lines = [
        "# Cascade MC frequency vs priced mean MILP — closed-season diagnostic",
        "",
        f"Protocol: `cascade_mc_vs_mean_hist` · season **{state['season']}** · "
        f"decision GW{state['decision_gw']} → horizon {state['horizon_start']}–"
        f"{state['horizon_end']} · N={state['n_strands']} · seed={state['seed']}",
        "",
        "**Diagnostic only** — strategy emergence, not a gated promote. "
        "Long arm uses **oracle post-GW38 ranks** + **LOSO prior GF/GA ± bands** "
        "(not live Opta). Not comparable to the live book/Opta season.",
        "",
        "## Design",
        "",
        "- **Path:** FD short on priced weeks → generative `k∼M`→attack; "
        "oracle-rank long for the rest; `xi_plus_captain_ft` per strand; frequencies.",
        "- **Mean:** single `xi_plus_captain` MILP on discounted FD mean boards "
        f"GW{mean['priced_gws']} only (hard stop).",
        f"- Pool: as-of minutes > {MIN_MINUTES} (n_apps≥{int(MIN_APPS)}); no FFScout.",
        f"- `lineup_cov={DEFAULT_LINEUP_COV}` (freeze as-of minutes on gen paths).",
        "- **Attack rates:** historical baseline — prior closed seasons per player "
        "(name key), else positional (`score_xp_v2_rate_baseline_prior`).",
        "",
        "## Path arm",
        "",
        f"- Squads solved: **{path.get('n_squads_solved')}** · "
        f"strands_with_squads={path.get('strands_with_squads')}",
        f"- Mean transfers / hits: {path.get('mean_transfers')} / {path.get('mean_hits')}",
        f"- Sources: {path.get('sources')}",
        f"- Elapsed: {path.get('elapsed_s'):.1f}s",
        "",
        "Top freq:",
        "",
    ]
    for r in freqs[:15]:
        lines.append(
            f"- {r.get('name')} ({r.get('team_norm')}): "
            f"freq={float(r.get('freq') or 0):.3f} "
            f"xi={float(r.get('freq_xi') or 0):.3f} "
            f"cap={float(r.get('freq_captain') or 0):.3f}"
        )
    lines += [
        "",
        "## Mean arm (priced t:t+1)",
        "",
        f"- Objective value: **{mean.get('objective_value'):.2f}**",
        f"- Captain id: `{mean.get('captain_id')}`",
        f"- Elapsed: {mean.get('elapsed_s'):.1f}s",
        "",
        "Squad:",
        "",
    ]
    for r in mean.get("squad") or []:
        tags = []
        if r.get("in_xi"):
            tags.append("XI")
        if r.get("captain"):
            tags.append("C")
        tag = f" [{','.join(tags)}]" if tags else ""
        lines.append(
            f"- {r.get('name')} ({r.get('team_norm')}){tag} · μ={float(r.get('mc_score') or 0):.2f}"
        )
    lines += [
        "",
        "## Overlap (strategy emergence)",
        "",
        f"- Jaccard path-top{cmp['top_n']} ∩ mean-15: **{cmp['jaccard_top_vs_mean15']:.3f}**",
        f"- Mean picks in path top{cmp['top_n']}: **{cmp['n_mean_in_path_top']}** / 15",
        f"- Path top captain: {cmp.get('path_top_captain')}",
        f"- Mean captain in path top: **{cmp.get('mean_captain_in_path_top')}**",
        "",
        "Mean picks that are also path-stable:",
        "",
    ]
    for r in cmp.get("mean_in_path_top") or []:
        lines.append(f"- {r.get('name')} (`{r.get('player_id')}`)")
    if not cmp.get("mean_in_path_top"):
        lines.append("- _(none)_")
    lines += [
        "",
        f"**Wall-clock:** {state.get('elapsed_s'):.1f}s",
        "",
        "## Artefacts",
        "",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{OUT_DIR.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_mc_vs_mean_hist "
        f"--season {state['season']} --n {state['n_strands']}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    print(f"[hist] season={season} e0={e0} decision={decision_gw} "
          f"horizon={start}-{horizon_end} N={args.n}", flush=True)
    t_all = _now()

    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    print(f"[hist] oracle ranks={len(ranks)} LOSO profile places={len(profile)}", flush=True)

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
    print(f"[hist] pool={len(pool)} alphas=({alpha_g:.3f},{alpha_a:.3f})", flush=True)

    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start, end=horizon_end)
    horizon_gws = sorted(fixtures_gw)
    if not horizon_gws:
        raise RuntimeError("no fixtures in horizon")
    snapshot_gws = [
        g for i, g in enumerate(horizon_gws) if i % max(1, int(args.snapshot_every)) == 0
    ]
    # Priced short = first two horizon GWs with FD fits (t:t+1 of the roll).
    priced_candidates = horizon_gws[:2]
    priced_by_gw = _fd_priced_by_gw(enriched, fd, priced_candidates)
    priced_gws = sorted(priced_by_gw)
    print(
        f"[hist] horizon_gws={horizon_gws} snapshots={snapshot_gws} "
        f"priced={priced_gws}",
        flush=True,
    )
    if len(priced_gws) < 1:
        raise RuntimeError("no FD priced fits in horizon head")

    side, _, _ = build_frozen_side_table(
        hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        season=season,
        historical_rate_baseline=True,
    )

    print("[hist] mean arm…", flush=True)
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
        f"[hist] mean done obj={mean['objective_value']:.1f} "
        f"cap={mean['captain_id']} ({mean['elapsed_s']:.1f}s)",
        flush=True,
    )

    print("[hist] path arm…", flush=True)
    path = run_path_arm(
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
        f"[hist] path done squads={path['n_squads_solved']} "
        f"hits={path.get('mean_hits')} ({path['elapsed_s']:.1f}s)",
        flush=True,
    )

    cmp = compare(path, mean)
    state = {
        "task": "cascade_mc_vs_mean_hist",
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
        "path": path,
        "mean": mean,
        "compare": cmp,
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    _save(state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "path_freq.csv").write_text(
        pd.DataFrame(path.get("frequencies") or []).to_csv(index=False)
    )
    (OUT_DIR / "mean_squad.json").write_text(
        json.dumps(mean, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "compare.json").write_text(
        json.dumps(cmp, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(state)
    print(
        json.dumps(
            {
                "elapsed_s": state["elapsed_s"],
                "compare": cmp,
                "path_squads": path.get("n_squads_solved"),
                "path_hits": path.get("mean_hits"),
                "mean_obj": mean.get("objective_value"),
                "report": str(REPORT_MD),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
