"""Captain / same-fixture stack diagnostics on fixture-covariance draws.

Pre-registered in ``protocol.fixture_planning_glue`` (locked before totals).
Not a μ likelihood gate and not a free-transfer climb.

Policies (as-of draws only):
  - captain_mu   vs captain_haul / captain_cvar
  - stack_same   vs diversify (home top + away top)

Realised: Vaastav ``total_points`` on the fixture row. Agree → fixture Δ = 0.

  python -m src.markets.fixture_planning_glue
  python -m src.markets.fixture_planning_glue --season 2025-26 --max-fixtures 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets.fixture_covariance import (
    build_club_side,
    draw_fixture_covariant,
)
from src.markets.fixture_covariance_audit import (
    _fixture_schedule,
    asof_groups_for_team,
    assert_asof_pairs_ignore_future,
)
from src.markets.minutes_covariance import load_merged, merged_path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Defaults match protocol.fixture_planning_glue (asserted at run).
MIN_HIST_GW = 8
N_SCENARIOS = 100
SEED = 0
HAUL_THRESHOLD = 8.0
CVAR_TOP_FRAC = 0.20
MIN_CAND_M = 45.0
CAPTAIN_MULT = 2
DEFAULT_LAM = (1.5, 1.2, -0.05)

# Vaastav / FD team-name bridge for λ lookup only.
FD_TEAM_ALIAS = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}


def _fd_name(team: str) -> str:
    return FD_TEAM_ALIAS.get(str(team), str(team))


def load_fd_lambda_index(season: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Map (home, away) Vaastav names → (lam_h, lam_a, rho) from FD moments CSV."""
    path = PROCESSED / f"scoreline_fd_moments_{season.replace('-', '_')}.csv"
    if not path.exists():
        return {}
    fd = pd.read_csv(path)
    # One row per side; take home-side row for fixture λ.
    home_rows = fd.loc[fd["side"] == "home"] if "side" in fd.columns else fd
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    # Build reverse alias: FD → Vaastav
    rev = {v: k for k, v in FD_TEAM_ALIAS.items()}
    for r in home_rows.itertuples(index=False):
        h_fd, a_fd = str(r.home), str(r.away)
        h = rev.get(h_fd, h_fd)
        a = rev.get(a_fd, a_fd)
        try:
            out[(h, a)] = (float(r.lam_h), float(r.lam_a), float(r.rho))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def fixture_lambda(
    home: str,
    away: str,
    index: dict[tuple[str, str], tuple[float, float, float]],
) -> tuple[float, float, float, str]:
    key = (str(home), str(away))
    if key in index:
        lh, la, rho = index[key]
        return lh, la, rho, "fd"
    # Try FD-native names
    key2 = (_fd_name(home), _fd_name(away))
    for h, a in (key, key2, (key2[0], away), (home, key2[1])):
        if (h, a) in index:
            lh, la, rho = index[(h, a)]
            return lh, la, rho, "fd"
    return (*DEFAULT_LAM, "default")


def _draw_index_by_element(
    draws, home: pd.DataFrame, away: pd.DataFrame
) -> dict[int, int]:
    """Map element id → column in draws.xp (player_ids are names from build_club_side)."""
    ids = list(draws.player_ids)
    name_to_i = {pid: i for i, pid in enumerate(ids)}
    out: dict[int, int] = {}
    for side in (home, away):
        for r in side.itertuples(index=False):
            el = int(r.element)
            name = str(r.name)
            if name in name_to_i:
                out[el] = name_to_i[name]
    return out


def attacker_candidates(side: pd.DataFrame, *, min_m: float = MIN_CAND_M) -> pd.DataFrame:
    sub = side.loc[
        side["position"].isin(["MID", "FWD"])
        & (pd.to_numeric(side["m"], errors="coerce").fillna(0.0) >= min_m)
    ].copy()
    return sub


def policy_captains(
    draws,
    el_to_i: dict[int, int],
    candidates: pd.DataFrame,
    mu_by_el: dict[int, float],
    *,
    haul_threshold: float = HAUL_THRESHOLD,
    cvar_frac: float = CVAR_TOP_FRAC,
) -> dict[str, int | None]:
    """Return element ids for mu / haul / cvar captain policies."""
    els = [int(e) for e in candidates["element"].tolist() if int(e) in el_to_i]
    if not els:
        return {"mu": None, "haul": None, "cvar": None}

    def _haul_p(el: int) -> float:
        col = draws.xp[:, el_to_i[el]]
        return float(np.mean(col >= haul_threshold))

    def _cvar(el: int) -> float:
        col = np.sort(draws.xp[:, el_to_i[el]])
        k = max(1, int(np.ceil(cvar_frac * len(col))))
        return float(col[-k:].mean())

    mu_el = max(els, key=lambda e: (mu_by_el.get(e, -1e9), -e))
    haul_el = max(els, key=lambda e: (_haul_p(e), mu_by_el.get(e, -1e9), -e))
    cvar_el = max(els, key=lambda e: (_cvar(e), mu_by_el.get(e, -1e9), -e))
    return {"mu": mu_el, "haul": haul_el, "cvar": cvar_el}


def policy_stacks(
    home: pd.DataFrame,
    away: pd.DataFrame,
    mu_by_el: dict[int, float],
    *,
    min_m: float = MIN_CAND_M,
) -> dict[str, tuple[int, int] | None]:
    """stack_same = top-2 on stronger side; diversify = top home + top away."""

    def top_two(side: pd.DataFrame) -> list[int]:
        c = attacker_candidates(side, min_m=min_m)
        if c.empty:
            return []
        ranked = sorted(
            (int(e) for e in c["element"].tolist()),
            key=lambda e: (mu_by_el.get(e, -1e9), -e),
            reverse=True,
        )
        return ranked[:2]

    h2, a2 = top_two(home), top_two(away)
    diversify = None
    if h2 and a2:
        diversify = (h2[0], a2[0])

    stack_same = None
    best_sum = -1e18
    for pair in (h2, a2):
        if len(pair) < 2:
            continue
        s = mu_by_el.get(pair[0], 0.0) + mu_by_el.get(pair[1], 0.0)
        if s > best_sum:
            best_sum = s
            stack_same = (pair[0], pair[1])

    return {"stack_same": stack_same, "diversify": diversify}


def realised_points(
    season_df: pd.DataFrame, fixture: int, element: int
) -> float | None:
    rows = season_df.loc[
        (season_df["fixture"] == fixture) & (season_df["element"] == int(element))
    ]
    if rows.empty:
        return None
    return float(pd.to_numeric(rows["total_points"], errors="coerce").iloc[0])


def evaluate_fixture(
    season_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    gw: int,
    fixture: int,
    *,
    lam_h: float,
    lam_a: float,
    rho: float,
    n_scenarios: int,
    seed: int,
    haul_threshold: float,
    cvar_frac: float,
    min_cand_m: float,
) -> dict[str, Any] | None:
    hist = season_df.loc[season_df["GW"] < int(gw)]
    if hist.empty:
        return None
    try:
        home, _ = build_club_side(hist, home_team)
        away, _ = build_club_side(hist, away_team)
    except (ValueError, KeyError):
        return None
    h_groups = asof_groups_for_team(season_df, home_team, gw)
    a_groups = asof_groups_for_team(season_df, away_team, gw)
    h_keep = set(home["element"].astype(int))
    a_keep = set(away["element"].astype(int))
    h_groups = [g for g in h_groups if set(g.elements) <= h_keep]
    a_groups = [g for g in a_groups if set(g.elements) <= a_keep]
    if home.empty or away.empty:
        return None

    draws = draw_fixture_covariant(
        lam_h,
        lam_a,
        rho,
        home,
        away,
        home_groups=h_groups,
        away_groups=a_groups,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    el_to_i = _draw_index_by_element(draws, home, away)
    # Point μ from draws.mu_point aligned to player_ids / names
    mu_by_el: dict[int, float] = {}
    ids = list(draws.player_ids)
    for side in (home, away):
        for r in side.itertuples(index=False):
            name = str(r.name)
            if name in ids:
                mu_by_el[int(r.element)] = float(draws.mu_point[ids.index(name)])

    cands = pd.concat(
        [
            attacker_candidates(home, min_m=min_cand_m),
            attacker_candidates(away, min_m=min_cand_m),
        ],
        ignore_index=True,
    )
    caps = policy_captains(
        draws,
        el_to_i,
        cands,
        mu_by_el,
        haul_threshold=haul_threshold,
        cvar_frac=cvar_frac,
    )
    stacks = policy_stacks(home, away, mu_by_el, min_m=min_cand_m)
    if caps["mu"] is None:
        return None

    def cap_pts(el: int | None) -> float | None:
        if el is None:
            return None
        y = realised_points(season_df, fixture, el)
        if y is None:
            return None
        return CAPTAIN_MULT * y

    y_mu = cap_pts(caps["mu"])
    y_haul = cap_pts(caps["haul"])
    y_cvar = cap_pts(caps["cvar"])
    if y_mu is None or y_haul is None or y_cvar is None:
        return None

    d_haul = float(y_haul - y_mu)
    d_cvar = float(y_cvar - y_mu)

    stack_pts = None
    div_pts = None
    d_stack = None
    if stacks["stack_same"] is not None and stacks["diversify"] is not None:
        a, b = stacks["stack_same"]
        c, d = stacks["diversify"]
        ya, yb = realised_points(season_df, fixture, a), realised_points(
            season_df, fixture, b
        )
        yc, yd = realised_points(season_df, fixture, c), realised_points(
            season_df, fixture, d
        )
        if None not in (ya, yb, yc, yd):
            stack_pts = float(ya + yb)  # type: ignore[operator]
            div_pts = float(yc + yd)  # type: ignore[operator]
            d_stack = float(stack_pts - div_pts)

    return {
        "fixture": int(fixture),
        "gw": int(gw),
        "home_team": home_team,
        "away_team": away_team,
        "captain_mu": caps["mu"],
        "captain_haul": caps["haul"],
        "captain_cvar": caps["cvar"],
        "captain_disagree_haul": int(caps["haul"] != caps["mu"]),
        "captain_disagree_cvar": int(caps["cvar"] != caps["mu"]),
        "y_captain_mu": y_mu,
        "y_captain_haul": y_haul,
        "y_captain_cvar": y_cvar,
        "d_captain_haul_minus_mu": d_haul,
        "d_captain_cvar_minus_mu": d_cvar,
        "stack_same": stacks["stack_same"],
        "diversify": stacks["diversify"],
        "y_stack_same": stack_pts,
        "y_diversify": div_pts,
        "d_stack_same_minus_diversify": d_stack,
        "n_candidates": int(len(cands)),
        "n_scenarios": n_scenarios,
    }


def walk_season(
    season: str,
    *,
    n_scenarios: int = N_SCENARIOS,
    seed: int = SEED,
    min_hist_gw: int = MIN_HIST_GW,
    max_fixtures: int | None = None,
    haul_threshold: float = HAUL_THRESHOLD,
    cvar_frac: float = CVAR_TOP_FRAC,
    min_cand_m: float = MIN_CAND_M,
) -> dict[str, Any]:
    path = merged_path(season)
    if not path.exists():
        return {"season": season, "skipped": True, "reason": "missing_merged"}
    season_df = load_merged(path)
    if "total_points" not in season_df.columns:
        return {"season": season, "skipped": True, "reason": "no_total_points"}

    teams = sorted(season_df["team"].astype(str).unique())[:5]
    asof_ok = all(
        assert_asof_pairs_ignore_future(season_df, t, min_hist_gw + 2) for t in teams
    )

    lam_index = load_fd_lambda_index(season)
    sched = _fixture_schedule(season_df)
    sched = sched.loc[sched["GW"] >= min_hist_gw].reset_index(drop=True)
    if max_fixtures is not None and len(sched) > max_fixtures:
        idx = np.linspace(0, len(sched) - 1, max_fixtures).astype(int)
        sched = sched.iloc[idx].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n_fd = 0
    for row in sched.itertuples(index=False):
        lh, la, rho, src = fixture_lambda(
            str(row.home_team), str(row.away_team), lam_index
        )
        if src == "fd":
            n_fd += 1
        ev = evaluate_fixture(
            season_df,
            str(row.home_team),
            str(row.away_team),
            int(row.GW),
            int(row.fixture),
            lam_h=lh,
            lam_a=la,
            rho=rho,
            n_scenarios=n_scenarios,
            seed=int(rng.integers(0, 1_000_000)),
            haul_threshold=haul_threshold,
            cvar_frac=cvar_frac,
            min_cand_m=min_cand_m,
        )
        if ev is None:
            continue
        ev["lam_source"] = src
        ev["season"] = season
        rows.append(ev)

    fx = pd.DataFrame(rows)
    gw_rows = []
    if not fx.empty:
        for gw, g in fx.groupby("gw"):
            stack_col = g["d_stack_same_minus_diversify"]
            gw_rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "n_fixtures": int(len(g)),
                    "frac_disagree_haul": float(g["captain_disagree_haul"].mean()),
                    "frac_disagree_cvar": float(g["captain_disagree_cvar"].mean()),
                    "d_captain_haul_minus_mu": float(
                        g["d_captain_haul_minus_mu"].mean()
                    ),
                    "d_captain_cvar_minus_mu": float(
                        g["d_captain_cvar_minus_mu"].mean()
                    ),
                    "d_stack_same_minus_diversify": float(
                        stack_col.dropna().mean()
                    )
                    if stack_col.notna().any()
                    else 0.0,
                }
            )
    gw_df = pd.DataFrame(gw_rows)
    return {
        "season": season,
        "skipped": False,
        "asof_passed": bool(asof_ok),
        "n_fixtures_scored": int(len(fx)),
        "n_fixtures_fd_lambda": int(n_fd),
        "n_gws": int(len(gw_df)),
        "frac_disagree_haul": float(fx["captain_disagree_haul"].mean())
        if len(fx)
        else None,
        "frac_disagree_cvar": float(fx["captain_disagree_cvar"].mean())
        if len(fx)
        else None,
        "fixtures": fx,
        "by_gw": gw_df,
    }


def _gw_deltas(
    by_season: dict[str, pd.DataFrame], col: str
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        g = by_season.get(season)
        if g is None or g.empty or col not in g.columns:
            out[season] = np.asarray([], dtype=float)
        else:
            out[season] = g[col].to_numpy(float)
    return out


def _certified_protocol_comparison() -> dict[str, Any]:
    certified = json.loads(
        (PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8")
    )
    key = comparison_key("score_xp", "score_exp_points")
    return {key: certified["comparisons"][key]}


def assert_protocol_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    screen = protocol.get("fixture_planning_glue") or {}
    if not screen:
        raise RuntimeError("protocol.fixture_planning_glue missing")
    if int(screen.get("min_hist_gw", -1)) != MIN_HIST_GW:
        raise RuntimeError("fixture_planning_glue.min_hist_gw lock mismatch")
    if int(screen.get("n_scenarios", -1)) != N_SCENARIOS:
        raise RuntimeError("fixture_planning_glue.n_scenarios lock mismatch")
    if float(screen.get("haul_threshold_pts", -1)) != HAUL_THRESHOLD:
        raise RuntimeError("fixture_planning_glue.haul_threshold_pts lock mismatch")
    if float(screen.get("cvar_top_frac", -1)) != CVAR_TOP_FRAC:
        raise RuntimeError("fixture_planning_glue.cvar_top_frac lock mismatch")
    if screen.get("milp") is not False or screen.get("climb") is not False:
        raise RuntimeError("fixture_planning_glue must keep milp/climb false")
    return screen


def write_report(
    doc: dict[str, Any],
    intervals: dict[str, Any],
    path: Path,
) -> None:
    comps = intervals["comparisons"]
    lines = [
        "# Fixture planning glue — captain / stack diagnostics",
        "",
        "Pre-registered decision screen on fixture-covariance draws "
        "(`protocol.fixture_planning_glue`). Not a μ likelihood gate; not a climb.",
        "",
        f"- Min history GW: **{doc['defs']['min_hist_gw']}**",
        f"- Scenarios / fixture: **{doc['defs']['n_scenarios']}**",
        f"- Haul threshold: **≥ {doc['defs']['haul_threshold_pts']:.0f}** pts",
        f"- CVaR: mean of top **{doc['defs']['cvar_top_frac']:.0%}** draw scenarios",
        f"- Captain multiplier: **×{doc['defs']['captain_mult']}**",
        f"- As-of poison checks: **{'PASS' if doc['asof_all_passed'] else 'FAIL'}**",
        "",
        "## Contrasts (GW-mean Δ, cluster interval)",
        "",
        "| contrast | mean | lo | hi | covers 0? |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key in (
        "captain_haul_minus_captain_mu",
        "captain_cvar_minus_captain_mu",
        "stack_same_minus_diversify",
    ):
        row = comps[key]
        covers = row["lo"] <= 0.0 <= row["hi"]
        lines.append(
            f"| `{key}` | {row['mean']:+.3f} | {row['lo']:+.3f} | {row['hi']:+.3f} | "
            f"{'yes' if covers else 'no'} |"
        )

    lines += [
        "",
        "## By season",
        "",
        "| season | as-of | fixtures | GWs | frac haul≠μ | frac CVaR≠μ | "
        "mean Δ haul−μ | mean Δ CVaR−μ | mean Δ stack−div |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in doc["seasons"]:
        if s.get("skipped"):
            lines.append(f"| {s['season']} | — | skip | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {s['season']} | {'PASS' if s['asof_passed'] else 'FAIL'} | "
            f"{s['n_fixtures_scored']} | {s['n_gws']} | "
            f"{s['frac_disagree_haul']:.0%} | {s['frac_disagree_cvar']:.0%} | "
            f"{s['mean_d_haul']:+.3f} | {s['mean_d_cvar']:+.3f} | "
            f"{s['mean_d_stack']:+.3f} |"
        )

    haul = comps["captain_haul_minus_captain_mu"]
    cvar = comps["captain_cvar_minus_captain_mu"]
    stack = comps["stack_same_minus_diversify"]
    haul_win = haul["lo"] > 0
    cvar_win = cvar["lo"] > 0
    stack_win = stack["lo"] > 0
    haul_lose = haul["hi"] < 0
    cvar_lose = cvar["hi"] < 0
    stack_lose = stack["hi"] < 0

    verdict_bits = []
    if haul_win:
        verdict_bits.append("captain_haul beats μ (interval above 0)")
    elif haul_lose:
        verdict_bits.append("captain_haul loses to μ (interval below 0)")
    else:
        verdict_bits.append("captain_haul vs μ inconclusive (covers 0)")
    if cvar_win:
        verdict_bits.append("captain_cvar beats μ")
    elif cvar_lose:
        verdict_bits.append("captain_cvar loses to μ")
    else:
        verdict_bits.append("captain_cvar vs μ inconclusive")
    if stack_win:
        verdict_bits.append("same-fixture stack beats diversify")
    elif stack_lose:
        verdict_bits.append("diversify beats same-fixture stack")
    else:
        verdict_bits.append("stack vs diversify inconclusive")

    if haul_win or cvar_win:
        next_step = (
            "Draw-based captain signal clears μ on at least one arm — "
            "keep as a planning diagnostic; do not fold into MILP μ yet."
        )
    elif haul_lose and cvar_lose:
        next_step = (
            "Draw captain arms lose to μ — park captain override; "
            "μ remains the captain rule for Layer 1–3."
        )
    else:
        next_step = (
            "No clear captain override — μ stays default; "
            "optional live shadow only. Proceed to Layer 4 horizon when ready."
        )

    lines += [
        "",
        "## Verdict",
        "",
        doc["verdict_prefix"] + " " + "; ".join(verdict_bits) + ".",
        "",
        next_step,
        "",
        f"JSON: `data/processed/fixture_planning_glue.json`",
    ]
    doc["verdict"] = " ".join(verdict_bits) + ". " + next_step
    write_gated_report(path, doc["audit"], intervals, lines)


def run(
    *,
    seasons: tuple[str, ...] | None = None,
    n_scenarios: int = N_SCENARIOS,
    seed: int = SEED,
    max_fixtures: int | None = None,
) -> dict[str, Any]:
    protocol = load_protocol()
    screen = assert_protocol_lock(protocol)
    seasons = seasons or CLOSED_SEASONS

    audit = run_asof_audit()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))

    season_docs = []
    by_gw: dict[str, pd.DataFrame] = {}
    for i, season in enumerate(seasons):
        print(f"planning-glue {season}...", flush=True)
        doc = walk_season(
            season,
            n_scenarios=n_scenarios,
            seed=seed + 31 * i,
            min_hist_gw=int(screen["min_hist_gw"]),
            max_fixtures=max_fixtures,
            haul_threshold=float(screen["haul_threshold_pts"]),
            cvar_frac=float(screen["cvar_top_frac"]),
            min_cand_m=MIN_CAND_M,
        )
        if not doc.get("skipped"):
            gw = doc["by_gw"]
            by_gw[season] = gw
            doc["mean_d_haul"] = (
                float(gw["d_captain_haul_minus_mu"].mean()) if len(gw) else 0.0
            )
            doc["mean_d_cvar"] = (
                float(gw["d_captain_cvar_minus_mu"].mean()) if len(gw) else 0.0
            )
            doc["mean_d_stack"] = (
                float(gw["d_stack_same_minus_diversify"].mean()) if len(gw) else 0.0
            )
            # Drop bulky frames from JSON summary
            fixtures_n = len(doc.pop("fixtures"))
            doc["n_fixture_rows"] = fixtures_n
            doc.pop("by_gw")
        season_docs.append(doc)

    asof_all = all(s.get("asof_passed") for s in season_docs if not s.get("skipped"))
    if not asof_all:
        raise RuntimeError("fixture planning as-of poison checks failed")

    n_boot = int(protocol.get("bootstrap", 1000))
    intervals_raw = {
        "captain_haul_minus_captain_mu": cluster_interval(
            _gw_deltas(by_gw, "d_captain_haul_minus_mu"),
            seasons=CLOSED_SEASONS,
            n_boot=n_boot,
            seed=int(screen.get("seed", 0)),
        ),
        "captain_cvar_minus_captain_mu": cluster_interval(
            _gw_deltas(by_gw, "d_captain_cvar_minus_mu"),
            seasons=CLOSED_SEASONS,
            n_boot=n_boot,
            seed=int(screen.get("seed", 0)) + 1,
        ),
        "stack_same_minus_diversify": cluster_interval(
            _gw_deltas(by_gw, "d_stack_same_minus_diversify"),
            seasons=CLOSED_SEASONS,
            n_boot=n_boot,
            seed=int(screen.get("seed", 0)) + 2,
        ),
    }
    for v in intervals_raw.values():
        v["incomplete_seasons"] = {}

    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol.get("min_gws_per_season", 20)),
        "comparisons": {
            **_certified_protocol_comparison(),
            **intervals_raw,
        },
    }

    out_doc = {
        "defs": {
            "min_hist_gw": int(screen["min_hist_gw"]),
            "n_scenarios": n_scenarios,
            "haul_threshold_pts": float(screen["haul_threshold_pts"]),
            "cvar_top_frac": float(screen["cvar_top_frac"]),
            "captain_mult": CAPTAIN_MULT,
            "min_cand_m": MIN_CAND_M,
            "max_fixtures": max_fixtures,
            "protocol_locked": screen.get("locked"),
        },
        "asof_all_passed": asof_all,
        "audit": {"passed": True, "failures": [], "engine_asof": audit},
        "seasons": season_docs,
        "verdict_prefix": "Planning-glue screen complete.",
        "intervals": {k: intervals_raw[k] for k in intervals_raw},
    }

    out_json = PROCESSED / "fixture_planning_glue.json"
    out_md = REPORTS / "fixture_planning_glue.md"
    # Persist GW series for reproducibility
    gw_all = pd.concat([by_gw[s] for s in CLOSED_SEASONS if s in by_gw], ignore_index=True)
    gw_all.to_csv(PROCESSED / "fixture_planning_glue_gw.csv", index=False)
    out_json.write_text(json.dumps(out_doc, indent=2, default=str), encoding="utf-8")
    write_report(out_doc, intervals, out_md)
    print(
        json.dumps(
            {
                "asof": asof_all,
                "intervals": {
                    k: {
                        "mean": v["mean"],
                        "lo": v["lo"],
                        "hi": v["hi"],
                    }
                    for k, v in intervals_raw.items()
                },
                "verdict": out_doc.get("verdict"),
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return out_doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", action="append", dest="seasons", default=None)
    p.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--max-fixtures",
        type=int,
        default=None,
        help="Optional stratified subsample per season (debug / smoke)",
    )
    args = p.parse_args(argv)
    seasons = tuple(args.seasons) if args.seasons else None
    locked_full = seasons is None and args.max_fixtures is None
    if locked_full and args.n_scenarios != N_SCENARIOS:
        raise SystemExit(
            f"locked full run requires n_scenarios={N_SCENARIOS} "
            f"(got {args.n_scenarios}); pass --max-fixtures for a debug subsample"
        )
    if seasons is not None and set(seasons) != set(CLOSED_SEASONS):
        raise SystemExit(
            "gated report needs all four closed seasons; omit --season for the locked run"
        )
    run(
        seasons=None,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        max_fixtures=args.max_fixtures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
