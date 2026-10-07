"""Fast eleven whose early priors come from the previous season only.

The historical builder fills a missing Gameweek 1 prior with the position
mean of the season being scored, including later weeks. This path does
what the live week does: a matched player keeps last season's shifted
prior, and a debutant takes the position mean of those matched first
rows. The score formula is unchanged.

Cold start is kept only if at least two of 2023/24, 2024/25, and 2025/26
clear both bars. Gameweeks 1–8 must trail Gameweeks 9–38 by at least 2
points a week, and Gameweeks 5–8 must trail the published eleven by at
least 1 point a week. The points are the eleven plus the captain, which
is the fast-eleven total.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.fpl_odds import join_players_to_fixtures, load_football_data, load_player_logs
from src.live.benchmark import stamp_unmatched_priors
from src.models.ridge_multiseason import CACHE, SEASONS, _attach_value_defcon, build_one_season
from src.models.season_climb import pick_xi
from src.models.xp_engine import (
    DEFCON_THRESH,
    MIN_HISTORY,
    MIN_MINUTES,
    add_market_pots,
    add_player_priors,
    compute_xp,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PAIRS: list[tuple[str, str, str]] = [
    ("2023-24", "2324", "2022-23"),
    ("2024-25", "2425", "2023-24"),
    ("2025-26", "2526", "2024-25"),
]
PRIOR_GW_SHIFT = 100
EARLY_GAP = 2.0
PUBLISHED_GAP = 1.0
EARLY_GWS = list(range(1, 9))
LATE_GWS = list(range(9, 39))
MID_GWS = list(range(5, 9))


def link_id(
    element: str,
    code: int | None,
    code_to_current: dict[int, str],
    current: str,
    prior: str,
) -> str:
    """Opta ``code`` joins a prior row onto this season. A miss stays prior."""
    if code is not None and code in code_to_current:
        return f"{current}:{code_to_current[code]}"
    return f"{prior}:{element}"


def season_kept(early_gap: float, published_gap: float) -> bool:
    """Both bars, on the captain-included weekly means."""
    if not np.isfinite(early_gap) or not np.isfinite(published_gap):
        return False
    return float(early_gap) >= EARLY_GAP and float(published_gap) >= PUBLISHED_GAP


def hypothesis_kept(flags: list[bool]) -> bool:
    """At least two of the three locked seasons."""
    return sum(bool(flag) for flag in flags) >= 2


def _codes(season: str) -> tuple[dict[str, int], dict[int, str]]:
    raw = pd.read_csv(
        CACHE / f"players_raw_{season.replace('-', '_')}.csv",
        usecols=["id", "code"],
    )
    by_element: dict[str, int] = {}
    by_code: dict[int, str] = {}
    elements = raw["id"].astype(str)
    codes = pd.to_numeric(raw["code"], errors="coerce")
    for element, code in zip(elements, codes, strict=True):
        if pd.isna(code):
            continue
        by_element[element] = int(code)
        by_code[int(code)] = element
    return by_element, by_code


def _joined(season: str, fd_code: str) -> pd.DataFrame:
    """One season of odds-joined rows, before any prior is filled."""
    fixtures = load_football_data(code=fd_code)
    players = load_player_logs(season=season)
    players = _attach_value_defcon(players, season)
    joined, _, stats = join_players_to_fixtures(players, fixtures)
    print(
        f"  {season}: joined={stats['n_player_joined']} rate={stats['join_rate']:.2f}",
        flush=True,
    )
    out = joined.copy()
    out["element"] = out["player_id"].astype(str)
    out["xG"] = pd.to_numeric(out["xG"], errors="coerce").fillna(0.0)
    out["xA"] = pd.to_numeric(out["xA"], errors="coerce").fillna(0.0)
    out["p_not_lose"] = pd.to_numeric(out["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        out["p_draw"], errors="coerce"
    )
    out["p_over"] = pd.to_numeric(out["p_over25"], errors="coerce")
    out["p_under"] = pd.to_numeric(out["p_under25"], errors="coerce")
    thr = out["position"].map(DEFCON_THRESH)
    out["defcon_hit"] = (
        out["position"].isin(DEFCON_THRESH)
        & (pd.to_numeric(out["minutes"], errors="coerce").fillna(0) >= MIN_MINUTES)
        & (pd.to_numeric(out["defcon_raw"], errors="coerce").fillna(0) >= thr.fillna(999))
    ).astype(float)
    return out


def _fd_code(season: str) -> str:
    for name, code in SEASONS:
        if name == season:
            return code
    raise KeyError(season)


def build_clean_season(season: str, fd_code: str, prior_season: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Current season scored with the previous season as the only fill."""
    current = _joined(season, fd_code)
    prior = _joined(prior_season, _fd_code(prior_season))
    prior_codes, current_by_code = _codes(prior_season)[0], _codes(season)[1]
    linked_ids = [
        link_id(
            element,
            prior_codes.get(element),
            current_by_code,
            season,
            prior_season,
        )
        for element in prior["element"].astype(str)
    ]
    prior = prior.copy()
    prior["player_id"] = linked_ids
    prior["gw"] = pd.to_numeric(prior["gw"], errors="coerce").astype(int) - PRIOR_GW_SHIFT
    current = current.copy()
    current["player_id"] = season + ":" + current["element"].astype(str)
    n_linked = int(prior["player_id"].astype(str).str.startswith(f"{season}:").sum())
    combined = pd.concat([prior, current], ignore_index=True, sort=False)
    scored = add_market_pots(combined)
    scored = add_player_priors(scored, fill_from=prior)
    scored, n_debut = stamp_unmatched_priors(scored)
    scored = compute_xp(scored)
    feat = scored.loc[scored["gw"].between(1, 38)].copy()
    feat = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()
    feat["score_xp"] = feat["xp"]
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    info = {"rows_linked": n_linked, "debuts": int(n_debut), "rows": int(len(feat))}
    return feat, info


def week_points(feat: pd.DataFrame, gw: int) -> dict[str, float] | None:
    """Captain-included points for the eligible eleven. None if it cannot form."""
    gw_df = feat.loc[(feat["gw"] == int(gw)) & feat["eligible"]].copy()
    if gw_df["position"].nunique() < 4:
        return None
    try:
        selected, form = pick_xi(gw_df, "score_xp")
    except RuntimeError:
        return None
    cap_idx = selected["score_xp"].idxmax()
    points = float(pd.to_numeric(selected["total_points"], errors="coerce").fillna(0).sum())
    captain = float(pd.to_numeric(selected.loc[cap_idx, "total_points"], errors="coerce"))
    return {
        "gw": float(gw),
        "xi_points": points,
        "xi_points_cap": points + captain,
        "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
    }


def _mean(rows: list[dict[str, float]], gws: list[int]) -> tuple[float, int]:
    chosen = [row for row in rows if int(row["gw"]) in gws]
    if not chosen:
        return float("nan"), 0
    return float(sum(row["xi_points_cap"] for row in chosen) / len(chosen)), len(chosen)


def published_weeks(season: str, fd_code: str) -> list[dict[str, float]]:
    """The fast eleven already published: within-season fill, Gameweeks 5–8."""
    feat = build_one_season(season, fd_code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    rows: list[dict[str, float]] = []
    for gw in MID_GWS:
        row = week_points(feat, gw)
        if row is not None:
            row["arm"] = "published"
            rows.append(row)
    return rows


def summarise_season(
    season: str,
    clean_rows: list[dict[str, float]],
    published_rows: list[dict[str, float]],
    info: dict[str, int],
) -> dict[str, float | str | bool]:
    early, n_early = _mean(clean_rows, EARLY_GWS)
    late, n_late = _mean(clean_rows, LATE_GWS)
    clean_mid, n_clean = _mean(clean_rows, MID_GWS)
    published, n_published = _mean(published_rows, MID_GWS)
    early_gap = late - early
    published_gap = published - clean_mid
    return {
        "season": season,
        "early_mean": early,
        "late_mean": late,
        "early_gap": early_gap,
        "clean_mid": clean_mid,
        "published_mid": published,
        "published_gap": published_gap,
        "n_early": float(n_early),
        "n_late": float(n_late),
        "n_clean_mid": float(n_clean),
        "n_published": float(n_published),
        "rows_linked": float(info["rows_linked"]),
        "debuts": float(info["debuts"]),
        "kept": season_kept(early_gap, published_gap),
    }


def _write(frame: pd.DataFrame, kept: bool) -> None:
    lines = [
        "# Clean fill",
        "",
        "The eleven is the fast eleven: no squad is carried, and the total "
        "is the eleven plus the captain. The clean arm fills a missing prior "
        "from the previous season only. A matched player keeps that season's "
        "shifted prior. A debutant takes the position mean of the matched "
        "players' first rows. The published arm is the within-season fill "
        "already used for the fast eleven, on Gameweeks 5–8 only.",
        "",
        f"A season is kept when Gameweeks 1–8 trail Gameweeks 9–38 by at least "
        f"{EARLY_GAP:.0f} points a week and Gameweeks 5–8 trail the published "
        f"eleven by at least {PUBLISHED_GAP:.0f} point a week. The hypothesis "
        "is kept when at least two seasons clear both.",
        "",
        "| season | GW1–8 | GW9–38 | early gap | clean GW5–8 | published GW5–8 | published gap | weeks | kept |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.early_mean:.2f} | {row.late_mean:.2f} | "
            f"{row.early_gap:.2f} | {row.clean_mid:.2f} | {row.published_mid:.2f} | "
            f"{row.published_gap:.2f} | "
            f"{row.n_early:.0f}/{row.n_late:.0f}/{row.n_clean_mid:.0f}/{row.n_published:.0f} | "
            f"{'yes' if row.kept else 'no'} |"
        )
    if kept:
        call = "At least two seasons cleared both bars. Cold start is kept as a measurement. The score is unchanged."
    else:
        call = "Fewer than two seasons cleared both bars. Cold start is rejected. The score is unchanged."
    lines += ["", call, ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "clean_fill.md").write_text("\n".join(lines), encoding="utf-8")


def run_season(season: str, fd_code: str, prior_season: str) -> dict[str, float | str | bool]:
    feat, info = build_clean_season(season, fd_code, prior_season)
    clean_rows: list[dict[str, float]] = []
    for gw in EARLY_GWS + LATE_GWS:
        row = week_points(feat, gw)
        if row is not None:
            row["arm"] = "clean"
            row["season"] = season
            clean_rows.append(row)
    published = published_weeks(season, fd_code)
    for row in published:
        row["season"] = season
    summary = summarise_season(season, clean_rows, published, info)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    code = fd_code
    pd.DataFrame([summary]).to_csv(PROCESSED / f"clean_fill_{code}.csv", index=False)
    pd.DataFrame(clean_rows + published).to_csv(
        PROCESSED / f"clean_fill_weeks_{code}.csv", index=False
    )
    return summary


def run() -> dict[str, object]:
    rows = [run_season(*pair) for pair in PAIRS]
    frame = pd.DataFrame(rows)
    kept = hypothesis_kept([bool(row["kept"]) for row in rows])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "clean_fill.csv", index=False)
    _write(frame, kept)
    return {"kept": kept, "seasons": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="")
    args = parser.parse_args()
    if args.season:
        matched = [pair for pair in PAIRS if pair[0] == args.season]
        if len(matched) != 1:
            raise SystemExit(f"unknown season {args.season}")
        print(run_season(*matched[0]), flush=True)
        return
    print(run(), flush=True)


if __name__ == "__main__":
    main()
