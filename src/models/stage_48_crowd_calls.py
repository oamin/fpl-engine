"""Stage 48 — whether crowd flow breaks a close call.

Same eligible rows as stage 47. The crowd adjustment uses that stage's
published coefficients, fit on the other three seasons. It is not refit
on the close calls. A flip is a same-position player inside 0.25 of the
leader on ``score_xp`` whom the crowd strictly prefers. Nothing here is
written onto the score, and no transfer climb is run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.stage_47_crowd_context import (
    SEASONS,
    aggregate_player_weeks,
    attach_market,
    dedupe_market,
    eligible_mask,
    norm_element,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
STAGE_47_CSV = PROCESSED / "stage_47_crowd_context.csv"

POSITIONS = ("GKP", "DEF", "MID", "FWD")
GAP = 0.25
DECISIVE_MIN = 20
DECISIVE_TOTAL_MIN = 100
WIN_RATE_MIN = 0.60
FLIP_GAP_MIN = 0.20
SLOT_MIN = 0.030

# Published stage 47 additive fit. The holdout names the season left out.
LOCKED_FITS: dict[str, dict[str, float]] = {
    "2022-23": {
        "c": 0.13856950312742033,
        "d": 0.07785301289759985,
        "own_mean_train": 0.5360842647033507,
        "n": 7085,
    },
    "2023-24": {
        "c": 0.2027342556831876,
        "d": 0.2529124594946375,
        "own_mean_train": 0.5370540393359826,
        "n": 7330,
    },
    "2024-25": {
        "c": 0.2123620032248541,
        "d": 0.2400537994732705,
        "own_mean_train": 0.5356544697544858,
        "n": 7427,
    },
    "2025-26": {
        "c": 0.17723626628485106,
        "d": 0.2559910461672511,
        "own_mean_train": 0.535655273361355,
        "n": 7464,
    },
}


class CrowdCallError(RuntimeError):
    """The close-call count is not allowed to refit or to drop rows."""


def locked_fits_from_csv(path: Path) -> dict[str, dict[str, float]]:
    """Read the published fit and refuse a file that does not match the lock."""
    frame = pd.read_csv(path)
    found: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        season = str(row.holdout)
        found[season] = {
            "c": float(row.c),
            "d": float(row.d),
            "own_mean_train": float(row.own_mean_train),
            "n": float(row.n_test),
        }
    for season, locked in LOCKED_FITS.items():
        got = found.get(season)
        if got is None:
            raise CrowdCallError(f"{season}: stage 47 coefficient row is missing")
        for key in ("c", "d", "own_mean_train", "n"):
            if abs(got[key] - locked[key]) > 1e-9:
                raise CrowdCallError(f"{season}: {key} does not match the published fit")
    return LOCKED_FITS


def first_position(feat: pd.DataFrame) -> pd.DataFrame:
    """Position on the earliest fixture of the player-week."""
    work = pd.DataFrame(
        {
            "season": feat["season"].astype(str),
            "element": norm_element(feat["element"]),
            "gw": pd.to_numeric(feat["gw"], errors="coerce"),
            "date": feat["date"],
            "position": feat["position"].astype(str),
        }
    ).dropna(subset=["gw"])
    work["gw"] = work["gw"].astype(int)
    work = work.sort_values(["season", "element", "gw", "date"], kind="mergesort")
    return work.drop_duplicates(["season", "element", "gw"], keep="first")[
        ["season", "element", "gw", "position"]
    ]


def attach_position(sample: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    merged = sample.merge(positions, on=["season", "element", "gw"], how="left")
    ok = merged["position"].isin(POSITIONS)
    for season, idx in merged.groupby("season", sort=False).groups.items():
        rate = float(ok.loc[idx].mean()) if len(idx) else 0.0
        if rate < 1.0:
            raise CrowdCallError(f"{season}: position merge rate {rate:.4f}")
    return merged.loc[ok].copy()


def assert_stage_47_counts(sample: pd.DataFrame) -> None:
    if len(sample) != 29306:
        raise CrowdCallError(f"eligible rows {len(sample)} != 29306")
    for season, locked in LOCKED_FITS.items():
        n = int((sample["season"] == season).sum())
        if n != int(locked["n"]):
            raise CrowdCallError(f"{season}: {n} eligible rows != {int(locked['n'])}")


def _element_num(values: pd.Series) -> pd.Series:
    num = pd.to_numeric(values, errors="coerce")
    if num.isna().any():
        raise CrowdCallError("element id is not numeric")
    return num


def resolve_band(group: pd.DataFrame, c: float, d: float, own_mean: float) -> dict[str, Any]:
    """One position-week. An equal crowd adjustment is not a flip."""
    work = group.copy()
    work["element_num"] = _element_num(work["element"])
    work["crowd"] = c * work["volume_z"].astype(float) + d * (
        work["own_10"].astype(float) - own_mean
    )
    work = work.sort_values(["score_xp", "element_num"], ascending=[False, True], kind="mergesort")
    leader = work.iloc[0]
    band = work.loc[work["score_xp"] >= float(leader["score_xp"]) - GAP]
    close = len(band) >= 2
    max_crowd = float(band["crowd"].max())
    leader_crowd = float(leader["crowd"])
    flip = bool(close and max_crowd > leader_crowd)
    if flip:
        contenders = band.loc[np.isclose(band["crowd"], max_crowd, rtol=0.0, atol=1e-12)]
        contenders = contenders.sort_values("element_num", kind="mergesort")
        chosen = contenders.iloc[0]
    else:
        chosen = leader
    gap = float(chosen["total_points"]) - float(leader["total_points"]) if flip else 0.0
    return {
        "season": str(leader["season"]),
        "gw": int(leader["gw"]),
        "position": str(leader["position"]),
        "n_band": int(len(band)),
        "close": bool(close),
        "flip": flip,
        "leader": str(leader["element"]),
        "chosen": str(chosen["element"]),
        "leader_points": float(leader["total_points"]),
        "chosen_points": float(chosen["total_points"]) if flip else float(leader["total_points"]),
        "point_gap": gap,
        "decisive": bool(flip and gap != 0.0),
        "crowd_win": bool(flip and gap > 0.0),
    }


def position_calls(sample: pd.DataFrame, fits: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Every gameweek in the sample, four positions. An empty slot is not a flip."""
    rows: list[dict[str, Any]] = []
    for season in SEASONS:
        part = sample.loc[sample["season"] == season]
        fit = fits[season]
        gws = sorted(int(g) for g in part["gw"].unique())
        for gw in gws:
            week = part.loc[part["gw"] == gw]
            for position in POSITIONS:
                group = week.loc[week["position"] == position]
                if group.empty:
                    rows.append(
                        {
                            "season": season,
                            "gw": gw,
                            "position": position,
                            "n_band": 0,
                            "close": False,
                            "flip": False,
                            "leader": "",
                            "chosen": "",
                            "leader_points": 0.0,
                            "chosen_points": 0.0,
                            "point_gap": 0.0,
                            "decisive": False,
                            "crowd_win": False,
                        }
                    )
                    continue
                rows.append(resolve_band(group, fit["c"], fit["d"], fit["own_mean_train"]))
    return pd.DataFrame(rows)


def summarise(calls: pd.DataFrame) -> list[dict[str, Any]]:
    seasons: list[dict[str, Any]] = []
    for season in SEASONS:
        part = calls.loc[calls["season"] == season]
        flips = part.loc[part["flip"]]
        decisive = int(part["decisive"].sum())
        wins = int(part["crowd_win"].sum())
        n_flips = int(len(flips))
        gap_sum = float(flips["point_gap"].sum()) if n_flips else 0.0
        n_pos = int(len(part))
        seasons.append(
            {
                "season": season,
                "n_pos_weeks": n_pos,
                "n_close": int(part["close"].sum()),
                "n_flips": n_flips,
                "n_decisive": decisive,
                "win_rate": (wins / decisive) if decisive else float("nan"),
                "mean_flip_gap": (gap_sum / n_flips) if n_flips else float("nan"),
                "slot_gain": gap_sum / n_pos if n_pos else float("nan"),
                "n_wins": wins,
            }
        )
    return seasons


def stays_open(seasons: list[dict[str, Any]]) -> bool:
    """Open only when every season clears the count, the rate, and both point bars."""
    if len(seasons) != len(SEASONS):
        raise CrowdCallError("the reading needs four seasons")
    if sum(int(row["n_decisive"]) for row in seasons) < DECISIVE_TOTAL_MIN:
        return False
    for row in seasons:
        win_rate = float(row["win_rate"])
        mean_gap = float(row["mean_flip_gap"])
        slot = float(row["slot_gain"])
        if int(row["n_decisive"]) < DECISIVE_MIN:
            return False
        if not np.isfinite(win_rate) or win_rate < WIN_RATE_MIN:
            return False
        if not np.isfinite(mean_gap) or mean_gap < FLIP_GAP_MIN:
            return False
        if not np.isfinite(slot) or slot < SLOT_MIN:
            return False
    return True


def write_report(path: Path, result: dict[str, Any], review: str | None = None) -> None:
    word = "stays open" if result["stays_open"] else "is closed"
    lines = [
        "# Stage 48 — crowd close calls",
        "",
        "Within one position and one gameweek, the leader is the highest `score_xp`. "
        "The band is everyone within 0.25 of that leader. The crowd adjustment is the "
        "stage 47 flow and ownership slopes from the other three seasons. A flip is "
        "a different player in the band whom that adjustment strictly prefers. "
        "An equal adjustment is not a flip. `score_xp` is not changed, and no "
        "transfer climb is run.",
        "",
        "The question stays open only when every season has at least 20 decisive "
        "flips, the four seasons together have at least 100, the crowd side scores "
        "more on at least 60% of decisive flips, the mean gap on flips is at least "
        "0.20, and the gaps average at least 0.03 per slot-week. A tie on points "
        "counts as zero in the gap and stays out of the win rate.",
        "",
        "## Counts",
        "",
        "| season | slot-weeks | close | flips | decisive | win rate | mean gap | per slot-week |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["seasons"]:
        lines.append(
            f"| {row['season']} | {row['n_pos_weeks']} | {row['n_close']} | {row['n_flips']} | "
            f"{row['n_decisive']} | {row['win_rate']:.3f} | {row['mean_flip_gap']:+.3f} | "
            f"{row['slot_gain']:+.4f} |"
        )
    lines += [
        "",
        f"Decisive flips in total: **{result['n_decisive_total']}**. The question {word}.",
        "",
        "`score_xp` stays the published score.",
        "",
    ]
    if review:
        lines += [review, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    from src.models.ridge_multiseason import CACHE, SEASONS as SEASON_CODES, build_one_season

    fits = locked_fits_from_csv(STAGE_47_CSV)
    frames: list[pd.DataFrame] = []
    markets: list[pd.DataFrame] = []
    for season, code in SEASON_CODES:
        if season not in SEASONS:
            continue
        frames.append(build_one_season(season, code))
        raw = pd.read_csv(CACHE / f"merged_gw_{season.replace('-', '_')}.csv")
        markets.append(dedupe_market(raw, season))
    feat = pd.concat(frames, ignore_index=True)
    weeks = aggregate_player_weeks(feat)
    eligible = weeks.loc[eligible_mask(weeks)].copy()
    sample, rates = attach_market(eligible, pd.concat(markets, ignore_index=True))
    for season, rate in rates.items():
        if float(rate) < 1.0:
            raise CrowdCallError(f"{season}: transfer merge rate {rate:.4f}")
    assert_stage_47_counts(sample)
    sample = attach_position(sample, first_position(feat))
    calls = position_calls(sample, fits)
    seasons = summarise(calls)
    result = {
        "seasons": seasons,
        "n_decisive_total": int(sum(row["n_decisive"] for row in seasons)),
        "stays_open": stays_open(seasons),
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seasons).to_csv(PROCESSED / "stage_48_crowd_calls.csv", index=False)
    calls.to_csv(PROCESSED / "stage_48_crowd_calls_weeks.csv", index=False)
    (PROCESSED / "stage_48_crowd_calls.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "seasons"}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(REPORTS / "stage_48_crowd_calls.md", result)
    return result


if __name__ == "__main__":
    run()
