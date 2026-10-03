"""Stage 45 — seven repairs to the published score, fast XI only.

Locked with Gemini before the totals were read. Each arm is a player
score. The fast XI is the best eleven each week, with no squad carried
forward. A score that finishes 100 or more points behind ``score_xp`` on
any season is killed. A free-transfer climb is allowed only when a score
is ahead on every season, and only on 2025/26, against the +34 bar.
Nothing in this batch replaces ``score_xp``.

The defensive-contribution term is the model's own expectation. A
positive mean before 2025/26 is the phantom the table is there to show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb import pick_xi
from src.models.stage_29_batch import fast_xi

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))
KILL_GAP = 100.0
REQUIRED = (
    "score_xp",
    "xp_defcon",
    "xp_cs",
    "xp_appear",
    "xmi",
    "xp_goals",
    "xp_gc_loss",
    "position",
)
ARMS = (
    "xp",
    "no_defcon",
    "no_cs",
    "cs_half",
    "appear_linear",
    "gk6",
    "gc_half",
    "pos_shift",
)


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def repair_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The seven locked scores, plus the published score under ``xp``."""
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise RuntimeError(f"score repair is missing {missing}")
    xp = _num(df, "score_xp")
    defcon = _num(df, "xp_defcon")
    cs = _num(df, "xp_cs")
    appear = _num(df, "xp_appear")
    xmi = _num(df, "xmi")
    goals = _num(df, "xp_goals")
    gc = _num(df, "xp_gc_loss")
    pos = df["position"].astype(str)
    gk_cut = np.where(pos.eq("GKP"), 0.472 * goals, 0.0)
    shift = np.where(pos.eq("DEF"), -0.25, np.where(pos.eq("FWD"), 0.25, 0.0))
    linear = 2.0 * (xmi / 90.0).clip(lower=0.0, upper=1.0)
    return {
        "xp": xp,
        "no_defcon": xp - defcon,
        "no_cs": xp - 1.08 * cs,
        "cs_half": xp - 0.54 * cs,
        "appear_linear": xp - appear + linear,
        "gk6": xp - gk_cut,
        "gc_half": xp + 0.5 * gc,
        "pos_shift": xp + shift,
    }


def attach_repairs(feat: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = feat.copy()
    scores = repair_scores(out)
    cols = {}
    for name, series in scores.items():
        column = "score_xp" if name == "xp" else f"screen__{name}"
        if name != "xp":
            out[column] = series.to_numpy()
        cols[name] = column
    return out, cols


def phantom_table(feat: pd.DataFrame, season: str) -> pd.DataFrame:
    """Mean model defensive-contribution points on the buy pool."""
    if "xp_defcon" not in feat.columns:
        raise RuntimeError("phantom table is missing xp_defcon")
    pool = feat.loc[feat["gw"].isin(GWS) & feat["eligible"].astype(bool)].copy()
    pool["xp_defcon"] = _num(pool, "xp_defcon")
    rows = []
    for position, block in pool.groupby(pool["position"].astype(str), sort=True):
        rows.append(
            {
                "season": season,
                "position": position,
                "n": int(len(block)),
                "mean_xp_defcon": float(block["xp_defcon"].mean()) if len(block) else float("nan"),
            }
        )
    rows.append(
        {
            "season": season,
            "position": "ALL",
            "n": int(len(pool)),
            "mean_xp_defcon": float(pool["xp_defcon"].mean()) if len(pool) else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def _captain_points(sel: pd.DataFrame, column: str) -> float:
    rank = pd.to_numeric(sel[column], errors="coerce").fillna(-1e18)
    points = pd.to_numeric(sel["total_points"], errors="coerce").fillna(0.0)
    return float(points.loc[rank.idxmax()])


def captain_and_shapes(feat: pd.DataFrame, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Captain extra and formation tally on the published fast XI only."""
    captains = []
    shapes = []
    for gw in GWS:
        gw_df = feat.loc[(feat["gw"] == gw) & feat["eligible"].astype(bool)]
        if gw_df["position"].nunique() < 4:
            continue
        sel, form = pick_xi(gw_df, "score_xp")
        captains.append(
            {
                "season": season,
                "gw": int(gw),
                "xp": _captain_points(sel, "score_xp"),
                "goals": _captain_points(sel, "xp_goals"),
                "oracle": _captain_points(sel, "total_points"),
            }
        )
        shapes.append(
            {
                "season": season,
                "gw": int(gw),
                "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                "xi_points_cap": float(pd.to_numeric(sel["total_points"], errors="coerce").fillna(0.0).sum())
                + _captain_points(sel, "score_xp"),
            }
        )
    return pd.DataFrame(captains), pd.DataFrame(shapes)


def _prepare(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    out["eligible"] = pd.to_numeric(out["xmi"], errors="coerce").fillna(0.0) >= 45.0
    return out


def run_season(season: str, code: str) -> dict[str, pd.DataFrame]:
    feat = _prepare(build_one_season(season, code))
    scored, cols = attach_repairs(feat)
    weekly = fast_xi(scored, GWS, cols)
    weekly.insert(0, "season", season)
    captains, shapes = captain_and_shapes(scored, season)
    print(f"{season}: weeks={weekly['gw'].nunique()}", flush=True)
    return {
        "weekly": weekly,
        "phantom": phantom_table(scored, season),
        "captains": captains,
        "shapes": shapes,
    }


def _totals(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, block in weekly.groupby("season", sort=False):
        sums = block.groupby("method", sort=False)["xi_points_cap"].sum()
        base = float(sums["xp"])
        for name in ARMS:
            total = float(sums.get(name, np.nan))
            rows.append(
                {
                    "season": season,
                    "arm": name,
                    "xi_points": total,
                    "delta_vs_xp": total - base,
                }
            )
    return pd.DataFrame(rows)


def _killed(totals: pd.DataFrame) -> pd.DataFrame:
    arms = totals.loc[totals["arm"] != "xp"]
    worst = arms.groupby("arm", sort=False)["delta_vs_xp"].min()
    ahead = arms.groupby("arm", sort=False)["delta_vs_xp"].apply(lambda s: bool((s > 0).all()))
    out = pd.DataFrame({"worst_delta": worst, "ahead_all": ahead})
    out["killed"] = out["worst_delta"] <= -KILL_GAP
    out["ft_queued"] = out["ahead_all"] & ~out["killed"]
    return out.reset_index()


def _write(
    totals: pd.DataFrame,
    verdict: pd.DataFrame,
    phantom: pd.DataFrame,
    captains: pd.DataFrame,
    shapes: pd.DataFrame,
) -> None:
    lines = [
        "# Stage 45 — score repairs, fast XI",
        "",
        "Seven scores, locked before these totals. Gameweeks 5–38. The fast XI "
        "does not carry a squad. A score 100 or more behind `score_xp` on any "
        "season is killed. Ahead on every season is the only route to a "
        "2025/26 free-transfer climb, and the pass bar there is +34. "
        "`score_xp` stays the published score.",
        "",
        "The 0.25 on defenders and forwards was read off the 2025/26 calibration. "
        "That season does not confirm `pos_shift`.",
        "",
        "| arm | 2022/23 | 2023/24 | 2024/25 | 2025/26 | worst | result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    order = [s for s, _ in SEASONS]
    wide = totals.pivot(index="arm", columns="season", values="delta_vs_xp")
    points = totals.pivot(index="arm", columns="season", values="xi_points")
    base = points.loc["xp"]
    for arm in ARMS:
        if arm == "xp":
            cells = " | ".join(f"{base[s]:.0f}" for s in order)
            lines.append(f"| xp | {cells} | — | baseline |")
            continue
        row = verdict.loc[verdict["arm"] == arm].iloc[0]
        cells = " | ".join(f"{wide.loc[arm, s]:+.0f}" for s in order)
        mark = "killed" if row["killed"] else ("climb" if row["ft_queued"] else "alive")
        lines.append(f"| {arm} | {cells} | {row['worst_delta']:+.0f} | {mark} |")
    lines.extend(
        [
            "",
            "Deltas are captained XI points minus the published fast XI.",
            "",
            "## Defensive contributions",
            "",
            "Mean `xp_defcon` on the buy pool. A positive number before 2025/26 "
            "is the model charging an award the official points did not pay.",
            "",
            "| season | position | n | mean |",
            "|---|---|---:|---:|",
        ]
    )
    for row in phantom.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.position} | {row.n} | {row.mean_xp_defcon:.3f} |"
        )
    lines.extend(["", "## Captain extra, published XI", ""])
    lines.append("| season | score_xp | goals | oracle | goals minus score | oracle minus score |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for season, block in captains.groupby("season", sort=False):
        xp = float(block["xp"].sum())
        goals = float(block["goals"].sum())
        oracle = float(block["oracle"].sum())
        lines.append(
            f"| {season} | {xp:.0f} | {goals:.0f} | {oracle:.0f} | "
            f"{goals - xp:+.0f} | {oracle - xp:+.0f} |"
        )
    gk = totals.loc[totals["arm"] == "gk6", "delta_vs_xp"]
    lines.extend(
        [
            "",
            f"Goalkeeper goals rescaled to 6 move the fast XI by "
            f"{gk.abs().max():.1f} points in the largest season. A move past "
            "1 point would mean the cut reached an outfield player.",
            "",
            "## Published shapes",
            "",
            "| season | shape | weeks | points |",
            "|---|---|---:|---:|",
        ]
    )
    for (season, shape), block in shapes.groupby(["season", "formation"], sort=True):
        lines.append(
            f"| {season} | {shape} | {len(block)} | {block['xi_points_cap'].sum():.0f} |"
        )
    lines.append("")
    (REPORTS / "stage_45_score_repair.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> pd.DataFrame:
    weekly_parts = []
    phantom_parts = []
    captain_parts = []
    shape_parts = []
    for season, code in SEASONS:
        print(f"building {season}", flush=True)
        part = run_season(season, code)
        weekly_parts.append(part["weekly"])
        phantom_parts.append(part["phantom"])
        captain_parts.append(part["captains"])
        shape_parts.append(part["shapes"])
    weekly = pd.concat(weekly_parts, ignore_index=True)
    totals = _totals(weekly)
    verdict = _killed(totals)
    phantom = pd.concat(phantom_parts, ignore_index=True)
    captains = pd.concat(captain_parts, ignore_index=True)
    shapes = pd.concat(shape_parts, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    totals.to_csv(PROCESSED / "stage_45_score_repair.csv", index=False)
    phantom.to_csv(PROCESSED / "stage_45_defcon.csv", index=False)
    captains.to_csv(PROCESSED / "stage_45_captain.csv", index=False)
    shapes.to_csv(PROCESSED / "stage_45_formation.csv", index=False)
    _write(totals, verdict, phantom, captains, shapes)
    print(verdict.to_string(index=False), flush=True)
    return verdict


if __name__ == "__main__":
    run()
