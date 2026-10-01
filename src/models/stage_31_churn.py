"""Stage 31 — transfer-penalty ablation.

Locked with Gemini before the totals were read. The fast XI climb cannot
see this, so there is no tier-1 screen.

XI, captain, and transfer value stay on score_xp. HOLD_EPS stays 1.25.
Only SWITCH_PENALTY changes: 0.0, 2.0, and 3.0, against the default 1.0
on the same runner. 2025/26, GW5–38. An arm passes if it beats that
paired climb by at least 34. If none pass, there is no holdout. If one
or more pass, the highest total goes to 2023/24 at the same penalty.
Fewer mean transfers breaks a tie. A winner also needs +34 on that holdout.

Blank intended and blank final XI slots are diagnostics. They do not rank.

Writes:
  data/processed/stage_31_churn.csv
  reports/stage_31_churn.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.stage_29_batch import GW_END, GW_START, PASS_MARGIN

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PENALTIES = (0.0, 2.0, 3.0)
SCREEN_SEASON = "2025-26"
HOLDOUT_SEASON = "2023-24"


def _gws(feat: pd.DataFrame) -> list[int]:
    return [
        g
        for g in sorted(int(x) for x in feat["gw"].unique())
        if GW_START <= g <= GW_END
    ]


def _prepare(season: str) -> pd.DataFrame:
    code = next(code for name, code in SEASONS if name == season)
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    return feat


def arm_stats(weekly: pd.DataFrame, method: str) -> dict[str, float]:
    frame = weekly.loc[weekly["method"] == method]
    if frame.empty:
        return {
            "xi_points": float("nan"),
            "mean_transfers": float("nan"),
            "hits": float("nan"),
            "blank_intended": float("nan"),
            "blank_final": float("nan"),
            "sub_points": float("nan"),
            "n_gw": 0.0,
        }
    return {
        "xi_points": float(frame["xi_points_cap"].sum()),
        "mean_transfers": float(frame["n_transfers"].mean()),
        "hits": float(frame["hits"].sum()),
        "blank_intended": float(frame["n_blank_intended"].sum()),
        "blank_final": float(frame["n_blank_final"].sum()),
        "sub_points": float(frame["sub_points"].sum()) if "sub_points" in frame.columns else float("nan"),
        "n_gw": float(len(frame)),
    }


def choose_holdout_arm(screen: pd.DataFrame, pass_margin: float = PASS_MARGIN) -> pd.Series | None:
    """Highest passing total. Fewer transfers, then the lower penalty, break ties."""
    passed = screen.loc[screen["delta_vs_xp"] >= float(pass_margin)]
    if passed.empty:
        return None
    ordered = passed.sort_values(
        ["xi_points", "mean_transfers", "penalty"],
        ascending=[False, True, True],
    )
    return ordered.iloc[0]


def _climb(feat: pd.DataFrame, gws: list[int], roster: pd.DataFrame, penalty: float | None, label: str) -> pd.DataFrame:
    return run_ft_season(
        feat,
        {label: "score_xp"},
        gws,
        roster=roster,
        horizon=HORIZON,
        switch_penalty=penalty,
    )


def run_screen(feat: pd.DataFrame, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gws = _gws(feat)
    roster = load_vaastav_roster(season)
    print("  xp penalty 1.0…", flush=True)
    frames = [_climb(feat, gws, roster, None, "xp")]
    base = arm_stats(frames[0], "xp_ft")
    rows = [
        {
            "season": season,
            "role": "screen",
            "penalty": 1.0,
            "method": "xp_ft",
            "delta_vs_xp": 0.0,
            **base,
        }
    ]
    for penalty in PENALTIES:
        label = f"pen_{penalty:.1f}"
        print(f"  penalty {penalty:.1f}…", flush=True)
        weekly = _climb(feat, gws, roster, penalty, label)
        frames.append(weekly)
        stats = arm_stats(weekly, f"{label}_ft")
        rows.append(
            {
                "season": season,
                "role": "screen",
                "penalty": penalty,
                "method": f"{label}_ft",
                "delta_vs_xp": stats["xi_points"] - base["xi_points"],
                **stats,
            }
        )
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def run_holdout(penalty: float) -> pd.DataFrame:
    print(f"Holdout {HOLDOUT_SEASON} penalty {penalty:.1f}…", flush=True)
    feat = _prepare(HOLDOUT_SEASON)
    gws = _gws(feat)
    roster = load_vaastav_roster(HOLDOUT_SEASON)
    xp = _climb(feat, gws, roster, None, "xp")
    arm = _climb(feat, gws, roster, penalty, "hold")
    base = arm_stats(xp, "xp_ft")
    stats = arm_stats(arm, "hold_ft")
    return pd.DataFrame(
        [
            {
                "season": HOLDOUT_SEASON,
                "role": "holdout",
                "penalty": 1.0,
                "method": "xp_ft",
                "delta_vs_xp": 0.0,
                **base,
            },
            {
                "season": HOLDOUT_SEASON,
                "role": "holdout",
                "penalty": penalty,
                "method": "hold_ft",
                "delta_vs_xp": stats["xi_points"] - base["xi_points"],
                **stats,
            },
        ]
    )


def write_report(path: Path, table: pd.DataFrame, chosen: pd.Series | None) -> None:
    screen = table.loc[table["role"] == "screen"]
    lines = [
        "# Stage 31 — transfer penalty",
        "",
        "XI and captain stay on expected points. The hold margin stays 1.25.",
        "Only the per-transfer penalty changes. The fast climb cannot see that, so this batch is the free-transfer climb.",
        "A screen arm passes at +34 versus the default penalty of 1.0. The best passer, if any, is checked on 2023/24 at the same penalty.",
        "Blank slots are recorded because a high penalty can look better only by blocking hits.",
        "",
        "## Screen 2025/26",
        "",
        "| penalty | points | vs default | transfers/GW | hits | blank XI (intended) | blank XI (after subs) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in screen.sort_values("penalty").itertuples(index=False):
        lines.append(
            f"| {row.penalty:.1f} | {row.xi_points:.0f} | {row.delta_vs_xp:+.0f} | "
            f"{row.mean_transfers:.2f} | {row.hits:.0f} | {row.blank_intended:.0f} | {row.blank_final:.0f} |"
        )
    lines.extend(["", "## Result", ""])
    if chosen is None:
        lines.append("No arm cleared +34. No holdout was run. This batch does not change the default penalty.")
    else:
        hold = table.loc[table["role"] == "holdout"]
        arm = hold.loc[hold["method"] == "hold_ft"].iloc[0]
        cleared = float(arm["delta_vs_xp"]) >= PASS_MARGIN
        label = "cleared the holdout bar" if cleared else "missed the holdout bar"
        lines.append(
            f"Screen pick: penalty {float(chosen['penalty']):.1f}, "
            f"{float(chosen['delta_vs_xp']):+.0f} on 2025/26."
        )
        lines.append(
            f"2023/24: {float(arm['xi_points']):.0f} versus default "
            f"{float(arm['xi_points']) - float(arm['delta_vs_xp']):.0f} "
            f"({float(arm['delta_vs_xp']):+.0f}). It {label}."
        )
        if not cleared:
            lines.append("The default penalty stays in place.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Stage 31 screen {SCREEN_SEASON}…", flush=True)
    screen, _weekly = run_screen(_prepare(SCREEN_SEASON), SCREEN_SEASON)
    chosen = choose_holdout_arm(screen.loc[screen["penalty"] != 1.0])
    frames = [screen]
    if chosen is not None:
        frames.append(run_holdout(float(chosen["penalty"])))
    table = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "stage_31_churn.csv", index=False)
    write_report(REPORTS / "stage_31_churn.md", table, chosen)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
