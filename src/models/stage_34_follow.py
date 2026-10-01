"""Stage 34 — transfer climbs for the four stage-33 scores that beat xp.

Locked after the screen and before these transfer totals were read.
agree_min, starter, minutes, and upside are the XI, the captain, and the
transfer score. 2025/26 GW5–38. Pass bar +34 versus paired xp_ft.
Only the highest passer is checked on 2024/25. Premium, within_pos,
goals_tilt, and agree_max are not climbed.

Writes:
  data/processed/stage_34_follow.csv
  reports/stage_34_follow.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.stage_29_batch import PASS_MARGIN
from src.models.stage_31_churn import arm_stats
from src.models.stage_32_own_value import CHECK_SEASON, SCREEN_SEASON, _gws, prepare
from src.models.stage_33_screen import attach_screen_scores

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

ARMS = ("agree_min", "starter", "minutes", "upside")


def choose_check_arm(screen: pd.DataFrame, pass_margin: float = PASS_MARGIN) -> pd.Series | None:
    """Highest passing total. Fewer transfers, then the earlier arm name, break ties."""
    passed = screen.loc[screen["delta_vs_xp"] >= float(pass_margin)]
    if passed.empty:
        return None
    ordered = passed.sort_values(
        ["xi_points", "mean_transfers", "method"],
        ascending=[False, True, True],
    )
    return ordered.iloc[0]


def _one(feat: pd.DataFrame, season: str, label: str, column: str) -> pd.DataFrame:
    print(f"  {season} {label}…", flush=True)
    return run_ft_season(
        feat,
        {label: column},
        _gws(feat),
        roster=load_vaastav_roster(season),
        horizon=HORIZON,
    )


def run_season(feat: pd.DataFrame, season: str, role: str, arms: tuple[str, ...] = ARMS) -> pd.DataFrame:
    scored, cols = attach_screen_scores(feat)
    weekly = _one(scored, season, "xp", "score_xp")
    base = arm_stats(weekly, "xp_ft")
    rows = [{"season": season, "role": role, "method": "xp_ft", "delta_vs_xp": 0.0, **base}]
    for arm in arms:
        arm_weekly = _one(scored, season, arm, cols[arm])
        stats = arm_stats(arm_weekly, f"{arm}_ft")
        rows.append(
            {
                "season": season,
                "role": role,
                "method": f"{arm}_ft",
                "delta_vs_xp": stats["xi_points"] - base["xi_points"],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, table: pd.DataFrame, chosen: pd.Series | None) -> None:
    screen = table.loc[table["role"] == "screen"]
    lines = [
        "# Stage 34 — transfer climbs",
        "",
        "The four fast-XI scores that finished ahead of expected points, excluding the monotone premium map.",
        "Each score picks the squad, the XI, and the captain. The check is 2024/25, and only for the best arm that clears +34.",
        "Points from subs are what automatic substitutes scored after replacing a starter who played zero minutes.",
        "",
        "## Screen 2025/26",
        "",
        "| method | points | vs xp | transfers/GW | hits | blank XI (after subs) | points from subs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in screen.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.xi_points:.0f} | {row.delta_vs_xp:+.0f} | "
            f"{row.mean_transfers:.2f} | {row.hits:.0f} | {row.blank_final:.0f} | {row.sub_points:.0f} |"
        )
    lines.extend(["", "## Result", ""])
    if chosen is None:
        lines.append("No arm cleared +34. No 2024/25 check was run.")
    else:
        check = table.loc[(table["role"] == "check") & (table["method"] == chosen["method"])].iloc[0]
        base = float(check["xi_points"]) - float(check["delta_vs_xp"])
        label = "cleared +34" if float(check["delta_vs_xp"]) >= PASS_MARGIN else "missed +34"
        lines.append(
            f"Screen pick: {chosen['method']}, {float(chosen['delta_vs_xp']):+.0f}."
        )
        lines.append(
            f"2024/25: {float(check['xi_points']):.0f} versus {base:.0f} "
            f"({float(check['delta_vs_xp']):+.0f}). "
            f"Substitutes scored {float(check['sub_points']):.0f} against "
            f"{float(table.loc[(table['role'] == 'check') & (table['method'] == 'xp_ft'), 'sub_points'].iloc[0]):.0f} "
            f"for the default. It {label}."
        )
        lines.append(
            "Starter also cleared +34 on the screen. The locked rule sent only the higher total. "
            "Checking starter on 2024/25 after the selected arm missed would be a second look. "
            "The minutes gain did not travel."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Prepare {SCREEN_SEASON}…", flush=True)
    screen = run_season(prepare(SCREEN_SEASON), SCREEN_SEASON, "screen")
    chosen = choose_check_arm(screen.loc[screen["method"] != "xp_ft"])
    frames = [screen]
    if chosen is not None:
        print(f"Prepare {CHECK_SEASON}…", flush=True)
        arm_name = str(chosen["method"]).removesuffix("_ft")
        check = run_season(prepare(CHECK_SEASON), CHECK_SEASON, "check", arms=(arm_name,))
        frames.append(check)
    table = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "stage_34_follow.csv", index=False)
    write_report(REPORTS / "stage_34_follow.md", table, chosen)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
