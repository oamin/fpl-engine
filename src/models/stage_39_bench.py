"""Stage 39 — bench weight inside transfer value.

Locked before the totals: ``bench_w_ft``, one weight ``w = 0.25``.
Only ``transfer_value`` changes. The opening squad, the XI that is fielded,
the captain, autosubs, and banked points stay on ``score_xp``. Chips stay
empty. ``HOLD_EPS``, ``SWITCH_PENALTY``, horizon, gamma, and the hit cost
stay where they are. The switch penalty stays outside ``transfer_value``.

2025/26, GW5–38, against the paired ``xp_ft`` on the same frame.
Kill if bench_w points with captain minus that climb is <= −100.
Otherwise the same pair on 2024/25. No other season. No second weight.
The pass bar of +34 is recorded on each season that is run.
A delta inside the 2025/26 opening-squad band of 1839 to 1972 is not an edge.

Known failure cases, left in the rule: the weight can spend starting-XI
money on the bench; it can take hits to fix bench players; all four bench
slots are weighted the same.

Writes:
  data/processed/stage_39_bench.csv
  reports/stage_39_bench.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.stage_29_batch import GW_END, GW_START, KILL_GAP, PASS_MARGIN
from src.models.stage_31_churn import arm_stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

BENCH_WEIGHT = 0.25
SCREEN_SEASON = "2025-26"
CHECK_SEASON = "2024-25"
OPENING_BAND = (1839.0, 1972.0)


def _gws(feat: pd.DataFrame) -> list[int]:
    return [g for g in sorted(int(x) for x in feat["gw"].unique()) if GW_START <= g <= GW_END]


def prepare(season: str) -> pd.DataFrame:
    code = next(code for name, code in SEASONS if name == season)
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    return feat


def _in_opening_band(points: float) -> bool:
    lo, hi = OPENING_BAND
    return lo <= float(points) <= hi


def _climb(
    feat: pd.DataFrame,
    season: str,
    label: str,
    bench_weight: float | None,
) -> pd.DataFrame:
    print(f"  {season} {label}…", flush=True)
    return run_ft_season(
        feat,
        {label: "score_xp"},
        _gws(feat),
        roster=load_vaastav_roster(season),
        horizon=HORIZON,
        bench_weight=bench_weight,
    )


def _row(
    weekly: pd.DataFrame,
    season: str,
    role: str,
    method: str,
    bench_weight: float | None,
    base_points: float,
) -> dict[str, object]:
    stats = arm_stats(weekly, method)
    frame = weekly.loc[weekly["method"] == method]
    delta = float(stats["xi_points"]) - float(base_points)
    return {
        "season": season,
        "role": role,
        "method": method,
        "bench_weight": bench_weight if bench_weight is not None else 0.0,
        "delta_vs_xp": delta,
        "transfers": float(frame["n_transfers"].sum()) if len(frame) else float("nan"),
        "pass_34": bool(delta >= PASS_MARGIN) if method != "xp_ft" else False,
        "clear_loss": bool(delta <= -KILL_GAP) if method != "xp_ft" else False,
        **stats,
    }


def run_pair(feat: pd.DataFrame, season: str, role: str) -> pd.DataFrame:
    xp = _climb(feat, season, "xp", None)
    arm = _climb(feat, season, "bench_w", BENCH_WEIGHT)
    base = arm_stats(xp, "xp_ft")
    rows = [
        _row(xp, season, role, "xp_ft", None, base["xi_points"]),
        _row(arm, season, role, "bench_w_ft", BENCH_WEIGHT, base["xi_points"]),
    ]
    return pd.DataFrame(rows)


def write_report(path: Path, table: pd.DataFrame, ran_check: bool) -> None:
    screen = table.loc[table["role"] == "screen"]
    xp = screen.loc[screen["method"] == "xp_ft"].iloc[0]
    arm = screen.loc[screen["method"] == "bench_w_ft"].iloc[0]
    lines = [
        "# Stage 39 — bench weight inside transfers",
        "",
        "One locked weight, `w = 0.25`, named `bench_w_ft`.",
        "The opening squad, the fielded XI, the captain, automatic substitutes, and banked points stay on expected points.",
        "Chips stay empty. The switch penalty stays outside transfer value, at 1.0.",
        "The screen is 2025/26 gameweeks 5–38. The comparator is the paired expected-points climb on that same frame.",
        "A clear loss is 100 or more points under that climb. Only then is 2024/25 skipped.",
        "The pass bar of +34 is recorded on each season that is run. It is not a second weight.",
        "",
        "A delta inside the 2025/26 opening-squad band of 1839 to 1972 is not an edge.",
        "",
        "The weight can spend starting-XI money on the bench. It can take hits to fix bench players. All four bench slots are weighted the same.",
        "",
        "## Screen 2025/26",
        "",
        "| method | weight | points | vs xp | transfers | transfers/GW | hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in screen.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.bench_weight:.2f} | {row.xi_points:.0f} | "
            f"{row.delta_vs_xp:+.0f} | {row.transfers:.0f} | {row.mean_transfers:.2f} | {row.hits:.0f} |"
        )
    xp_band = "inside" if _in_opening_band(float(xp["xi_points"])) else "outside"
    arm_band = "inside" if _in_opening_band(float(arm["xi_points"])) else "outside"
    lines.extend(
        [
            "",
            f"Paired xp_ft is {float(xp['xi_points']):.0f} ({xp_band} the opening-squad band).",
            f"bench_w_ft is {float(arm['xi_points']):.0f} ({arm_band} the opening-squad band), "
            f"delta {float(arm['delta_vs_xp']):+.0f}.",
            f"Pass bar +34 on 2025/26: {'cleared' if bool(arm['pass_34']) else 'missed'}.",
            "",
            "## 2024/25",
            "",
        ]
    )
    if not ran_check:
        lines.append(
            f"2024/25 was not run. The screen delta was {float(arm['delta_vs_xp']):+.0f}, "
            "which is a clear loss (at or under −100)."
        )
    else:
        check = table.loc[table["role"] == "check"]
        lines.extend(
            [
                "2024/25 was run. The screen was not a clear loss.",
                "",
                "| method | weight | points | vs xp | transfers | transfers/GW | hits |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in check.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.bench_weight:.2f} | {row.xi_points:.0f} | "
                f"{row.delta_vs_xp:+.0f} | {row.transfers:.0f} | {row.mean_transfers:.2f} | {row.hits:.0f} |"
            )
        check_arm = check.loc[check["method"] == "bench_w_ft"].iloc[0]
        lines.extend(
            [
                "",
                f"2024/25 delta {float(check_arm['delta_vs_xp']):+.0f}. "
                f"Pass bar +34 on 2024/25: {'cleared' if bool(check_arm['pass_34']) else 'missed'}.",
            ]
        )
    both_clear = bool(arm["pass_34"]) and ran_check and bool(
        table.loc[(table["role"] == "check") & (table["method"] == "bench_w_ft"), "pass_34"].iloc[0]
    )
    lines.extend(["", "## Diagnostics", ""])
    lines.append(
        f"Screen hits {float(xp['hits']):.0f} to {float(arm['hits']):.0f}, "
        f"transfers {float(xp['transfers']):.0f} to {float(arm['transfers']):.0f}."
    )
    lines.append(
        f"Screen points from substitutes {float(xp['sub_points']):.0f} to {float(arm['sub_points']):.0f}. "
        f"Unfilled XI slots after substitutes {float(xp['blank_final']):.0f} to {float(arm['blank_final']):.0f}."
    )
    if ran_check:
        check_xp = table.loc[(table["role"] == "check") & (table["method"] == "xp_ft")].iloc[0]
        check_arm = table.loc[(table["role"] == "check") & (table["method"] == "bench_w_ft")].iloc[0]
        lines.append(
            f"2024/25 hits {float(check_xp['hits']):.0f} to {float(check_arm['hits']):.0f}, "
            f"transfers {float(check_xp['transfers']):.0f} to {float(check_arm['transfers']):.0f}, "
            f"points from substitutes {float(check_xp['sub_points']):.0f} to {float(check_arm['sub_points']):.0f}, "
            f"unfilled XI slots after substitutes {float(check_xp['blank_final']):.0f} to {float(check_arm['blank_final']):.0f}."
        )
    lines.append(
        "The pre-registered failure cases stay open. "
        "Extra hits are consistent with paying to fix bench players. "
        "The same weight on all four bench slots does not separate a playing substitute from a dead one. "
        "Nothing in the rule stops the search from spending starting-XI money on the bench."
    )
    lines.extend(["", "## Result", ""])
    in_band = _in_opening_band(float(arm["xi_points"])) and _in_opening_band(float(xp["xi_points"]))
    if in_band:
        lines.append(
            f"Both 2025/26 totals sit in 1839 to 1972, so the {float(arm['delta_vs_xp']):+.0f} screen delta is not an edge."
        )
    if both_clear and not in_band:
        lines.append(
            "bench_w_ft is the best of this batch. It cleared +34 on both seasons that were run. "
            "That is the batch label, not a confirmed edge."
        )
    elif both_clear:
        lines.append(
            "bench_w_ft cleared +34 on both seasons. It is the only method in the batch, and the screen delta is not an edge."
        )
    else:
        lines.append(
            "bench_w_ft is the best of this batch, and the only method in it. It is not carried forward. The weight stays 0.25."
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"Prepare {SCREEN_SEASON}…", flush=True)
    screen = run_pair(prepare(SCREEN_SEASON), SCREEN_SEASON, "screen")
    arm = screen.loc[screen["method"] == "bench_w_ft"].iloc[0]
    ran_check = float(arm["delta_vs_xp"]) > -KILL_GAP
    frames = [screen]
    if ran_check:
        print(f"Prepare {CHECK_SEASON}…", flush=True)
        frames.append(run_pair(prepare(CHECK_SEASON), CHECK_SEASON, "check"))
    else:
        print("Clear loss. 2024/25 not run.", flush=True)
    table = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "stage_39_bench.csv", index=False)
    write_report(REPORTS / "stage_39_bench.md", table, ran_check)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
