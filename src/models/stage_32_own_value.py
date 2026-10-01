"""Stage 32 and 33 in one season build.

Stage 32: XI and captain stay on score_xp. Transfer value uses
score_xp * (1 - 0.5 * ow). One weight. 2025/26 GW5–38. Pass bar +34.
The check is 2024/25, and only if the screen passes.

Stage 33: twelve fast-XI scores. No transfer climb.

Writes:
  data/processed/stage_32_own_value.csv
  data/processed/stage_33_screen.csv
  reports/stage_32_own_value.md
  reports/stage_33_screen.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.sharpe_u import add_causal_sharpe_u
from src.models.stage_29_batch import GW_END, GW_START, PASS_MARGIN
from src.models.stage_30_followup import attach_ownership
from src.models.stage_31_churn import arm_stats
from src.models.stage_33_screen import run_screen

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SCREEN_SEASON = "2025-26"
CHECK_SEASON = "2024-25"


def _gws(feat: pd.DataFrame) -> list[int]:
    return [g for g in sorted(int(x) for x in feat["gw"].unique()) if GW_START <= g <= GW_END]


def prepare(season: str) -> pd.DataFrame:
    code = next(code for name, code in SEASONS if name == season)
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    feat = add_causal_sharpe_u(feat)
    return attach_ownership(feat, season)


def _climb(feat: pd.DataFrame, season: str, *, owned: bool) -> pd.DataFrame:
    gws = _gws(feat)
    roster = load_vaastav_roster(season)
    label = "own" if owned else "xp"
    return run_ft_season(
        feat,
        {label: "score_xp"},
        gws,
        roster=roster,
        horizon=HORIZON,
        value_col="score_own" if owned else None,
    )


def run_ownership(feat: pd.DataFrame, season: str, role: str) -> pd.DataFrame:
    print(f"  {role} xp…", flush=True)
    xp = _climb(feat, season, owned=False)
    print(f"  {role} ownership value…", flush=True)
    arm = _climb(feat, season, owned=True)
    base = arm_stats(xp, "xp_ft")
    stats = arm_stats(arm, "own_ft")
    return pd.DataFrame(
        [
            {"season": season, "role": role, "method": "xp_ft", "delta_vs_xp": 0.0, **base},
            {
                "season": season,
                "role": role,
                "method": "own_ft",
                "delta_vs_xp": stats["xi_points"] - base["xi_points"],
                **stats,
            },
        ]
    )


def write_ownership_report(path: Path, table: pd.DataFrame) -> None:
    screen = table.loc[table["role"] == "screen"]
    own = screen.loc[screen["method"] == "own_ft"].iloc[0]
    lines = [
        "# Stage 32 — ownership inside transfers",
        "",
        "The starting XI and the captain stay on expected points.",
        "Only the transfer score changes: expected points times (1 - 0.5 × ownership share).",
        "The screen is 2025/26. The check is 2024/25, and only if the screen clears +34.",
        "",
        "## Screen 2025/26",
        "",
        "| method | points | vs xp | transfers/GW | hits | blank XI (intended) | blank XI (after subs) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in screen.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.xi_points:.0f} | {row.delta_vs_xp:+.0f} | "
            f"{row.mean_transfers:.2f} | {row.hits:.0f} | {row.blank_intended:.0f} | {row.blank_final:.0f} |"
        )
    lines.extend(["", "## Result", ""])
    if float(own["delta_vs_xp"]) < PASS_MARGIN:
        lines.append(
            f"The screen was {float(own['delta_vs_xp']):+.0f}, under +34. No 2024/25 check was run."
        )
    else:
        check = table.loc[(table["role"] == "check") & (table["method"] == "own_ft")].iloc[0]
        base = float(check["xi_points"]) - float(check["delta_vs_xp"])
        cleared = float(check["delta_vs_xp"]) >= PASS_MARGIN
        label = "cleared +34" if cleared else "missed +34"
        lines.append(f"Screen {float(own['delta_vs_xp']):+.0f}, so the 2024/25 check was run.")
        lines.append(
            f"2024/25: {float(check['xi_points']):.0f} versus {base:.0f} "
            f"({float(check['delta_vs_xp']):+.0f}, {check.mean_transfers:.2f} transfers/GW, "
            f"{check.hits:.0f} hits). It {label}."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_screen_report(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Stage 33 — fast XI screen",
        "",
        "Twelve scores, one setting each, on 2025/26 gameweeks 5–38.",
        "Clearly behind means 100 or more points under expected points.",
        "No transfer climb in this batch. `within_pos` rescales each position inside the gameweek, so a formation can win on tail shape rather than points.",
        "",
        "| score | XI points | vs xp | queued |",
        "|---|---:|---:|---|",
    ]
    for row in table.itertuples(index=False):
        queued = "—" if row.candidate == "xp" else ("no" if row.killed else "yes")
        lines.append(
            f"| {row.candidate} | {row.xi_points:.0f} | {row.delta_vs_xp:+.0f} | {queued} |"
        )
    alive = table.loc[(table["candidate"] != "xp") & ~table["killed"].astype(bool), "candidate"]
    names = ", ".join(alive.tolist()) if len(alive) else "none"
    lines.extend(["", f"Queued for a later transfer climb: {names}.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Prepare {SCREEN_SEASON}…", flush=True)
    feat = prepare(SCREEN_SEASON)
    gws = _gws(feat)
    print("Stage 33 fast XI…", flush=True)
    screen = run_screen(feat, gws)
    print(screen.to_string(index=False), flush=True)
    print("Stage 32 ownership transfers…", flush=True)
    own = run_ownership(feat, SCREEN_SEASON, "screen")
    arm = own.loc[own["method"] == "own_ft"].iloc[0]
    if float(arm["delta_vs_xp"]) >= PASS_MARGIN:
        print(f"Prepare {CHECK_SEASON}…", flush=True)
        own = pd.concat([own, run_ownership(prepare(CHECK_SEASON), CHECK_SEASON, "check")], ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    own.to_csv(PROCESSED / "stage_32_own_value.csv", index=False)
    screen.to_csv(PROCESSED / "stage_33_screen.csv", index=False)
    write_ownership_report(REPORTS / "stage_32_own_value.md", own)
    write_screen_report(REPORTS / "stage_33_screen.md", screen)
    print(own.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
