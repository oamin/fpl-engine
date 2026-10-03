"""Where the Gameweek 1–5 gap against ojaminFC comes from.

Starts from their Gameweek 1 fifteen and runs the published climb.
Does not rewrite the leaked 327 record. Five weeks are a diagnosis,
not a verdict that the rule is worse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, _opening_state, build_frames, player_key
from src.live.entry import load_entry
from src.models.season_climb import apply_autosubs
from src.models.season_climb_ft import run_ft_season
from src.rules.fpl_2026 import captain_extra_points

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ENTRY_ID = 2632584


def _their_final(week: dict[str, Any], roster: pd.DataFrame) -> pd.DataFrame:
    """Their picked 15, with automatic substitutes in their bench-slot order."""
    gw = int(week["gw"])
    sheet = roster.loc[roster["gw"] == gw].drop_duplicates("player_id", keep="first")
    by_id = {str(r.player_id): r for r in sheet.itertuples(index=False)}
    rows = []
    for group, role in ((week["xi"], "xi"), (week["bench"], "bench")):
        for player in group:
            pid = player_key(player["id"])
            src = by_id.get(pid)
            rows.append(
                {
                    "player_id": pid,
                    "name": player["name"],
                    "position": player["position"],
                    "role": role,
                    "slot": int(player["slot"]),
                    "minutes": float(getattr(src, "minutes", 0) or 0) if src is not None else 0.0,
                    "total_points": float(getattr(src, "total_points", 0) or 0) if src is not None else 0.0,
                }
            )
    squad = pd.DataFrame(rows)
    xi = squad.loc[squad["role"] == "xi"].sort_values("slot")
    bench = squad.loc[squad["role"] == "bench"].sort_values("slot")
    final, _n = apply_autosubs(xi, bench)
    return final


def _names(ids: set[str], roster: pd.DataFrame, gw: int) -> str:
    sheet = roster.loc[roster["gw"] == gw].drop_duplicates("player_id", keep="first")
    names = dict(zip(sheet["player_id"].astype(str), sheet["player_name"], strict=False))
    return ", ".join(names.get(pid, pid) for pid in sorted(ids))


def run() -> dict[str, Any]:
    feat, roster, _info = build_frames()
    entry = load_entry(ENTRY_ID)
    opening = _opening_state(entry, roster)
    trace: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    weekly = run_ft_season(
        feat,
        {"xp": "score_xp"},
        list(GWS),
        roster=roster,
        trace=trace,
        opening=opening,
        decisions=decisions,
    )
    by_gw = {int(g["gw"]): g for g in entry["gameweeks"]}
    theirs_tx = {}
    for row in entry["transfers"]:
        theirs_tx.setdefault(int(row["gw"]), []).append(f"in {row['in']}; out {row['out']}")
    rows = []
    owned: set[str] | None = None
    for step, week in zip(trace, weekly.sort_values("gw").itertuples(index=False), strict=True):
        gw = int(step["gw"])
        theirs = by_gw[gw]
        final = step["final_xi"]
        model_ids = set(final["player_id"].astype(str))
        cap_id = str(step["captain_id"])
        model_captain = _names({cap_id}, roster, gw)
        nan_score = 0
        if "score_xp" in step["squad"].columns:
            nan_score = int(pd.to_numeric(step["squad"]["score_xp"], errors="coerce").isna().sum())
        model_base = float(pd.to_numeric(final["total_points"], errors="coerce").fillna(0).sum())
        model_extra = float(step["cap_extra"])
        model_hits = float(week.hit_cost)
        model_points = float(week.xi_points_cap)
        their_final = _their_final(theirs, roster)
        their_ids = set(their_final["player_id"].astype(str))
        their_base = float(their_final["total_points"].sum())
        cap_row = their_final.loc[their_final["name"] == theirs["captain"]]
        vic_row = their_final.loc[their_final["name"] == theirs["vice"]]
        cap_pts = float(cap_row["total_points"].iloc[0]) if len(cap_row) else 0.0
        vic_pts = float(vic_row["total_points"].iloc[0]) if len(vic_row) else 0.0
        cap_played = bool(len(cap_row) and float(cap_row["minutes"].iloc[0]) > 0)
        vic_played = bool(len(vic_row) and float(vic_row["minutes"].iloc[0]) > 0)
        their_extra = captain_extra_points(
            cap_pts, vic_pts, captain_played=cap_played, vice_played=vic_played, chip=None
        )
        chip_premium = 0.0
        if theirs.get("chip") == "triple_captain":
            doubled = captain_extra_points(
                cap_pts, vic_pts, captain_played=cap_played, vice_played=vic_played, chip="triple_captain"
            )
            chip_premium = doubled - their_extra
        their_hits = float(theirs["transfer_cost"])
        their_built = their_base + their_extra + chip_premium - their_hits
        only_model = model_ids - their_ids
        only_theirs = their_ids - model_ids
        model_only_pts = float(
            pd.to_numeric(final.loc[final["player_id"].astype(str).isin(only_model), "total_points"], errors="coerce").fillna(0).sum()
        )
        their_only_pts = float(their_final.loc[their_final["player_id"].isin(only_theirs), "total_points"].sum())
        squad_ids = set(step["squad"]["player_id"].astype(str))
        if owned is None:
            bought, sold = set(), set()
        else:
            bought, sold = squad_ids - owned, owned - squad_ids
        owned = squad_ids
        hold = next((d for d in decisions if int(d["gw"]) == gw and d["role"] == "hold"), None)
        move = next((d for d in decisions if int(d["gw"]) == gw and d["role"] == "move"), None)
        rows.append(
            {
                "gw": gw,
                "model_points": model_points,
                "their_points": float(theirs["points"]),
                "gap": model_points - float(theirs["points"]),
                "model_xi": model_base,
                "their_xi": their_base,
                "xi_gap": model_base - their_base,
                "model_captain_extra": model_extra,
                "their_captain_extra": their_extra,
                "captain_gap": model_extra - their_extra,
                "their_chip_premium": chip_premium,
                "model_hits": model_hits,
                "their_hits": their_hits,
                "hit_gap": -(model_hits - their_hits),
                "only_model_points": model_only_pts,
                "only_their_points": their_only_pts,
                "their_built": their_built,
                "their_residual": their_built - float(theirs["points"]),
                "model_in": _names(bought, roster, gw),
                "model_out": _names(sold, roster, gw),
                "their_transfers": "; ".join(theirs_tx.get(gw, [])) or "none",
                "model_captain": model_captain,
                "their_captain": theirs["captain"],
                "nan_score": nan_score,
                "hold_value": None if hold is None else float(hold["value"]),
                "move_value": None if move is None else float(move["value"]),
                "move_margin": None if move is None else float(move["margin"]),
            }
        )
    table = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "stage_40_gw15_gap.csv", index=False)
    _report(REPORTS / "stage_40_gw15_gap.md", table)
    return {
        "model": float(table["model_points"].sum()),
        "theirs": float(table["their_points"].sum()),
        "gap": float(table["gap"].sum()),
    }


def _report(path: Path, table: pd.DataFrame) -> None:
    gap = float(table["gap"].sum())
    lines = [
        "# Gameweeks 1–5, same opening 15",
        "",
        "The published climb starts from ojaminFC's Gameweek 1 fifteen. The chip map is empty, so their Triple Captain is not copied. Five weeks are a split of the gap, not a verdict on the rule.",
        "",
        f"Model **{table['model_points'].sum():.0f}**. ojaminFC **{table['their_points'].sum():.0f}**. Gap **{gap:+.0f}**.",
        "",
        "The gap is the model's week total minus theirs. A negative number is points they scored and the model did not.",
        "",
        "| GW | Model | Theirs | Gap | XI gap | Captain gap | Their chip | Hit gap | Only model | Only theirs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {int(row.gw)} | {row.model_points:.0f} | {row.their_points:.0f} | {row.gap:+.0f} | "
            f"{row.xi_gap:+.0f} | {row.captain_gap:+.0f} | {row.their_chip_premium:.0f} | {row.hit_gap:+.0f} | "
            f"{row.only_model_points:.0f} | {row.only_their_points:.0f} |"
        )
    lines += [
        "",
        f"| Total | {table['model_points'].sum():.0f} | {table['their_points'].sum():.0f} | {gap:+.0f} | "
        f"{table['xi_gap'].sum():+.0f} | {table['captain_gap'].sum():+.0f} | {table['their_chip_premium'].sum():.0f} | "
        f"{table['hit_gap'].sum():+.0f} | {table['only_model_points'].sum():.0f} | {table['only_their_points'].sum():.0f} |",
        "",
        "XI gap is the final elevens after automatic substitutes, before the captain extra and before hits. Captain gap is the extra copy only. Their chip is the third copy on the Triple Captain week. Hit gap is theirs minus the model's charge, so a model hit is negative. Only-model and only-theirs are the points of players who finished in one scoring eleven and not the other. Those two columns are the XI gap split in two; they are not a further subtraction.",
        "",
        f"Rebuilding their week from the sheet leaves a mean residual of **{table['their_residual'].mean():+.3f}** against the official gameweek total.",
        "",
        "## Transfers",
        "",
        "| GW | Model in | Model out | Theirs | Model captain | Their captain | Hold value | Move value | Margin |",
        "|---:|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        hold = "—" if pd.isna(row.hold_value) else f"{row.hold_value:.2f}"
        move = "—" if pd.isna(row.move_value) else f"{row.move_value:.2f}"
        margin = "—" if pd.isna(row.move_margin) else f"{row.move_margin:.2f}"
        lines.append(
            f"| {int(row.gw)} | {row.model_in or 'none'} | {row.model_out or 'none'} | {row.their_transfers} | {row.model_captain} | {row.their_captain} | {hold} | {move} | {margin} |"
        )
    lines.append("")
    lines += [
        "Hold value and move value are the three-week number the search maximised. A missing `score_xp` is 0. It is not the player's price. Before that change, M.Sangaré's price was his score and he was captain for the first three weeks, and the search value sat near 680.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(run())
