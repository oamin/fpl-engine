"""His chip weeks, with the illegal ones left unset.

The squad is not read. ``score_xp``, the hold, and both chip margins stay
where they are. A week this filter drops is reported and not played.
The carry below plays that calendar and still picks the squad itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.rules.fpl_2026 import ChipWallet

ROOT = Path(__file__).resolve().parents[2]
AUTO_PATH = ROOT / "data" / "processed" / "cohort_carry_gw15.csv"
OUT_CSV = ROOT / "data" / "processed" / "forced_chips_gw15.csv"
OUT_REPORT = ROOT / "reports" / "forced_chips_gw15.md"

# Locked before the forced calendar is scored. Chip value is the forced
# gap minus the autonomous gap. A non-positive mean, or a mean forced gap
# at or below the floor, kills the claim that his chip weeks explain the
# deficit. Neither number is a new margin.
CHIP_VALUE_FLOOR = 0.0
FORCED_GAP_FLOOR = -20.0


def legal_calendar(
    played: Mapping[int, str],
) -> tuple[dict[int, str], tuple[tuple[int, str], ...]]:
    """Keep the chips the wallet allows, in week order.

    A Gameweek 1 wildcard or free hit is dropped. So is a second chip in
    the same half, a second chip in a week, and a free hit the week after
    a free hit. The returned map is the one the carry may play. The
    dropped pairs are ``(gameweek, chip)``.
    """
    wallet = ChipWallet()
    kept: dict[int, str] = {}
    dropped: list[tuple[int, str]] = []
    for gw in sorted(int(week) for week in played):
        chip = str(played[gw])
        if chip not in wallet.available(gw):
            dropped.append((gw, chip))
            continue
        wallet.play(gw, chip)
        kept[gw] = chip
    return kept, tuple(dropped)


def hypothesis_killed(mean_chip_value: float, mean_forced_gap: float) -> bool:
    """True when his chip weeks do not explain the deficit."""
    if float(mean_chip_value) <= CHIP_VALUE_FLOOR:
        return True
    return float(mean_forced_gap) <= FORCED_GAP_FLOOR


def chips_from_entry(entry: Mapping[str, Any], weeks: tuple[int, ...]) -> dict[int, str]:
    """The chip he played in each listed week. An empty week is absent."""
    out: dict[int, str] = {}
    wanted = {int(gw) for gw in weeks}
    for row in entry["gameweeks"]:
        gw = int(row["gw"])
        chip = row.get("chip")
        if gw in wanted and chip:
            out[gw] = str(chip)
    return out


def auto_gaps(path: Path = AUTO_PATH) -> dict[int, float]:
    """The stored carry gap, summed over the five weeks."""
    frame = pd.read_csv(path)
    grouped = frame.groupby("entry_id", as_index=False)["gap"].sum()
    return {int(row.entry_id): float(row.gap) for row in grouped.itertuples(index=False)}


def carry_forced(
    spec: Mapping[str, Any],
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    clubs: dict[int, set[str]],
    roster_by_gw: dict[int, set[str]],
    horizon_scores: Any,
) -> dict[str, Any]:
    """One manager on his chip weeks. The squad is still the model's."""
    from src.live.benchmark import GWS
    from src.live.entry import load_entry
    from src.models.cohort_carry import failed_result, manager_result
    from src.models.friend_start import assert_carried, one_week
    from src.models.reset_gap import pre_deadline

    try:
        entry = load_entry(int(spec["entry_id"]))
        kept, dropped = legal_calendar(chips_from_entry(entry, GWS))
        state = pre_deadline(entry, roster, 1)
        if int(state.ft) != 0:
            raise RuntimeError("Gameweek 1 did not start with 0 free transfers")
        wallet = ChipWallet()
        weeks: list[dict[str, Any]] = []
        for gw in GWS:
            row, state = one_week(
                feat,
                roster,
                entry,
                gw,
                state,
                wallet,
                clubs=clubs,
                roster_by_gw=roster_by_gw,
                horizon_scores=horizon_scores,
                forced_chip=kept.get(int(gw)),
            )
            weeks.append(row)
        assert_carried(weeks)
        played = [row["chip"] for row in weeks if row["chip"]]
        if len(played) != len(set(played)):
            raise RuntimeError("a chip was played twice")
        if weeks[0]["chip"] in {"wildcard", "free_hit"}:
            raise RuntimeError("Gameweek 1 played a wildcard or a free hit")
        result = manager_result(dict(spec), weeks)
        result["dropped"] = dropped
        result["kept"] = tuple(sorted(kept.items()))
        return result
    except Exception as exc:
        failed = failed_result(dict(spec), f"{type(exc).__name__}: {exc}")
        failed["dropped"] = ()
        failed["kept"] = ()
        return failed


def _fmt(value: float) -> str:
    text = f"{float(value):.2f}"
    return text[:-3] if text.endswith(".00") else text.rstrip("0").rstrip(".")


def render_report(
    rows: list[dict[str, Any]],
    autos: Mapping[int, float],
) -> str:
    """The five-week gaps and the chip value. The reference stays out of the mean."""
    cohort = [row for row in rows if row["group"] != "reference"]
    if any(not row["finished"] for row in cohort):
        raise RuntimeError("a cohort manager did not finish")
    if len(cohort) != 14:
        raise RuntimeError("the cohort is not 14 managers")
    paired = []
    for row in rows:
        auto = float(autos[int(row["entry_id"])])
        forced = float(row["gap"])
        paired.append((row, auto, forced, forced - auto))
    body = [item for item in paired if item[0]["group"] != "reference"]
    mean_auto = sum(item[1] for item in body) / len(body)
    mean_forced = sum(item[2] for item in body) / len(body)
    mean_value = sum(item[3] for item in body) / len(body)
    killed = hypothesis_killed(mean_value, mean_forced)
    if killed:
        verdict = (
            f"The mean chip value is {_fmt(mean_value)} and the mean forced gap is "
            f"{_fmt(mean_forced)}. His chip weeks do not explain the deficit. "
            "The margins stay 12 and 16."
        )
    else:
        verdict = (
            f"The mean chip value is {_fmt(mean_value)} and the mean forced gap is "
            f"{_fmt(mean_forced)}. The kill bar was not met."
        )
    lines = [
        "# Forced chip calendar, Gameweeks 1–5",
        "",
        "Each manager starts from his own Gameweek 1 fifteen. His legal chip weeks replace the model's chip choice. The model still picks the players, the eleven, and the captain. His squad inside the chip week is not copied. The autonomous gaps are the stored carry.",
        "",
        verdict,
        "",
        f"Autonomous mean {_fmt(mean_auto)}. Forced mean {_fmt(mean_forced)}. Chip value {_fmt(mean_value)}.",
        "",
        "| Manager | Group | Auto | Forced | Chip value | Captain | Transfers | Lineup | Hits | Bench | Chips |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row, auto, forced, value in paired:
        chips = ", ".join(row["model_chips"]) if row["model_chips"] else "none"
        lines.append(
            f"| {row['label']} | {row['group']} | {_fmt(auto)} | {_fmt(forced)} | "
            f"{_fmt(value)} | {_fmt(row['captain_gap'])} | {_fmt(row['transfer_gap'])} | "
            f"{_fmt(row['lineup_gap'])} | {_fmt(row['hit_gap'])} | {_fmt(row['bench_gap'])} | {chips} |"
        )
    dropped = [
        f"{row['label']} GW{gw} {chip}"
        for row, _, _, _ in paired
        for gw, chip in row["dropped"]
    ]
    lines += [
        "",
        "Dropped chips: " + (", ".join(dropped) if dropped else "none") + ".",
        "",
        "ojaminFC is the last row and is outside the mean. Entry 1078627 is not in the run.",
        "",
    ]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    """Score the locked calendar. Does not retune."""
    from src.live.benchmark import build_frames
    from src.models.cohort_carry import cohort_specs
    from src.models.open_horizon import attach_opening_horizon
    from src.models.reset_gap import merged_clubs

    specs = cohort_specs()
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    autos = auto_gaps()
    results: list[dict[str, Any]] = []
    for spec in specs:
        print(f"forced {spec['entry_id']} {spec['label']}", flush=True)
        results.append(
            carry_forced(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_CSV, results)
    unfinished = [
        row for row in results if row["group"] != "reference" and not row["finished"]
    ]
    if unfinished:
        for row in unfinished:
            print(f"failed {row['label']}: {row['error']}", flush=True)
        raise RuntimeError("a cohort manager did not finish")
    text = render_report(results, autos)
    OUT_REPORT.write_text(text, encoding="utf-8")
    print(text, flush=True)
    cohort = [row for row in results if row["group"] != "reference"]
    values = [float(row["gap"]) - float(autos[int(row["entry_id"])]) for row in cohort]
    mean_value = sum(values) / len(values)
    mean_forced = sum(float(row["gap"]) for row in cohort) / len(cohort)
    return {
        "killed": hypothesis_killed(mean_value, mean_forced),
        "mean_chip_value": mean_value,
        "mean_forced_gap": mean_forced,
        "finished": sum(1 for row in cohort if row["finished"]),
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    flat = []
    for manager in results:
        for row in manager["weeks"]:
            flat.append(
                {
                    "entry_id": manager["entry_id"],
                    "label": manager["label"],
                    "group": manager["group"],
                    "gw": row["gw"],
                    "chip": row["chip"] or "",
                    "his_chip": row["his_chip"] or "",
                    "gap": row["gap"],
                    "captain_gap": row["captain_gap"],
                    "transfer_gap": row["transfer_gap"],
                    "lineup_gap": row["lineup_gap"],
                    "hit_gap": row["hit_gap"],
                    "bench_gap": row["bench_gap"],
                    "model_points": row["model_points"],
                    "their_points": row["their_points"],
                }
            )
    pd.DataFrame(flat).to_csv(path, index=False)


if __name__ == "__main__":
    print(run())
