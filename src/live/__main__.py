"""Collect the live FPL slate. A team is not picked without a minutes file."""

from __future__ import annotations

import argparse
import sys

from src.live.fpl_snapshot import fixture_counts, next_event, refresh
from src.live.xmi import LiveInputError, load_xmi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect inputs for the live FPL planner.")
    parser.add_argument("--xmi", help="CSV of player_id, gw, xmi supplied by Gemini")
    args = parser.parse_args(argv)
    slate = refresh()
    event = next_event(slate["bootstrap"])
    gw = int(event["id"])
    counts = fixture_counts(slate["fixtures"], gw)
    clubs = [int(row["id"]) for row in slate["bootstrap"]["teams"]]
    blanks = sum(1 for club in clubs if counts.get(club, 0) == 0)
    doubles = sum(1 for club in clubs if counts.get(club, 0) >= 2)
    print(
        f"GW{gw} deadline {event['deadline_time']}. "
        f"Fixtures {sum(counts.values()) // 2}. "
        f"Blank clubs {blanks}. Double clubs {doubles}."
    )
    if not args.xmi:
        print(
            "No minutes file. Gemini supplies expected minutes before a team is picked. "
            "No XI, captain, bench, or chip was chosen."
        )
        return 2
    try:
        minutes = load_xmi(args.xmi, gw)
    except LiveInputError as exc:
        print(exc)
        return 2
    print(f"Minutes loaded for {len(minutes)} players in GW{gw}.")
    print("Odds are not fetched. A snapshot CSV is required before λ is built. No team was picked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
