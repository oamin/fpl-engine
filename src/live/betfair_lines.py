"""Refresh live gw_lines from Betfair Exchange into a predictions folder.

Does not overwrite the frozen ``data/live/`` holdout files. Run this from an
IP Betfair allows (UK / non-restricted). US cloud VMs receive HTTP 403.

Usage::

    python3 -m src.live.betfair_lines
    python3 -m src.live.betfair_lines --out data/predictions/2026-27/gw06/betfair_20261008
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.live.fpl_snapshot import load, refresh
from src.live.lines import refresh_lines

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT
    / "data"
    / "predictions"
    / "2026-27"
    / "gw06"
    / f"betfair_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for bootstrap, fixtures, gw_lines, and Betfair meta",
    )
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    refresh(out)
    snap = load(out)
    names = {int(t["id"]): str(t["name"]) for t in snap["bootstrap"]["teams"]}
    result = refresh_lines(
        fixtures=snap["fixtures"],
        team_names=names,
        lines_path=out / "gw_lines.csv",
        raw_path=out / "betfair_trial.json",
        meta_path=out / "betfair_meta.json",
    )
    meta = result["meta"]
    print(
        f"out={out} rows={result['rows']} reason={meta.get('reason')} "
        f"tiers={meta.get('tiers') or {}}"
    )
    if not meta.get("ok"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
