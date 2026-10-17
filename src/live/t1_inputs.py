"""Files the live ``score_xp`` and ``forecast_xp`` read.

The compiled minutes sheet and the T−1 Exchange folder. The frozen holdout
files are refused. An earlier ``betfair_*`` pull is not a substitute for
the T−1 sheet.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data" / "predictions" / "2026-27"
FROZEN_MINUTES = (ROOT / "data" / "live" / "xmi_gw6.csv").resolve()
FROZEN_LINES = (ROOT / "data" / "live" / "gw_lines.csv").resolve()


class LiveScoreInputError(RuntimeError):
    """The live score is missing the T−1 file it is not allowed to replace."""


def minutes_sheet(gw: int) -> Path:
    """Compiled expected minutes for this gameweek."""
    return PREDICTIONS / f"gw{int(gw):02d}" / "xmi_t1.csv"


def exchange_dir(gw: int) -> Path:
    """T−1 Exchange pull. Not an earlier ``betfair_*`` folder."""
    return PREDICTIONS / f"gw{int(gw):02d}" / "betfair_t1"


def exchange_sheet(gw: int) -> Path:
    """T−1 ``gw_lines.csv``."""
    return exchange_dir(gw) / "gw_lines.csv"


def refuse_frozen(path: Path) -> None:
    """Raise when ``path`` is a frozen holdout file."""
    resolved = Path(path).resolve()
    if resolved == FROZEN_MINUTES or resolved == FROZEN_LINES:
        raise LiveScoreInputError(f"refusing frozen holdout file {path}")


def live_score_book(gw: int, live_path: Path | None) -> Path | None:
    """T−1 ``gw_lines.csv``, or a caller file that is not the frozen slate.

    A missing caller file stays missing. An earlier ``betfair_*`` pull is
    not a fallback, and the frozen holdout file is not opened.
    """
    if live_path is not None:
        candidate = Path(live_path)
        if candidate.resolve() == FROZEN_LINES:
            return None
        return candidate if candidate.is_file() else None
    sheet = exchange_sheet(gw)
    if sheet.is_file() and sheet.resolve() != FROZEN_LINES:
        return sheet
    return None


def require_score_inputs(gw: int) -> tuple[Path, Path]:
    """Minutes sheet and Exchange lines. Missing means stop, not an older file."""
    minutes = minutes_sheet(gw)
    sheet = exchange_sheet(gw)
    refuse_frozen(minutes)
    refuse_frozen(sheet)
    if not minutes.is_file():
        raise LiveScoreInputError(f"xmi_t1 is missing: {minutes}")
    if not sheet.is_file():
        raise LiveScoreInputError(
            f"updated Exchange sheet is missing: {sheet}. "
            "An earlier betfair folder is not used."
        )
    return minutes, sheet
