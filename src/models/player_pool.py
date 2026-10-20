"""Shared cascade player pool — FFScout firm ∩ expanding minutes.

Canonical name for the season-sticky selectable universe used by
``cascade_short`` and ``cascade_long``. Implementation lives in
``selectable_pool`` (build / snapshot / report); this module is the
public import surface.

Rules (unchanged):
1. Expanding mean minutes > 60 (as-of).
2. FFScout firm / nailed_on (predicted XI minus Out / Banned / Doubts).
3. Sticky for the season — plans do not re-scrape or re-gate per future GW.

Snapshot: ``data/predictions/2026-27/selectable_pool/selectable_pool.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.models.selectable_pool import (
    DEFAULT_POOL_DIR,
    DEFAULT_POOL_JSON,
    MIN_APPS,
    MIN_EXPANDING_MINUTES,
    apply_selectable_pool,
    build_selectable_pool,
    load_season_selectable_ids,
    load_selectable_ids,
    write_pool_outputs,
)

# Public aliases
POOL_JSON = DEFAULT_POOL_JSON
POOL_DIR = DEFAULT_POOL_DIR


def load_player_pool_ids(
    path: Path | None = None,
    *,
    missing_ok: bool = False,
) -> set[str]:
    """Load sticky season ``selectable_ids`` (bare FPL element ids as strings)."""
    if missing_ok:
        ids = load_season_selectable_ids(path, missing_ok=True)
        return ids or set()
    return load_selectable_ids(path)


def filter_board_to_pool(
    board: pd.DataFrame,
    selectable_ids: set[str] | None,
    *,
    force_ids: Sequence[str] | None = None,
    id_col: str = "player_id",
    element_col: str = "element",
) -> pd.DataFrame:
    """Keep rows in the player pool; always keep ``force_ids`` (owned squad).

    Matches bare element ids and ``season:element`` player_id forms.
    """
    if board is None or board.empty or not selectable_ids:
        return board
    out = board.copy()
    force = {str(x) for x in (force_ids or ())}
    pids = out[id_col].astype(str) if id_col in out.columns else pd.Series("", index=out.index)
    bare_from_pid = pids.str.split(":").str[-1]
    if element_col in out.columns:
        bare_el = pd.to_numeric(out[element_col], errors="coerce").fillna(-1).astype(int).astype(str)
    else:
        bare_el = bare_from_pid
    in_pool = (
        bare_el.isin(selectable_ids)
        | bare_from_pid.isin(selectable_ids)
        | pids.isin(selectable_ids)
        | pids.isin(force)
        | bare_from_pid.isin({f.split(":")[-1] for f in force})
    )
    return out.loc[in_pool].reset_index(drop=True)


def build_and_write_pool(
    *,
    bootstrap: Mapping[str, Any],
    ffscout: Mapping[str, Any] | None = None,
    html: str | None = None,
    history_rows: list[Mapping[str, Any]] | None = None,
    before_gw: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Build FFScout ∩ minutes pool and write JSON/CSV/report under ``out_dir``."""
    result = build_selectable_pool(
        bootstrap=bootstrap,
        ffscout=ffscout,
        html=html,
        history_rows=history_rows,
        before_gw=before_gw,
        min_minutes=MIN_EXPANDING_MINUTES,
        min_apps=MIN_APPS,
    )
    write_pool_outputs(result, out_dir=out_dir or POOL_DIR)
    return result


__all__ = [
    "POOL_DIR",
    "POOL_JSON",
    "MIN_EXPANDING_MINUTES",
    "MIN_APPS",
    "apply_selectable_pool",
    "build_and_write_pool",
    "build_selectable_pool",
    "filter_board_to_pool",
    "load_player_pool_ids",
    "load_season_selectable_ids",
    "load_selectable_ids",
]
