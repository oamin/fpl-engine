"""Graded minutes / availability overlay for cascade scoring.

The season selectable pool already hard-filters to FFScout firm ∩ expanding
minutes > 60. Players can still sit in that set with FPL ``status`` /
``chance_of_playing_*`` flags (doubtful, injured, suspended). This module
scales involvement (xmi + shares + defcon) by that availability so the
multistage MILP does not treat a 0% chance player as a full starter.

Not a second hard filter. Not a gated likelihood promotion by itself.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

# Default multipliers when chance_of_playing_* is missing.
STATUS_AVAIL = {
    "a": 1.0,
    "d": 0.50,
    "i": 0.0,
    "s": 0.0,
    "u": 0.0,
    "n": 0.0,
}


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def availability_factor(
    element: Mapping[str, Any],
    *,
    prefer_next: bool = True,
) -> float:
    """Fraction of full involvement expected this decision horizon.

    Prefers ``chance_of_playing_next_round`` for long-horizon cascade steps
    (GW t+2…), falling back to this-round chance, then status defaults.
    """
    if prefer_next:
        chance = element.get("chance_of_playing_next_round")
        if chance is None:
            chance = element.get("chance_of_playing_this_round")
    else:
        chance = element.get("chance_of_playing_this_round")
        if chance is None:
            chance = element.get("chance_of_playing_next_round")
    if chance is not None:
        try:
            return _clip01(float(chance) / 100.0)
        except (TypeError, ValueError):
            pass
    status = str(element.get("status") or "a")
    return float(STATUS_AVAIL.get(status, 1.0))


def apply_availability_overlay(
    pool: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    *,
    prefer_next: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``pool`` with involvement scaled by availability.

    Scales involvement columns by the same factor. Prefer Numagent 2.0 ``m``
    (minutes into ``score_team_side``); also scales legacy ``xmi`` /
    ``share_xG`` / ``share_xA`` / ``exp_defcon_hit`` when present. Adds
    ``avail_factor`` and ``avail_status`` columns.
    """
    by_id = {str(el["id"]): el for el in bootstrap.get("elements") or []}
    out = pool.copy()
    factors: list[float] = []
    statuses: list[str] = []
    for pid in out["player_id"].astype(str):
        el = by_id.get(pid, {})
        factors.append(availability_factor(el, prefer_next=prefer_next))
        statuses.append(str(el.get("status") or "a"))
    fac = np.asarray(factors, dtype=float)
    out["avail_factor"] = fac
    out["avail_status"] = statuses
    for col in ("m", "xmi", "share_xG", "share_xA", "exp_defcon_hit"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(float) * fac
    return out


def overlay_summary(pool: pd.DataFrame) -> dict[str, Any]:
    """Compact stats for reports."""
    if "avail_factor" not in pool.columns:
        return {"n": int(len(pool)), "overlay": False}
    fac = pd.to_numeric(pool["avail_factor"], errors="coerce").fillna(1.0)
    status = pool.get("avail_status", pd.Series(["a"] * len(pool)))
    return {
        "n": int(len(pool)),
        "overlay": True,
        "n_scaled": int((fac < 0.999).sum()),
        "n_zeroed": int((fac <= 1e-9).sum()),
        "mean_factor": float(fac.mean()),
        "status_counts": {str(k): int(v) for k, v in status.value_counts().to_dict().items()},
    }
