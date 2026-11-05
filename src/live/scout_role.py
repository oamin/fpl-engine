"""Closed scout role tags from as-of minutes.

Handful of roles so the live packet can isolate similar players quickly.
Role is the **playing** pattern when available — not injury/suspension state
(that stays on ``context``).

Tags
----
- ``firm_starter`` — nailed when fit (Virgil)
- ``tactical_starter`` — often starts, managed/early-sub minutes (Cherki)
- ``rotational`` — shares starts; volatile week-to-week
- ``impact_sub`` — rarely starts; short/mid cameo minutes
- ``fringe`` — rarely used

  Within-season only (``GW < decision_gw``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ROLE_FIRM = "firm_starter"
ROLE_TACTICAL = "tactical_starter"
ROLE_ROTATIONAL = "rotational"
ROLE_IMPACT = "impact_sub"
ROLE_FRINGE = "fringe"

ROLE_TAGS: tuple[str, ...] = (
    ROLE_FIRM,
    ROLE_TACTICAL,
    ROLE_ROTATIONAL,
    ROLE_IMPACT,
    ROLE_FRINGE,
)


@dataclass(frozen=True)
class RoleFeatures:
    n_gws: int
    expand_m: float
    start60_rate: float
    start45_rate: float
    early_sub_rate: float  # among appearances ≥45′: share with 45–75′
    zero_rate: float
    on_mean: float
    std: float


@dataclass(frozen=True)
class RoleRow:
    element: int
    name: str
    club: str
    role: str
    reason: str
    features: RoleFeatures


def minutes_features(series: Sequence[float]) -> RoleFeatures:
    xs = [float(x) for x in series]
    n = len(xs)
    if n == 0:
        return RoleFeatures(0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    arr = xs
    expand = sum(arr) / n
    start60 = sum(1 for m in arr if m >= 60.0) / n
    start45 = sum(1 for m in arr if m >= 45.0) / n
    zero = sum(1 for m in arr if m <= 0.0) / n
    on = [m for m in arr if m > 0.0]
    on_mean = sum(on) / len(on) if on else 0.0
    started = [m for m in arr if m >= 45.0]
    early = (
        sum(1 for m in started if m <= 75.0) / len(started) if started else 0.0
    )
    mean = expand
    var = sum((m - mean) ** 2 for m in arr) / n
    std = var**0.5
    return RoleFeatures(
        n_gws=n,
        expand_m=float(expand),
        start60_rate=float(start60),
        start45_rate=float(start45),
        early_sub_rate=float(early),
        zero_rate=float(zero),
        on_mean=float(on_mean),
        std=float(std),
    )


def classify_role(
    series: Sequence[float],
    *,
    xi_omit: bool = False,
) -> tuple[str, str]:
    """Return ``(role_tag, short_reason)`` from as-of minutes (+ light cues)."""
    f = minutes_features(series)
    if f.n_gws < 2:
        return ROLE_FRINGE, "too_few_gws"

    # Firm starter: almost always starts and finishes (or huge expand).
    if f.start60_rate >= 0.75 and f.expand_m >= 65.0 and (
        f.early_sub_rate < 0.55 or f.on_mean >= 78.0
    ):
        return ROLE_FIRM, f"start60={f.start60_rate:.2f} expand={f.expand_m:.0f}"

    # Tactical starter: regularly involved in XI but managed minutes.
    managed = f.early_sub_rate >= 0.35 and f.start45_rate >= 0.35
    if f.expand_m >= 40.0 and f.start45_rate >= 0.35 and managed:
        return ROLE_TACTICAL, "managed_minutes"

    # Impact sub: rare starts, cameo on-pitch mean.
    if f.start60_rate <= 0.20 and f.zero_rate >= 0.35 and 10.0 <= f.on_mean <= 45.0:
        return ROLE_IMPACT, f"on_mean={f.on_mean:.0f} start60={f.start60_rate:.2f}"
    if f.start60_rate <= 0.15 and f.expand_m < 25.0 and f.on_mean > 0 and f.on_mean < 50.0:
        return ROLE_IMPACT, f"bench_cameo on_mean={f.on_mean:.0f}"

    # Rotational: shares starts / volatile (incl. many XI-omit regulars).
    if f.expand_m >= 25.0 and 0.20 <= f.start60_rate <= 0.70:
        return ROLE_ROTATIONAL, f"start60={f.start60_rate:.2f} std={f.std:.0f}"
    if xi_omit and f.expand_m >= 40.0:
        return ROLE_ROTATIONAL, "xi_omit_regular"

    if f.expand_m < 20.0 or f.zero_rate >= 0.75:
        return ROLE_FRINGE, f"expand={f.expand_m:.0f} zero_rate={f.zero_rate:.2f}"

    # Soft fallback by expand.
    if f.expand_m >= 65.0 and f.start45_rate >= 0.60:
        return ROLE_FIRM, "high_expand_fallback"
    if f.expand_m >= 40.0:
        return ROLE_ROTATIONAL, "mid_expand_fallback"
    if f.on_mean > 0 and f.on_mean < 50.0:
        return ROLE_IMPACT, "low_expand_cameo_fallback"
    return ROLE_FRINGE, "fallback"


def attach_roles(
    *,
    bootstrap: Mapping[str, Any],
    series_by_element: Mapping[int, Sequence[float]],
    baseline_m: Mapping[int, float],
    overlays: Mapping[int, Any] | None = None,
) -> tuple[dict[int, RoleRow], dict[str, int]]:
    """Classify roles for the full as-of pool; stamp ``role`` onto overlays."""
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    overlays = overlays or {}
    rows: dict[int, RoleRow] = {}
    counts: dict[str, int] = {r: 0 for r in ROLE_TAGS}

    for el in bootstrap["elements"]:
        eid = int(el["id"])
        name = str(el.get("web_name") or "")
        club = teams.get(int(el["team"]), "")
        series = list(series_by_element.get(eid) or [])
        ov = overlays.get(eid)
        xi_omit = bool(
            ov is not None and str(getattr(ov, "source", "")) == "ffscout_xi_omit"
        )
        role, reason = classify_role(series, xi_omit=xi_omit)
        feat = minutes_features(series)
        rows[eid] = RoleRow(
            element=eid,
            name=name,
            club=club,
            role=role,
            reason=reason,
            features=feat,
        )
        counts[role] = counts.get(role, 0) + 1
        if ov is not None and hasattr(ov, "role"):
            ov.role = role
            if hasattr(ov, "role_reason"):
                ov.role_reason = reason

    return rows, counts
