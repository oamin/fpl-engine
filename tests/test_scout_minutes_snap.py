"""Scout horizon law: pin known steps, then snap to expanding mean (no ρ)."""

from __future__ import annotations

from src.live.scout_minutes import ScoutMinutesPacket, ScoutOverlay


def _packet(*ovs: ScoutOverlay) -> ScoutMinutesPacket:
    overlays = {ov.element: ov for ov in ovs}
    baseline = {eid: 60.0 for eid in overlays}
    baseline[999] = 70.0  # no overlay
    return ScoutMinutesPacket(
        season="2026-27",
        decision_gw=9,
        horizon=4,
        source="test",
        fetched_at="t",
        baseline_m=baseline,
        overlays=overlays,
    )


def test_no_overlay_is_expand() -> None:
    p = _packet()
    assert p.m_at(999, 0) == 70.0
    assert p.m_at(999, 3) == 70.0


def test_single_week_pin_then_snap() -> None:
    ov = ScoutOverlay(
        element=399,
        name="Cherki",
        club="Man City",
        tag="benched",
        m0=15.0,
        reason="xi_omit",
        force_steps={0: 15.0},
        return_gws=1,
    )
    p = _packet(ov)
    assert p.m_at(399, 0) == 15.0
    assert p.m_at(399, 1) == 60.0
    assert p.m_at(399, 2) == 60.0
    assert p.minutes_path(399) == [15.0, 60.0, 60.0, 60.0]


def test_multi_week_force_then_snap() -> None:
    ov = ScoutOverlay(
        element=102,
        name="Yarmoliuk",
        club="Brentford",
        tag="injured",
        m0=0.0,
        reason="banned",
        force_steps={0: 0.0, 1: 0.0, 2: 0.0},
        return_gws=3,
    )
    p = _packet(ov)
    assert p.m_at(102, 0) == 0.0
    assert p.m_at(102, 2) == 0.0
    assert p.m_at(102, 3) == 60.0


def test_doubt_step0_then_snap() -> None:
    ov = ScoutOverlay(
        element=8,
        name="Calafiori",
        club="Arsenal",
        tag="injured",
        m0=48.0,
        reason="doubt",
        chance=75.0,
        force_steps={},
        return_gws=None,
    )
    p = _packet(ov)
    assert p.m_at(8, 0) == 48.0
    assert p.m_at(8, 1) == 60.0


def test_return_gws_without_force_map() -> None:
    ov = ScoutOverlay(
        element=1,
        name="X",
        club="Y",
        tag="injured",
        m0=0.0,
        reason="out",
        force_steps={},
        return_gws=2,
    )
    p = _packet(ov)
    assert p.m_at(1, 0) == 0.0
    assert p.m_at(1, 1) == 0.0
    assert p.m_at(1, 2) == 60.0


def test_unknown_return_stays_zero() -> None:
    ov = ScoutOverlay(
        element=12,
        name="Saka",
        club="Arsenal",
        tag="injured",
        m0=0.0,
        reason="out",
        context="injured_unknown_return",
        force_steps={},
        return_gws=None,
    )
    p = _packet(ov)
    assert p.minutes_path(12) == [0.0, 0.0, 0.0, 0.0]
