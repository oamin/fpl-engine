"""Closed scout role tags."""

from __future__ import annotations

from src.live.scout_role import (
    ROLE_CUP,
    ROLE_FIRM,
    ROLE_FRINGE,
    ROLE_IMPACT,
    ROLE_ROTATIONAL,
    ROLE_TACTICAL,
    blurb_cup_cue,
    classify_role,
)


def test_virgil_firm_starter() -> None:
    role, _ = classify_role([90.0] * 8)
    assert role == ROLE_FIRM


def test_cherki_tactical_starter() -> None:
    # Starts often but managed / early-sub band.
    series = [27.0, 81.0, 65.0, 45.0, 84.0, 90.0, 0.0, 45.0]
    role, reason = classify_role(series, anti_corr=-0.76)
    assert role == ROLE_TACTICAL
    assert reason in {"managed_minutes", "anti_corr_partner"}


def test_impact_sub() -> None:
    series = [0.0, 0.0, 18.0, 0.0, 25.0, 0.0, 12.0, 0.0]
    role, _ = classify_role(series)
    assert role == ROLE_IMPACT


def test_fringe() -> None:
    role, _ = classify_role([0.0] * 8)
    assert role == ROLE_FRINGE


def test_rotational_mid() -> None:
    series = [90.0, 0.0, 90.0, 0.0, 70.0, 20.0, 90.0, 0.0]
    role, _ = classify_role(series)
    assert role == ROLE_ROTATIONAL


def test_cup_starter_cue() -> None:
    series = [0.0, 0.0, 90.0, 0.0, 0.0, 20.0, 0.0, 0.0]
    assert blurb_cup_cue(
        "Lewis",
        "Pep confirmed Rico Lewis started the cup side midweek and may be rested.",
    )
    # Club-wide EFL Cup mention must not tag unrelated players.
    assert not blurb_cup_cue(
        "Travers",
        "David Moyes' presser was filmed after Thursday's EFL Cup win over Newcastle.",
    )
    assert not blurb_cup_cue("", "EFL Cup win midweek.")
    role, _ = classify_role(series, cup_cue=True)
    assert role == ROLE_CUP
