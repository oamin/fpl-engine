"""Unit tests for scout blurb / FPL return parsing."""

from __future__ import annotations

from datetime import date

from src.live.scout_context import (
    CTX_INJURED_BACK,
    CTX_ROTATION_SINGLE,
    CTX_SUSPENDED,
    classify_context,
    parse_blurb_return,
    parse_fpl_return_date,
    return_gws_from_date,
)


def test_parse_fpl_expected_back_date() -> None:
    d = parse_fpl_return_date("Hamstring injury - Expected back 21 Nov", year_hint=2026)
    assert d == date(2026, 11, 21)


def test_return_gws_from_date_across_break() -> None:
    deadlines = {
        9: date(2026, 10, 31),
        10: date(2026, 11, 6),
        11: date(2026, 11, 21),
        12: date(2026, 11, 27),
    }
    n, rgw = return_gws_from_date(
        date(2026, 11, 21), decision_gw=9, deadlines=deadlines
    )
    assert rgw == 11
    assert n == 2


def test_blurb_next_two_gameweeks() -> None:
    h = parse_blurb_return(
        "Lacroix is out of the next two Gameweeks with a hamstring issue",
        decision_gw=9,
        deadlines={},
    )
    assert h is not None
    assert h.return_gws == 2
    assert h.source == "blurb_next_n_gw"


def test_blurb_until_gw() -> None:
    h = parse_blurb_return(
        "Scott won't return until Gameweek 10 at the earliest",
        decision_gw=9,
        deadlines={},
    )
    assert h is not None
    assert h.return_gws == 1
    assert h.return_gw == 10


def test_blurb_three_match_ban() -> None:
    h = parse_blurb_return(
        "Dowman begins a three-match ban",
        decision_gw=9,
        deadlines={},
        player_name="Dowman",
    )
    assert h is not None
    assert h.return_gws == 3


def test_blurb_ban_does_not_attach_to_other_player() -> None:
    sent = (
        "Bukayo Saka (quad) and William Saliba (back) remain out, while "
        "Arsenal's appeal against Max Dowman's midweek red card has failed, "
        "so the teenager begins a three-match ban."
    )
    # Clause split drops the ban half for Saka-only lookups via sentences_about;
    # even on the full string, ban requires name proximity.
    h = parse_blurb_return(
        sent, decision_gw=9, deadlines={}, player_name="Saka"
    )
    assert h is None or h.source != "blurb_n_match_ban"


def test_blurb_internationals() -> None:
    h = parse_blurb_return(
        "He won't play again before the November internationals",
        decision_gw=9,
        deadlines={},
    )
    assert h is not None
    assert h.return_gw == 11
    assert h.return_gws == 2


def test_classify_injured_back_label() -> None:
    from src.live.scout_context import ReturnHint, ScoutContextRow

    hint = ReturnHint(return_gws=2, source="blurb_next_n_gw", snippet="next two")
    ctx, _ = classify_context(
        bucket="out",
        tag="injured",
        chance=0.0,
        fpl_status="i",
        fpl_news="Hamstring",
        hint=hint,
        expand_m=70.0,
        recent=[90, 90, 0],
    )
    assert ctx == CTX_INJURED_BACK
    row = ScoutContextRow(
        element=1,
        name="Lacroix",
        club="Chelsea",
        bucket="out",
        context=ctx,
        context_detail="",
        tag="injured",
        m0=0.0,
        return_gws=2,
    )
    assert row.human_label() == "injured, expected back in 2 weeks"


def test_classify_rotation_single() -> None:
    ctx, _ = classify_context(
        bucket="xi_omit",
        tag="benched",
        chance=None,
        fpl_status="a",
        fpl_news="",
        hint=None,
        expand_m=70.0,
        recent=[90.0, 85.0, 90.0],
    )
    assert ctx == CTX_ROTATION_SINGLE


def test_classify_suspended() -> None:
    from src.live.scout_context import ReturnHint

    ctx, _ = classify_context(
        bucket="banned",
        tag="injured",
        chance=0.0,
        fpl_status="s",
        fpl_news="",
        hint=ReturnHint(return_gws=1, source="blurb_n_match_ban"),
        expand_m=60.0,
        recent=[90.0],
    )
    assert ctx == CTX_SUSPENDED
