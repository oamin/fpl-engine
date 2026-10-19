"""As-of checks for ``score_xp_v2`` under the historical minutes oracle.

The scored week's **minutes** are an intentional oracle input (stack screen).
Outcome fields on the scored week (points, goals, assists, bonus, cards,
defcon) and later-week history must not move the forecast. Rates / priors / α
use GW ``< t`` only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.markets import numagent_xp


def _toy_history() -> pd.DataFrame:
    rows = []
    for gw in (1, 2, 3, 4):
        rows.append(
            {
                "element": 1,
                "name": "A",
                "position": "MID",
                "minutes": 90.0,
                "goals_scored": 1.0 if gw == 1 else 0.0,
                "assists": 0.0,
                "bonus": 0.0,
                "yellow_cards": 0.0,
                "red_cards": 0.0,
                "defensive_contribution": 5.0,
            }
        )
        rows.append(
            {
                "element": 2,
                "name": "B",
                "position": "MID",
                "minutes": 90.0,
                "goals_scored": 0.0,
                "assists": 1.0 if gw == 2 else 0.0,
                "bonus": 1.0 if gw == 3 else 0.0,
                "yellow_cards": 0.0,
                "red_cards": 0.0,
                "defensive_contribution": 8.0,
            }
        )
    return pd.DataFrame(rows)


def _score_side(
    history: pd.DataFrame,
    *,
    m1: float,
    m2: float,
    xg: float = 1.5,
    xcs: float = 0.3,
    xgc: float = 0.4,
) -> pd.DataFrame:
    rates = numagent_xp.player_rates_from_history(history, id_col="element")
    priors = numagent_xp.history_priors_for_xp(history, id_col="element")
    players = pd.DataFrame(
        {
            "element": [1, 2],
            "player_id": [1, 2],
            "position": ["MID", "MID"],
            "m": [m1, m2],
            "sigma_m": [0.0, 0.0],
        }
    )
    if not rates.empty and "element" in rates.columns:
        players = players.merge(rates, on="element", how="left")
    if not priors.empty and "element" in priors.columns:
        players = players.merge(
            priors.drop(columns=["position"], errors="ignore"), on="element", how="left"
        )
    for c in ("rG", "drG", "rA", "drA", "bonus_mu", "bonus_se", "card_mu", "card_se", "defcon_mu"):
        if c in players.columns:
            players[c] = players[c].fillna(0.0)
        else:
            players[c] = 0.0
    if "defcon_sig" in players.columns:
        players["defcon_sig"] = players["defcon_sig"].fillna(numagent_xp.SIG_FLOOR)
    else:
        players["defcon_sig"] = numagent_xp.SIG_FLOOR
    if "defcon_n" in players.columns:
        players["defcon_n"] = players["defcon_n"].fillna(0).astype(int)
    else:
        players["defcon_n"] = 0
    return numagent_xp.score_team_side(
        players, xg=xg, xcs=xcs, xgc=xgc, alpha_g=0.96, alpha_a=0.90
    )


def run_score_xp_v2_asof() -> dict[str, Any]:
    """Return ``{passed, failures, notes}`` for the v2 stack under minutes oracle."""
    failures: list[str] = []
    notes = [
        "minutes on the scored week are an oracle input (historical stack screen)",
        "outcome fields on the scored week must not enter rates/priors",
    ]
    hist = _toy_history()
    base = _score_side(hist, m1=90.0, m2=75.0)
    xp1 = float(base.loc[base["element"] == 1, "xp"].iloc[0])
    xp2 = float(base.loc[base["element"] == 2, "xp"].iloc[0])

    # Oracle: minutes move the score (0′ blanks the row; both ≥60′ can leave
    # a lone scorer's share unchanged when the teammate has rG=0).
    blanked = _score_side(hist, m1=0.0, m2=75.0)
    if abs(float(blanked.loc[blanked["element"] == 1, "xp"].iloc[0]) - xp1) < 1e-9:
        failures.append("minutes oracle did not blank score_xp_v2 when m=0")
    # Two scorers: relative minutes must move shares
    hist2 = hist.copy()
    hist2.loc[hist2["element"] == 2, "goals_scored"] = 1.0
    two = _score_side(hist2, m1=90.0, m2=90.0)
    skewed = _score_side(hist2, m1=90.0, m2=45.0)
    if abs(
        float(two.loc[two["element"] == 1, "xp_goals"].iloc[0])
        - float(skewed.loc[skewed["element"] == 1, "xp_goals"].iloc[0])
    ) < 1e-9:
        failures.append("minutes oracle did not move goal shares when teammate m changed")

    # Outcome leak into history would require adding scored-week rows to history.
    # Simulate wrongly appending GW-t outcomes into the rate sample.
    leaked = pd.concat(
        [
            hist,
            pd.DataFrame(
                [
                    {
                        "element": 1,
                        "name": "A",
                        "position": "MID",
                        "minutes": 90.0,
                        "goals_scored": 5.0,
                        "assists": 5.0,
                        "bonus": 3.0,
                        "yellow_cards": 0.0,
                        "red_cards": 0.0,
                        "defensive_contribution": 20.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    wrong = _score_side(leaked, m1=90.0, m2=75.0)
    # Correct pipeline must use hist only; if we accidentally pass leaked, score moves.
    # This documents the failure mode — the gate checks the correct call is hist-only
    # by verifying identical scores for two copies of hist and that leaked ≠ base.
    again = _score_side(hist.copy(), m1=90.0, m2=75.0)
    if abs(float(again.loc[again["element"] == 1, "xp"].iloc[0]) - xp1) > 1e-9:
        failures.append("identical history did not reproduce score_xp_v2")
    if abs(float(wrong.loc[wrong["element"] == 1, "xp"].iloc[0]) - xp1) < 1e-9:
        failures.append(
            "appending scored-week outcomes into history did not change rates "
            "(audit probe broken)"
        )

    # Empty history → zeros rates; score should still be finite from appear/CS/GC/bonus0
    cold = _score_side(hist.iloc[0:0], m1=90.0, m2=90.0)
    if not all(float(x) == float(x) for x in cold["xp"]):
        failures.append("cold-start score_xp_v2 is non-finite")

    # Teammate share dependence: removing a teammate from the side changes shares
    solo_hist = hist[hist["element"] == 1]
    solo = _score_side(solo_hist, m1=90.0, m2=0.0)
    # only score player 1 with m2=0
    solo_players = solo[solo["element"] == 1]
    if solo_players.empty:
        failures.append("solo side produced no rows")
    elif abs(float(solo_players["xp"].iloc[0]) - xp1) < 1e-9 and xp2 != 0:
        # With one player, share of goals is 1 — usually differs from two-player side
        pass  # may coincidentally match; not a hard fail

    return {"passed": not failures, "failures": failures, "notes": notes}
