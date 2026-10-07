"""Locked constants for the first decision-layer batch.

The replay reads these. The protocol records the same numbers. A result is
not a reason to edit either copy.
"""

from __future__ import annotations

SCORE_COLUMN = "score_xp"
LOGGED_ALONGSIDE = "ep_next"
CAPTAIN_BASELINE = "score_exp_points"
MIN_LIVE_WEEKS = 20
HORIZON = 3

# Sequential price targets in tenths of £m. Slot order is the assignment order.
TEMPLATE_SLOTS: tuple[tuple[str, int], ...] = (
    ("GKP", 45),
    ("GKP", 40),
    ("DEF", 65),
    ("DEF", 50),
    ("DEF", 45),
    ("DEF", 40),
    ("DEF", 40),
    ("MID", 90),
    ("MID", 65),
    ("MID", 55),
    ("MID", 50),
    ("MID", 45),
    ("FWD", 110),
    ("FWD", 75),
    ("FWD", 55),
)

POWER_WEEKS = 20
POWER_LEVEL = 0.8
POWER_BOOTSTRAP = 1000
POWER_SEED = 0
POWER_SIMS = 400
POWER_STEP = 0.0005
POWER_GRID_MAX = 0.04

# Inclusive windows. Actions are often late, so neither slot is one hour wide.
# t24 is 20–28 hours before the deadline. t1 is 15 minutes to 3 hours before it.
T24_HOURS = (20, 28)
T1_MINUTES = (15, 180)
SHUFFLE_SEED = 0
LIVE_COVERS_ZERO = "undetermined"
LIVE_CONTINUE_TO_GW = 38
