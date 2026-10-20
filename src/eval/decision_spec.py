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
# Numagent 2.0 product lock (protocol.live_primary / numagent_2_0).
# Numagent 1.0 used ep_next from GW6; archived under protocol.numagent_1_0.
LIVE_PRIMARY = "score_xp_v2"
LIVE_SHADOW = None
LIVE_HISTORICAL_SCORE = "score_xp_v2"
LIVE_PRIMARY_FROM_GW = 7
LIVE_PRIMARY_WIRED = True
# GW6 week_freeze ledger start (ep_next-era bar). Not the product from_gw.
LIVE_FREEZE_BAR_FROM_GW = 6
# PI lock file under data/predictions/<season>/gwNN/numagent_lock.json.
# An earlier board CSV is not a fallback.
LIVE_DECISION_CAPTURE = "numagent_lock"
LIVE_CAPTURE_FALLBACK = "none"
LIVE_ENTRYPOINT = "src/live/numagent_live_pick.py"
# A logged judgement. It is not a chip-rule result and it does not count.
GW6_SUBMITTED_CHIP = "logged_judgement"
GW6_CHIP_COUNTS = False
# A written zero in the minutes file stays zero. A manual override is a log
# entry that names the player, the value, and the reason.
LIVE_INJURY_FLAGS = "minutes_file_zero_stays_zero"
LIVE_MANUAL_OVERRIDE = "logged_with_reason"
LIVE_CHOOSE_AFTER_SCORES = False
LIVE_REVIEW_GW = 26
LOSO_BOOTSTRAP = 1000
LOSO_SEED = 0
LOSO_FLOOR = 20
LOSO_CONDITIONAL_FLOOR = 5
LOSO_MIN_SEASONS = 2
