"""Expected minutes supplied before the deadline.

The live planner does not estimate minutes. A missing file is an error.
A player omitted from the file is treated as zero minutes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ("player_id", "gw", "xmi")
# Optional Numagent 2.0 minutes uncertainty (regime SD).
SIGMA_COL = "xmi_sigma"
# One fixture is at most 90. A double gameweek can list up to 180.
XMI_MAX = 180.0


class LiveInputError(ValueError):
    """The live planner is missing an input it is not allowed to invent."""


def load_xmi(path: Path | str, gw: int) -> pd.DataFrame:
    """Rows for one gameweek. Empty, duplicate, or out-of-range files raise.

    ``xmi_sigma`` is optional; missing values become 0.
    """
    file = Path(path)
    if not file.is_file():
        raise LiveInputError(
            f"No minutes file at {file}. Gemini supplies xmi before a live plan. "
            "The historical rolling minutes model is not used."
        )
    frame = pd.read_csv(file)
    missing = [col for col in COLUMNS if col not in frame.columns]
    if missing:
        raise LiveInputError(f"minutes file is missing columns: {missing}")
    cols = list(COLUMNS) + ([SIGMA_COL] if SIGMA_COL in frame.columns else [])
    use = frame.loc[pd.to_numeric(frame["gw"], errors="coerce") == int(gw), cols].copy()
    if use.empty:
        raise LiveInputError(f"minutes file has no rows for GW{gw}.")
    use["player_id"] = use["player_id"].astype(str)
    use["xmi"] = pd.to_numeric(use["xmi"], errors="coerce")
    if use["xmi"].isna().any():
        raise LiveInputError("xmi must be numeric.")
    if (use["xmi"] < 0).any() or (use["xmi"] > XMI_MAX).any():
        raise LiveInputError(f"xmi must sit between 0 and {XMI_MAX:g}.")
    if use["player_id"].duplicated().any():
        raise LiveInputError(f"duplicate player_id in the GW{gw} minutes file.")
    if SIGMA_COL in use.columns:
        use[SIGMA_COL] = pd.to_numeric(use[SIGMA_COL], errors="coerce").fillna(0.0)
        if (use[SIGMA_COL] < 0).any():
            raise LiveInputError("xmi_sigma must be non-negative.")
    else:
        use[SIGMA_COL] = 0.0
    use["gw"] = int(gw)
    return use.reset_index(drop=True)


def apply_supplied_xmi(features: pd.DataFrame, supplied: pd.DataFrame, *, id_col: str = "player_id") -> pd.DataFrame:
    """Replace the minutes prior. Anyone not in the file gets 0.

    Call this after the historical priors and before ``compute_xp``.
    It does not change goal, assist, or clean-sheet shares.
    When ``xmi_sigma`` is present on ``supplied``, it is copied too (else 0).
    """
    out = features.copy()
    lookup = supplied.set_index("player_id")["xmi"]
    ids = out[id_col].astype(str)
    out["xmi"] = ids.map(lookup).fillna(0.0).astype(float)
    if SIGMA_COL in supplied.columns:
        sig = supplied.set_index("player_id")[SIGMA_COL]
        out[SIGMA_COL] = ids.map(sig).fillna(0.0).astype(float)
    else:
        out[SIGMA_COL] = 0.0
    return out
