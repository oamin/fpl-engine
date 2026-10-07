"""First-half chips left after Gameweek 5. No point value is added.

The five-week gap is points scored. A chip still in the wallet is not
points. Second-half chips are a separate copy and are full on both sides,
so they cancel. An unused first-half chip dies at Gameweek 19. The hurdles
are the rule for playing, not a price for holding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rules.fpl_2026 import CHIPS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CARRY = PROCESSED / "cohort_carry_gw15.csv"


def remaining(played: set[str]) -> frozenset[str]:
    """First-half chips not in ``played``. Unknown names are ignored."""
    known = {chip for chip in played if chip in CHIPS}
    return frozenset(chip for chip in CHIPS if chip not in known)


def count_diff(model_played: set[str], human_played: set[str]) -> int:
    """Model chips left minus human chips left. Positive means the model holds more."""
    return len(remaining(model_played)) - len(remaining(human_played))


def _names(series: pd.Series) -> set[str]:
    return {str(chip) for chip in series.dropna() if str(chip) and str(chip) != "nan"}


def inventory(carry: pd.DataFrame) -> pd.DataFrame:
    """One row per manager. The gap column is the five-week points gap, unchanged."""
    rows: list[dict[str, object]] = []
    keys = ["entry_id", "label", "group"]
    for key, frame in carry.groupby(keys, sort=False):
        entry_id, label, group = key
        model = _names(frame["chip"])
        human = _names(frame["his_chip"])
        model_left = remaining(model)
        human_left = remaining(human)
        rows.append(
            {
                "entry_id": entry_id,
                "label": label,
                "group": group,
                "model_played": len(model),
                "human_played": len(human),
                "model_left": len(model_left),
                "human_left": len(human_left),
                "model_only": ",".join(sorted(model_left - human_left)),
                "human_only": ",".join(sorted(human_left - model_left)),
                "count_diff": count_diff(model, human),
                "gap": float(frame["gap"].sum()),
            }
        )
    return pd.DataFrame(rows)


def cohort_mean(frame: pd.DataFrame) -> dict[str, float]:
    """The 14. The reference row is left out, as in the carry."""
    cohort = frame.loc[frame["group"] != "reference"]
    return {
        "n": float(len(cohort)),
        "mean_gap": float(cohort["gap"].mean()),
        "mean_count_diff": float(cohort["count_diff"].mean()),
        "model_played": float(cohort["model_played"].mean()),
        "human_played": float(cohort["human_played"].mean()),
    }


def _write(frame: pd.DataFrame, summary: dict[str, float]) -> None:
    lines = [
        "# Chips left after Gameweek 5",
        "",
        "The five-week gap is points scored. A chip still held is not added "
        "to it. Each side started the half with one of each chip. The second "
        "half is a separate wallet and is still full on both sides, so those "
        "four chips cancel. What can differ is the first-half chip that has "
        "not been played. It dies if it is still unused at Gameweek 19.",
        "",
        "The model played Bench Boost and Triple Captain on every squad. "
        "It played no wildcard and no free hit. The humans played between "
        "one and four chips. No legal week in this window cleared a wildcard "
        "lead of 16 or a free-hit lead of 12, so the model did not spend those two.",
        "",
        "| manager | group | model played | human played | model holds extra | human holds extra | count | gap |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        model_only = row.model_only or "—"
        human_only = row.human_only or "—"
        lines.append(
            f"| {row.label} | {row.group} | {row.model_played:.0f} | "
            f"{row.human_played:.0f} | {model_only} | {human_only} | "
            f"{row.count_diff:+.0f} | {row.gap:+.0f} |"
        )
    diff = summary["mean_count_diff"]
    gap = summary["mean_gap"]
    lines += [
        "",
        f"On the 14, the model played {summary['model_played']:.1f} chips and "
        f"the humans played {summary['human_played']:.1f}. The model holds "
        f"{diff:.1f} more first-half chips. The mean gap stays {gap:.2f}.",
        "",
        "Closing that gap with the extra half chip would need the chip to be "
        f"worth {abs(gap) / diff:.0f} points. Nothing on file is that price. "
        "The hurdle of 16 is the lead required to play, not points in the bank. "
        "The priced leads inside this window, a median wildcard sum of about 7 "
        "and about 3 by Gameweek 5, are the value of playing then. The model "
        "turned those plays down. Gameweek 6 has not been played, so a chip "
        "held past Gameweek 5 has no realised points to add.",
        "",
        "Two of the 14, and the reference entry, still hold Bench Boost. "
        "The model has spent it. Stored chips do not all sit on the model's side. "
        "Gemini left the points unchanged "
        "([chip stock](bc-7121b96b-db3a-552d-ade8-335c01d37eda)).",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "chip_stock_gw15.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, float]:
    frame = inventory(pd.read_csv(CARRY))
    summary = cohort_mean(frame)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "chip_stock_gw15.csv", index=False)
    _write(frame, summary)
    return summary


if __name__ == "__main__":
    print(run())
