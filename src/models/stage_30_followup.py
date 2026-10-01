"""Stage 30 — arm-2 replication, then an ownership screen.

Replication (locked before the run): arm 2 on 2022/23, 2023/24, 2024/25.
XI and captain stay on score_xp. Transfer value uses score_xp - 0.25*sigma.
A season passes if it beats paired xp_ft by at least 34.
Replicated if at least 2 of the 3 seasons pass. Exactly one is inconclusive.
None means the 2025/26 result stays one season. Do not rerun 2025/26 arm 2.

Ownership (2025/26 only): deadline `selected` gives
ow = 15 * selected / sum(selected) within the gameweek, summed over the
full Vaastav list. Score = score_xp * (1 - 0.5 * ow).
Fast XI screen first. Skip the transfer climb if it trails xp by 100 or more.

Writes:
  data/processed/stage_30_followup.csv
  reports/stage_30_followup.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.ridge_multiseason import CACHE, SEASONS, build_one_season
from src.models.season_climb import summarize
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.sharpe_u import LAMBDA_RISK, add_causal_sharpe_u
from src.models.stage_29_batch import GW_END, GW_START, KILL_GAP, PASS_MARGIN, fast_xi

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PRIOR = [pair for pair in SEASONS if pair[0] != "2025-26"]
OWN_WEIGHT = 0.5


def _gws(feat: pd.DataFrame) -> list[int]:
    return [
        g
        for g in sorted(int(x) for x in feat["gw"].unique())
        if GW_START <= g <= GW_END
    ]


def _prepare(season: str, code: str) -> pd.DataFrame:
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    return add_causal_sharpe_u(feat)


def _tot(summary: pd.DataFrame, method: str) -> float:
    hit = summary.loc[summary["method"] == method, "total_points"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def attach_ownership(feat: pd.DataFrame, season: str) -> pd.DataFrame:
    """Deadline ownership share from Vaastav `selected`. Sum is the full list."""
    raw = pd.read_csv(CACHE / f"merged_gw_{season.replace('-', '_')}.csv")
    sel = pd.DataFrame(
        {
            "element": raw["element"].astype(str),
            "gw": pd.to_numeric(raw["GW"], errors="coerce").astype(int),
            "selected": pd.to_numeric(raw["selected"], errors="coerce").fillna(0.0),
        }
    ).drop_duplicates(["element", "gw"], keep="first")
    totals = sel.groupby("gw")["selected"].transform("sum")
    sel["ow"] = (15.0 * sel["selected"] / totals.replace(0, pd.NA)).fillna(0.0)
    out = feat.copy()
    if "element" not in out.columns:
        out["element"] = out["player_id"].astype(str).str.split(":").str[-1]
    out = out.merge(sel[["element", "gw", "ow"]], on=["element", "gw"], how="left")
    out["ow"] = out["ow"].fillna(out.groupby("gw")["ow"].transform("median")).fillna(0.0)
    out["score_own"] = pd.to_numeric(out["score_xp"], errors="coerce") * (
        1.0 - OWN_WEIGHT * out["ow"]
    )
    return out


def run_arm2(feat: pd.DataFrame, season: str, gws: list[int]) -> pd.DataFrame:
    roster = load_vaastav_roster(season)
    print(f"  {season} xp_ft…", flush=True)
    xp = run_ft_season(feat, {"xp": "score_xp"}, gws, roster=roster, horizon=HORIZON)
    print(f"  {season} arm2…", flush=True)
    arm = run_ft_season(
        feat,
        {"risk_value": "score_xp"},
        gws,
        roster=roster,
        horizon=HORIZON,
        value_col="score_risk",
    )
    both = pd.concat([xp, arm], ignore_index=True)
    both.insert(0, "season", season)
    return both


def write_report(
    path: Path,
    rep_summary: pd.DataFrame,
    own_fast: pd.DataFrame | None,
    own_ft: pd.DataFrame | None,
    own_killed: bool | None,
) -> None:
    lines = [
        "# Stage 30 — replication and ownership",
        "",
        "Locked with the Co-PI before these totals were read. "
        f"Lambda stays {LAMBDA_RISK}. Pass bar stays +{PASS_MARGIN:.0f}. "
        "2025/26 arm 2 was not rerun.",
        "",
        "A prior season passes if arm 2 beats that season's xp_ft by at least 34. "
        "Replicated if at least 2 of the 3 seasons pass. Exactly one is inconclusive. "
        "None keeps the 2025/26 result as one season.",
        "",
        "## Arm 2 on earlier seasons",
        "",
        "| season | xp_ft | arm2 | delta | pass |",
        "|---|---:|---:|---:|---|",
    ]
    n_pass = 0
    for season, _code in PRIOR:
        chunk = rep_summary.loc[rep_summary["season"] == season]
        xp = _tot(chunk, "xp_ft")
        arm = _tot(chunk, "risk_value_ft")
        delta = arm - xp
        passed = bool(delta >= PASS_MARGIN)
        n_pass += int(passed)
        lines.append(
            f"| {season} | {xp:.0f} | {arm:.0f} | {delta:+.0f} | "
            f"{'yes' if passed else 'no'} |"
        )
    if n_pass >= 2:
        verdict = f"REPLICATED — {n_pass} of 3 prior seasons cleared +34."
    elif n_pass == 1:
        verdict = "INCONCLUSIVE — exactly 1 of 3 prior seasons cleared +34."
    else:
        verdict = "ONE SEASON — 0 of 3 prior seasons cleared +34. Park the 2025/26 +90 as unreplicated."
    lines += ["", f"**{verdict}**", ""]

    lines += ["## Ownership, 2025/26", ""]
    if own_fast is None:
        lines.append("Not run.")
    else:
        xp_fast = _tot(own_fast, "xp_fast")
        own = _tot(own_fast, "own_fast")
        lines += [
            "Score = score_xp * (1 - 0.5 * ow), with ow = 15 * selected / sum(selected) "
            "over the full gameweek list.",
            "",
            "| method | total | vs xp_fast |",
            "|---|---:|---:|",
            f"| xp_fast | {xp_fast:.0f} | +0 |",
            f"| own_fast | {own:.0f} | {own - xp_fast:+.0f} |",
            "",
            f"Transfer climb: **{'skipped' if own_killed else 'run'}**.",
            "",
        ]
        if own_ft is not None and not own_killed:
            xp_ft = _tot(own_ft, "xp_ft")
            own_ft_tot = _tot(own_ft, "own_ft")
            delta = own_ft_tot - xp_ft
            lines += [
                "| method | total | vs xp_ft |",
                "|---|---:|---:|",
                f"| xp_ft | {xp_ft:.0f} | +0 |",
                f"| own_ft | {own_ft_tot:.0f} | {delta:+.0f} |",
                "",
            ]
            if delta >= PASS_MARGIN:
                lines.append(
                    "Ownership cleared +34 on this one season. Best of its own screen, "
                    "not a confirmed edge."
                )
            else:
                lines.append("Ownership did not clear +34. Park this formula.")
    lines += ["", "- `data/processed/stage_30_followup.csv`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    rep_frames: list[pd.DataFrame] = []
    rep_rows: list[pd.DataFrame] = []
    for season, code in PRIOR:
        print(f"Building {season}…", flush=True)
        feat = _prepare(season, code)
        gws = _gws(feat)
        if len(gws) < 10:
            raise RuntimeError(f"{season} has too few gameweeks: {gws}")
        weekly = run_arm2(feat, season, gws)
        rep_frames.append(weekly)
        summary = summarize(weekly)
        summary.insert(0, "season", season)
        rep_rows.append(summary)
        print(summary.to_string(index=False), flush=True)
    rep_summary = pd.concat(rep_rows, ignore_index=True)
    rep_weekly = pd.concat(rep_frames, ignore_index=True)

    print("Ownership screen 2025/26…", flush=True)
    own_feat = attach_ownership(_prepare("2025-26", "2526"), "2025-26")
    gws = _gws(own_feat)
    fast = fast_xi(own_feat, gws, {"xp_fast": "score_xp", "own_fast": "score_own"})
    fast_summary = summarize(fast)
    print(fast_summary.to_string(index=False), flush=True)
    own_killed = _tot(fast_summary, "own_fast") <= _tot(fast_summary, "xp_fast") - KILL_GAP
    own_ft_summary = None
    own_weekly = fast
    if not own_killed:
        roster = load_vaastav_roster("2025-26")
        print("Ownership FT climbs…", flush=True)
        xp = run_ft_season(own_feat, {"xp": "score_xp"}, gws, roster=roster, horizon=HORIZON)
        own = run_ft_season(
            own_feat, {"own": "score_own"}, gws, roster=roster, horizon=HORIZON
        )
        own_weekly = pd.concat([fast, xp, own], ignore_index=True)
        own_ft_summary = summarize(own_weekly.loc[own_weekly["mode"] == "ft"])
        print(own_ft_summary.to_string(index=False), flush=True)

    weekly = pd.concat([rep_weekly, own_weekly], ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "stage_30_followup.csv", index=False)
    write_report(
        REPORTS / "stage_30_followup.md",
        rep_summary,
        fast_summary,
        own_ft_summary,
        own_killed,
    )
    print(f"Wrote {REPORTS / 'stage_30_followup.md'}", flush=True)
    return {"replication": rep_summary, "own_killed": own_killed}


if __name__ == "__main__":
    run()
