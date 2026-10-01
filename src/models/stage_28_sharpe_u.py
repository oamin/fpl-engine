"""Stage 28 — Sharpe-u free-transfer climb.

Same FT rules as season_climb_ft (H=3, γ=0.9, hold 1.25, hit −4).
The only change is the score: u = score_xp / (σ + 1), with σ causal.

Pass bar, pre-registered: sharpe_u_ft total >= paired xp_ft + 34
on the same season, same gameweeks, same rules.

Writes:
  data/processed/stage_28_sharpe_u.csv
  data/plots/stage_28_sharpe_u.png
  reports/stage_28_sharpe_u.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.ridge_multiseason import EVAL_SEASON, SEASONS, build_one_season
from src.models.season_climb import summarize
from src.models.season_climb_ft import (
    HOLD_EPS,
    HORIZON,
    load_vaastav_roster,
    run_ft_season,
)
from src.models.sharpe_u import EPS, add_causal_sharpe_u

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

PASS_MARGIN = 34.0
GW_START = 5
GW_END = 38


def _fd_code(season: str) -> str:
    for s, code in SEASONS:
        if s == season:
            return code
    raise KeyError(season)


def _played_points(xi: pd.DataFrame, player_id: str) -> float:
    row = xi.loc[xi["player_id"].astype(str) == str(player_id)]
    if row.empty:
        return 0.0
    minutes = float(pd.to_numeric(row["minutes"], errors="coerce").fillna(0.0).iloc[0])
    points = float(pd.to_numeric(row["total_points"], errors="coerce").fillna(0.0).iloc[0])
    if minutes <= 0:
        return 0.0
    return points


def captain_gap(trace: list[dict[str, Any]], method: str = "sharpe_u_ft") -> pd.DataFrame:
    """On the u-picked XI, actual points of argmax u versus argmax score_xp."""
    rows = []
    for item in trace:
        if item["method"] != method:
            continue
        xi = item["xi"]
        if "score_xp" not in xi.columns or "score_u" not in xi.columns:
            continue
        mu = pd.to_numeric(xi["score_xp"], errors="coerce")
        util = pd.to_numeric(xi["score_u"], errors="coerce")
        cap_mu = str(xi.loc[mu.idxmax(), "player_id"])
        cap_u = str(xi.loc[util.idxmax(), "player_id"])
        rows.append(
            {
                "gw": int(item["gw"]),
                "captain_u": cap_u,
                "captain_mu": cap_mu,
                "same_captain": cap_u == cap_mu,
                "points_u": _played_points(xi, cap_u),
                "points_mu": _played_points(xi, cap_mu),
            }
        )
    return pd.DataFrame(rows)


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4), constrained_layout=True)
    colors = {"xp_ft": "steelblue", "sharpe_u_ft": "crimson"}
    for method, color in colors.items():
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        ax.plot(
            g["gw"],
            g["xi_points_cap"].cumsum(),
            "-o",
            ms=3,
            lw=2.2,
            color=color,
            label=method,
        )
    ax.set_xlabel(f"Gameweek ({EVAL_SEASON})")
    ax.set_ylabel("Cumulative XI points (captain ×2 − hits)")
    ax.set_title(f"Stage 28 — Sharpe-u vs xP (H={HORIZON})")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    gap: pd.DataFrame,
    gws: list[int],
) -> None:
    def tot(name: str) -> float:
        hit = summary.loc[summary["method"] == name, "total_points"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    xp = tot("xp_ft")
    sharpe = tot("sharpe_u_ft")
    delta = sharpe - xp

    def arm(name: str) -> pd.DataFrame:
        return weekly.loc[weekly["method"] == name]

    xp_w = arm("xp_ft")
    sh_w = arm("sharpe_u_ft")

    def tx(frame: pd.DataFrame) -> tuple[float, float, float]:
        if frame.empty:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(frame["n_transfers"].mean()),
            float(frame["hits"].sum()),
            float(frame["hit_cost"].sum()),
        )

    xp_tx, xp_hits, xp_hit_pts = tx(xp_w)
    sh_tx, sh_hits, sh_hit_pts = tx(sh_w)
    cap_delta = float((gap["points_mu"] - gap["points_u"]).sum()) if len(gap) else float("nan")
    n_disagree = int((~gap["same_captain"]).sum()) if len(gap) else 0

    if np.isfinite(delta) and delta >= PASS_MARGIN:
        verdict = f"PASS — sharpe_u_ft beats paired xp_ft by {delta:+.0f} (bar +{PASS_MARGIN:.0f})"
    else:
        verdict = (
            f"FAIL — sharpe_u_ft vs paired xp_ft is {delta:+.0f} "
            f"(bar +{PASS_MARGIN:.0f}). Park the objective."
        )

    lines = [
        "# Stage 28 — Sharpe-u free-transfer climb",
        "",
        "Agreed with the Co-PI before the run. The score is",
        rf"\(u = \mu / (\sigma + {EPS:.0f})\), \(\mu =\) `score_xp`.",
        r"\(\sigma\) is the expanding sample std of `total_points − score_xp`",
        "after a one-gameweek shift, needing 3 prior residuals.",
        "A missing sigma uses the position median of earlier gameweeks only, else 3.0,",
        "then clipped to [0.5, 8]. That u is frozen for the whole transfer horizon.",
        "",
        "Rules held fixed from `season_climb_ft`: no chips, H="
        f"{HORIZON}, hold unless ΔV ≥ {HOLD_EPS}, hit −4, formations unchanged.",
        f"Gameweeks **{gws[0]}–{gws[-1]}** (n={len(gws)}), {EVAL_SEASON}.",
        "The comparator is the paired `xp_ft` from this same run, not an older published total.",
        "",
        "## Standings (captain ×2 − hits)",
        "",
        "| method | total | mean/GW | vs xp_ft |",
        "|---|---:|---:|---:|",
        f"| xp_ft | {xp:.0f} | {xp / len(gws):.1f} | +0 |",
        f"| sharpe_u_ft | {sharpe:.0f} | {sharpe / len(gws):.1f} | {delta:+.0f} |",
        "",
        "## Transfers (the disagreeing diagnostic)",
        "",
        "| method | mean transfers/GW | hits | hit points |",
        "|---|---:|---:|---:|",
        f"| xp_ft | {xp_tx:.2f} | {xp_hits:.0f} | {xp_hit_pts:.0f} |",
        f"| sharpe_u_ft | {sh_tx:.2f} | {sh_hits:.0f} | {sh_hit_pts:.0f} |",
        "",
        "If the total rises while hits and transfers both rise, treat it as churn, not a risk edge.",
        "",
        "## Captain on the same XI",
        "",
        f"On the XI picked by u, argmax u and argmax mu differed in **{n_disagree}** "
        f"/ {len(gap)} gameweeks.",
        f"Played-minutes points of the mu captain minus the u captain: **{cap_delta:+.0f}**.",
        "Positive means the expected-points captain scored more on that same XI.",
        "This does not re-optimise transfers, and it does not apply the vice-captain.",
        "",
        "## Gate",
        "",
        f"**{verdict}**",
        "",
        "Pre-registered bar: +34 versus the paired xp_ft. The bar was not moved after the result.",
        "",
        "## Output",
        "",
        "- `data/processed/stage_28_sharpe_u.csv`",
        "- `data/plots/stage_28_sharpe_u.png`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print(f"Building {EVAL_SEASON}…", flush=True)
    feat = build_one_season(EVAL_SEASON, _fd_code(EVAL_SEASON))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    feat = add_causal_sharpe_u(feat)
    gws = [
        g
        for g in sorted(int(x) for x in feat["gw"].unique())
        if GW_START <= g <= GW_END
    ]
    if len(gws) < 10:
        raise RuntimeError(f"Too few gameweeks: {gws}")

    roster = load_vaastav_roster(EVAL_SEASON)
    trace: list[dict[str, Any]] = []
    print(f"Paired FT climbs GW {gws[0]}–{gws[-1]} H={HORIZON}…", flush=True)
    weekly_xp = run_ft_season(
        feat,
        {"xp": "score_xp"},
        gws,
        roster=roster,
        horizon=HORIZON,
        trace=trace,
    )
    print("  xp_ft done", flush=True)
    weekly_u = run_ft_season(
        feat,
        {"sharpe_u": "score_u"},
        gws,
        roster=roster,
        horizon=HORIZON,
        trace=trace,
    )
    print("  sharpe_u_ft done", flush=True)
    weekly = pd.concat([weekly_xp, weekly_u], ignore_index=True)
    summary = summarize(weekly)
    gap = captain_gap(trace)
    print(summary.to_string(index=False), flush=True)
    if len(gap):
        print(
            f"captain disagreements {(gap['same_captain'] == False).sum()} / {len(gap)}",
            flush=True,
        )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "stage_28_sharpe_u.csv", index=False)
    if len(gap):
        gap.to_csv(PROCESSED / "stage_28_captain_gap.csv", index=False)
    plot_climb(weekly, PLOTS / "stage_28_sharpe_u.png")
    write_report(REPORTS / "stage_28_sharpe_u.md", summary, weekly, gap, gws)
    return {"summary": summary, "weekly": weekly, "gap": gap, "gws": gws}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
