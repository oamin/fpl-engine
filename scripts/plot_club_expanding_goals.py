#!/usr/bin/env python3
"""Expanding mean / sd / median GF and GA for four clubs, 2025/26.

Reads ``data/cache/merged_gw_2025_26.csv`` (one row per team–fixture), builds
match-order expanding stats, aligns to GW1–38 with forward-fill on blanks,
writes JSON + PNG figures under ``reports/figures/``.

Clubs default: Arsenal, Liverpool, Man City, Chelsea.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "cache" / "merged_gw_2025_26.csv"
DEFAULT_OUT = ROOT / "reports" / "figures"
DEFAULT_CLUBS = ("Arsenal", "Liverpool", "Man City", "Chelsea")
GWS = list(range(1, 39))


def load_team_matches(path: Path) -> dict[str, list[dict]]:
    by_team: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["team"], row["fixture"])
            if key in seen:
                continue
            seen.add(key)
            th, ta = row["team_h_score"], row["team_a_score"]
            if th == "" or ta == "":
                continue
            th_i, ta_i = int(float(th)), int(float(ta))
            home = str(row["was_home"]).lower() in {"true", "1"}
            by_team[row["team"]].append(
                {
                    "gw": int(row["GW"]),
                    "fixture": int(row["fixture"]),
                    "kickoff": row["kickoff_time"],
                    "gf": th_i if home else ta_i,
                    "ga": ta_i if home else th_i,
                }
            )
    for team in by_team:
        by_team[team].sort(key=lambda m: (m["kickoff"], m["fixture"]))
    return by_team


def expanding(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    means: list[float] = []
    sds: list[float] = []
    meds: list[float] = []
    for i in range(len(values)):
        window = values[: i + 1]
        means.append(statistics.fmean(window))
        meds.append(float(statistics.median(window)))
        sds.append(0.0 if i == 0 else statistics.stdev(window))
    return means, sds, meds


def forward_fill(seq: list[float | None]) -> list[float]:
    out: list[float] = []
    last = 0.0
    for v in seq:
        if v is None:
            out.append(last)
        else:
            last = float(v)
            out.append(last)
    return out


def club_series(matches: list[dict]) -> dict:
    gfs: list[float] = []
    gas: list[float] = []
    snap: dict[int, dict] = {}
    gw_totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {"gf": 0.0, "ga": 0.0, "n": 0.0}
    )

    for m in matches:
        gfs.append(float(m["gf"]))
        gas.append(float(m["ga"]))
        mgf, sgf, medgf = expanding(gfs)
        mga, sga, medga = expanding(gas)
        i = len(gfs) - 1
        snap[m["gw"]] = {
            "exp_mean_gf": mgf[i],
            "exp_sd_gf": sgf[i],
            "exp_median_gf": medgf[i],
            "exp_mean_ga": mga[i],
            "exp_sd_ga": sga[i],
            "exp_median_ga": medga[i],
        }
        gw_totals[m["gw"]]["gf"] += float(m["gf"])
        gw_totals[m["gw"]]["ga"] += float(m["ga"])
        gw_totals[m["gw"]]["n"] += 1.0

    series: dict[str, list] = {
        "exp_mean_gf": [],
        "exp_sd_gf": [],
        "exp_median_gf": [],
        "exp_mean_ga": [],
        "exp_sd_ga": [],
        "exp_median_ga": [],
        "mean_plus_sd_gf": [],
        "mean_minus_sd_gf": [],
        "mean_plus_sd_ga": [],
        "mean_minus_sd_ga": [],
        "gw_gf": [],
        "gw_ga": [],
        "played": [],
    }
    last: dict | None = None
    for g in GWS:
        if g in snap:
            last = snap[g]
            series["played"].append(True)
            n = gw_totals[g]["n"]
            series["gw_gf"].append(gw_totals[g]["gf"] / n)
            series["gw_ga"].append(gw_totals[g]["ga"] / n)
        else:
            series["played"].append(False)
            series["gw_gf"].append(None)
            series["gw_ga"].append(None)
        if last is None:
            for key in (
                "exp_mean_gf",
                "exp_sd_gf",
                "exp_median_gf",
                "exp_mean_ga",
                "exp_sd_ga",
                "exp_median_ga",
                "mean_plus_sd_gf",
                "mean_minus_sd_gf",
                "mean_plus_sd_ga",
                "mean_minus_sd_ga",
            ):
                series[key].append(None)
            continue
        series["exp_mean_gf"].append(round(last["exp_mean_gf"], 3))
        series["exp_sd_gf"].append(round(last["exp_sd_gf"], 3))
        series["exp_median_gf"].append(round(last["exp_median_gf"], 3))
        series["exp_mean_ga"].append(round(last["exp_mean_ga"], 3))
        series["exp_sd_ga"].append(round(last["exp_sd_ga"], 3))
        series["exp_median_ga"].append(round(last["exp_median_ga"], 3))
        series["mean_plus_sd_gf"].append(
            round(last["exp_mean_gf"] + last["exp_sd_gf"], 3)
        )
        series["mean_minus_sd_gf"].append(
            round(max(0.0, last["exp_mean_gf"] - last["exp_sd_gf"]), 3)
        )
        series["mean_plus_sd_ga"].append(
            round(last["exp_mean_ga"] + last["exp_sd_ga"], 3)
        )
        series["mean_minus_sd_ga"].append(
            round(max(0.0, last["exp_mean_ga"] - last["exp_sd_ga"]), 3)
        )

    for key, values in list(series.items()):
        if key == "played":
            continue
        series[key] = forward_fill(values)

    return {
        "n_matches": len(matches),
        "final_mean_gf": round(statistics.fmean(gfs), 3),
        "final_mean_ga": round(statistics.fmean(gas), 3),
        "final_median_gf": float(statistics.median(gfs)),
        "final_median_ga": float(statistics.median(gas)),
        "final_sd_gf": round(statistics.stdev(gfs), 3),
        "final_sd_ga": round(statistics.stdev(gas), 3),
        "blank_gws": [g for g in GWS if g not in snap],
        "series": series,
    }


def build_payload(csv_path: Path, clubs: tuple[str, ...]) -> dict:
    by_team = load_team_matches(csv_path)
    missing = [c for c in clubs if c not in by_team]
    if missing:
        raise SystemExit(f"clubs not in sheet: {missing}; have {sorted(by_team)}")
    return {
        "season": "2025/26",
        "source": str(csv_path.relative_to(ROOT)),
        "categories": [f"GW{g}" for g in GWS],
        "clubs": {c: club_series(by_team[c]) for c in clubs},
    }


def render_figures(payload: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    clubs = list(payload["clubs"].keys())
    gws = GWS
    x = np.array(gws)

    fig, axes = plt.subplots(len(clubs), 2, figsize=(14, 4 * len(clubs)), sharex=True)
    if len(clubs) == 1:
        axes = np.array([axes])
    fig.suptitle(
        "2025/26 expanding goals for / against by gameweek\n"
        + " · ".join(clubs),
        fontsize=14,
        y=0.995,
    )
    for i, club in enumerate(clubs):
        s = payload["clubs"][club]["series"]
        blanks = [g for g, p in zip(gws, s["played"]) if not p]
        ax = axes[i, 0]
        lo = np.array(s["mean_minus_sd_gf"])
        hi = np.array(s["mean_plus_sd_gf"])
        ax.fill_between(x, lo, hi, alpha=0.22, color="#2a6f97", label="mean ± 1 sd")
        ax.plot(x, s["exp_mean_gf"], color="#01497c", lw=2, label="expanding mean GF")
        ax.plot(
            x,
            s["exp_median_gf"],
            color="#61a5c2",
            lw=1.5,
            ls="--",
            label="expanding median GF",
        )
        for b in blanks:
            ax.axvline(b, color="#bbb", lw=0.8, ls=":")
        ax.set_ylabel(club)
        ax.set_ylim(0, max(5.5, float(hi.max()) + 0.2))
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title("Goals for (expanding)")
            ax.legend(loc="upper right", fontsize=8, frameon=False)

        ax = axes[i, 1]
        lo = np.array(s["mean_minus_sd_ga"])
        hi = np.array(s["mean_plus_sd_ga"])
        ax.fill_between(x, lo, hi, alpha=0.22, color="#9b2226", label="mean ± 1 sd")
        ax.plot(x, s["exp_mean_ga"], color="#641220", lw=2, label="expanding mean GA")
        ax.plot(
            x,
            s["exp_median_ga"],
            color="#e5383b",
            lw=1.5,
            ls="--",
            label="expanding median GA",
        )
        for b in blanks:
            ax.axvline(b, color="#bbb", lw=0.8, ls=":")
        ax.set_ylim(0, max(4.5, float(hi.max()) + 0.2))
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title("Goals against (expanding)")
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes[-1]:
        ax.set_xlabel("Gameweek")
        ax.set_xticks(range(1, 39, 2))
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    panel = out_dir / "club_expanding_gf_ga_2025_26.png"
    fig.savefig(panel, dpi=140)
    plt.close(fig)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    colors = ["#01497c", "#c1121f", "#2a9d8f", "#e9c46a"]
    for club, color in zip(clubs, colors):
        s = payload["clubs"][club]["series"]
        ax1.plot(gws, s["exp_mean_gf"], color=color, lw=2, label=club)
        ax2.plot(gws, s["exp_mean_ga"], color=color, lw=2, label=club)
    ax1.set_title("Expanding mean goals for")
    ax2.set_title("Expanding mean goals against")
    for ax in (ax1, ax2):
        ax.set_xlabel("Gameweek")
        ax.set_ylabel("Goals per match")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(1, 39, 2))
        ax.legend(frameon=False, fontsize=9)
    fig2.suptitle("2025/26 — four clubs expanding mean GF / GA", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    overview = out_dir / "club_expanding_mean_overview_2025_26.png"
    fig2.savefig(overview, dpi=140)
    plt.close(fig2)
    print(f"wrote {panel}")
    print(f"wrote {overview}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--clubs", nargs="+", default=list(DEFAULT_CLUBS))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    payload = build_payload(args.csv, tuple(args.clubs))
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "club_expanding_gf_ga_2025_26.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {json_path}")
    if not args.no_plot:
        render_figures(payload, args.out)


if __name__ == "__main__":
    main()
