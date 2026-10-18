#!/usr/bin/env python3
"""Expanding mean / sd / median / dispersion GF and GA for four clubs, 2025/26.

Reads ``data/cache/merged_gw_2025_26.csv`` and seeds each club's expanding
window with that club's scored matches from ``merged_gw_2024_25.csv``, so the
series opens at the 2024/25 season mean and dispersion instead of a 1-match
spike. Aligns to GW1–38 with forward-fill on blanks; writes JSON + PNG figures
under ``reports/figures/``.

Dispersion is the index of dispersion: expanding sample variance / expanding
mean (Fano factor). When the mean is 0 the ratio is set to 0.

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
DEFAULT_PRIOR_CSV = ROOT / "data" / "cache" / "merged_gw_2024_25.csv"
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


def season_moments(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        var = 0.0
        sd = 0.0
    else:
        sd = statistics.stdev(values)
        var = sd * sd
    return {
        "n": len(values),
        "mean": round(mean, 3),
        "sd": round(sd, 3),
        "var": round(var, 3),
        "median": float(statistics.median(values)),
        "disp": round(index_of_dispersion(var, mean), 3),
    }


def expanding(
    values: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    means: list[float] = []
    sds: list[float] = []
    vars_: list[float] = []
    meds: list[float] = []
    for i in range(len(values)):
        window = values[: i + 1]
        means.append(statistics.fmean(window))
        meds.append(float(statistics.median(window)))
        if i == 0:
            sds.append(0.0)
            vars_.append(0.0)
        else:
            sd = statistics.stdev(window)
            sds.append(sd)
            vars_.append(sd * sd)
    return means, sds, vars_, meds


def index_of_dispersion(variance: float, mean: float) -> float:
    """Sample variance / mean. Zero mean with zero variance → 0."""
    if mean <= 0.0:
        return 0.0
    return variance / mean


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


def club_series(matches: list[dict], prior_matches: list[dict]) -> dict:
    """Expanding stats on prior season matches, then 2025/26 matches.

    Snapshots are taken only after 2025/26 fixtures, so GW1 already sits on
    the 2024/25 mean and dispersion rather than a single-match spike.
    """
    prior_gfs = [float(m["gf"]) for m in prior_matches]
    prior_gas = [float(m["ga"]) for m in prior_matches]
    if not prior_gfs:
        raise SystemExit("prior season has no scored matches for club")

    gfs = list(prior_gfs)
    gas = list(prior_gas)
    n_prior = len(prior_gfs)
    snap: dict[int, dict] = {}
    gw_totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {"gf": 0.0, "ga": 0.0, "n": 0.0}
    )

    for m in matches:
        gfs.append(float(m["gf"]))
        gas.append(float(m["ga"]))
        mgf, sgf, vgf, medgf = expanding(gfs)
        mga, sga, vga, medga = expanding(gas)
        i = len(gfs) - 1
        snap[m["gw"]] = {
            "n_total": i + 1,
            "n_prior": n_prior,
            "n_season": i + 1 - n_prior,
            "exp_mean_gf": mgf[i],
            "exp_sd_gf": sgf[i],
            "exp_var_gf": vgf[i],
            "exp_median_gf": medgf[i],
            "exp_disp_gf": index_of_dispersion(vgf[i], mgf[i]),
            "exp_mean_ga": mga[i],
            "exp_sd_ga": sga[i],
            "exp_var_ga": vga[i],
            "exp_median_ga": medga[i],
            "exp_disp_ga": index_of_dispersion(vga[i], mga[i]),
        }
        gw_totals[m["gw"]]["gf"] += float(m["gf"])
        gw_totals[m["gw"]]["ga"] += float(m["ga"])
        gw_totals[m["gw"]]["n"] += 1.0

    series: dict[str, list] = {
        "exp_mean_gf": [],
        "exp_sd_gf": [],
        "exp_var_gf": [],
        "exp_median_gf": [],
        "exp_disp_gf": [],
        "exp_mean_ga": [],
        "exp_sd_ga": [],
        "exp_var_ga": [],
        "exp_median_ga": [],
        "exp_disp_ga": [],
        "mean_plus_sd_gf": [],
        "mean_minus_sd_gf": [],
        "mean_plus_sd_ga": [],
        "mean_minus_sd_ga": [],
        "gw_gf": [],
        "gw_ga": [],
        "played": [],
    }
    last: dict | None = None
    # Before GW1: seed with pure prior so forward-fill before first played GW
    # is the 2024/25 season totals (not zeros).
    prior_mgf, prior_sgf, prior_vgf, prior_medgf = expanding(prior_gfs)
    prior_mga, prior_sga, prior_vga, prior_medga = expanding(prior_gas)
    i_p = n_prior - 1
    seed = {
        "exp_mean_gf": prior_mgf[i_p],
        "exp_sd_gf": prior_sgf[i_p],
        "exp_var_gf": prior_vgf[i_p],
        "exp_median_gf": prior_medgf[i_p],
        "exp_disp_gf": index_of_dispersion(prior_vgf[i_p], prior_mgf[i_p]),
        "exp_mean_ga": prior_mga[i_p],
        "exp_sd_ga": prior_sga[i_p],
        "exp_var_ga": prior_vga[i_p],
        "exp_median_ga": prior_medga[i_p],
        "exp_disp_ga": index_of_dispersion(prior_vga[i_p], prior_mga[i_p]),
    }
    last = seed

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
        assert last is not None
        series["exp_mean_gf"].append(round(last["exp_mean_gf"], 3))
        series["exp_sd_gf"].append(round(last["exp_sd_gf"], 3))
        series["exp_var_gf"].append(round(last["exp_var_gf"], 3))
        series["exp_median_gf"].append(round(last["exp_median_gf"], 3))
        series["exp_disp_gf"].append(round(last["exp_disp_gf"], 3))
        series["exp_mean_ga"].append(round(last["exp_mean_ga"], 3))
        series["exp_sd_ga"].append(round(last["exp_sd_ga"], 3))
        series["exp_var_ga"].append(round(last["exp_var_ga"], 3))
        series["exp_median_ga"].append(round(last["exp_median_ga"], 3))
        series["exp_disp_ga"].append(round(last["exp_disp_ga"], 3))
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

    # Season-only finals (2025/26 matches alone) plus seeded finals (prior+season).
    season_gfs = [float(m["gf"]) for m in matches]
    season_gas = [float(m["ga"]) for m in matches]
    final_mean_gf = statistics.fmean(gfs)
    final_mean_ga = statistics.fmean(gas)
    final_var_gf = statistics.stdev(gfs) ** 2
    final_var_ga = statistics.stdev(gas) ** 2
    return {
        "n_matches": len(matches),
        "n_prior": n_prior,
        "prior_2024_25": {
            "gf": season_moments(prior_gfs),
            "ga": season_moments(prior_gas),
        },
        "final_mean_gf": round(final_mean_gf, 3),
        "final_mean_ga": round(final_mean_ga, 3),
        "final_median_gf": float(statistics.median(gfs)),
        "final_median_ga": float(statistics.median(gas)),
        "final_sd_gf": round(statistics.stdev(gfs), 3),
        "final_sd_ga": round(statistics.stdev(gas), 3),
        "final_var_gf": round(final_var_gf, 3),
        "final_var_ga": round(final_var_ga, 3),
        "final_disp_gf": round(index_of_dispersion(final_var_gf, final_mean_gf), 3),
        "final_disp_ga": round(index_of_dispersion(final_var_ga, final_mean_ga), 3),
        "season_only_mean_gf": round(statistics.fmean(season_gfs), 3),
        "season_only_mean_ga": round(statistics.fmean(season_gas), 3),
        "blank_gws": [g for g in GWS if g not in snap],
        "series": series,
    }


def build_payload(
    csv_path: Path, prior_csv: Path, clubs: tuple[str, ...]
) -> dict:
    by_team = load_team_matches(csv_path)
    by_prior = load_team_matches(prior_csv)
    missing = [c for c in clubs if c not in by_team]
    if missing:
        raise SystemExit(f"clubs not in sheet: {missing}; have {sorted(by_team)}")
    missing_prior = [c for c in clubs if c not in by_prior]
    if missing_prior:
        raise SystemExit(
            f"clubs not in prior sheet: {missing_prior}; have {sorted(by_prior)}"
        )
    return {
        "season": "2025/26",
        "prior_season": "2024/25",
        "seed": "2024/25 match GF/GA prepended so expanding mean and "
        "dispersion open at that season's totals",
        "source": str(csv_path.relative_to(ROOT)),
        "prior_source": str(prior_csv.relative_to(ROOT)),
        "categories": [f"GW{g}" for g in GWS],
        "clubs": {
            c: club_series(by_team[c], by_prior[c]) for c in clubs
        },
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
    seed_note = "seeded with 2024/25"

    fig, axes = plt.subplots(len(clubs), 2, figsize=(14, 4 * len(clubs)), sharex=True)
    if len(clubs) == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"2025/26 expanding goals for / against by gameweek ({seed_note})\n"
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

    colors = ["#01497c", "#c1121f", "#2a9d8f", "#e9c46a"]
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
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
    fig2.suptitle(
        f"2025/26 — four clubs expanding mean GF / GA ({seed_note})",
        fontsize=13,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    overview = out_dir / "club_expanding_mean_overview_2025_26.png"
    fig2.savefig(overview, dpi=140)
    plt.close(fig2)

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    for club, color in zip(clubs, colors):
        s = payload["clubs"][club]["series"]
        ax1.plot(gws, s["exp_disp_gf"], color=color, lw=2, label=club)
        ax2.plot(gws, s["exp_disp_ga"], color=color, lw=2, label=club)
    for ax, title in (
        (ax1, "Goals for"),
        (ax2, "Goals against"),
    ):
        ax.axhline(1.0, color="#888", lw=1.0, ls=":", label="Poisson (=1)")
        ax.set_title(f"Expanding dispersion — {title}")
        ax.set_xlabel("Gameweek")
        ax.set_ylabel("Var / mean")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(1, 39, 2))
        ax.set_ylim(bottom=0)
        handles, labels = ax.get_legend_handles_labels()
        seen: dict[str, object] = {}
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen[lab] = h
        ax.legend(
            list(seen.values()),
            list(seen.keys()),
            frameon=False,
            fontsize=9,
        )
    fig3.suptitle(
        f"2025/26 — expanding dispersion (var / mean, {seed_note})",
        fontsize=13,
    )
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    dispersion = out_dir / "club_expanding_dispersion_2025_26.png"
    fig3.savefig(dispersion, dpi=140)
    plt.close(fig3)

    fig4, axes = plt.subplots(len(clubs), 2, figsize=(14, 3.2 * len(clubs)), sharex=True)
    if len(clubs) == 1:
        axes = np.array([axes])
    fig4.suptitle(
        f"2025/26 expanding dispersion (var / mean, {seed_note})\n"
        + " · ".join(clubs),
        fontsize=14,
        y=0.995,
    )
    for i, club in enumerate(clubs):
        s = payload["clubs"][club]["series"]
        blanks = [g for g, p in zip(gws, s["played"]) if not p]
        for j, (key, color, title) in enumerate(
            (
                ("exp_disp_gf", "#01497c", "GF dispersion"),
                ("exp_disp_ga", "#641220", "GA dispersion"),
            )
        ):
            ax = axes[i, j]
            ax.plot(x, s[key], color=color, lw=2, label="var / mean")
            ax.axhline(1.0, color="#888", lw=1.0, ls=":", label="Poisson (=1)")
            for b in blanks:
                ax.axvline(b, color="#bbb", lw=0.8, ls=":")
            if j == 0:
                ax.set_ylabel(club)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(title)
                ax.legend(loc="upper right", fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Gameweek")
        ax.set_xticks(range(1, 39, 2))
    fig4.tight_layout(rect=[0, 0.01, 1, 0.97])
    disp_panel = out_dir / "club_expanding_dispersion_panel_2025_26.png"
    fig4.savefig(disp_panel, dpi=140)
    plt.close(fig4)

    print(f"wrote {panel}")
    print(f"wrote {overview}")
    print(f"wrote {dispersion}")
    print(f"wrote {disp_panel}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--prior-csv", type=Path, default=DEFAULT_PRIOR_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--clubs", nargs="+", default=list(DEFAULT_CLUBS))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    payload = build_payload(args.csv, args.prior_csv, tuple(args.clubs))
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "club_expanding_gf_ga_2025_26.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {json_path}")
    if not args.no_plot:
        render_figures(payload, args.out)


if __name__ == "__main__":
    main()
