#!/usr/bin/env python3
"""Predict 2026/27 season GF/GA from predicted final ranks.

Mapping fitted on closed seasons only (2022/23–2025/26). Does not fit on
the live 2026/27 holdout.

Sources for 2026/27 ranks:
  - Betfair Exchange outrights → E[rank] (repo snapshot 2026-10-08)
  - Opta supercomputer predicted table (2026-10-12, public press)

Writes:
  data/processed/predict_gf_ga_from_rank_2026_27.csv
  reports/figures/predict_gf_ga_from_rank_2026_27.json
  reports/predict_gf_ga_from_rank_2026_27.md
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CLOSED = [
    ("2022/23", "E0_2223.csv"),
    ("2023/24", "E0_2324.csv"),
    ("2024/25", "E0_2425.csv"),
    ("2025/26", "E0_2526.csv"),
]

# Opta supercomputer predicted final table (SI, 2026-10-11) — full 1–20.
# YEP 2026-10-12 flipped only the title race to City 78.43 / Arsenal 77.17;
# we apply that flip and keep SI ranks/points for places 3–20.
OPTA_SI_2026_10_11 = [
    (1, "Arsenal", 76.99),
    (2, "Man City", 76.81),
    (3, "Brighton", 63.24),
    (4, "Liverpool", 61.19),
    (5, "Chelsea", 56.54),
    (6, "Brentford", 54.00),
    (7, "Newcastle", 53.09),
    (8, "Everton", 53.04),
    (9, "Man United", 51.90),
    (10, "Aston Villa", 51.43),
    (11, "Leeds", 51.13),
    (12, "Hull", 45.69),
    (13, "Ipswich", 45.56),
    (14, "Crystal Palace", 45.00),
    (15, "Bournemouth", 44.86),
    (16, "Nott'm Forest", 44.07),
    (17, "Tottenham", 43.95),
    (18, "Sunderland", 42.18),
    (19, "Coventry", 39.63),
    (20, "Fulham", 38.26),
]


def load_season_ranks(path: Path) -> list[dict]:
    stats: dict[str, dict] = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("FTHG"):
                continue
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            for team, gf, ga in (
                (row["HomeTeam"], hg, ag),
                (row["AwayTeam"], ag, hg),
            ):
                s = stats[team]
                s["gf"] += gf
                s["ga"] += ga
                s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    rows = [
        {"team": t, "pts": s["pts"], "gf": s["gf"], "ga": s["ga"], "gd": s["gf"] - s["ga"]}
        for t, s in stats.items()
    ]
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def ols(y: list[float], x: list[float]) -> tuple[float, float, float, float]:
    """Return intercept, slope, R², MAE of y ~ a + b x."""
    n = len(y)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    varx = sum((xi - mx) ** 2 for xi in x)
    if varx <= 0:
        return my, 0.0, 0.0, float("nan")
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / varx
    a = my - b * mx
    pred = [a + b * xi for xi in x]
    ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, pred))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    mae = statistics.fmean(abs(yi - pi) for yi, pi in zip(y, pred))
    return a, b, r2, mae


def mean_by_rank(rows: list[dict]) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"gf": [], "ga": []}
    )
    for r in rows:
        buckets[int(r["rank"])]["gf"].append(float(r["gf"]))
        buckets[int(r["rank"])]["ga"].append(float(r["ga"]))
    out = {}
    for rank, d in buckets.items():
        out[rank] = {
            "mean_gf": statistics.fmean(d["gf"]),
            "mean_ga": statistics.fmean(d["ga"]),
            "sd_gf": statistics.stdev(d["gf"]) if len(d["gf"]) > 1 else 0.0,
            "sd_ga": statistics.stdev(d["ga"]) if len(d["ga"]) > 1 else 0.0,
            "n": len(d["gf"]),
        }
    return out


def interpolate_rank(profile: dict[int, dict[str, float]], rank: float) -> tuple[float, float]:
    """Linear interpolate GF/GA between integer finishing positions."""
    r = max(1.0, min(20.0, float(rank)))
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    if lo == hi:
        p = profile[lo]
        return p["mean_gf"], p["mean_ga"]
    w = r - lo
    gf = (1 - w) * profile[lo]["mean_gf"] + w * profile[hi]["mean_gf"]
    ga = (1 - w) * profile[lo]["mean_ga"] + w * profile[hi]["mean_ga"]
    return gf, ga


def load_betfair_ranks() -> list[dict]:
    path = (
        ROOT
        / "data/predictions/2026-27/gw06/betfair_20261008/outrights_ranks.json"
    )
    rows = json.loads(path.read_text())
    # Bournemouth missing in Oct-8 pull → default E[rank]=18.5 (engine convention).
    present = {r["club_norm"] for r in rows}
    if "bournemouth" not in present:
        rows.append(
            {
                "club": "Bournemouth",
                "club_norm": "bournemouth",
                "E_rank": 18.5,
                "strength": -0.84,
                "p_win": 0.0,
                "p_top6": 0.0,
                "p_rel": None,
                "note": "missing_from_outrights_default",
            }
        )
    rows = sorted(rows, key=lambda r: float(r["E_rank"]))
    # Assign ordinal predicted rank by sorting E_rank (1 = best).
    for i, r in enumerate(rows, start=1):
        r["pred_rank_ord"] = i
        r["pred_rank"] = float(r["E_rank"])
        r["source"] = "betfair_E_rank_20261008"
    return rows


def main() -> None:
    hist: list[dict] = []
    for label, fname in CLOSED:
        for r in load_season_ranks(CACHE / fname):
            hist.append({"season": label, **r})

    profile = mean_by_rank(hist)
    ranks = [float(r["rank"]) for r in hist]
    gfs = [float(r["gf"]) for r in hist]
    gas = [float(r["ga"]) for r in hist]
    a_gf, b_gf, r2_gf, mae_gf = ols(gfs, ranks)
    a_ga, b_ga, r2_ga, mae_ga = ols(gas, ranks)

    # Leave-one-season-out MAE for rank-mean profile.
    loso = []
    for hold in CLOSED:
        train = [r for r in hist if r["season"] != hold[0]]
        test = [r for r in hist if r["season"] == hold[0]]
        prof = mean_by_rank(train)
        err_gf = []
        err_ga = []
        for r in test:
            pgf, pga = interpolate_rank(prof, r["rank"])
            err_gf.append(abs(pgf - r["gf"]))
            err_ga.append(abs(pga - r["ga"]))
        loso.append(
            {
                "holdout": hold[0],
                "mae_gf": round(statistics.fmean(err_gf), 2),
                "mae_ga": round(statistics.fmean(err_ga), 2),
            }
        )

    # --- Apply to Betfair E[rank] ---
    betfair = load_betfair_ranks()
    bf_out = []
    for r in betfair:
        gf_m, ga_m = interpolate_rank(profile, r["pred_rank"])
        gf_ols = a_gf + b_gf * r["pred_rank"]
        ga_ols = a_ga + b_ga * r["pred_rank"]
        bf_out.append(
            {
                "club": r["club"],
                "club_norm": r["club_norm"],
                "source": r["source"],
                "pred_rank_E": round(r["pred_rank"], 3),
                "pred_rank_ord": r["pred_rank_ord"],
                "pred_gf_rankmean": round(gf_m, 1),
                "pred_ga_rankmean": round(ga_m, 1),
                "pred_gd_rankmean": round(gf_m - ga_m, 1),
                "pred_gf_ols": round(gf_ols, 1),
                "pred_ga_ols": round(ga_ols, 1),
                "pred_gd_ols": round(gf_ols - ga_ols, 1),
            }
        )

    # --- Apply to Opta ranks (YEP 2026-10-12 title flip on SI 2026-10-11 table) ---
    opta_rows = []
    for rank, club, pts in OPTA_SI_2026_10_11:
        if club == "Man City":
            rank, pts = 1, 78.43
        elif club == "Arsenal":
            rank, pts = 2, 77.17
        gf_m, ga_m = interpolate_rank(profile, rank)
        gf_ols = a_gf + b_gf * rank
        ga_ols = a_ga + b_ga * rank
        opta_rows.append(
            {
                "club": club,
                "source": "opta_yep_2026_10_12",
                "pred_rank": rank,
                "pred_pts": pts,
                "pred_gf_rankmean": round(gf_m, 1),
                "pred_ga_rankmean": round(ga_m, 1),
                "pred_gd_rankmean": round(gf_m - ga_m, 1),
                "pred_gf_ols": round(gf_ols, 1),
                "pred_ga_ols": round(ga_ols, 1),
                "pred_gd_ols": round(gf_ols - ga_ols, 1),
            }
        )
    opta_rows.sort(key=lambda r: r["pred_rank"])
    assert len({r["pred_rank"] for r in opta_rows}) == 20

    profile_out = {
        str(k): {
            "mean_gf": round(v["mean_gf"], 2),
            "mean_ga": round(v["mean_ga"], 2),
            "sd_gf": round(v["sd_gf"], 2),
            "sd_ga": round(v["sd_ga"], 2),
            "n": v["n"],
        }
        for k, v in sorted(profile.items())
    }

    payload = {
        "mapping": {
            "train_seasons": [s[0] for s in CLOSED],
            "fit_on_2026_27": False,
            "ols_gf": {
                "intercept": round(a_gf, 3),
                "slope": round(b_gf, 3),
                "r2": round(r2_gf, 4),
                "mae": round(mae_gf, 2),
                "formula": f"GF ≈ {a_gf:.1f} {b_gf:+.2f}·rank",
            },
            "ols_ga": {
                "intercept": round(a_ga, 3),
                "slope": round(b_ga, 3),
                "r2": round(r2_ga, 4),
                "mae": round(mae_ga, 2),
                "formula": f"GA ≈ {a_ga:.1f} {b_ga:+.2f}·rank",
            },
            "loso_mae": loso,
            "mean_mae_loso_gf": round(statistics.fmean(x["mae_gf"] for x in loso), 2),
            "mean_mae_loso_ga": round(statistics.fmean(x["mae_ga"] for x in loso), 2),
            "rank_profile": profile_out,
        },
        "rank_sources": {
            "betfair": {
                "path": "data/predictions/2026-27/gw06/betfair_20261008/outrights_ranks.json",
                "as_of": "2026-10-08",
                "field": "E_rank",
            },
            "opta": {
                "as_of": "2026-10-12",
                "note": "Opta via SI 2026-10-11 full table; YEP 2026-10-12 title flip (City 78.43, Arsenal 77.17).",
            },
        },
        "predictions_betfair": bf_out,
        "predictions_opta": opta_rows,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Flat CSV: Opta primary + Betfair side columns where name-matched
    bf_by_norm = {r["club_norm"]: r for r in bf_out}
    alias = {
        "man united": "man utd",
        "manchester united": "man utd",
        "manchester city": "man city",
        "tottenham": "spurs",
        "nottingham forest": "nottm forest",
        "nott'm forest": "nottm forest",
        "nottm forest": "nottm forest",
    }

    def _norm(name: str) -> str:
        key = " ".join(str(name).lower().replace(".", "").split())
        return alias.get(key, key)

    flat = []
    for r in opta_rows:
        bn = _norm(r["club"])
        b = bf_by_norm.get(bn, {})
        flat.append(
            {
                "club": r["club"],
                "opta_rank": r["pred_rank"],
                "opta_pts": r["pred_pts"],
                "pred_gf": r["pred_gf_rankmean"],
                "pred_ga": r["pred_ga_rankmean"],
                "pred_gd": r["pred_gd_rankmean"],
                "pred_gf_ols": r["pred_gf_ols"],
                "pred_ga_ols": r["pred_ga_ols"],
                "betfair_E_rank": b.get("pred_rank_E"),
                "betfair_ord": b.get("pred_rank_ord"),
                "betfair_pred_gf": b.get("pred_gf_rankmean"),
                "betfair_pred_ga": b.get("pred_ga_rankmean"),
            }
        )
    csv_path = PROCESSED / "predict_gf_ga_from_rank_2026_27.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    json_path = FIGURES / "predict_gf_ga_from_rank_2026_27.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# 2026/27 GF/GA from predicted rankings",
        "",
        "Yes — with a caveat. Final table rank is a strong proxy for season GF/GA "
        "on closed seasons (ρ ≈ 0.84 / 0.78). Mapping is fit **only** on "
        "2022/23–2025/26; the live 2026/27 holdout is not used to fit.",
        "",
        "## Rank sources (most recent found)",
        "",
        "| source | as-of | what |",
        "|---|---|---|",
        "| **Opta supercomputer** (press) | **2026-10-12** | predicted final positions + points |",
        "| Betfair Exchange outrights → `E_rank` | 2026-10-08 | in-repo engine snapshot |",
        "",
        "No newer full 1–20 table was found after 12 Oct in open press; club-level "
        "snippets exist through ~19 Oct (e.g. Chelsea still ~4th–5th in Opta).",
        "",
        "## Mapping quality (closed seasons)",
        "",
        f"- OLS: `{payload['mapping']['ols_gf']['formula']}` "
        f"(R²={payload['mapping']['ols_gf']['r2']}, MAE={payload['mapping']['ols_gf']['mae']})",
        f"- OLS: `{payload['mapping']['ols_ga']['formula']}` "
        f"(R²={payload['mapping']['ols_ga']['r2']}, MAE={payload['mapping']['ols_ga']['mae']})",
        f"- Leave-one-season-out MAE (rank-mean profile): "
        f"GF **{payload['mapping']['mean_mae_loso_gf']}**, "
        f"GA **{payload['mapping']['mean_mae_loso_ga']}** goals/season",
        "",
        "Primary predictor below = **historical mean GF/GA at that finishing place** "
        "(interpolated for fractional Betfair E[rank]).",
        "",
        "## Opta predicted table → GF / GA (2026-10-12)",
        "",
        "| Rk | Club | Pred pts | Pred GF | Pred GA | Pred GD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in opta_rows:
        lines.append(
            f"| {r['pred_rank']} | {r['club']} | {r['pred_pts']:.0f} | "
            f"{r['pred_gf_rankmean']:.0f} | {r['pred_ga_rankmean']:.0f} | "
            f"{r['pred_gd_rankmean']:+.0f} |"
        )

    lines += [
        "",
        "## Betfair E[rank] → GF / GA (2026-10-08)",
        "",
        "| Ord | Club | E[rank] | Pred GF | Pred GA | Pred GD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in bf_out:
        lines.append(
            f"| {r['pred_rank_ord']} | {r['club']} | {r['pred_rank_E']:.2f} | "
            f"{r['pred_gf_rankmean']:.0f} | {r['pred_ga_rankmean']:.0f} | "
            f"{r['pred_gd_rankmean']:+.0f} |"
        )

    lines += [
        "",
        "## Historical GF/GA by finishing place (train mean)",
        "",
        "| Rank | mean GF | mean GA | sd GF | sd GA |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, v in sorted(profile.items()):
        lines.append(
            f"| {k} | {v['mean_gf']:.1f} | {v['mean_ga']:.1f} | "
            f"{v['sd_gf']:.1f} | {v['sd_ga']:.1f} |"
        )

    lines += [
        "",
        "## Artefacts",
        "",
        f"- `{csv_path.relative_to(ROOT)}`",
        f"- `{json_path.relative_to(ROOT)}`",
        "",
        "Reproduce: `python3 scripts/predict_gf_ga_from_rank_2026_27.py`",
        "",
    ]
    report_path = REPORTS / "predict_gf_ga_from_rank_2026_27.md"
    report_path.write_text("\n".join(lines))

    print(json.dumps(payload["mapping"]["ols_gf"], indent=2))
    print(json.dumps(payload["mapping"]["ols_ga"], indent=2))
    print("loso", loso)
    print("wrote", csv_path, report_path)


if __name__ == "__main__":
    main()
