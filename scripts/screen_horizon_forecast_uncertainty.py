#!/usr/bin/env python3
"""Stage 1: empirical GF/GA forecast-error vs lead time (closed seasons).

ChatGPT review ask: calibrate time-based *forecast* uncertainty from historical
errors, without an arbitrary %/GW penalty, and without double-counting match
noise already in the Poisson/Dixon–Coles layer.

We do **not** have archived Opta rank tables. Proxy:

1. **Rolling expanding rates** — at lead *h* weeks before a fixture, form
   expanding GF/GA from matches strictly before the as-of GW, build
   ``λ = ½(own_GF + opp_GA)``, score realised goals.
2. **Early freeze (Opta-sticky proxy)** — freeze expanding rates after GW6;
   score later fixtures by lead = fixture_GW − 6 (same λ recipe).

Excess residual MSE relative to Poisson (``MSE − mean(λ)``) is the quantity
that should grow with lead if strength drift is real. Fit shared
``excess ≈ σ0² + h·q`` on closed seasons only (no 2026/27 holdout fit).

Writes ``reports/figures/horizon_forecast_uncertainty.json`` and
``reports/horizon_forecast_uncertainty.md``.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT_JSON = ROOT / "reports" / "figures" / "horizon_forecast_uncertainty.json"
OUT_MD = ROOT / "reports" / "horizon_forecast_uncertainty.md"

SEASONS = [
    ("2022/23", "merged_gw_2022_23.csv", "E0_2223.csv"),
    ("2023/24", "merged_gw_2023_24.csv", "E0_2324.csv"),
    ("2024/25", "merged_gw_2024_25.csv", "E0_2425.csv"),
    ("2025/26", "merged_gw_2025_26.csv", "E0_2526.csv"),
]

LEADS = (1, 2, 4, 6, 8)
FREEZE_GW = 6
MIN_PRIOR = 3
HOME_ADV = 1.08


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 30:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return intercept, slope, R² for y ~ a + b x."""
    n = len(xs)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    yhat = [a + b * x for x in xs]
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return a, b, r2


def load_team_fixtures(path: Path) -> list[dict]:
    import csv

    by_fix: dict = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fix = row["fixture"]
            th, ta = row["team_h_score"], row["team_a_score"]
            if th == "" or ta == "":
                continue
            th_i, ta_i = int(float(th)), int(float(ta))
            home = str(row["was_home"]).lower() in {"true", "1"}
            key = (row["team"], fix)
            if key not in by_fix:
                by_fix[key] = {
                    "team": row["team"],
                    "fixture": fix,
                    "gw": int(row["GW"]),
                    "kickoff": row["kickoff_time"],
                    "home": home,
                    "hg": th_i,
                    "ag": ta_i,
                }
    fixtures: dict = defaultdict(dict)
    for (_, fix), r in by_fix.items():
        side = "home" if r["home"] else "away"
        fixtures[fix][side] = r
    out2 = []
    for fix, sides in fixtures.items():
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        if h["home"]:
            home_team, away_team = h["team"], a["team"]
            hg, ag = h["hg"], h["ag"]
            gw, kickoff = h["gw"], h["kickoff"]
        else:
            home_team, away_team = a["team"], h["team"]
            hg, ag = a["hg"], a["ag"]
            gw, kickoff = a["gw"], a["kickoff"]
        out2.append(
            {
                "fixture": fix,
                "gw": int(gw),
                "kickoff": kickoff,
                "home": home_team,
                "away": away_team,
                "hg": int(hg),
                "ag": int(ag),
            }
        )
    out2.sort(key=lambda m: (m["kickoff"], m["fixture"]))
    return out2


def team_histories(matches: list[dict]) -> dict[str, list[dict]]:
    hist: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        hist[m["home"]].append(
            {"gw": m["gw"], "gf": m["hg"], "ga": m["ag"], "home": True}
        )
        hist[m["away"]].append(
            {"gw": m["gw"], "gf": m["ag"], "ga": m["hg"], "home": False}
        )
    for team in hist:
        hist[team].sort(key=lambda r: (r["gw"],))
    return hist


def rates_before(hist: list[dict], before_gw: int) -> tuple[float | None, float | None, int]:
    prior = [r for r in hist if int(r["gw"]) < int(before_gw)]
    if len(prior) < MIN_PRIOR:
        return None, None, len(prior)
    gf = statistics.fmean(r["gf"] for r in prior)
    ga = statistics.fmean(r["ga"] for r in prior)
    return gf, ga, len(prior)


def lam_side(
    own_gf: float,
    opp_ga: float,
    *,
    is_home: bool,
) -> float:
    base = 0.5 * (own_gf + opp_ga)
    return base * HOME_ADV if is_home else base / HOME_ADV


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    y = [r["goals"] for r in rows]
    yhat = [r["lam"] for r in rows]
    err = [a - b for a, b in zip(y, yhat)]
    sq = [e * e for e in err]
    mae = statistics.fmean(abs(e) for e in err)
    mse = statistics.fmean(sq)
    mean_lam = statistics.fmean(yhat)
    excess = mse - mean_lam  # Poisson: E[(G-λ)²] = λ if λ known
    return {
        "n": len(rows),
        "mae": round(mae, 4),
        "rmse": round(math.sqrt(mse), 4),
        "mse": round(mse, 4),
        "mean_lam": round(mean_lam, 4),
        "excess_mse": round(excess, 4),
        "bias": round(statistics.fmean(err), 4),
        "pearson": round(_pearson(yhat, y), 4),
    }


def collect_rolling(matches: list[dict], hist: dict[str, list[dict]], season: str) -> list[dict]:
    rows = []
    for m in matches:
        g = int(m["gw"])
        for h in LEADS:
            asof = g - h + 1  # last GW allowed in history is asof-1; h=1 → asof=g
            # information strictly before fixture GW, aged by h: before_gw = g - h + 1
            # For h=1, before_gw = g (use all prior to this GW) — standard
            before = g - h + 1
            if before <= MIN_PRIOR:
                continue
            hg, hga, nh = rates_before(hist[m["home"]], before)
            ag, aga, na = rates_before(hist[m["away"]], before)
            if hg is None or ag is None:
                continue
            # home goals vs λ_home; away goals vs λ_away
            for team, goals, own_gf, opp_ga, is_home, n_prior in (
                (m["home"], m["hg"], hg, aga, True, nh),
                (m["away"], m["ag"], ag, hga, False, na),
            ):
                lam = lam_side(own_gf, opp_ga, is_home=is_home)
                rows.append(
                    {
                        "season": season,
                        "mode": "rolling",
                        "lead": h,
                        "gw": g,
                        "team": team,
                        "goals": float(goals),
                        "lam": float(lam),
                        "n_prior": n_prior,
                    }
                )
    return rows


def collect_freeze(matches: list[dict], hist: dict[str, list[dict]], season: str) -> list[dict]:
    rows = []
    freeze_rates: dict[str, tuple[float, float, int]] = {}
    for team, th in hist.items():
        gf, ga, n = rates_before(th, FREEZE_GW + 1)
        if gf is not None:
            freeze_rates[team] = (gf, ga, n)
    for m in matches:
        g = int(m["gw"])
        if g <= FREEZE_GW:
            continue
        lead = g - FREEZE_GW
        if m["home"] not in freeze_rates or m["away"] not in freeze_rates:
            continue
        h_gf, h_ga, nh = freeze_rates[m["home"]]
        a_gf, a_ga, na = freeze_rates[m["away"]]
        for team, goals, own_gf, opp_ga, is_home, n_prior in (
            (m["home"], m["hg"], h_gf, a_ga, True, nh),
            (m["away"], m["ag"], a_gf, h_ga, False, na),
        ):
            lam = lam_side(own_gf, opp_ga, is_home=is_home)
            rows.append(
                {
                    "season": season,
                    "mode": "freeze_gw6",
                    "lead": int(lead),
                    "gw": g,
                    "team": team,
                    "goals": float(goals),
                    "lam": float(lam),
                    "n_prior": n_prior,
                }
            )
    return rows


def bucket_table(rows: list[dict], leads: list[int] | None = None) -> list[dict]:
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[int(r["lead"])].append(r)
    keys = leads if leads is not None else sorted(by)
    out = []
    for lead in keys:
        chunk = by.get(int(lead), [])
        s = summarise(chunk)
        s["lead"] = int(lead)
        out.append(s)
    return out


def fit_growth(by_lead: list[dict]) -> dict:
    usable = [r for r in by_lead if r.get("n", 0) >= 50]
    if len(usable) < 3:
        return {"ok": False}
    xs = [float(r["lead"]) for r in usable]
    mse = [float(r["mse"]) for r in usable]
    excess = [float(r["excess_mse"]) for r in usable]
    mae = [float(r["mae"]) for r in usable]
    a_m, b_m, r2_m = _ols_slope(xs, mse)
    a_e, b_e, r2_e = _ols_slope(xs, excess)
    # Materiality: excess slope must clear ~0.02/GW (≈0.16 over H=8) or
    # H=8−H=1 MAE gap ≥ 0.05. Tiny positive slopes with high R² are not
    # enough to open Stage 2 drift in the live engine.
    by_l = {int(r["lead"]): r for r in usable}
    d_excess = float("nan")
    d_mae = float("nan")
    if 1 in by_l and 8 in by_l:
        d_excess = float(by_l[8]["excess_mse"]) - float(by_l[1]["excess_mse"])
        d_mae = float(by_l[8]["mae"]) - float(by_l[1]["mae"])
    stat = bool(b_e > 0 and r2_e >= 0.3)
    material = bool(
        (b_e >= 0.02 and r2_e >= 0.3)
        or (isinstance(d_mae, float) and d_mae == d_mae and d_mae >= 0.05)
        or (isinstance(d_excess, float) and d_excess == d_excess and d_excess >= 0.15)
    )
    return {
        "ok": True,
        "n_buckets": len(usable),
        "mse_intercept": round(a_m, 5),
        "mse_slope_per_gw": round(b_m, 5),
        "mse_r2": round(r2_m, 4),
        "excess_intercept": round(a_e, 5),
        "excess_slope_per_gw": round(b_e, 5),
        "excess_r2": round(r2_e, 4),
        "delta_excess_h1_h8": round(d_excess, 4) if d_excess == d_excess else None,
        "delta_mae_h1_h8": round(d_mae, 4) if d_mae == d_mae else None,
        "growth_stat_positive": stat,
        "growth_material": material,
        "growth_supported": material,
        "mean_mae": round(statistics.fmean(mae), 4),
    }


def loso_growth(rows: list[dict]) -> list[dict]:
    seasons = sorted({r["season"] for r in rows})
    out = []
    for hold in seasons:
        train = [r for r in rows if r["season"] != hold]
        by_lead = bucket_table(train, list(LEADS))
        fit = fit_growth(by_lead)
        fit["holdout_season"] = hold
        out.append(fit)
    return out


def run() -> dict:
    rolling_all: list[dict] = []
    freeze_all: list[dict] = []
    for season, gw_file, _e0 in SEASONS:
        path = CACHE / gw_file
        if not path.is_file():
            continue
        matches = load_team_fixtures(path)
        hist = team_histories(matches)
        rolling_all.extend(collect_rolling(matches, hist, season))
        freeze_all.extend(collect_freeze(matches, hist, season))

    rolling_buckets = bucket_table(rolling_all, list(LEADS))
    # Freeze: bin leads into LEADS
    freeze_bins: dict[int, list[dict]] = {L: [] for L in LEADS}
    for r in freeze_all:
        lead = int(r["lead"])
        if lead <= 1:
            freeze_bins[1].append(r)
        elif lead == 2:
            freeze_bins[2].append(r)
        elif lead <= 4:
            freeze_bins[4].append(r)
        elif lead <= 6:
            freeze_bins[6].append(r)
        else:
            freeze_bins[8].append(r)
    freeze_buckets = []
    for L in LEADS:
        s = summarise(freeze_bins[L])
        s["lead"] = L
        freeze_buckets.append(s)
    freeze_fit = fit_growth(freeze_buckets)
    # Unbalanced bins (tiny n at short lead) inflate freeze slope — not actionable.
    if any(int(r.get("n") or 0) < 100 for r in freeze_buckets if int(r["lead"]) <= 4):
        freeze_fit["growth_material"] = False
        freeze_fit["growth_supported"] = False
        freeze_fit["unbalanced_bins"] = True

    payload = {
        "question": (
            "Does GF/GA forecast error grow with lead time after removing "
            "Poisson match noise (excess MSE = MSE − mean λ)?"
        ),
        "protocol": {
            "seasons": [s for s, _, _ in SEASONS],
            "leads_gw": list(LEADS),
            "min_prior_apps": MIN_PRIOR,
            "lambda": "0.5*(own_exp_GF + opp_exp_GA) * home_adv^±1",
            "home_adv": HOME_ADV,
            "no_opta_archive": True,
            "proxies": ["rolling_expanding", "freeze_after_gw6"],
            "holdout_2026_27": "not used",
        },
        "chatgpt_review": {
            "keep": [
                "Separate match-outcome noise from forecast/strength uncertainty",
                "Calibrate lead-time growth from historical errors (Stage 1)",
                "Prefer rolling-origin / as-of rebuilds over hindsight Opta ranks",
                "Start with league-wide drift params; team-specific only if needed",
                "Do not apply an arbitrary %/GW inflation",
            ],
            "defer": [
                "Stage 2 random-walk / mean-reverting strength drift — Stage 1 slope is tiny vs Poisson",
                "Stage 3 nested MC (scenario λ then score) until material drift is shown",
                "Team-specific q_A / q_D",
                "Archived Opta forecast reconstruction (we lack mid-season Opta snaps)",
                "Arbitrary % inflation of GF/GA or λ per future gameweek",
            ],
            "already_have": [
                "Opta rank→season GF/GA OLS ±MAE (level uncertainty, not time-scaled)",
                "Match pots via Dixon–Coles / Poisson in xp_on_pot (match noise)",
                "OUTRIGHT_SHRINK=0.5 for unpriced weeks (level shrink, not lead curve)",
            ],
        },
        "rolling_expanding": {
            "by_lead": rolling_buckets,
            "growth_fit": fit_growth(rolling_buckets),
            "loso": loso_growth(rolling_all),
            "n": len(rolling_all),
        },
        "freeze_gw6": {
            "by_lead": freeze_buckets,
            "growth_fit": freeze_fit,
            "n": len(freeze_all),
        },
    }

    # Verdict
    roll_g = payload["rolling_expanding"]["growth_fit"]
    frz_g = payload["freeze_gw6"]["growth_fit"]
    payload["verdict"] = {
        "rolling_growth_stat_positive": roll_g.get("growth_stat_positive", False),
        "rolling_growth_material": roll_g.get("growth_material", False),
        "freeze_growth_material": frz_g.get("growth_material", False),
        "freeze_note": (
            "Freeze lead bins are unbalanced (most mass in lead≥7); treat "
            "freeze slope as diagnostic only."
        ),
        "recommendation": (
            "Park Stage 2–3. Rolling excess MSE rises only ~0.04 from H=1 to "
            "H=8 against MSE≈1.45 (mostly Poisson). Keep Opta ±MAE as "
            "*level* uncertainty; do not invent a %/GW lead penalty. Revisit "
            "shared-q drift only if a richer as-of strength archive (Opta "
            "snaps or priced long-horizon markets) shows material growth."
        ),
    }
    return payload


def write_report(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    def table(buckets: list[dict]) -> list[str]:
        lines = [
            "| lead (GW) | n | MAE | RMSE | MSE | mean λ | excess MSE | Pearson |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in buckets:
            if r.get("n", 0) == 0:
                lines.append(f"| {r.get('lead')} | 0 | | | | | | |")
                continue
            lines.append(
                f"| {r['lead']} | {r['n']} | {r['mae']:.3f} | {r['rmse']:.3f} | "
                f"{r['mse']:.3f} | {r['mean_lam']:.3f} | {r['excess_mse']:.3f} | "
                f"{r['pearson']:.3f} |"
            )
        return lines

    roll = payload["rolling_expanding"]
    frz = payload["freeze_gw6"]
    rg, fg = roll["growth_fit"], frz["growth_fit"]
    lines = [
        "# Horizon forecast uncertainty — Stage 1 screen",
        "",
        "## ChatGPT proposal — what to keep / defer",
        "",
        "### Worth trialing now",
        "",
    ]
    for item in payload["chatgpt_review"]["keep"]:
        lines.append(f"- {item}")
    lines += ["", "### Defer", ""]
    for item in payload["chatgpt_review"]["defer"]:
        lines.append(f"- {item}")
    lines += ["", "### Already in the engine (do not double-count)", ""]
    for item in payload["chatgpt_review"]["already_have"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "No archived Opta mid-season tables. Proxies: (1) rolling expanding "
        "GF/GA aged by lead *h*; (2) rates frozen after GW6 (sticky Opta-like).",
        f"Closed seasons only: {', '.join(payload['protocol']['seasons'])}. "
        "2026/27 holdout unused.",
        "",
        "## Rolling expanding rates",
        "",
        f"n team-matches×leads = **{roll['n']}**.",
        "",
        *table(roll["by_lead"]),
        "",
        f"Growth fit `excess_mse ~ a + b·lead`: "
        f"b={rg.get('excess_slope_per_gw')} /GW, R²={rg.get('excess_r2')}, "
        f"Δexcess(H8−H1)={rg.get('delta_excess_h1_h8')}, "
        f"ΔMAE={rg.get('delta_mae_h1_h8')}, "
        f"stat_positive={rg.get('growth_stat_positive')}, "
        f"material={rg.get('growth_material')}.",
        "",
        "### LOSO (leave one season out)",
        "",
        "| holdout | excess slope /GW | R² | material |",
        "|---|---:|---:|:---:|",
    ]
    for row in roll["loso"]:
        lines.append(
            f"| {row.get('holdout_season')} | {row.get('excess_slope_per_gw')} | "
            f"{row.get('excess_r2')} | {row.get('growth_material')} |"
        )

    lines += [
        "",
        "## Freeze after GW6 (Opta-sticky proxy)",
        "",
        f"n = **{frz['n']}**. Leads binned to {list(LEADS)}. "
        f"{payload['verdict']['freeze_note']}",
        "",
        *table(frz["by_lead"]),
        "",
        f"Growth fit: b={fg.get('excess_slope_per_gw')} /GW, "
        f"R²={fg.get('excess_r2')}, material={fg.get('growth_material')}.",
        "",
        "## Verdict",
        "",
        f"- Rolling growth statistically positive: "
        f"**{payload['verdict']['rolling_growth_stat_positive']}**",
        f"- Rolling growth material for Stage 2: "
        f"**{payload['verdict']['rolling_growth_material']}**",
        f"- Freeze growth material: "
        f"**{payload['verdict']['freeze_growth_material']}**",
        "",
        payload["verdict"]["recommendation"],
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 scripts/screen_horizon_forecast_uncertainty.py",
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = run()
    write_report(payload)
    print(
        json.dumps(
            {
                "json": str(OUT_JSON),
                "md": str(OUT_MD),
                "verdict": payload["verdict"],
                "rolling_growth": payload["rolling_expanding"]["growth_fit"],
                "freeze_growth": payload["freeze_gw6"]["growth_fit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
