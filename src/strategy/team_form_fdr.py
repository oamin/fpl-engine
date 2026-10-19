"""Team-form FDR beyond the priced window: expanding actuals → future margins.

At decision week t (as-of, GW < t only):
  team_actual[g] = Σ player total_points for the club in GW g
                   (exclude assistant managers: 0 minutes + pts > 0)
  team_mu        = expanding mean of team_actual over g < t
  team_se        = sample SD / √n   (diagnostic only — not in FDR)

Locked FDR (2026-10-16): **raw μ only**
  fdr = team_mu_own(t) − team_mu_opp(t)

For each fixture in GW ∈ [t+LEAD, t+LEAD+DEPTH):
  y = team_actual[target] − opp_actual[target]

Screen: Spearman / Pearson of fdr vs y; binned mean y by fdr decile.
Not a promote arm — measurement only.

  python -m src.strategy.team_form_fdr
  python -m src.strategy.team_form_fdr --lead 3 --depth 5 --seasons 2025-26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS
from src.strategy.layer4_bookie_path import _load_merged_enriched, _schedule
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "data" / "plots"

DEFAULT_LEAD = 3
DEFAULT_DEPTH = 5
GW_START = 8
MIN_HISTORY = 2  # need ≥2 GWs for a sample SD
SE_FLOOR = 0.5


def _is_assistant_manager_row(df: pd.DataFrame) -> pd.Series:
    """2024/25 Assistant Managers: 0 minutes but positive FPL points."""
    minutes = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0)
    pts = pd.to_numeric(df.get("total_points"), errors="coerce").fillna(0.0)
    return (minutes <= 0) & (pts > 0)


def _team_gw_actuals(merged: pd.DataFrame) -> pd.DataFrame:
    """Panel: gw, team (norm), pts — managers stripped."""
    df = merged.copy()
    n_drop = int(_is_assistant_manager_row(df).sum())
    if n_drop:
        df = df.loc[~_is_assistant_manager_row(df)].copy()
    df["gw"] = pd.to_numeric(df["GW"], errors="coerce").astype(int)
    df["team"] = df["team"].astype(str).map(norm_team)
    df["pts"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0.0)
    out = (
        df.groupby(["gw", "team"], as_index=False)["pts"]
        .sum()
        .sort_values(["team", "gw"])
        .reset_index(drop=True)
    )
    out.attrs["n_manager_rows_dropped"] = n_drop
    return out


def _form_asof(panel: pd.DataFrame, decision_gw: int) -> pd.DataFrame:
    """Expanding mean (+ diagnostic SE) of team GW points for gw < decision_gw."""
    hist = panel.loc[panel["gw"] < int(decision_gw)]
    rows: list[dict[str, Any]] = []
    for team, g in hist.groupby("team", sort=False):
        vals = g["pts"].to_numpy(float)
        n = int(len(vals))
        if n < MIN_HISTORY:
            continue
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n >= 2 else 0.0
        se = max(sd / np.sqrt(n), SE_FLOOR)
        rows.append(
            {
                "team": str(team),
                "n_gw": n,
                "team_mu": mu,
                "team_se": se,
                "team_sd": sd,
            }
        )
    return pd.DataFrame(rows)


def _actual_map(panel: pd.DataFrame, gw: int) -> dict[str, float]:
    sub = panel.loc[panel["gw"] == int(gw)]
    if sub.empty:
        return {}
    return dict(zip(sub["team"].astype(str), sub["pts"].astype(float), strict=True))


def collect_pairs(
    *,
    seasons: list[str],
    lead: int = DEFAULT_LEAD,
    depth: int = DEFAULT_DEPTH,
    gw_start: int = GW_START,
    gw_end: int = 38,
) -> pd.DataFrame:
    """Rows: one side of one future fixture, form frozen at decision_gw."""
    rows: list[dict[str, Any]] = []

    for season in seasons:
        merged = _load_merged_enriched(season)
        sched = _schedule(merged)
        panel = _team_gw_actuals(merged)
        avail = sorted(int(g) for g in panel["gw"].unique())
        decision_gws = [g for g in avail if gw_start <= g <= gw_end]

        n_mgr = int(panel.attrs.get("n_manager_rows_dropped", 0))
        if n_mgr:
            print(f"  {season}: stripped {n_mgr} assistant-manager rows", flush=True)

        for t in decision_gws:
            form = _form_asof(panel, t)
            if form.empty:
                continue
            mu_map = dict(zip(form["team"], form["team_mu"], strict=True))
            se_map = dict(zip(form["team"], form["team_se"], strict=True))
            n_map = dict(zip(form["team"], form["n_gw"], strict=True))

            for k in range(int(lead), int(lead) + int(depth)):
                tgt = int(t) + k
                week = sched.loc[sched["GW"] == tgt]
                if week.empty:
                    continue
                actual = _actual_map(panel, tgt)
                if not actual:
                    continue
                for r in week.itertuples(index=False):
                    home = norm_team(str(r.home_team))
                    away = norm_team(str(r.away_team))
                    if home not in mu_map or away not in mu_map:
                        continue
                    y_h = float(actual.get(home, 0.0))
                    y_a = float(actual.get(away, 0.0))
                    for team, opp, side, y, y_opp in (
                        (home, away, "home", y_h, y_a),
                        (away, home, "away", y_a, y_h),
                    ):
                        rows.append(
                            {
                                "season": season,
                                "decision_gw": int(t),
                                "target_gw": tgt,
                                "lead": k,
                                "team": team,
                                "opp": opp,
                                "side": side,
                                "fdr": float(mu_map[team] - mu_map[opp]),
                                "team_mu": float(mu_map[team]),
                                "team_mu_opp": float(mu_map[opp]),
                                "team_se": float(se_map[team]),
                                "n_gw": int(n_map[team]),
                                "pts": y,
                                "pts_opp": y_opp,
                                "pts_margin": y - y_opp,
                            }
                        )
            print(f"  {season} t={t:02d} form teams={len(form)}", flush=True)

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "n": 0,
        "vs_pts_margin": {"pearson": float("nan"), "spearman": float("nan")},
        "vs_pts": {"pearson": float("nan"), "spearman": float("nan")},
        "mu_vs_pts": {"pearson": float("nan"), "spearman": float("nan")},
        "by_lead": [],
        "by_season": [],
        "deciles": [],
    }
    if df.empty:
        return empty

    def _corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return {"pearson": float("nan"), "spearman": float("nan")}
        pear = float(np.corrcoef(x, y)[0, 1])
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        spear = float(np.corrcoef(rx, ry)[0, 1])
        return {"pearson": pear, "spearman": spear}

    x = df["fdr"].to_numpy(float)
    out: dict[str, Any] = {
        "n": int(len(df)),
        "vs_pts_margin": _corr(x, df["pts_margin"].to_numpy(float)),
        "vs_pts": _corr(x, df["pts"].to_numpy(float)),
        "mu_vs_pts": _corr(df["team_mu"].to_numpy(float), df["pts"].to_numpy(float)),
    }

    by_lead = []
    for lead, g in df.groupby("lead"):
        c = _corr(g["fdr"].to_numpy(float), g["pts_margin"].to_numpy(float))
        by_lead.append(
            {
                "lead": int(lead),
                "n": int(len(g)),
                "pearson_margin": c["pearson"],
                "spearman_margin": c["spearman"],
                "mean_pts": float(g["pts"].mean()),
            }
        )
    out["by_lead"] = by_lead

    by_season = []
    for season, g in df.groupby("season"):
        c = _corr(g["fdr"].to_numpy(float), g["pts_margin"].to_numpy(float))
        by_season.append(
            {
                "season": season,
                "n": int(len(g)),
                "pearson_margin": c["pearson"],
                "spearman_margin": c["spearman"],
            }
        )
    out["by_season"] = by_season

    q = pd.qcut(df["fdr"], 10, duplicates="drop")
    cal = (
        df.assign(_q=q)
        .groupby("_q", observed=True)
        .agg(
            n=("fdr", "size"),
            fdr_mean=("fdr", "mean"),
            margin_mean=("pts_margin", "mean"),
            pts_mean=("pts", "mean"),
        )
        .reset_index(drop=True)
    )
    out["deciles"] = cal.to_dict(orient="records")
    return out


def plot_calibration(df: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    sample = df.sample(n=min(4000, len(df)), random_state=0)
    ax.scatter(sample["fdr"], sample["pts_margin"], s=4, alpha=0.15, c="C0")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("raw-μ FDR at t  (μ_own − μ_opp)")
    ax.set_ylabel("pts margin at t+k  (own − opp)")
    sp = summary["vs_pts_margin"]["spearman"]
    ax.set_title(f"Expanding-mean FDR vs margin  (Spearman {sp:+.3f})")

    ax = axes[1]
    dec = pd.DataFrame(summary["deciles"])
    ax.plot(dec["fdr_mean"], dec["margin_mean"], "o-", color="C1")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("mean FDR in decile")
    ax.set_ylabel("mean pts margin")
    ax.set_title("Decile calibration")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path, *, lead: int, depth: int) -> None:
    lines = [
        "# Team-form FDR → future team performance",
        "",
        "Form at decision GW **t** (GW < t only): "
        "`team_actual[g] = Σ player total_points` that GW "
        "(**assistant managers stripped**: 0 minutes + pts > 0); "
        "**`team_mu` = expanding mean** of those GW totals. "
        "`team_se = SD/√n` kept as diagnostic only. "
        f"**Locked FDR:** `fdr = team_mu_own(t) − team_mu_opp(t)` on fixtures "
        f"**t+{lead} … t+{lead + depth - 1}**. "
        "Outcome: same `team_actual` at target GW; margin = own − opp.",
        "",
        f"n sides = **{summary.get('n', 0)}**",
        "",
        "## Correlations (raw-μ FDR)",
        "",
        f"- FDR vs pts margin: Pearson "
        f"**{summary['vs_pts_margin']['pearson']:+.3f}** · Spearman "
        f"**{summary['vs_pts_margin']['spearman']:+.3f}**",
        f"- FDR vs own pts: Pearson "
        f"**{summary['vs_pts']['pearson']:+.3f}** · Spearman "
        f"**{summary['vs_pts']['spearman']:+.3f}**",
        f"- team_mu alone vs own pts: Pearson "
        f"**{summary['mu_vs_pts']['pearson']:+.3f}** · Spearman "
        f"**{summary['mu_vs_pts']['spearman']:+.3f}**",
        "",
        "## By lead (k = target − t)",
        "",
        "| lead | n | Pearson(margin) | Spearman(margin) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in summary.get("by_lead") or []:
        lines.append(
            f"| {r['lead']} | {r['n']} | {r['pearson_margin']:+.3f} | "
            f"{r['spearman_margin']:+.3f} |"
        )
    lines += [
        "",
        "## By season",
        "",
        "| season | n | Pearson(margin) | Spearman(margin) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in summary.get("by_season") or []:
        lines.append(
            f"| {r['season']} | {r['n']} | {r['pearson_margin']:+.3f} | "
            f"{r['spearman_margin']:+.3f} |"
        )
    lines += [
        "",
        "## Decile calibration (mean margin by FDR decile)",
        "",
        "| decile μ(FDR) | n | mean margin | mean pts |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in summary.get("deciles") or []:
        lines.append(
            f"| {r['fdr_mean']:+.3f} | {r['n']} | {r['margin_mean']:+.2f} | "
            f"{r['pts_mean']:.1f} |"
        )
    lines += [
        "",
        "Plot: `data/plots/team_form_fdr_calibration.png`",
        "",
        "JSON: `data/processed/team_form_fdr_summary.json` · "
        "pairs: `data/processed/team_form_fdr_pairs.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--lead", type=int, default=DEFAULT_LEAD)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    print(
        f"collecting expanding-actual form FDR lead={args.lead} depth={args.depth} …",
        flush=True,
    )
    df = collect_pairs(
        seasons=seasons,
        lead=args.lead,
        depth=args.depth,
        gw_start=args.gw_start,
        gw_end=args.gw_end,
    )
    summary = summarise(df)
    summary["defs"] = {
        "team_actual": "Σ player total_points by club GW; strip 0-min + pts>0 (assistant managers)",
        "team_mu": "expanding mean of team_actual for GW < t",
        "team_se": "sample SD / √n — diagnostic only",
        "fdr": "team_mu_own(t) - team_mu_opp(t)  [LOCKED raw μ]",
        "outcome": "team_actual at target; margin = own - opp",
        "lead": args.lead,
        "depth": args.depth,
        "window": f"t+{args.lead} … t+{args.lead + args.depth - 1}",
        "min_history": MIN_HISTORY,
        "seasons": seasons,
    }

    pairs_path = PROCESSED / "team_form_fdr_pairs.csv"
    sum_path = PROCESSED / "team_form_fdr_summary.json"
    md_path = REPORTS / "team_form_fdr.md"
    plot_path = PLOTS / "team_form_fdr_calibration.png"

    df.to_csv(pairs_path, index=False)
    sum_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(summary, md_path, lead=args.lead, depth=args.depth)
    if not df.empty:
        plot_calibration(df, summary, plot_path)

    print(
        json.dumps(
            {
                "n": summary.get("n"),
                "spearman_margin": summary.get("vs_pts_margin", {}).get("spearman"),
                "pearson_margin": summary.get("vs_pts_margin", {}).get("pearson"),
                "by_season": summary.get("by_season"),
                "by_lead": summary.get("by_lead"),
            },
            indent=2,
        )
    )
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
