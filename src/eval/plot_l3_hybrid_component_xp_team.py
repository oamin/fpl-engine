"""Club xP components: GW8 freeze hybrid L3(t:t+1)+Opta(t+2…) vs actuals.

Freeze rates/shares/α at ``--freeze-gw`` (default 8). For each target g:

* ``g ∈ {t, t+1}`` — FD centreline λ → ``score_team_side`` (L3 near)
* ``g ≥ t+2`` — Opta blind λ → same ``score_team_side`` (L3+Opta far)

One season, one team. Plots every ``xp_*`` component vs actual FPL points.

  PYTHONPATH=. python3 -m src.eval.plot_l3_hybrid_component_xp_team \\
      --season 2025-26 --team Arsenal --freeze-gw 8 --horizon-end 38
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import SEASON_E0
from src.eval.l3_opta_lambda_horizon import (
    _l3_component_board_from_fits,
    _opta_fits_for_gw,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    load_prices,
)
from src.rules.fpl_2026 import (
    appearance_points,
    assist_points,
    card_points,
    clean_sheet_points,
    defcon_points,
    goal_points,
    goals_conceded_points,
    normalize_position,
    save_points,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]

XP_COMPS = (
    "xp_appear",
    "xp_goals",
    "xp_assists",
    "xp_cs",
    "xp_gc",
    "xp_defcon",
    "xp_bonus",
    "xp_cards",
    "xp_saves",
    "xp",
)
ACT_COMPS = (
    "P_appear",
    "P_G",
    "P_A",
    "P_CS",
    "P_GC",
    "P_defcon",
    "P_bonus",
    "P_cards",
    "P_saves",
    "P_total",
)
COMP_LABELS = (
    "Appear",
    "Goals",
    "Assists",
    "CS",
    "GC",
    "DefCon",
    "Bonus",
    "Cards",
    "Saves",
    "Total xP",
)


def _full_board_from_fits(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    """Like horizon helper but keep every xp_* component (DGW-safe sum)."""
    if side.empty or not fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fits:
        for team, team_n, prefix in (
            (str(fit["home"]), str(fit["home_n"]), "home"),
            (str(fit["away"]), str(fit["away_n"]), "away"),
        ):
            club = side.loc[side["team"].astype(str) == team].copy()
            if club.empty and "team_n" in side.columns:
                club = side.loc[side["team_n"] == team_n].copy()
            if club.empty:
                continue
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fit[f"xg_{prefix}"]),
                xcs=float(fit[f"xcs_{prefix}"]),
                xgc=float(fit[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            ).copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["team_n"] = norm_team(team_n)
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    all_p = pd.concat(parts, ignore_index=True)
    agg = {c: (c, "sum") for c in XP_COMPS if c in all_p.columns}
    agg.update(
        {
            "name": ("name", "first"),
            "position": ("position", "first"),
            "team": ("team", "first"),
            "team_n": ("team_n", "first"),
        }
    )
    return all_p.groupby("element", as_index=False).agg(**agg)


def _actual_full(merged: pd.DataFrame, gw: int) -> pd.DataFrame:
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)].copy()
    if g.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for r in g.itertuples(index=False):
        eid = int(r.element) if pd.notna(r.element) else None
        if eid is None:
            continue
        mins = int(float(r.minutes or 0))
        pos = normalize_position(str(r.position))
        goals = int(float(getattr(r, "goals_scored", 0) or 0))
        assists = int(float(getattr(r, "assists", 0) or 0))
        if bool(r.was_home):
            gc = int(float(r.team_a_score or 0))
        else:
            gc = int(float(r.team_h_score or 0))
        cs = mins >= 60 and gc == 0
        yellows = int(float(getattr(r, "yellow_cards", 0) or 0))
        reds = int(float(getattr(r, "red_cards", 0) or 0))
        saves = int(float(getattr(r, "saves", 0) or 0))
        bonus = int(float(getattr(r, "bonus", 0) or 0))
        defcon = int(float(getattr(r, "defensive_contribution", 0) or 0))
        rows.append(
            {
                "element": eid,
                "team_n": norm_team(str(r.team)),
                "minutes": mins,
                "P_appear": float(appearance_points(mins)),
                "P_G": float(goal_points(pos, goals)),
                "P_A": float(assist_points(assists)),
                "P_CS": float(clean_sheet_points(pos, mins, cs)),
                "P_GC": float(goals_conceded_points(pos, mins, gc)),
                "P_defcon": float(defcon_points(pos, defcon)),
                "P_bonus": float(bonus),
                "P_cards": float(card_points(yellows, reds)),
                "P_saves": float(save_points(saves)),
                "P_total": float(getattr(r, "total_points", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def collect(
    *,
    season: str,
    team: str,
    freeze_gw: int,
    horizon_end: int,
) -> pd.DataFrame:
    team_n = norm_team(team)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)
    hist = numagent_xp.asof_share_history(
        merged, before_gw=int(freeze_gw), season=season
    )
    if hist.empty:
        raise RuntimeError(f"empty hist before GW{freeze_gw}")
    alpha_g, alpha_a = _asof_alphas(hist)
    side, alpha_g, alpha_a = build_frozen_side_table(
        hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        season=season,
        historical_rate_baseline=True,
    )
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = float(league_avg_gf(profile))

    rows: list[dict[str, Any]] = []
    for g in range(int(freeze_gw), int(horizon_end) + 1):
        offset = int(g) - int(freeze_gw)
        if offset <= 1:
            source = "l3_fd"
            fits = centreline_fixture_fits(fixtures_for_gw(enriched, int(g)), fd)
        else:
            source = "l3_opta"
            fits = _opta_fits_for_gw(
                enriched,
                gw=int(g),
                ranks=ranks,
                profile=profile,
                league_gf=league_gf,
            )
        board = _full_board_from_fits(
            side, fits, alpha_g=float(alpha_g), alpha_a=float(alpha_a)
        )
        if board.empty:
            continue
        club_xp = board.loc[board["team_n"] == team_n]
        if club_xp.empty:
            # team string fallback
            club_xp = board.loc[
                board["team"].map(lambda t: norm_team(str(t)) == team_n)
            ]
        act = _actual_full(merged, int(g))
        club_act = act.loc[act["team_n"] == team_n] if len(act) else act
        if club_xp.empty or club_act.empty:
            continue
        row: dict[str, Any] = {
            "gw": int(g),
            "offset": offset,
            "source": source,
            "n_xp": int(len(club_xp)),
            "n_act": int(len(club_act)),
            "opp": "",
        }
        # opponent label from fits
        for fit in fits:
            if norm_team(str(fit.get("home_n") or "")) == team_n:
                row["opp"] = str(fit.get("away") or fit.get("away_n") or "")
            elif norm_team(str(fit.get("away_n") or "")) == team_n:
                row["opp"] = str(fit.get("home") or fit.get("home_n") or "")
        for xc, ac in zip(XP_COMPS, ACT_COMPS):
            row[xc] = float(club_xp[xc].sum()) if xc in club_xp.columns else float("nan")
            row[ac] = float(club_act[ac].sum()) if ac in club_act.columns else float("nan")
        rows.append(row)
        if g % 5 == 0:
            print(f"  GW{g} source={source} xp={row['xp']:.1f} act={row['P_total']:.1f}", flush=True)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, *, season: str, team: str, freeze_gw: int, out: Path) -> None:
    n = len(XP_COMPS)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 2.15 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(
        f"{team} {season} — hybrid L3 centreline xP components vs actuals",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.978,
        f"Freeze GW{freeze_gw}: L3/FD for g∈{{{freeze_gw},{freeze_gw+1}}} · "
        f"L3+Opta blind for g≥{freeze_gw+2} · club Σ",
        ha="center",
        va="top",
        fontsize=9,
        color="#444",
    )
    gws = df["gw"].to_numpy()
    split = float(freeze_gw) + 1.5
    for ax, xc, ac, lab in zip(axes, XP_COMPS, ACT_COMPS, COMP_LABELS):
        ax.axvspan(gws.min() - 0.5, split, color="#1f77b4", alpha=0.06, lw=0)
        ax.axvspan(split, gws.max() + 0.5, color="#9467bd", alpha=0.06, lw=0)
        ax.axvline(split, color="#666", ls=":", lw=1.0)
        ax.plot(
            gws,
            df[xc],
            color="#1f77b4",
            marker="s",
            ms=3.5,
            lw=1.4,
            label="Forecast xP",
        )
        ax.plot(
            gws,
            df[ac],
            color="#222",
            marker="o",
            ms=3.5,
            lw=1.1,
            label="Actual P",
        )
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[min(n - 1, len(axes) - 1)].set_xlabel("GW")
    # source legend proxies on last used axis
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", marker="s", lw=1.4, label="Forecast xP"),
            plt.Line2D([0], [0], color="#222", marker="o", lw=1.1, label="Actual P"),
            Patch(facecolor="#1f77b4", alpha=0.15, label="L3/FD (t:t+1)"),
            Patch(facecolor="#9467bd", alpha=0.15, label="L3+Opta (t+2…)"),
        ],
        loc="upper right",
        fontsize=7,
        frameon=False,
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".jpg"), dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--team", default="Arsenal")
    p.add_argument("--freeze-gw", type=int, default=8)
    p.add_argument("--horizon-end", type=int, default=38)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    team_slug = norm_team(args.team).replace(" ", "_")
    stem = (
        f"l3_hybrid_component_xp_{team_slug}_{args.season}_"
        f"freeze{args.freeze_gw}_gw{args.freeze_gw}-{args.horizon_end}"
    )
    out = args.out or (ROOT / "reports" / f"{stem}.png")
    art = Path("/opt/cursor/artifacts") / f"{stem}.png"

    print(
        f"[hybrid-xp] {args.team} {args.season} freeze={args.freeze_gw} "
        f"→{args.horizon_end}",
        flush=True,
    )
    df = collect(
        season=args.season,
        team=args.team,
        freeze_gw=args.freeze_gw,
        horizon_end=args.horizon_end,
    )
    if df.empty:
        raise SystemExit("no rows")
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    plot(
        df,
        season=args.season,
        team=args.team,
        freeze_gw=args.freeze_gw,
        out=out,
    )
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(out.read_bytes())
    art.with_suffix(".jpg").write_bytes(out.with_suffix(".jpg").read_bytes())
    (Path("/opt/cursor/artifacts") / csv_path.name).write_text(csv_path.read_text())

    # summary table
    lines = [
        f"# {args.team} {args.season} — hybrid L3 component xP vs actuals",
        "",
        f"Freeze **GW{args.freeze_gw}**: L3/FD for `g∈{{{args.freeze_gw},{args.freeze_gw+1}}}`; "
        f"L3+Opta blind for `g≥{args.freeze_gw+2}` (locked `l3_centreline_forecast`).",
        "",
        f"![plot]({out.name})",
        "",
        "| Component | Σ forecast | Σ actual | MAE/GW |",
        "| --- | ---: | ---: | ---: |",
    ]
    for xc, ac, lab in zip(XP_COMPS, ACT_COMPS, COMP_LABELS):
        mae = float(np.mean(np.abs(df[xc] - df[ac])))
        lines.append(
            f"| {lab} | {df[xc].sum():.1f} | {df[ac].sum():.1f} | {mae:.2f} |"
        )
    near = df.loc[df["source"] == "l3_fd"]
    far = df.loc[df["source"] == "l3_opta"]
    lines += [
        "",
        f"| Band | n GW | Σ xP | Σ actual |",
        f"| --- | ---: | ---: | ---: |",
        f"| L3/FD (t:t+1) | {len(near)} | {near['xp'].sum():.1f} | {near['P_total'].sum():.1f} |",
        f"| L3+Opta (t+2…) | {len(far)} | {far['xp'].sum():.1f} | {far['P_total'].sum():.1f} |",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.plot_l3_hybrid_component_xp_team "
        f"--season {args.season} --team {args.team!r} "
        f"--freeze-gw {args.freeze_gw} --horizon-end {args.horizon_end}",
        "```",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(lines))
    print(f"[hybrid-xp] wrote {out} n={len(df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
