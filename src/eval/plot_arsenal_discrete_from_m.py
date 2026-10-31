"""Plot Arsenal discrete CS/GC laws from Dixon–Coles M (FD centreline).

Shows what the promoted same-skeleton screen scores:

* top — Bernoulli P(CS) from M vs actual CS ∈ {0,1}
* bottom — categorical P(⌊GA/2⌋=k) heatmap vs actual k

  PYTHONPATH=. python3 -m src.eval.plot_arsenal_discrete_from_m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.m_discrete_event_likelihood_screen import _fixture_event_pack
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
TEAM = "arsenal"  # norm_team key
TEAM_LABEL = "Arsenal"
SEASON_DEFAULT = "2025-26"
GW_START = 8
GW_END = 38
K_MAX_SHOW = 4  # FPL GC bins 0..4 (−0..−4 pts); rare higher folded


def _arsenal_side(fit: dict) -> str | None:
    if norm_team(str(fit.get("home_n") or fit.get("home") or "")) == TEAM:
        return "home"
    if norm_team(str(fit.get("away_n") or fit.get("away") or "")) == TEAM:
        return "away"
    return None


def _actual_arsenal_cs_gc(week: pd.DataFrame) -> tuple[float, int]:
    """Club CS (0/1 if any Arsenal starter ≥60' with GA=0) and GA for the side."""
    ars = week.loc[week["team"].map(lambda t: norm_team(str(t)) == TEAM)].copy()
    if ars.empty:
        return float("nan"), -1
    # Prefer a GKP/DEF row with minutes for GA; else any row.
    def_side = ars.loc[ars["position"].isin(["GKP", "DEF", "GK", "G"])]
    row = def_side.iloc[0] if len(def_side) else ars.iloc[0]
    if bool(row["was_home"]):
        ga = int(float(row["team_a_score"] or 0))
    else:
        ga = int(float(row["team_h_score"] or 0))
    starters = ars.loc[pd.to_numeric(ars["minutes"], errors="coerce").fillna(0) >= 60]
    cs = 1.0 if (len(starters) and ga == 0) else 0.0
    # If blank / no 60' Arsenal players, still report GA from the fixture.
    if starters.empty:
        cs = 1.0 if ga == 0 else 0.0
    return cs, ga


def collect(season: str) -> pd.DataFrame:
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    rows: list[dict] = []
    for gw in range(GW_START, GW_END + 1):
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw), season=season)
        week = merged.loc[merged["GW"] == int(gw)].copy()
        if week.empty or hist.empty:
            continue
        alpha_g, alpha_a = _asof_alphas(hist)
        side, alpha_g, alpha_a = build_frozen_side_table(
            hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            season=season,
            historical_rate_baseline=True,
        )
        if side.empty:
            continue
        fx = fixtures_for_gw(enriched, int(gw))
        fits = centreline_fixture_fits(fx, fd)
        ars_fits = [f for f in fits if _arsenal_side(f) is not None]
        if not ars_fits:
            continue
        act_cs, act_ga = _actual_arsenal_cs_gc(week)
        if act_ga < 0:
            continue
        # Aggregate DGW: product of independent fixture laws → sum logs; for
        # display use mean P(CS) and average PMF; actual CS = 1 only if all
        # fixtures CS, actual k = sum of per-fixture ⌊GA/2⌋.
        p_cs = 0.0
        pmf_acc = None
        opp_names: list[str] = []
        k_act_sum = 0
        for fit in ars_fits:
            prefix = _arsenal_side(fit)
            assert prefix is not None
            pack = _fixture_event_pack(fit)
            p_cs += float(pack[f"p_cs_{prefix}"])
            pmf = np.asarray(pack[f"gc_pmf_{prefix}"], dtype=float)
            if pmf_acc is None:
                pmf_acc = np.zeros_like(pmf)
            if len(pmf) > len(pmf_acc):
                pmf_acc = np.pad(pmf_acc, (0, len(pmf) - len(pmf_acc)))
            if len(pmf) < len(pmf_acc):
                pmf = np.pad(pmf, (0, len(pmf_acc) - len(pmf)))
            pmf_acc = pmf_acc + pmf
            opp = pack["away"] if prefix == "home" else pack["home"]
            opp_names.append(str(opp))
            # Per-fixture actual GA from week join on side.
            ars = week.loc[week["team"].map(lambda t: norm_team(str(t)) == TEAM)]
            if prefix == "home":
                ga_fx = int(float(ars["team_a_score"].iloc[0])) if len(ars) else act_ga
            else:
                ga_fx = int(float(ars["team_h_score"].iloc[0])) if len(ars) else act_ga
            k_act_sum += int(ga_fx // 2)
        n_fx = len(ars_fits)
        p_cs /= n_fx
        pmf_acc = pmf_acc / n_fx
        # Single-fixture CS actual already; multi: 1 if every fixture CS.
        if n_fx == 1:
            cs_out, k_out = act_cs, int(act_ga // 2)
        else:
            cs_out = act_cs  # club-week CS flag from helpers
            k_out = k_act_sum
        row = {
            "gw": int(gw),
            "n_fx": n_fx,
            "opp": "+".join(opp_names),
            "p_cs": p_cs,
            "act_cs": float(cs_out),
            "act_ga": int(act_ga) if n_fx == 1 else int(k_out * 2),  # approx for label
            "act_k": int(k_out),
            "e_k": float(np.dot(np.arange(len(pmf_acc)), pmf_acc)),
        }
        for k in range(K_MAX_SHOW + 1):
            row[f"p_k{k}"] = float(pmf_acc[k]) if k < len(pmf_acc) else 0.0
        # fold tail
        if len(pmf_acc) > K_MAX_SHOW + 1:
            row[f"p_k{K_MAX_SHOW}"] += float(pmf_acc[K_MAX_SHOW + 1 :].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, *, season: str, out: Path) -> None:
    gws = df["gw"].to_numpy()
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.35]}
    )
    fig.suptitle(
        f"{TEAM_LABEL} {season} — discrete CS/GC from Dixon–Coles M (FD) · GW{GW_START}–{GW_END}",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.995,
        "Top: Bernoulli P(CS) from M vs actual CS.  "
        "Bottom: categorical P(⌊GA/2⌋=k) from M; white × = actual k.",
        ha="center",
        va="top",
        fontsize=9,
        color="#444",
    )

    ax0 = axes[0]
    ax0.plot(
        gws,
        df["p_cs"],
        color="#1f77b4",
        marker="s",
        ms=4,
        lw=1.6,
        label="FD P(CS) from M",
    )
    ax0.plot(
        gws,
        df["act_cs"],
        color="#222",
        marker="o",
        ms=4.5,
        lw=1.2,
        label="Actual CS (0/1)",
    )
    ax0.set_ylabel("Clean sheet")
    ax0.set_ylim(-0.05, 1.08)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8, frameon=False)

    ax1 = axes[1]
    mat = np.vstack([df[f"p_k{k}"].to_numpy(float) for k in range(K_MAX_SHOW + 1)])
    im = ax1.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.55, float(mat.max())),
        extent=(gws.min() - 0.5, gws.max() + 0.5, -0.5, K_MAX_SHOW + 0.5),
        interpolation="nearest",
    )
    ax1.scatter(
        df["gw"],
        df["act_k"].clip(upper=K_MAX_SHOW),
        marker="x",
        s=55,
        c="white",
        linewidths=2.0,
        label="Actual ⌊GA/2⌋",
        zorder=5,
    )
    ax1.scatter(
        df["gw"],
        df["act_k"].clip(upper=K_MAX_SHOW),
        marker="x",
        s=55,
        c="#111",
        linewidths=0.9,
        zorder=6,
    )
    ax1.set_ylabel("GC bin k = ⌊GA/2⌋  (FPL −k pts)")
    ax1.set_yticks(range(K_MAX_SHOW + 1))
    ax1.set_yticklabels([f"{k}  (−{k})" for k in range(K_MAX_SHOW + 1)])
    ax1.set_xlabel("GW")
    ax1.set_xticks(gws[::2])
    ax1.grid(False)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
    cbar.set_label("P(k) from M", fontsize=8)
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.85)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".jpg"), dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default=SEASON_DEFAULT)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG path (default under reports/ + artifacts/)",
    )
    args = p.parse_args(argv)
    season = args.season
    tag = season.replace("/", "-")
    stem = f"l3_fd_arsenal_discrete_cs_gc_from_M_{tag}_gw{GW_START}-{GW_END}"
    out = args.out or (ROOT / "reports" / f"{stem}.png")
    art = Path("/opt/cursor/artifacts") / f"{stem}.png"

    print(f"[plot] collect {TEAM_LABEL} {season} …", flush=True)
    df = collect(season)
    if df.empty:
        raise SystemExit(f"no {TEAM_LABEL} rows for {season}")
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    plot(df, season=season, out=out)
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(out.read_bytes())
    art.with_suffix(".jpg").write_bytes(out.with_suffix(".jpg").read_bytes())
    (Path("/opt/cursor/artifacts") / f"{stem}.csv").write_text(csv_path.read_text())

    md = out.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                f"# {TEAM_LABEL} {season} — discrete CS/GC from M (FD)",
                "",
                "Visual companion to the promoted same-skeleton screen:",
                "Bernoulli `P(CS)` and categorical `P(⌊GA/2⌋=k)` from Dixon–Coles `M`.",
                "",
                f"![plot]({out.name})",
                "",
                "| | mean |",
                "| --- | ---: |",
                f"| FD P(CS) | {df['p_cs'].mean():.3f} |",
                f"| Actual CS rate | {df['act_cs'].mean():.3f} |",
                f"| FD E[⌊GA/2⌋] | {df['e_k'].mean():.3f} |",
                f"| Actual mean ⌊GA/2⌋ | {df['act_k'].mean():.3f} |",
                "",
                "```bash",
                f"PYTHONPATH=. python3 -m src.eval.plot_arsenal_discrete_from_m --season {season}",
                "```",
                "",
            ]
        )
    )
    print(f"[plot] wrote {out} · n={len(df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
