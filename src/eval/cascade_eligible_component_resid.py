"""Stepped eligible-pool E[G, CS, GC]: cascade_short / cascade_long vs L3.

Pre-registered: ``protocol.cascade_eligible_component_resid``.

At each decision GW ``t ∈ {8,…,38}``:

1. Refresh as-of hist / share pool (full club ``m>0``).
2. Score **cascade_short** — FD centreline λ → generative MC-mean
   ``xp_goals`` / ``xp_cs`` / ``xp_gc``.
3. Score **cascade_long** — Opta place GF/GA (``venue_mode=blind``) → same
   generative stack.
4. Score **L3** — as-of ``score_xp_v2`` centreline components.
5. Residual rows = players with **actual minutes > 60** that GW
   (and ``n_prior ≥ 3``) who appear on all three boards.

Does **not** change defaults. Diagnostic residual quality vs L3 — not a
player-GW likelihood gate.

  PYTHONPATH=. python3 -m src.eval.cascade_eligible_component_resid --n 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import (
    DEFAULT_SEASON,
    SEASON_E0,
    SEED,
    SHARE_MIN_APPS,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_xi_ft_component_resid import actual_components
from src.eval.cascade_mean_xi_ft_hist import HORIZON_END
from src.eval.cascade_mean_xi_ft_step_pool_resid import mean_component_boards_split
from src.eval.cascade_vs_l3_component_xp import _v2_component_board
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import (
    centreline_fixture_fits,
    fixtures_for_gw,
    load_prices,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_eligible_component_resid.md"
STATE = ROOT / "REPORT_STATE.json"
PROTOCOL_KEY = "cascade_eligible_component_resid"

GW_START = 8
GW_END = 38
MIN_MINUTES = 60.0
MIN_HISTORY = 3
N_STRANDS = 40
COMPS = ("G", "CS", "GC")
BLOCKS = ((8, 12), (13, 19), (20, 26), (27, 33), (34, 38))


def _now() -> float:
    return time.time()


def _n_prior(hist: pd.DataFrame) -> pd.Series:
    if hist.empty or "minutes" not in hist.columns:
        return pd.Series(dtype=float)
    return (
        hist.loc[pd.to_numeric(hist["minutes"], errors="coerce").fillna(0) > 0]
        .groupby("element")["GW"]
        .nunique()
    )


def _board_by_element(board: pd.DataFrame) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if board is None or board.empty:
        return out
    for r in board.itertuples(index=False):
        eid = pd.to_numeric(getattr(r, "element", None), errors="coerce")
        if not pd.notna(eid):
            continue
        out[int(eid)] = {
            "xp_goals": float(getattr(r, "xp_goals", float("nan"))),
            "xp_cs": float(getattr(r, "xp_cs", float("nan"))),
            "xp_gc": float(getattr(r, "xp_gc", float("nan"))),
            "xp": float(
                getattr(r, "xp", getattr(r, "score_xp_v2", float("nan")))
            ),
        }
    return out


def _resid_stats(
    pred: np.ndarray, act: np.ndarray
) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(act)
    if int(mask.sum()) == 0:
        return {
            "n": 0.0,
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "corr": float("nan"),
            "sum_pred": float("nan"),
            "sum_act": float("nan"),
        }
    p = pred[mask]
    a = act[mask]
    resid = p - a
    corr = float(np.corrcoef(p, a)[0, 1]) if len(p) >= 2 and np.std(p) > 0 and np.std(a) > 0 else float("nan")
    return {
        "n": float(len(p)),
        "bias": float(np.mean(resid)),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "corr": corr,
        "sum_pred": float(np.sum(p)),
        "sum_act": float(np.sum(a)),
    }


def _arm_metrics(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {"arm": prefix, "n": int(len(df))}
    for comp, pcol, acol in (
        ("G", f"{prefix}_G", "P_G"),
        ("CS", f"{prefix}_CS", "P_CS"),
        ("GC", f"{prefix}_GC", "P_GC"),
    ):
        st = _resid_stats(df[pcol].to_numpy(float), df[acol].to_numpy(float))
        out[comp] = st
    return out


def _plot_gw_bias(gw_df: pd.DataFrame, path: Path) -> None:
    if gw_df.empty:
        return
    gws = gw_df["gw"].to_numpy(int)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    series = (
        ("bias_short_G", "bias_long_G", "bias_l3_G", "bias G (xP−P)"),
        ("bias_short_CS", "bias_long_CS", "bias_l3_CS", "bias CS"),
        ("bias_short_GC", "bias_long_GC", "bias_l3_GC", "bias GC"),
    )
    colors = {"short": "#2980b9", "long": "#c0392b", "l3": "#27ae60"}
    for ax, (ks, kl, k3, lab) in zip(axes, series):
        ax.axhline(0.0, color="#888", lw=0.8)
        ax.plot(gws, gw_df[ks], "o-", color=colors["short"], ms=3.5, lw=1.3, label="short FD")
        ax.plot(gws, gw_df[kl], "s-", color=colors["long"], ms=3.5, lw=1.3, label="long Opta")
        ax.plot(gws, gw_df[k3], "^-", color=colors["l3"], ms=3.5, lw=1.3, label="L3")
        ax.set_ylabel(lab)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(framealpha=0.95, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("GW")
    axes[0].set_title("Eligible m>60 residual bias by GW (2025-26)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _fmt(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if nd >= 0 and x != 0 else f"{x:.{abs(nd)}f}"


def _fmt_mae(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.3f}"


def run(
    *,
    season: str,
    step_gws: list[int],
    n_strands: int,
    seed0: int,
    goal_weight: str,
) -> dict[str, Any]:
    e0 = SEASON_E0[season]
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    row_parts: list[pd.DataFrame] = []
    gw_rows: list[dict[str, Any]] = []
    t_all = _now()

    for t in step_gws:
        print(f"[elig-comp] === GW{t} ===", flush=True)
        hist = numagent_xp.asof_share_history(merged, before_gw=int(t), season=season)
        if hist.empty:
            print(f"[elig-comp] GW{t} empty hist — skip", flush=True)
            continue
        alpha_g, alpha_a = _asof_alphas(hist)
        team_hist = hist_by_team_norm(hist)
        prior = _n_prior(hist)

        fx = fixtures_for_gw(enriched, int(t))
        fits = centreline_fixture_fits(fx, fd)
        if len(fits) < 5:
            print(
                f"[elig-comp] GW{t} FD fits={len(fits)} < 5 — skip",
                flush=True,
            )
            continue

        share_pool = _build_hist_share_pool(
            hist, prices, decision_gw=int(t), season=season
        )
        # Eligible generation board = full share pool (m>0). Residual filter later.
        select_pool = share_pool.copy()
        fixtures_gw = _hist_fixtures_by_gw(enriched, start=int(t), end=int(t))
        plan_gws = sorted(fixtures_gw)
        if not plan_gws:
            print(f"[elig-comp] GW{t} no fixtures — skip", flush=True)
            continue

        # L3 centreline components (same side as FD short)
        l3_board = _v2_component_board(
            hist=hist,
            enriched=enriched,
            fd=fd,
            prices=prices,
            season=season,
            decision_gw=int(t),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            force_ids=None,
        )
        if l3_board.empty:
            print(f"[elig-comp] GW{t} empty L3 board — skip", flush=True)
            continue

        step_seed = int(seed0) + 10_000 * int(t)
        priced_short = _fd_priced_by_gw(enriched, fd, plan_gws)
        print(
            f"[elig-comp] GW{t} share={len(share_pool)} fits={len(fits)} "
            f"N={n_strands}",
            flush=True,
        )
        short_boards, _, short_meta = mean_component_boards_split(
            select_pool=select_pool,
            share_pool=share_pool,
            fixtures_gw=fixtures_gw,
            priced_by_gw=priced_short,
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            plan_gws=plan_gws,
            n_strands=n_strands,
            seed0=step_seed,
            goal_weight=goal_weight,
        )
        long_boards, _, long_meta = mean_component_boards_split(
            select_pool=select_pool,
            share_pool=share_pool,
            fixtures_gw=fixtures_gw,
            priced_by_gw={},  # Opta / cascade_long
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            plan_gws=plan_gws,
            n_strands=n_strands,
            seed0=step_seed + 1,
            goal_weight=goal_weight,
        )
        short_board = short_boards[int(t)]
        long_board = long_boards[int(t)]
        print(
            f"[elig-comp] GW{t} boards short={short_meta['elapsed_s']:.1f}s "
            f"long={long_meta['elapsed_s']:.1f}s",
            flush=True,
        )

        act = actual_components(merged, int(t))
        if act.empty:
            continue
        short_m = _board_by_element(short_board)
        long_m = _board_by_element(long_board)
        l3_m = _board_by_element(l3_board)

        rows: list[dict[str, Any]] = []
        for r in act.itertuples(index=False):
            mins = float(r.minutes)
            if mins <= MIN_MINUTES:
                continue
            eid = int(r.element)
            if float(prior.get(eid, 0) or 0) < MIN_HISTORY:
                continue
            if eid not in short_m or eid not in long_m or eid not in l3_m:
                continue
            s, o, l3 = short_m[eid], long_m[eid], l3_m[eid]
            if not all(
                np.isfinite(v)
                for v in (
                    s["xp_goals"],
                    s["xp_cs"],
                    s["xp_gc"],
                    o["xp_goals"],
                    o["xp_cs"],
                    o["xp_gc"],
                    l3["xp_goals"],
                    l3["xp_cs"],
                    l3["xp_gc"],
                )
            ):
                continue
            rows.append(
                {
                    "season": season,
                    "gw": int(t),
                    "element": eid,
                    "name": str(r.name),
                    "position": str(r.position),
                    "minutes": mins,
                    "n_prior": float(prior.get(eid, 0) or 0),
                    "short_G": s["xp_goals"],
                    "short_CS": s["xp_cs"],
                    "short_GC": s["xp_gc"],
                    "long_G": o["xp_goals"],
                    "long_CS": o["xp_cs"],
                    "long_GC": o["xp_gc"],
                    "l3_G": l3["xp_goals"],
                    "l3_CS": l3["xp_cs"],
                    "l3_GC": l3["xp_gc"],
                    "P_G": float(r.P_G),
                    "P_CS": float(r.P_CS),
                    "P_GC": float(r.P_GC),
                }
            )
        if not rows:
            print(f"[elig-comp] GW{t} no residual rows — skip", flush=True)
            continue
        gw_df = pd.DataFrame(rows)
        row_parts.append(gw_df)

        def _b(prefix: str, comp: str) -> float:
            return float(
                (gw_df[f"{prefix}_{comp}"] - gw_df[f"P_{comp}"]).mean()
            )

        def _mae(prefix: str, comp: str) -> float:
            return float(
                (gw_df[f"{prefix}_{comp}"] - gw_df[f"P_{comp}"]).abs().mean()
            )

        gw_rows.append(
            {
                "gw": int(t),
                "n": int(len(gw_df)),
                "short_board_s": float(short_meta["elapsed_s"]),
                "long_board_s": float(long_meta["elapsed_s"]),
                **{
                    f"bias_{arm}_{comp}": _b(arm, comp)
                    for arm in ("short", "long", "l3")
                    for comp in COMPS
                },
                **{
                    f"mae_{arm}_{comp}": _mae(arm, comp)
                    for arm in ("short", "long", "l3")
                    for comp in COMPS
                },
                **{
                    f"sum_{arm}_{comp}": float(gw_df[f"{arm}_{comp}"].sum())
                    for arm in ("short", "long", "l3")
                    for comp in COMPS
                },
                **{
                    f"sum_P_{comp}": float(gw_df[f"P_{comp}"].sum())
                    for comp in COMPS
                },
            }
        )
        print(
            f"[elig-comp] GW{t} n={len(gw_df)} "
            f"bias G short/long/l3="
            f"{gw_rows[-1]['bias_short_G']:+.3f}/"
            f"{gw_rows[-1]['bias_long_G']:+.3f}/"
            f"{gw_rows[-1]['bias_l3_G']:+.3f}",
            flush=True,
        )

    if not row_parts:
        raise RuntimeError("no eligible residual rows")
    players = pd.concat(row_parts, ignore_index=True)
    gw_df = pd.DataFrame(gw_rows)

    overall = {
        "short": _arm_metrics(players, "short"),
        "long": _arm_metrics(players, "long"),
        "l3": _arm_metrics(players, "l3"),
    }
    # Paired Δ MAE and Δ |bias| vs L3
    paired: dict[str, Any] = {}
    for arm in ("short", "long"):
        paired[arm] = {}
        for comp in COMPS:
            pred_a = players[f"{arm}_{comp}"].to_numpy(float)
            pred_l = players[f"l3_{comp}"].to_numpy(float)
            act = players[f"P_{comp}"].to_numpy(float)
            err_a = np.abs(pred_a - act)
            err_l = np.abs(pred_l - act)
            resid_a = pred_a - act
            resid_l = pred_l - act
            paired[arm][comp] = {
                "delta_mae": float(np.mean(err_a - err_l)),
                "delta_bias": float(np.mean(resid_a - resid_l)),
                "mae_arm": float(np.mean(err_a)),
                "mae_l3": float(np.mean(err_l)),
            }

    blocks: list[dict[str, Any]] = []
    for lo, hi in BLOCKS:
        sub = players.loc[(players["gw"] >= lo) & (players["gw"] <= hi)]
        if sub.empty:
            continue
        b: dict[str, Any] = {"block": f"GW{lo}-{hi}", "n": int(len(sub))}
        for arm in ("short", "long", "l3"):
            for comp in COMPS:
                st = _resid_stats(
                    sub[f"{arm}_{comp}"].to_numpy(float),
                    sub[f"P_{comp}"].to_numpy(float),
                )
                b[f"{arm}_{comp}_bias"] = st["bias"]
                b[f"{arm}_{comp}_mae"] = st["mae"]
                b[f"{arm}_{comp}_sum"] = st["sum_pred"]
                b[f"P_{comp}_sum"] = st["sum_act"]
        blocks.append(b)

    return {
        "season": season,
        "goal_weight": goal_weight,
        "n_strands": n_strands,
        "seed": seed0,
        "n_player_gw": int(len(players)),
        "n_gw": int(gw_df["gw"].nunique()) if not gw_df.empty else 0,
        "gws": [int(x) for x in gw_df["gw"].tolist()] if not gw_df.empty else [],
        "elapsed_s": float(_now() - t_all),
        "overall": overall,
        "paired_vs_l3": paired,
        "blocks": blocks,
        "players": players,
        "gw": gw_df,
    }


def _write_report(payload: dict[str, Any], plot_path: Path) -> str:
    ov = payload["overall"]
    paired = payload["paired_vs_l3"]
    lines = [
        "# Eligible-pool cascade_short / cascade_long vs L3 — E[G, CS, GC]",
        "",
        f"Season **{payload['season']}** · GW{min(payload['gws'])}–{max(payload['gws'])} "
        f"({payload['n_gw']} weeks) · N={payload['n_strands']} · "
        f"goal_weight=`{payload['goal_weight']}` · "
        f"n={payload['n_player_gw']} player-GWs · wall {payload['elapsed_s']:.1f}s",
        "",
        "Pre-registered: `protocol.cascade_eligible_component_resid`.",
        "",
        "## Design",
        "",
        "- **Pool (generate):** full-club share pool `m>0` as-of before each decision GW.",
        "- **Pool (residual):** actual `minutes>60`, `n_prior≥3`, scored by all three arms.",
        "- **cascade_short:** FD AH+OU centreline λ → generative MC-mean components.",
        "- **cascade_long:** Opta place GF/GA (`venue_mode=blind`) → same generative stack.",
        "- **L3:** as-of `score_xp_v2` centreline `xp_goals` / `xp_cs` / `xp_gc`.",
        "- Stepped: as-of refresh each GW; decision-GW only.",
        "- Defaults **unchanged** (diagnostic).",
        "",
        f"![bias]({plot_path})",
        "",
        "## Overall residual vs actual (player-GW)",
        "",
        "| arm | n | G bias | G MAE | G corr | CS bias | CS MAE | CS corr | GC bias | GC MAE | GC corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("short", "long", "l3"):
        m = ov[arm]
        lines.append(
            f"| {arm} | {int(m['n'])} | "
            f"{_fmt(m['G']['bias'])} | {_fmt_mae(m['G']['mae'])} | {_fmt(m['G']['corr'])} | "
            f"{_fmt(m['CS']['bias'])} | {_fmt_mae(m['CS']['mae'])} | {_fmt(m['CS']['corr'])} | "
            f"{_fmt(m['GC']['bias'])} | {_fmt_mae(m['GC']['mae'])} | {_fmt(m['GC']['corr'])} |"
        )
    lines += [
        "",
        "## Quality vs L3 (paired Δ on same rows)",
        "",
        "ΔMAE = MAE(arm) − MAE(L3); negative means arm closer to actual than L3.",
        "",
        "| arm | G ΔMAE | G Δbias | CS ΔMAE | CS Δbias | GC ΔMAE | GC Δbias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("short", "long"):
        p = paired[arm]
        lines.append(
            f"| {arm}−L3 | {_fmt(p['G']['delta_mae'])} | {_fmt(p['G']['delta_bias'])} | "
            f"{_fmt(p['CS']['delta_mae'])} | {_fmt(p['CS']['delta_bias'])} | "
            f"{_fmt(p['GC']['delta_mae'])} | {_fmt(p['GC']['delta_bias'])} |"
        )
    lines += [
        "",
        "## Block totals (Σ pred − Σ actual bias implied)",
        "",
        "| block | n | short G bias | long G bias | L3 G bias | short CS bias | long CS bias | L3 CS bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in payload["blocks"]:
        lines.append(
            f"| {b['block']} | {b['n']} | "
            f"{_fmt(b['short_G_bias'])} | {_fmt(b['long_G_bias'])} | {_fmt(b['l3_G_bias'])} | "
            f"{_fmt(b['short_CS_bias'])} | {_fmt(b['long_CS_bias'])} | {_fmt(b['l3_CS_bias'])} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    # Auto-summarise G MAE ranking
    g_mae = {arm: ov[arm]["G"]["mae"] for arm in ("short", "long", "l3")}
    order = sorted(g_mae, key=lambda a: g_mae[a])
    lines.append(
        f"- G MAE ranking (lower better): "
        + " < ".join(f"**{a}** ({g_mae[a]:.3f})" for a in order)
        + "."
    )
    cs_mae = {arm: ov[arm]["CS"]["mae"] for arm in ("short", "long", "l3")}
    order_cs = sorted(cs_mae, key=lambda a: cs_mae[a])
    lines.append(
        f"- CS MAE ranking: "
        + " < ".join(f"**{a}** ({cs_mae[a]:.3f})" for a in order_cs)
        + "."
    )
    lines += [
        f"- short−L3 G ΔMAE **{_fmt(paired['short']['G']['delta_mae'])}**; "
        f"long−L3 G ΔMAE **{_fmt(paired['long']['G']['delta_mae'])}**.",
        "- Not a promote; defaults unchanged.",
        "",
        "## Reproduce",
        "",
        "```",
        f"PYTHONPATH=. python3 -m src.eval.cascade_eligible_component_resid "
        f"--n {payload['n_strands']} --goal-weight {payload['goal_weight']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument(
        "--steps",
        type=str,
        default=",".join(str(g) for g in range(GW_START, GW_END + 1)),
    )
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--goal-weight",
        choices=("rate_minutes", "rate"),
        default="rate_minutes",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_eligible_component_resid.json"),
    )
    args = ap.parse_args(argv)

    step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    # Touch HORIZON_END so import stays used for protocol alignment.
    _ = int(HORIZON_END)
    _ = float(SHARE_MIN_APPS)

    print(
        f"[elig-comp] season={args.season} steps={step_gws[0]}…{step_gws[-1]} "
        f"N={args.n} goal_weight={args.goal_weight}",
        flush=True,
    )
    payload = run(
        season=str(args.season),
        step_gws=step_gws,
        n_strands=int(args.n),
        seed0=int(args.seed),
        goal_weight=str(args.goal_weight),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    players: pd.DataFrame = payload.pop("players")
    gw: pd.DataFrame = payload.pop("gw")
    plot_path = out.with_name(out.stem + "_bias.png")
    _plot_gw_bias(gw, plot_path)

    players_path = out.with_name(out.stem + "_players.csv")
    gw_path = out.with_name(out.stem + "_gw.csv")
    players.to_csv(players_path, index=False)
    gw.to_csv(gw_path, index=False)

    slim = dict(payload)
    slim["players_csv"] = str(players_path)
    slim["gw_csv"] = str(gw_path)
    slim["plot"] = str(plot_path)
    out.write_text(json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8")

    # Also stash under data/processed for repo-local replay
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    players.to_csv(proc / "cascade_eligible_component_resid_players.csv", index=False)
    gw.to_csv(proc / "cascade_eligible_component_resid_gw.csv", index=False)
    (proc / "cascade_eligible_component_resid.json").write_text(
        json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8"
    )

    report = _write_report({**payload, "players": players, "gw": gw}, plot_path)
    REPORT.write_text(report + "\n", encoding="utf-8")

    state = {
        "task": PROTOCOL_KEY,
        "status": "done",
        "change": "Stepped eligible-pool E[G,CS,GC] short/long vs L3 (m>60)",
        "season": payload["season"],
        "n_player_gw": payload["n_player_gw"],
        "n_gw": payload["n_gw"],
        "overall_G_mae": {
            arm: payload["overall"][arm]["G"]["mae"] for arm in ("short", "long", "l3")
        },
        "paired_G_delta_mae": {
            arm: payload["paired_vs_l3"][arm]["G"]["delta_mae"]
            for arm in ("short", "long")
        },
        "report": str(REPORT),
        "out": str(out),
        "elapsed_s": payload["elapsed_s"],
    }
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)
    print(f"[elig-comp] report {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
