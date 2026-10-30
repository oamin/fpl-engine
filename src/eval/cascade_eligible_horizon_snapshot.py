"""As-of-t horizon snapshots: cascade_short/long vs contemporaneous L3 + actuals.

Pre-registered: ``protocol.cascade_eligible_horizon_snapshot``.

This is the intended design (supersedes decision-GW-only
``cascade_eligible_component_resid``).

At each decision GW ``t``:

1. Freeze as-of hist / full-club share pool / alphas at ``t``.
2. Build a **snapshot** of generative E[G, CS, GC] for every target
   ``g ∈ {t,…,horizon_end}``:
   - **cascade_long** — Opta place GF/GA (``venue_mode=blind``) for all ``g``.
   - **cascade_short** — FD centreline λ only for ``g ∈ {t, t+1}`` (causal
     short book; no future-FD leak into the freeze).
3. For each target ``g``, score **L3** contemporaneously: as-of hist before
   ``g`` + FD priced near ``g`` (``score_xp_v2`` centreline components).
4. Residual rows at target ``g``: actual ``minutes>60``, ``n_prior≥3`` on the
   freeze at ``t``, present on the cascade board and on L3.
5. Step ``t → t+1`` and repeat.

Primary lens: residual quality by ``horizon_offset = g − t`` (how well the
freeze forecasts into the future vs L3-near-g and vs actuals).

  PYTHONPATH=. python3 -m src.eval.cascade_eligible_horizon_snapshot --n 20
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
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_xi_ft_component_resid import actual_components
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
from src.models.short_horizon_cascade import load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_eligible_horizon_snapshot.md"
STATE = ROOT / "REPORT_STATE.json"
PROTOCOL_KEY = "cascade_eligible_horizon_snapshot"

GW_START = 8
HORIZON_END = 38
MIN_MINUTES = 60.0
MIN_HISTORY = 3
N_STRANDS = 20
COMPS = ("G", "CS", "GC")


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
        }
    return out


def _resid_stats(pred: np.ndarray, act: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(act)
    if int(mask.sum()) == 0:
        return {
            "n": 0.0,
            "bias": float("nan"),
            "mae": float("nan"),
            "corr": float("nan"),
        }
    p = pred[mask]
    a = act[mask]
    resid = p - a
    corr = (
        float(np.corrcoef(p, a)[0, 1])
        if len(p) >= 2 and float(np.std(p)) > 0 and float(np.std(a)) > 0
        else float("nan")
    )
    return {
        "n": float(len(p)),
        "bias": float(np.mean(resid)),
        "mae": float(np.mean(np.abs(resid))),
        "corr": corr,
    }


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:+.3f}"


def _fmt_mae(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.3f}"


def _plot_offset_mae(offset_df: pd.DataFrame, path: Path) -> None:
    if offset_df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, comp in zip(axes, COMPS):
        sub = offset_df.loc[offset_df["comp"] == comp].sort_values("offset")
        ax.plot(
            sub["offset"],
            sub["mae_long"],
            "s-",
            color="#c0392b",
            ms=3.5,
            lw=1.3,
            label="long Opta @t",
        )
        ax.plot(
            sub["offset"],
            sub["mae_l3"],
            "^-",
            color="#27ae60",
            ms=3.5,
            lw=1.3,
            label="L3 FD @g",
        )
        short = sub.loc[np.isfinite(sub["mae_short"])]
        if not short.empty:
            ax.plot(
                short["offset"],
                short["mae_short"],
                "o-",
                color="#2980b9",
                ms=3.5,
                lw=1.3,
                label="short FD @t (≤t+1)",
            )
        ax.set_ylabel(f"{comp} MAE")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(framealpha=0.95, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("horizon offset g − t")
    axes[0].set_title("Eligible m>60 MAE by forecast horizon (freeze at t)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _l3_board_at_g(
    *,
    cache: dict[int, dict[int, dict[str, float]]],
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    season: str,
    g: int,
) -> dict[int, dict[str, float]]:
    if int(g) in cache:
        return cache[int(g)]
    hist_g = numagent_xp.asof_share_history(merged, before_gw=int(g), season=season)
    if hist_g.empty:
        cache[int(g)] = {}
        return cache[int(g)]
    alpha_g, alpha_a = _asof_alphas(hist_g)
    board = _v2_component_board(
        hist=hist_g,
        enriched=enriched,
        fd=fd,
        prices=prices,
        season=season,
        decision_gw=int(g),
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        force_ids=None,
    )
    cache[int(g)] = _board_by_element(board)
    return cache[int(g)]


def run(
    *,
    season: str,
    step_gws: list[int],
    horizon_end: int,
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

    l3_cache: dict[int, dict[int, dict[str, float]]] = {}
    row_parts: list[dict[str, Any]] = []
    decision_meta: list[dict[str, Any]] = []
    t_all = _now()

    for t in step_gws:
        if int(t) > int(horizon_end):
            continue
        print(
            f"[horiz-snap] === decision GW{t} → snapshot {t}…{horizon_end} ===",
            flush=True,
        )
        hist_t = numagent_xp.asof_share_history(
            merged, before_gw=int(t), season=season
        )
        if hist_t.empty:
            print(f"[horiz-snap] GW{t} empty hist — skip", flush=True)
            continue
        alpha_g, alpha_a = _asof_alphas(hist_t)
        team_hist = hist_by_team_norm(hist_t)
        prior_t = _n_prior(hist_t)

        share_pool = _build_hist_share_pool(
            hist_t, prices, decision_gw=int(t), season=season
        )
        select_pool = share_pool.copy()
        fixtures_gw = _hist_fixtures_by_gw(
            enriched, start=int(t), end=int(horizon_end)
        )
        plan_gws = sorted(fixtures_gw)
        if not plan_gws:
            print(f"[horiz-snap] GW{t} no fixtures — skip", flush=True)
            continue

        step_seed = int(seed0) + 10_000 * int(t)
        # Long: Opta for entire snapshot
        print(
            f"[horiz-snap] GW{t} long Opta plan={plan_gws[0]}…{plan_gws[-1]} "
            f"n_plan={len(plan_gws)} share={len(share_pool)} N={n_strands}",
            flush=True,
        )
        long_boards, _, long_meta = mean_component_boards_split(
            select_pool=select_pool,
            share_pool=share_pool,
            fixtures_gw=fixtures_gw,
            priced_by_gw={},
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
        # Short: FD only for causal t:t+1
        short_gws = [g for g in plan_gws if g <= int(t) + 1]
        short_fx = {g: fixtures_gw[g] for g in short_gws}
        short_priced = _fd_priced_by_gw(enriched, fd, short_gws)
        print(
            f"[horiz-snap] GW{t} short FD plan={short_gws} "
            f"priced_gws={sorted(short_priced)}",
            flush=True,
        )
        short_boards, _, short_meta = mean_component_boards_split(
            select_pool=select_pool,
            share_pool=share_pool,
            fixtures_gw=short_fx,
            priced_by_gw=short_priced,
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            plan_gws=short_gws,
            n_strands=n_strands,
            seed0=step_seed + 1,
            goal_weight=goal_weight,
        )
        print(
            f"[horiz-snap] GW{t} boards long={long_meta['elapsed_s']:.1f}s "
            f"short={short_meta['elapsed_s']:.1f}s",
            flush=True,
        )

        n_rows_t = 0
        for g in plan_gws:
            offset = int(g) - int(t)
            long_m = _board_by_element(long_boards.get(int(g), pd.DataFrame()))
            short_m = (
                _board_by_element(short_boards.get(int(g), pd.DataFrame()))
                if int(g) in short_boards
                else {}
            )
            l3_m = _l3_board_at_g(
                cache=l3_cache,
                merged=merged,
                enriched=enriched,
                fd=fd,
                prices=prices,
                season=season,
                g=int(g),
            )
            act = actual_components(merged, int(g))
            if act.empty or not long_m or not l3_m:
                continue
            for r in act.itertuples(index=False):
                mins = float(r.minutes)
                if mins <= MIN_MINUTES:
                    continue
                eid = int(r.element)
                if float(prior_t.get(eid, 0) or 0) < MIN_HISTORY:
                    continue
                if eid not in long_m or eid not in l3_m:
                    continue
                lo = long_m[eid]
                l3 = l3_m[eid]
                if not all(
                    np.isfinite(v)
                    for v in (
                        lo["xp_goals"],
                        lo["xp_cs"],
                        lo["xp_gc"],
                        l3["xp_goals"],
                        l3["xp_cs"],
                        l3["xp_gc"],
                    )
                ):
                    continue
                row: dict[str, Any] = {
                    "season": season,
                    "decision_gw": int(t),
                    "target_gw": int(g),
                    "offset": offset,
                    "element": eid,
                    "name": str(r.name),
                    "position": str(r.position),
                    "minutes": mins,
                    "n_prior_at_t": float(prior_t.get(eid, 0) or 0),
                    "long_G": lo["xp_goals"],
                    "long_CS": lo["xp_cs"],
                    "long_GC": lo["xp_gc"],
                    "l3_G": l3["xp_goals"],
                    "l3_CS": l3["xp_cs"],
                    "l3_GC": l3["xp_gc"],
                    "P_G": float(r.P_G),
                    "P_CS": float(r.P_CS),
                    "P_GC": float(r.P_GC),
                    "short_G": float("nan"),
                    "short_CS": float("nan"),
                    "short_GC": float("nan"),
                    "has_short": False,
                }
                if eid in short_m:
                    sh = short_m[eid]
                    if all(
                        np.isfinite(v)
                        for v in (sh["xp_goals"], sh["xp_cs"], sh["xp_gc"])
                    ):
                        row["short_G"] = sh["xp_goals"]
                        row["short_CS"] = sh["xp_cs"]
                        row["short_GC"] = sh["xp_gc"]
                        row["has_short"] = True
                row_parts.append(row)
                n_rows_t += 1

        decision_meta.append(
            {
                "decision_gw": int(t),
                "n_plan": len(plan_gws),
                "n_rows": n_rows_t,
                "long_board_s": float(long_meta["elapsed_s"]),
                "short_board_s": float(short_meta["elapsed_s"]),
                "plan_lo": int(plan_gws[0]),
                "plan_hi": int(plan_gws[-1]),
            }
        )
        print(f"[horiz-snap] GW{t} residual rows={n_rows_t}", flush=True)

    if not row_parts:
        raise RuntimeError("no horizon residual rows")
    players = pd.DataFrame(row_parts)

    # By horizon offset: long vs L3 (all offsets); short where present
    offset_rows: list[dict[str, Any]] = []
    for offset, sub in players.groupby("offset"):
        for comp in COMPS:
            st_long = _resid_stats(
                sub[f"long_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            st_l3 = _resid_stats(
                sub[f"l3_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            sh = sub.loc[sub["has_short"]]
            if len(sh):
                st_short = _resid_stats(
                    sh[f"short_{comp}"].to_numpy(float),
                    sh[f"P_{comp}"].to_numpy(float),
                )
                d_mae_short = float(st_short["mae"] - st_l3["mae"]) if len(sh) else float("nan")
                # L3 on same short rows
                st_l3_short = _resid_stats(
                    sh[f"l3_{comp}"].to_numpy(float),
                    sh[f"P_{comp}"].to_numpy(float),
                )
                d_mae_short = float(st_short["mae"] - st_l3_short["mae"])
                mae_short = st_short["mae"]
                bias_short = st_short["bias"]
            else:
                mae_short = float("nan")
                bias_short = float("nan")
                d_mae_short = float("nan")
            offset_rows.append(
                {
                    "offset": int(offset),
                    "comp": comp,
                    "n": int(st_long["n"]),
                    "mae_long": st_long["mae"],
                    "bias_long": st_long["bias"],
                    "corr_long": st_long["corr"],
                    "mae_l3": st_l3["mae"],
                    "bias_l3": st_l3["bias"],
                    "corr_l3": st_l3["corr"],
                    "delta_mae_long_minus_l3": float(st_long["mae"] - st_l3["mae"]),
                    "mae_short": mae_short,
                    "bias_short": bias_short,
                    "delta_mae_short_minus_l3": d_mae_short,
                }
            )
    offset_df = pd.DataFrame(offset_rows)

    # Overall long vs L3 on all rows; short on has_short
    overall: dict[str, Any] = {"long": {}, "l3": {}, "short": {}}
    for comp in COMPS:
        overall["long"][comp] = _resid_stats(
            players[f"long_{comp}"].to_numpy(float),
            players[f"P_{comp}"].to_numpy(float),
        )
        overall["l3"][comp] = _resid_stats(
            players[f"l3_{comp}"].to_numpy(float),
            players[f"P_{comp}"].to_numpy(float),
        )
        sh = players.loc[players["has_short"]]
        overall["short"][comp] = _resid_stats(
            sh[f"short_{comp}"].to_numpy(float),
            sh[f"P_{comp}"].to_numpy(float),
        )
        overall["l3_on_short_rows"] = overall.get("l3_on_short_rows", {})
        overall["l3_on_short_rows"][comp] = _resid_stats(
            sh[f"l3_{comp}"].to_numpy(float),
            sh[f"P_{comp}"].to_numpy(float),
        )

    # Near (offset≤1) vs far (offset≥2) long vs L3
    bands = {
        "near_0_1": players.loc[players["offset"] <= 1],
        "far_2plus": players.loc[players["offset"] >= 2],
    }
    band_stats: dict[str, Any] = {}
    for name, sub in bands.items():
        if sub.empty:
            continue
        band_stats[name] = {"n": int(len(sub))}
        for comp in COMPS:
            sl = _resid_stats(
                sub[f"long_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            s3 = _resid_stats(
                sub[f"l3_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            band_stats[name][comp] = {
                "long": sl,
                "l3": s3,
                "delta_mae": float(sl["mae"] - s3["mae"]),
            }

    return {
        "season": season,
        "goal_weight": goal_weight,
        "n_strands": n_strands,
        "seed": seed0,
        "horizon_end": int(horizon_end),
        "n_player_gw": int(len(players)),
        "n_decision": int(players["decision_gw"].nunique()),
        "n_short_rows": int(players["has_short"].sum()),
        "elapsed_s": float(_now() - t_all),
        "overall": overall,
        "bands": band_stats,
        "decisions": decision_meta,
        "players": players,
        "offset": offset_df,
    }


def _write_report(payload: dict[str, Any], plot_path: Path) -> str:
    ov = payload["overall"]
    bands = payload["bands"]
    off = payload["offset"]
    lines = [
        "# Eligible-pool horizon snapshots — cascade @t vs L3 @g vs actuals",
        "",
        f"Season **{payload['season']}** · decisions "
        f"{payload['n_decision']} · horizon_end={payload['horizon_end']} · "
        f"N={payload['n_strands']} · goal_weight=`{payload['goal_weight']}` · "
        f"n={payload['n_player_gw']} player-GW cells · "
        f"short cells={payload['n_short_rows']} · wall {payload['elapsed_s']:.1f}s",
        "",
        "Pre-registered: `protocol.cascade_eligible_horizon_snapshot`.",
        "",
        "## Intent (correct design)",
        "",
        "At decision GW **t**, freeze as-of state and forecast **every target** "
        "`g ∈ [t, horizon_end]` with cascade mechanics. Compare each cell to:",
        "",
        "- **L3 @g** — contemporaneous `score_xp_v2` using FD priced near `g`",
        "  (should be the sharper near-term predictor), and",
        "- **actuals** at `g` (residual only if `minutes>60`).",
        "",
        "Then step `t → t+1` and repeat. This supersedes the decision-GW-only "
        "`cascade_eligible_component_resid` dig.",
        "",
        "## Arms",
        "",
        "- **cascade_long @t** — Opta blind place λ → generative MC-mean for all `g`.",
        "- **cascade_short @t** — FD λ only for `g ∈ {t, t+1}` (no future-FD leak).",
        "- **L3 @g** — as-of before `g` + that GW's FD centreline components.",
        "",
        f"![offset mae]({plot_path})",
        "",
        "## Overall (all freeze×target cells)",
        "",
        "| arm | G bias | G MAE | CS bias | CS MAE | GC bias | GC MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in (
        ("long", "long Opta @t"),
        ("l3", "L3 FD @g"),
    ):
        m = ov[arm]
        lines.append(
            f"| {label} | {_fmt(m['G']['bias'])} | {_fmt_mae(m['G']['mae'])} | "
            f"{_fmt(m['CS']['bias'])} | {_fmt_mae(m['CS']['mae'])} | "
            f"{_fmt(m['GC']['bias'])} | {_fmt_mae(m['GC']['mae'])} |"
        )
    sh = ov["short"]
    l3s = ov["l3_on_short_rows"]
    lines += [
        "",
        f"Short cells only (offset ≤1, n≈{int(sh['G']['n'])}):",
        "",
        "| arm | G MAE | CS MAE | GC MAE |",
        "| --- | ---: | ---: | ---: |",
        f"| short FD @t | {_fmt_mae(sh['G']['mae'])} | {_fmt_mae(sh['CS']['mae'])} | {_fmt_mae(sh['GC']['mae'])} |",
        f"| L3 FD @g (same rows) | {_fmt_mae(l3s['G']['mae'])} | {_fmt_mae(l3s['CS']['mae'])} | {_fmt_mae(l3s['GC']['mae'])} |",
        "",
        "## Near vs far (long Opta @t vs L3 @g)",
        "",
        "| band | n | G ΔMAE (long−L3) | CS ΔMAE | GC ΔMAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for band in ("near_0_1", "far_2plus"):
        if band not in bands:
            continue
        b = bands[band]
        lines.append(
            f"| {band} | {b['n']} | {_fmt(b['G']['delta_mae'])} | "
            f"{_fmt(b['CS']['delta_mae'])} | {_fmt(b['GC']['delta_mae'])} |"
        )
    # Offset table (G only, every offset) — compact every 2nd if many
    g_off = off.loc[off["comp"] == "G"].sort_values("offset")
    lines += [
        "",
        "## G MAE by horizon offset",
        "",
        "| offset | n | long MAE | L3 MAE | Δ (long−L3) | short MAE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in g_off.itertuples(index=False):
        lines.append(
            f"| {int(r.offset)} | {int(r.n)} | {_fmt_mae(r.mae_long)} | "
            f"{_fmt_mae(r.mae_l3)} | {_fmt(r.delta_mae_long_minus_l3)} | "
            f"{_fmt_mae(r.mae_short)} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if "far_2plus" in bands and "near_0_1" in bands:
        lines.append(
            f"- Far (offset≥2) long−L3 G ΔMAE "
            f"**{_fmt(bands['far_2plus']['G']['delta_mae'])}**; "
            f"near (0–1) **{_fmt(bands['near_0_1']['G']['delta_mae'])}**."
        )
    lines += [
        f"- Overall long G MAE {_fmt_mae(ov['long']['G']['mae'])} vs L3 "
        f"{_fmt_mae(ov['l3']['G']['mae'])}.",
        f"- Short vs L3 on causal t:t+1 cells: G MAE "
        f"{_fmt_mae(sh['G']['mae'])} vs {_fmt_mae(l3s['G']['mae'])}.",
        "- Defaults unchanged (diagnostic).",
        "",
        "## Reproduce",
        "",
        "```",
        f"PYTHONPATH=. python3 -m src.eval.cascade_eligible_horizon_snapshot "
        f"--n {payload['n_strands']} --horizon-end {payload['horizon_end']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma decision GWs (default: 8…horizon-end).",
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
        default=Path("/opt/cursor/artifacts/cascade_eligible_horizon_snapshot.json"),
    )
    args = ap.parse_args(argv)
    horizon_end = int(args.horizon_end)
    if str(args.steps).strip():
        step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    else:
        step_gws = list(range(GW_START, horizon_end + 1))

    print(
        f"[horiz-snap] season={args.season} steps={step_gws[0]}…{step_gws[-1]} "
        f"horizon_end={horizon_end} N={args.n} goal_weight={args.goal_weight}",
        flush=True,
    )
    payload = run(
        season=str(args.season),
        step_gws=step_gws,
        horizon_end=horizon_end,
        n_strands=int(args.n),
        seed0=int(args.seed),
        goal_weight=str(args.goal_weight),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    players: pd.DataFrame = payload.pop("players")
    offset: pd.DataFrame = payload.pop("offset")
    plot_path = out.with_name(out.stem + "_offset_mae.png")
    _plot_offset_mae(offset, plot_path)

    players_path = out.with_name(out.stem + "_players.csv")
    offset_path = out.with_name(out.stem + "_offset.csv")
    players.to_csv(players_path, index=False)
    offset.to_csv(offset_path, index=False)

    slim = dict(payload)
    slim["players_csv"] = str(players_path)
    slim["offset_csv"] = str(offset_path)
    slim["plot"] = str(plot_path)
    out.write_text(json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8")

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    # Players can be large — keep offset + summary in repo; players under artifacts.
    offset.to_csv(proc / "cascade_eligible_horizon_snapshot_offset.csv", index=False)
    (proc / "cascade_eligible_horizon_snapshot.json").write_text(
        json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8"
    )

    report = _write_report({**payload, "players": players, "offset": offset}, plot_path)
    REPORT.write_text(report + "\n", encoding="utf-8")

    state = {
        "task": PROTOCOL_KEY,
        "status": "done",
        "change": "Horizon snapshots: cascade @t vs L3 @g vs actuals (eligible m>60)",
        "supersedes": "cascade_eligible_component_resid",
        "season": payload["season"],
        "n_player_gw": payload["n_player_gw"],
        "n_decision": payload["n_decision"],
        "bands": {
            k: {
                "n": v["n"],
                "G_delta_mae": v["G"]["delta_mae"],
                "CS_delta_mae": v["CS"]["delta_mae"],
            }
            for k, v in payload["bands"].items()
        },
        "overall_G_mae": {
            "long": payload["overall"]["long"]["G"]["mae"],
            "l3": payload["overall"]["l3"]["G"]["mae"],
            "short": payload["overall"]["short"]["G"]["mae"],
        },
        "report": str(REPORT),
        "out": str(out),
        "elapsed_s": payload["elapsed_s"],
    }
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)
    print(f"[horiz-snap] report {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
