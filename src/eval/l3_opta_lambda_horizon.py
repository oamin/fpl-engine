"""L3 centreline at freeze t, driven by Opta λs over g=t…end.

Pre-registered: ``protocol.l3_opta_lambda_horizon``.

Alternative to cascade_short / cascade_long **generative** scores for
transfer / strategy feasibility.

At each decision GW ``t``:

1. Freeze L3 state: ``build_frozen_side_table`` + alphas as-of before ``t``.
2. For every target ``g ∈ [t, horizon_end]``, build Opta **centreline** λ
   (place rankmean GF/GA → Dixon–Coles; ``venue_mode=blind``) for each
   fixture, convert λ → moments, score with ``score_team_side`` (same
   mechanics as L3/L4 mean boards).
3. Compare to contemporaneous **L3 @g** (as-of before ``g`` + FD λ near
   ``g``) and to **actuals** (residual only if ``minutes>60``).
4. Step ``t → t+1`` and repeat.

No generative MC. Defaults unchanged.

  PYTHONPATH=. python3 -m src.eval.l3_opta_lambda_horizon
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
)
from src.eval.cascade_mean_xi_ft_component_resid import actual_components
from src.eval.cascade_vs_l3_component_xp import _v2_component_board
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    match_prediction,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    fixtures_for_gw,
    load_prices,
    moments_from_lams,
)
from src.rules.fpl_2026 import normalize_position
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "l3_opta_lambda_horizon.md"
STATE = ROOT / "REPORT_STATE.json"
PROTOCOL_KEY = "l3_opta_lambda_horizon"

GW_START = 8
HORIZON_END = 38
MIN_MINUTES = 60.0
MIN_HISTORY = 3
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


def _opta_fits_for_gw(
    enriched: pd.DataFrame,
    *,
    gw: int,
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
) -> list[dict[str, Any]]:
    """Fixture list with Opta centreline λ → Dixon–Coles moments (L3 inputs)."""
    out: list[dict[str, Any]] = []
    for fx in fixtures_for_gw(enriched, int(gw)):
        h = norm_team(str(fx["home_n"]))
        a = norm_team(str(fx["away_n"]))
        if h not in ranks or a not in ranks:
            continue
        pred = match_prediction(
            h,
            a,
            ranks=ranks,
            profile=profile,
            league_gf=float(league_gf),
            sd_league_gf=0.0,
            venue_mode="blind",
        )
        lam_h = float(pred["lam_home"])
        lam_a = float(pred["lam_away"])
        mom = moments_from_lams(lam_h, lam_a, 0.0)
        out.append(
            {
                **dict(fx),
                "lam_h": lam_h,
                "lam_a": lam_a,
                "rho": 0.0,
                "xg_home": float(mom["xg_home"]),
                "xcs_home": float(mom["xcs_home"]),
                "xgc_home": float(mom["xgc_home"]),
                "xg_away": float(mom["xg_away"]),
                "xcs_away": float(mom["xcs_away"]),
                "xgc_away": float(mom["xgc_away"]),
                "source": "opta_centreline_blind",
            }
        )
    return out


def _l3_component_board_from_fits(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    season: str,
    prices: pd.DataFrame,
    decision_gw: int,
) -> pd.DataFrame:
    """score_team_side on each fixture side; keep G/CS/GC (DGW-safe sum)."""
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
            )
            scored = scored.copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    all_p = pd.concat(parts, ignore_index=True)
    board = (
        all_p.groupby("element", as_index=False)
        .agg(
            name=("name", "first"),
            position=("position", "first"),
            team=("team", "first"),
            xp_goals=("xp_goals", "sum"),
            xp_cs=("xp_cs", "sum"),
            xp_gc=("xp_gc", "sum"),
            xp=("xp", "sum"),
        )
    )
    board["score_xp_v2"] = board["xp"]
    board["element"] = pd.to_numeric(board["element"], errors="coerce")
    board["position"] = board["position"].map(
        lambda p: normalize_position(str(p)) if pd.notna(p) else p
    )
    pr = prices.loc[prices["gw"] == int(decision_gw)].drop_duplicates("element")
    if not pr.empty and "element" in pr.columns:
        board = board.merge(
            pr[["element", "value", "player_id", "team_norm"]],
            on="element",
            how="left",
        )
    if "player_id" not in board.columns:
        board["player_id"] = board["element"].map(
            lambda e: f"{season}:{int(e)}" if pd.notna(e) else ""
        )
    else:
        board["player_id"] = board["player_id"].fillna(
            board["element"].map(
                lambda e: f"{season}:{int(e)}" if pd.notna(e) else ""
            )
        ).astype(str)
    return board


def _l3_fd_board_at_g(
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


def _plot_offset_mae(offset_df: pd.DataFrame, path: Path) -> None:
    if offset_df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, comp in zip(axes, COMPS):
        sub = offset_df.loc[offset_df["comp"] == comp].sort_values("offset")
        ax.plot(
            sub["offset"],
            sub["mae_opta"],
            "s-",
            color="#8e44ad",
            ms=3.5,
            lw=1.3,
            label="L3@t + Opta λ",
        )
        ax.plot(
            sub["offset"],
            sub["mae_l3"],
            "^-",
            color="#27ae60",
            ms=3.5,
            lw=1.3,
            label="L3@g + FD λ",
        )
        ax.set_ylabel(f"{comp} MAE")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(framealpha=0.95, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("horizon offset g − t")
    axes[0].set_title("Eligible m>60 MAE: L3 freeze+Opta λ vs contemporaneous L3+FD")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(
    *,
    season: str,
    step_gws: list[int],
    horizon_end: int,
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
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    t_all = _now()

    for t in step_gws:
        if int(t) > int(horizon_end):
            continue
        print(
            f"[l3-opta] === decision GW{t} → L3@t + Opta λ for {t}…{horizon_end} ===",
            flush=True,
        )
        hist_t = numagent_xp.asof_share_history(
            merged, before_gw=int(t), season=season
        )
        if hist_t.empty:
            continue
        alpha_g, alpha_a = _asof_alphas(hist_t)
        prior_t = _n_prior(hist_t)
        side, _, _ = build_frozen_side_table(
            hist_t,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            season=season,
            historical_rate_baseline=True,
        )
        if side.empty:
            print(f"[l3-opta] GW{t} empty side — skip", flush=True)
            continue

        n_rows_t = 0
        n_fit_gws = 0
        for g in range(int(t), int(horizon_end) + 1):
            fits = _opta_fits_for_gw(
                enriched,
                gw=int(g),
                ranks=ranks,
                profile=profile,
                league_gf=league_gf,
            )
            if len(fits) < 1:
                continue
            n_fit_gws += 1
            opta_board = _l3_component_board_from_fits(
                side,
                fits,
                alpha_g=alpha_g,
                alpha_a=alpha_a,
                season=season,
                prices=prices,
                decision_gw=int(t),
            )
            opta_m = _board_by_element(opta_board)
            l3_m = _l3_fd_board_at_g(
                cache=l3_cache,
                merged=merged,
                enriched=enriched,
                fd=fd,
                prices=prices,
                season=season,
                g=int(g),
            )
            act = actual_components(merged, int(g))
            if act.empty or not opta_m or not l3_m:
                continue
            offset = int(g) - int(t)
            for r in act.itertuples(index=False):
                mins = float(r.minutes)
                if mins <= MIN_MINUTES:
                    continue
                eid = int(r.element)
                if float(prior_t.get(eid, 0) or 0) < MIN_HISTORY:
                    continue
                if eid not in opta_m or eid not in l3_m:
                    continue
                o, l3 = opta_m[eid], l3_m[eid]
                if not all(
                    np.isfinite(v)
                    for v in (
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
                        "decision_gw": int(t),
                        "target_gw": int(g),
                        "offset": offset,
                        "element": eid,
                        "name": str(r.name),
                        "position": str(r.position),
                        "minutes": mins,
                        "opta_G": o["xp_goals"],
                        "opta_CS": o["xp_cs"],
                        "opta_GC": o["xp_gc"],
                        "l3_G": l3["xp_goals"],
                        "l3_CS": l3["xp_cs"],
                        "l3_GC": l3["xp_gc"],
                        "P_G": float(r.P_G),
                        "P_CS": float(r.P_CS),
                        "P_GC": float(r.P_GC),
                    }
                )
                n_rows_t += 1

        decisions.append(
            {
                "decision_gw": int(t),
                "n_target_gws": n_fit_gws,
                "n_rows": n_rows_t,
                "side_n": int(len(side)),
            }
        )
        print(
            f"[l3-opta] GW{t} side={len(side)} targets={n_fit_gws} rows={n_rows_t}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("no L3+Opta residual rows")
    players = pd.DataFrame(rows)

    offset_rows: list[dict[str, Any]] = []
    for offset, sub in players.groupby("offset"):
        for comp in COMPS:
            st_o = _resid_stats(
                sub[f"opta_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            st_l = _resid_stats(
                sub[f"l3_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            offset_rows.append(
                {
                    "offset": int(offset),
                    "comp": comp,
                    "n": int(st_o["n"]),
                    "mae_opta": st_o["mae"],
                    "bias_opta": st_o["bias"],
                    "corr_opta": st_o["corr"],
                    "mae_l3": st_l["mae"],
                    "bias_l3": st_l["bias"],
                    "corr_l3": st_l["corr"],
                    "delta_mae_opta_minus_l3": float(st_o["mae"] - st_l["mae"]),
                }
            )
    offset_df = pd.DataFrame(offset_rows)

    overall: dict[str, Any] = {"opta": {}, "l3": {}}
    for comp in COMPS:
        overall["opta"][comp] = _resid_stats(
            players[f"opta_{comp}"].to_numpy(float),
            players[f"P_{comp}"].to_numpy(float),
        )
        overall["l3"][comp] = _resid_stats(
            players[f"l3_{comp}"].to_numpy(float),
            players[f"P_{comp}"].to_numpy(float),
        )

    bands: dict[str, Any] = {}
    for name, mask in (
        ("near_0_1", players["offset"] <= 1),
        ("far_2plus", players["offset"] >= 2),
    ):
        sub = players.loc[mask]
        if sub.empty:
            continue
        bands[name] = {"n": int(len(sub))}
        for comp in COMPS:
            so = _resid_stats(
                sub[f"opta_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            sl = _resid_stats(
                sub[f"l3_{comp}"].to_numpy(float),
                sub[f"P_{comp}"].to_numpy(float),
            )
            bands[name][comp] = {
                "opta": so,
                "l3": sl,
                "delta_mae": float(so["mae"] - sl["mae"]),
            }

    return {
        "season": season,
        "horizon_end": int(horizon_end),
        "n_player_gw": int(len(players)),
        "n_decision": int(players["decision_gw"].nunique()),
        "elapsed_s": float(_now() - t_all),
        "overall": overall,
        "bands": bands,
        "decisions": decisions,
        "players": players,
        "offset": offset_df,
        "seed": SEED,
    }


def _write_report(payload: dict[str, Any], plot_path: Path) -> str:
    ov = payload["overall"]
    bands = payload["bands"]
    off = payload["offset"]
    lines = [
        "# L3@t + Opta λ horizon — vs contemporaneous L3@g + actuals",
        "",
        f"Season **{payload['season']}** · decisions {payload['n_decision']} · "
        f"horizon_end={payload['horizon_end']} · "
        f"n={payload['n_player_gw']} player-GW cells · "
        f"wall {payload['elapsed_s']:.1f}s",
        "",
        "Pre-registered: `protocol.l3_opta_lambda_horizon`.",
        "",
        "## Intent",
        "",
        "Alternative to cascade generative short/long for strategy scoring:",
        "keep **L3/L4 centreline mechanics** (`score_team_side`), but drive "
        "far (and near) fixtures with **Opta centreline λ** from a freeze at "
        "decision GW `t`, forecasting every `g ∈ [t, end]`.",
        "",
        "## Arms",
        "",
        "- **L3@t + Opta λ** — rates/shares/alphas frozen at `t`; Opta blind "
        "place λ → DC moments → `score_team_side`.",
        "- **L3@g + FD λ** — contemporaneous centreline (FD priced near `g`).",
        "- **Actuals** — `P_G` / `P_CS` / `P_GC` on `minutes>60`.",
        "",
        f"![offset mae]({plot_path})",
        "",
        "## Overall",
        "",
        "| arm | G bias | G MAE | CS bias | CS MAE | GC bias | GC MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in (
        ("opta", "L3@t + Opta λ"),
        ("l3", "L3@g + FD λ"),
    ):
        m = ov[arm]
        lines.append(
            f"| {label} | {_fmt(m['G']['bias'])} | {_fmt_mae(m['G']['mae'])} | "
            f"{_fmt(m['CS']['bias'])} | {_fmt_mae(m['CS']['mae'])} | "
            f"{_fmt(m['GC']['bias'])} | {_fmt_mae(m['GC']['mae'])} |"
        )
    lines += [
        "",
        "## Near vs far",
        "",
        "| band | n | G ΔMAE (Opta−L3) | CS ΔMAE | GC ΔMAE |",
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
    g_off = off.loc[off["comp"] == "G"].sort_values("offset")
    lines += [
        "",
        "## G MAE by horizon offset",
        "",
        "| offset | n | Opta MAE | L3 MAE | Δ (Opta−L3) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in g_off.itertuples(index=False):
        lines.append(
            f"| {int(r.offset)} | {int(r.n)} | {_fmt_mae(r.mae_opta)} | "
            f"{_fmt_mae(r.mae_l3)} | {_fmt(r.delta_mae_opta_minus_l3)} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if "far_2plus" in bands and "near_0_1" in bands:
        lines.append(
            f"- Near G ΔMAE **{_fmt(bands['near_0_1']['G']['delta_mae'])}**; "
            f"far **{_fmt(bands['far_2plus']['G']['delta_mae'])}** "
            "(positive ⇒ L3@g closer to actuals)."
        )
    lines += [
        f"- Overall G MAE: L3@t+Opta **{_fmt_mae(ov['opta']['G']['mae'])}** vs "
        f"L3@g+FD **{_fmt_mae(ov['l3']['G']['mae'])}**.",
        "- Compare also to prior generative cascade_long @t "
        "(G MAE ≈0.846 overall) — centreline+Opta should isolate whether "
        "dice were the problem vs Opta λ itself.",
        "- Defaults unchanged.",
        "",
        "## Reproduce",
        "",
        "```",
        "PYTHONPATH=. python3 -m src.eval.l3_opta_lambda_horizon "
        f"--horizon-end {payload['horizon_end']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--steps", type=str, default="")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/l3_opta_lambda_horizon.json"),
    )
    args = ap.parse_args(argv)
    horizon_end = int(args.horizon_end)
    if str(args.steps).strip():
        step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    else:
        step_gws = list(range(GW_START, horizon_end + 1))

    print(
        f"[l3-opta] season={args.season} steps={step_gws[0]}…{step_gws[-1]} "
        f"horizon_end={horizon_end}",
        flush=True,
    )
    payload = run(
        season=str(args.season),
        step_gws=step_gws,
        horizon_end=horizon_end,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    players: pd.DataFrame = payload.pop("players")
    offset: pd.DataFrame = payload.pop("offset")
    plot_path = out.with_name(out.stem + "_offset_mae.png")
    _plot_offset_mae(offset, plot_path)

    players.to_csv(out.with_name(out.stem + "_players.csv"), index=False)
    offset.to_csv(out.with_name(out.stem + "_offset.csv"), index=False)

    slim = dict(payload)
    slim["players_csv"] = str(out.with_name(out.stem + "_players.csv"))
    slim["offset_csv"] = str(out.with_name(out.stem + "_offset.csv"))
    slim["plot"] = str(plot_path)
    out.write_text(json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8")

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    offset.to_csv(proc / "l3_opta_lambda_horizon_offset.csv", index=False)
    (proc / "l3_opta_lambda_horizon.json").write_text(
        json.dumps(slim, indent=2, default=str) + "\n", encoding="utf-8"
    )

    report = _write_report({**payload, "players": players, "offset": offset}, plot_path)
    REPORT.write_text(report + "\n", encoding="utf-8")

    state = {
        "task": PROTOCOL_KEY,
        "status": "done",
        "change": "L3@t + Opta λ horizon vs L3@g + actuals (eligible m>60)",
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
            "l3_opta_at_t": payload["overall"]["opta"]["G"]["mae"],
            "l3_fd_at_g": payload["overall"]["l3"]["G"]["mae"],
        },
        "report": str(REPORT),
        "out": str(out),
        "elapsed_s": payload["elapsed_s"],
    }
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)
    print(f"[l3-opta] report {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
