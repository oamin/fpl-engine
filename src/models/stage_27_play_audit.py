"""Stage 27 — Game-engine play audit (locked xP v2).

Instrument one FT season under pure xP (no level-add, no horizon fade):
  - per-GW decision log (transfers, hits, hold, C/VC, bank, autosubs, blanks)
  - leak attribution: hit cost, captain regret, myopic hold regret
  - free-rebuild upper bound (budgeted pick each GW, no continuity)
  - policy sensitivities: HOLD_EPS × {0.5, 1.25, 2.5}, H × {3, 5}

Writes:
  data/processed/stage_27_decision_log.csv
  data/processed/stage_27_leaks.csv
  data/processed/stage_27_sensitivity.csv
  data/plots/stage_27_play_audit.png
  reports/stage_27_play_audit.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.ridge_multiseason import EVAL_SEASON, SEASONS, build_one_season
from src.models.season_climb import bank_squad_gw
from src.models.season_climb_budget import pick_squad, run_budgeted_season
from src.models.season_climb_ft import (
    HIT_COST,
    HOLD_EPS,
    HORIZON,
    SquadState,
    _fill_score,
    _gw_pool,
    advance_ft,
    choose_transfers,
    initial_squad,
    load_vaastav_roster,
    run_ft_season,
    sell_price,
)
from src.models.season_climb_budget import BUDGET

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

SCORE_COL = "score_xp"


def _fd_code(season: str) -> str:
    for s, code in SEASONS:
        if s == season:
            return code
    raise KeyError(season)


def _as_int_value(v: Any) -> int:
    x = pd.to_numeric(v, errors="coerce")
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return 50
    return int(x)


def run_instrumented_ft(
    feat: pd.DataFrame,
    gws: list[int],
    roster: pd.DataFrame,
    *,
    horizon: int = 5,
    hold_eps: float = HOLD_EPS,
    label: str = "xp_ft",
) -> pd.DataFrame:
    """FT climb with rich per-GW fields for audit."""
    roster_by_gw: dict[int, set[str]] = {
        int(g): set(gdf["player_id"].astype(str))
        for g, gdf in roster.groupby("gw")
    }
    rows: list[dict[str, Any]] = []
    state: SquadState | None = None
    col = SCORE_COL

    for i, gw in enumerate(gws):
        owned = state.ids() if state else set()
        pool = _gw_pool(feat, roster, gw, owned)
        if pool["position"].nunique() < 4:
            continue

        hold_pts = float("nan")
        hold_cap = float("nan")
        if state is None:
            try:
                state = initial_squad(pool, col)
            except RuntimeError:
                break
            n_tx, hits = 0, 0
            ft_before = 0
            pre_ids = state.ids()
        else:
            ft_before = state.ft
            pre_ids = state.ids()
            # Counterfactual: bank hold squad before transfers
            hold_df = pool.loc[pool["player_id"].isin(pre_ids)].copy()
            hold_df = hold_df.drop_duplicates("player_id", keep="first")
            if len(hold_df) >= 11:
                hold_df[col] = _fill_score(hold_df, col)
                try:
                    hb = bank_squad_gw(hold_df, col, use_autosubs=True)
                    hold_pts = float(hb["xi_points"] + hb["cap_extra"])
                    hold_cap = float(hb["cap_extra"])
                except RuntimeError:
                    pass
            new_state, n_tx, hits = choose_transfers(
                state,
                pool,
                col,
                gw=int(gw),
                future_gws=list(gws),
                roster_by_gw=roster_by_gw,
                horizon=horizon,
                hold_eps=hold_eps,
            )
            state = new_state

        squad_df = pool.loc[pool["player_id"].isin(state.ids())].copy()
        squad_df = squad_df.drop_duplicates("player_id", keep="first")
        if len(squad_df) < 11 or len(state.purchase) != 15:
            break
        squad_df[col] = _fill_score(squad_df, col)
        try:
            banked = bank_squad_gw(squad_df, col, use_autosubs=True)
        except RuntimeError:
            break

        form = banked["form"]
        pts_sum = float(banked["xi_points"])
        cap_extra = float(banked["cap_extra"])
        hit_pts = HIT_COST * hits
        gw_pts = pts_sum + cap_extra - hit_pts

        # Captain regret: best actual points in final XI vs awarded cap_extra
        xi = banked["xi"]
        xi_pts = pd.to_numeric(xi["total_points"], errors="coerce").fillna(0.0)
        best_in_xi = float(xi_pts.max()) if len(xi_pts) else 0.0
        cap_regret = max(0.0, best_in_xi - cap_extra)

        # Bench points left unused (simple: squad pts − XI pts, not formation-aware)
        squad_pts_all = float(
            pd.to_numeric(squad_df["total_points"], errors="coerce").fillna(0.0).sum()
        )
        bench_pts = max(0.0, squad_pts_all - pts_sum)

        squad_val = float(
            sum(
                sell_price(state.purchase[str(r.player_id)], _as_int_value(r.value))
                for r in squad_df.itertuples()
            )
        )
        bank_tenths = int(state.bank)
        names = {
            str(r.player_id): str(getattr(r, "player_name", r.player_id))
            for r in squad_df.itertuples()
        }
        sold = sorted(pre_ids - state.ids()) if i > 0 else []
        bought = sorted(state.ids() - pre_ids) if i > 0 else []

        rows.append(
            {
                "gw": int(gw),
                "method": label,
                "xi_points": pts_sum,
                "cap_extra": cap_extra,
                "hit_cost": hit_pts,
                "gw_points": gw_pts,
                "n_transfers": n_tx,
                "hits": hits,
                "held": int(n_tx == 0 and i > 0),
                "ft_before": ft_before,
                "ft_after": 1 if i == 0 else advance_ft(ft_before, n_tx),
                "bank": bank_tenths,
                "squad_sell_value": squad_val,
                "itb": bank_tenths / 10.0,
                "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                "n_autosubs": int(banked["n_autosubs"]),
                "n_blank_intended": int(banked["n_blank_intended"]),
                "n_blank_final": int(banked["n_blank_final"]),
                "captain_id": banked["captain_id"],
                "vice_id": banked["vice_id"],
                "captain_name": names.get(banked["captain_id"], ""),
                "best_in_xi_pts": best_in_xi,
                "cap_regret": cap_regret,
                "bench_pts": bench_pts,
                "hold_gw_points": hold_pts,
                "hold_cap_extra": hold_cap,
                "myopic_hold_delta": (hold_pts - (pts_sum + cap_extra))
                if np.isfinite(hold_pts)
                else float("nan"),
                "sold": ",".join(sold),
                "bought": ",".join(bought),
                "n_eligible": int(pool["eligible"].sum()),
            }
        )
        if i == 0:
            state.ft = 1
        else:
            state.ft = advance_ft(ft_before, n_tx)

    return pd.DataFrame(rows)


def summarise_leaks(log: pd.DataFrame) -> dict[str, float]:
    total = float(log["gw_points"].sum())
    hit_cost = float(log["hit_cost"].sum())
    cap_regret = float(log["cap_regret"].sum())
    # Myopic: when we transferred and hold would have scored more this GW
    tx = log.loc[log["n_transfers"] > 0]
    hold_regret = float(
        tx.loc[tx["myopic_hold_delta"] > 0, "myopic_hold_delta"].sum()
    ) if len(tx) else 0.0
    # When we held: leave as 0 for myopic (need swap counterfactual — skip)
    blanks = float(log["n_blank_final"].sum())
    autosubs = float(log["n_autosubs"].sum())
    n_hold = int(log["held"].sum())
    n_tx_gw = int((log["n_transfers"] > 0).sum())
    return {
        "total_points": total,
        "hit_cost": hit_cost,
        "cap_regret": cap_regret,
        "myopic_hold_regret": hold_regret,
        "n_blank_final_sum": blanks,
        "n_autosubs_sum": autosubs,
        "n_hold_gws": float(n_hold),
        "n_transfer_gws": float(n_tx_gw),
        "mean_tx": float(log["n_transfers"].mean()),
        "total_transfers": float(log["n_transfers"].sum()),
        "total_hits": float(log["hits"].sum()),
        "mean_itb": float(log["itb"].mean()),
        "final_squad_value": float(log["squad_sell_value"].iloc[-1]) / 10.0
        if len(log)
        else float("nan"),
    }


def plot_audit(
    log: pd.DataFrame,
    sens: pd.DataFrame,
    budget_total: float,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    fig.suptitle("Stage 27 — FT play audit (locked xP)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(log["gw"], log["gw_points"].cumsum(), "-o", ms=3, color="steelblue", label="xp_ft")
    ax.axhline(budget_total, color="gray", ls="--", lw=1, label=f"budget rebuild ({budget_total:.0f})")
    ax.set_title("Cumulative GW points (cap − hits)")
    ax.set_xlabel("GW")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    colors = ["seagreen" if h else ("crimson" if t > 0 else "steelblue") for h, t in zip(log["hits"], log["n_transfers"])]
    ax.bar(log["gw"], log["n_transfers"], color=colors, alpha=0.85)
    ax.set_title("Transfers / GW (red = took a hit)")
    ax.set_xlabel("GW")
    ax.set_ylabel("n_transfers")

    ax = axes[1, 0]
    ax.bar(log["gw"], log["cap_regret"], color="darkorange", alpha=0.85, label="cap regret")
    ax.bar(log["gw"], log["hit_cost"], color="crimson", alpha=0.55, label="hit cost")
    ax.set_title("Per-GW leaks")
    ax.set_xlabel("GW")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    if not sens.empty:
        for label, g in sens.groupby("setting"):
            ax.bar(label, float(g["total_points"].iloc[0]), color="steelblue", alpha=0.8)
        ax.axhline(float(log["gw_points"].sum()), color="crimson", ls="--", lw=1)
        ax.set_title("Sensitivity totals")
        ax.tick_params(axis="x", rotation=30)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    log: pd.DataFrame,
    leaks: dict[str, float],
    sens: pd.DataFrame,
    budget_total: float,
    gws: list[int],
) -> None:
    lines = [
        "# Stage 27 — Game-engine play audit (locked xP v2)",
        "",
        "Pure xP scorer (`FWD_LEVEL_CAL=False`, no horizon fade). Instrument the FT "
        "agent to see **where the game engine leaks points**, not whether μ ranks.",
        "",
        f"- Season: **{EVAL_SEASON}**, GWs **{gws[0]}–{gws[-1]}** (n={len(gws)})",
        f"- Defaults: H={HORIZON} (audit uses H=5), HOLD_EPS={HOLD_EPS}, HIT_COST={HIT_COST}",
        f"- FT total: **{leaks['total_points']:.0f}**",
        f"- Budget rebuild (no continuity): **{budget_total:.0f}** "
        f"(gap {leaks['total_points'] - budget_total:+.0f})",
        "",
        "## Season summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Total pts (cap − hits) | {leaks['total_points']:.0f} |",
        f"| Hit cost | −{leaks['hit_cost']:.0f} |",
        f"| Transfers | {leaks['total_transfers']:.0f} "
        f"({leaks['n_transfer_gws']:.0f} GWs) |",
        f"| Hold GWs | {leaks['n_hold_gws']:.0f} |",
        f"| Hits taken | {leaks['total_hits']:.0f} |",
        f"| Mean ITB | £{leaks['mean_itb']:.1f}m |",
        f"| Final squad SV | £{leaks['final_squad_value']:.1f}m |",
        f"| Autosubs fired | {leaks['n_autosubs_sum']:.0f} |",
        f"| Blank XI slots left | {leaks['n_blank_final_sum']:.0f} |",
        "",
        "## Leak attribution (additive diagnostics)",
        "",
        "| leak | pts | note |",
        "|---|---:|---|",
        f"| Hit cost | {leaks['hit_cost']:.0f} | paid −4s |",
        f"| Captain regret | {leaks['cap_regret']:.0f} | best-in-XI actual − awarded C |",
        f"| Myopic hold regret | {leaks['myopic_hold_regret']:.0f} | "
        f"transfer GWs where hold scored more *this* GW |",
        "",
        "These are **not** fully additive (captain regret ignores formation; hold "
        "regret is myopic). Use them to rank failure modes.",
        "",
        "## Worst GWs (by gw_points)",
        "",
        "| gw | pts | tx | hits | cap regret | blanks | held | captain |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    worst = log.nsmallest(8, "gw_points")
    for r in worst.itertuples():
        lines.append(
            f"| {r.gw} | {r.gw_points:.0f} | {r.n_transfers} | {r.hits} | "
            f"{r.cap_regret:.0f} | {r.n_blank_final} | {r.held} | {r.captain_name} |"
        )

    lines += [
        "",
        "## Transfer behaviour",
        "",
        f"- Hold rate: **{100 * leaks['n_hold_gws'] / max(1, len(log) - 1):.0f}%** of post-GW1 weeks",
        f"- Mean transfers/GW: **{leaks['mean_tx']:.2f}**",
        f"- Hit rate: **{leaks['total_hits']:.0f}** hits over season",
        "",
    ]

    # Hit GWs
    hit_gws = log.loc[log["hits"] > 0]
    if len(hit_gws):
        lines += ["### Hit weeks", "", "| gw | tx | hits | cost | Δ vs hold (myopic) |", "|---:|---:|---:|---:|---:|"]
        for r in hit_gws.itertuples():
            lines.append(
                f"| {r.gw} | {r.n_transfers} | {r.hits} | {r.hit_cost:.0f} | "
                f"{r.myopic_hold_delta:+.1f} |"
            )
        lines.append("")

    lines += [
        "## Policy sensitivity",
        "",
        "| setting | total | vs baseline | transfers | hits |",
        "|---|---:|---:|---:|---:|",
    ]
    base = float(sens.loc[sens["setting"] == "baseline", "total_points"].iloc[0]) if (
        not sens.empty and (sens["setting"] == "baseline").any()
    ) else leaks["total_points"]
    for r in sens.itertuples():
        lines.append(
            f"| {r.setting} | {r.total_points:.0f} | {r.total_points - base:+.0f} | "
            f"{r.total_transfers:.0f} | {r.total_hits:.0f} |"
        )

    # Verdict
    dominant = max(
        [
            ("hits", leaks["hit_cost"]),
            ("captain", leaks["cap_regret"]),
            ("myopic_hold", leaks["myopic_hold_regret"]),
        ],
        key=lambda x: x[1],
    )
    gap = budget_total - leaks["total_points"]
    lines += [
        "",
        "## Gate read",
        "",
        f"- Continuity tax vs free budget rebuild: **{gap:+.0f}** pts "
        f"(rebuild {budget_total:.0f} − FT {leaks['total_points']:.0f})",
        f"- Largest measured leak bucket: **{dominant[0]}** ({dominant[1]:.0f} pts)",
        "",
        "### Recommended next levers (from this audit)",
        "",
    ]
    recs = []
    if leaks["hit_cost"] >= 40:
        recs.append("- Tighten hit policy (`MAX_HITS` / higher `HOLD_EPS`) — hits are expensive.")
    if leaks["cap_regret"] >= 50:
        recs.append("- Captain model (not just top μ) — large C regret vs best-in-XI actual.")
    if leaks["myopic_hold_regret"] >= 40:
        recs.append("- Hold threshold / horizon — transfers often lose the current GW.")
    if gap >= 150:
        recs.append("- Continuity is the main tax; chips (WC/FH) likely high leverage next.")
    if leaks["mean_itb"] >= 1.5:
        recs.append("- Dead money in the bank — transfer valuation may undervalue upgrades.")
    if not recs:
        recs.append("- Policy looks balanced; move to chip timing next.")
    lines.extend(recs)
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/stage_27_decision_log.csv`",
        "- `data/processed/stage_27_sensitivity.csv`",
        "- `data/plots/stage_27_play_audit.png`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print(f"Building {EVAL_SEASON} locked xP features…", flush=True)
    feat = build_one_season(EVAL_SEASON, _fd_code(EVAL_SEASON))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster(EVAL_SEASON)
    gws = [g for g in sorted(int(x) for x in feat["gw"].unique()) if g >= 4]

    print(f"Instrumented FT audit GWs {gws[0]}–{gws[-1]} (H=5)…", flush=True)
    log = run_instrumented_ft(feat, gws, roster, horizon=5, hold_eps=HOLD_EPS)
    leaks = summarise_leaks(log)
    print(
        f"FT total={leaks['total_points']:.0f} hits=-{leaks['hit_cost']:.0f} "
        f"cap_regret={leaks['cap_regret']:.0f}",
        flush=True,
    )

    print("Budget rebuild upper bound…", flush=True)
    weekly_b = run_budgeted_season(
        feat, {"xp": SCORE_COL}, gws
    )
    budget_total = float(
        weekly_b.loc[weekly_b["method"] == "xp_budget", "xi_points_cap"].sum()
    )
    if not np.isfinite(budget_total) or budget_total == 0:
        budget_total = float(weekly_b["xi_points_cap"].sum())

    # Sensitivities
    sens_rows: list[dict[str, Any]] = [
        {
            "setting": "baseline",
            "total_points": leaks["total_points"],
            "total_transfers": leaks["total_transfers"],
            "total_hits": leaks["total_hits"],
            "horizon": 5,
            "hold_eps": HOLD_EPS,
        }
    ]
    for eps in (0.5, 2.5):
        print(f"  sensitivity HOLD_EPS={eps}…", flush=True)
        lg = run_instrumented_ft(
            feat, gws, roster, horizon=5, hold_eps=eps, label=f"eps_{eps}"
        )
        lk = summarise_leaks(lg)
        sens_rows.append(
            {
                "setting": f"HOLD_EPS={eps}",
                "total_points": lk["total_points"],
                "total_transfers": lk["total_transfers"],
                "total_hits": lk["total_hits"],
                "horizon": 5,
                "hold_eps": eps,
            }
        )
    print("  sensitivity H=3…", flush=True)
    lg = run_instrumented_ft(
        feat, gws, roster, horizon=3, hold_eps=HOLD_EPS, label="H3"
    )
    lk = summarise_leaks(lg)
    sens_rows.append(
        {
            "setting": "H=3",
            "total_points": lk["total_points"],
            "total_transfers": lk["total_transfers"],
            "total_hits": lk["total_hits"],
            "horizon": 3,
            "hold_eps": HOLD_EPS,
        }
    )
    sens = pd.DataFrame(sens_rows)
    print(sens.to_string(index=False), flush=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    log.to_csv(PROCESSED / "stage_27_decision_log.csv", index=False)
    sens.to_csv(PROCESSED / "stage_27_sensitivity.csv", index=False)
    pd.DataFrame([leaks]).to_csv(PROCESSED / "stage_27_leaks.csv", index=False)
    plot_audit(log, sens, budget_total, PLOTS / "stage_27_play_audit.png")
    write_report(
        REPORTS / "stage_27_play_audit.md", log, leaks, sens, budget_total, gws
    )
    return {"log": log, "leaks": leaks, "sens": sens, "budget_total": budget_total}


if __name__ == "__main__":
    out = run()
    print(f"\nWrote {REPORTS}/stage_27_play_audit.md")
