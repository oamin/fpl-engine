"""Stage 14 — Stripped season climb (no budget / chips / transfer state).

Each GW: pick a position-legal XI from the full player pool by a score,
bank *actual* FPL points, plot cumulative total vs baselines.

Constraints kept (only what's needed for a fair XI):
  1 GKP, 3–5 DEF, 2–5 MID, 1–3 FWD, exactly 11 players.
No: budget, free hits, chips, transfer continuity, bench.

Scores (leakage-free at GW t):
  xP engine, exp points, roll3 points, xMi, price, random

Writes:
  data/processed/season_climb.csv
  data/plots/season_climb.png
  reports/stage_14_season_climb.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.xp_engine import (
    compute_xp,
    load_joined,
    add_market_pots,
    add_player_priors,
    add_team_prior_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

MIN_HISTORY = 3
FORMATIONS = [
    # (def, mid, fwd) — GKP always 1
    (3, 4, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 3, 3),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1),
]


def build_scores() -> pd.DataFrame:
    raw = load_joined()
    feat = add_market_pots(raw)
    feat = add_player_priors(feat)
    feat = compute_xp(feat)
    feat = add_team_prior_score(feat)
    # Only rows with enough history for priors
    feat = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()
    # Random baseline: fixed seed noise (not predictive)
    rng = np.random.default_rng(42)
    feat["score_random"] = rng.random(len(feat))
    feat["score_xp"] = feat["xp"]
    feat["score_exp_points"] = feat["exp_points"]
    feat["score_team_prior"] = feat["xp_team_prior"]
    feat["score_roll3_points"] = feat["roll3_points"]
    feat["score_xmi"] = feat["xmi"]
    feat["score_price"] = feat["value"]
    return feat


def pick_xi(
    gw_df: pd.DataFrame,
    score_col: str,
    formations: list[tuple[int, int, int]] | None = None,
) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """Pick best formation XI by score_col. Returns selected rows + formation.

    The default list is the historical climb list, which omits 5-2-3.
    A live plan passes the official list instead. That does not change
    the published climb.
    """
    forms = FORMATIONS if formations is None else formations
    best_pts_proxy = -1.0
    best_sel: pd.DataFrame | None = None
    best_form = forms[0]

    for n_def, n_mid, n_fwd in forms:
        gkp = gw_df.loc[gw_df["position"] == "GKP"].nlargest(1, score_col)
        deff = gw_df.loc[gw_df["position"] == "DEF"].nlargest(n_def, score_col)
        mid = gw_df.loc[gw_df["position"] == "MID"].nlargest(n_mid, score_col)
        fwd = gw_df.loc[gw_df["position"] == "FWD"].nlargest(n_fwd, score_col)
        if len(gkp) < 1 or len(deff) < n_def or len(mid) < n_mid or len(fwd) < n_fwd:
            continue
        sel = pd.concat([gkp, deff, mid, fwd], axis=0)
        # Proxy for selection quality = sum of scores (not actual pts — no leakage)
        proxy = float(sel[score_col].sum())
        if proxy > best_pts_proxy:
            best_pts_proxy = proxy
            best_sel = sel
            best_form = (n_def, n_mid, n_fwd)

    if best_sel is None:
        raise RuntimeError("Could not form an XI")
    return best_sel, best_form


def formation_legal(positions: list[str]) -> bool:
    """True iff positions are a legal FPL XI shape."""
    n_gkp = sum(1 for p in positions if p == "GKP")
    n_def = sum(1 for p in positions if p == "DEF")
    n_mid = sum(1 for p in positions if p == "MID")
    n_fwd = sum(1 for p in positions if p == "FWD")
    return (
        len(positions) == 11
        and n_gkp == 1
        and 3 <= n_def <= 5
        and 2 <= n_mid <= 5
        and 1 <= n_fwd <= 3
    )


def ordered_bench(squad_df: pd.DataFrame, xi: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Bench = squad \\ XI; outfield ordered by score (FPL bench priority)."""
    xi_ids = set(xi["player_id"].astype(str))
    bench = squad_df.loc[~squad_df["player_id"].astype(str).isin(xi_ids)].copy()
    bench = bench.drop_duplicates("player_id", keep="first")
    gkp = bench.loc[bench["position"] == "GKP"]
    out = bench.loc[bench["position"] != "GKP"].sort_values(
        score_col, ascending=False, kind="mergesort"
    )
    return pd.concat([gkp, out], axis=0)


def apply_autosubs(
    xi: pd.DataFrame,
    bench: pd.DataFrame,
    *,
    minutes_col: str = "minutes",
) -> tuple[pd.DataFrame, int]:
    """FPL-style autosubs for 0-minute starters.

    - Outfield blanks filled from bench order if the sub played and formation stays legal.
    - GK blanks only filled by the bench GK (if they played).
    Returns (final XI, number of successful subs).
    """
    out = xi.copy()
    if bench.empty:
        return out, 0

    bench = bench.copy()
    bench_mins = pd.to_numeric(bench[minutes_col], errors="coerce").fillna(0.0)
    xi_mins = pd.to_numeric(out[minutes_col], errors="coerce").fillna(0.0)

    bench_gk = bench.loc[bench["position"] == "GKP"]
    bench_out = bench.loc[bench["position"] != "GKP"]  # already ordered by caller

    used: set[str] = set()
    n_subs = 0
    blank_idxs = list(out.index[xi_mins <= 0])

    for bidx in blank_idxs:
        blank_pos = str(out.at[bidx, "position"])
        candidates = bench_gk if blank_pos == "GKP" else bench_out
        for cidx in candidates.index:
            cid = str(candidates.at[cidx, "player_id"])
            if cid in used:
                continue
            if float(bench_mins.loc[cidx]) <= 0:
                continue
            new_pos = [
                str(candidates.at[cidx, "position"]) if i == bidx else str(out.at[i, "position"])
                for i in out.index
            ]
            if not formation_legal(new_pos):
                continue
            for col in out.columns:
                if col in candidates.columns:
                    out.at[bidx, col] = candidates.at[cidx, col]
            used.add(cid)
            n_subs += 1
            break

    return out, n_subs


def points_from_subs(
    final_xi: pd.DataFrame,
    intended_ids: set[str],
    *,
    points_col: str = "total_points",
) -> float:
    """Points scored by players who replaced a blank starter.

    A blank starter has zero minutes, so these points are what the
    automatic substitutes recouped.
    """
    arrived = final_xi.loc[~final_xi["player_id"].astype(str).isin(intended_ids)]
    return float(pd.to_numeric(arrived[points_col], errors="coerce").fillna(0.0).sum())


def bank_squad_gw(
    squad_df: pd.DataFrame,
    score_col: str,
    *,
    use_autosubs: bool = True,
    minutes_col: str = "minutes",
    points_col: str = "total_points",
) -> dict[str, Any]:
    """Pick XI from 15, optional autosubs, captain + vice-captain doubling.

    Captain / VC chosen on the *intended* XI (by ``score_col``) before autosubs.
    If captain played 0 minutes, VC receives the double (FPL rule).
    """
    xi, form = pick_xi(squad_df, score_col)
    xi = xi.copy()
    score = pd.to_numeric(xi[score_col], errors="coerce")
    # Captain = best μ; VC = second-best μ on intended XI
    order = score.sort_values(ascending=False, kind="mergesort")
    cap_id = str(xi.loc[order.index[0], "player_id"])
    vc_id = str(xi.loc[order.index[1], "player_id"]) if len(order) > 1 else cap_id

    n_subs = 0
    n_blank_intended = int(
        (pd.to_numeric(xi[minutes_col], errors="coerce").fillna(0.0) <= 0).sum()
    )
    intended_ids = set(xi["player_id"].astype(str))
    if use_autosubs:
        bench = ordered_bench(squad_df, xi, score_col)
        xi, n_subs = apply_autosubs(xi, bench, minutes_col=minutes_col)

    pts = pd.to_numeric(xi[points_col], errors="coerce").fillna(0.0)
    mins = pd.to_numeric(xi[minutes_col], errors="coerce").fillna(0.0)
    pts_sum = float(pts.sum())

    # Map player_id → points/minutes on final XI (after possible sub for blank captain)
    by_id_pts = {
        str(r.player_id): float(getattr(r, points_col))
        for r in xi.itertuples()
    }
    # Minutes for captain/VC from *original* intent: use squad rows (pre-sub identities)
    squad_mins = {
        str(r.player_id): float(getattr(r, minutes_col) or 0)
        for r in squad_df.itertuples()
    }
    squad_pts = {
        str(r.player_id): float(getattr(r, points_col) or 0)
        for r in squad_df.itertuples()
    }

    if squad_mins.get(cap_id, 0.0) > 0:
        cap_extra = squad_pts.get(cap_id, 0.0)
    elif squad_mins.get(vc_id, 0.0) > 0:
        cap_extra = squad_pts.get(vc_id, 0.0)
    else:
        cap_extra = 0.0

    n_blank_final = int((mins <= 0).sum())
    n_def = int((xi["position"] == "DEF").sum())
    n_mid = int((xi["position"] == "MID").sum())
    n_fwd = int((xi["position"] == "FWD").sum())
    return {
        "xi": xi,
        "form": (n_def, n_mid, n_fwd),
        "form_intended": form,
        "xi_points": pts_sum,
        "cap_extra": float(cap_extra),
        "n_autosubs": n_subs,
        "n_blank_intended": n_blank_intended,
        "n_blank_final": n_blank_final,
        "sub_points": points_from_subs(xi, intended_ids, points_col=points_col),
        "captain_id": cap_id,
        "vice_id": vc_id,
    }


def run_season(
    feat: pd.DataFrame,
    extra_score_cols: dict[str, str] | None = None,
    gws: list[int] | None = None,
) -> pd.DataFrame:
    """Run stripped climb. ``extra_score_cols`` maps method name → column."""
    score_cols = {
        "xp": "score_xp",
        "exp_points": "score_exp_points",
        "roll3_points": "score_roll3_points",
        "xmi": "score_xmi",
        "price": "score_price",
        "random": "score_random",
    }
    if extra_score_cols:
        score_cols.update(extra_score_cols)

    feat = feat.copy()
    rows: list[dict[str, Any]] = []
    gw_list = gws if gws is not None else sorted(feat["gw"].unique())

    for gw in gw_list:
        gw_df = feat.loc[feat["gw"] == gw]
        if gw_df["position"].nunique() < 4:
            continue
        z_xp = (gw_df["score_xp"] - gw_df["score_xp"].mean()) / (
            gw_df["score_xp"].std() + 1e-6
        )
        z_exp = (gw_df["score_exp_points"] - gw_df["score_exp_points"].mean()) / (
            gw_df["score_exp_points"].std() + 1e-6
        )
        gw_df = gw_df.copy()
        gw_df["score_blend"] = 0.5 * z_xp + 0.5 * z_exp

        methods = dict(score_cols)
        methods["blend_xp_exp"] = "score_blend"

        for name, col in methods.items():
            if col not in gw_df.columns or gw_df[col].isna().all():
                continue
            # Skip GW if this method has no finite scores (ML warm-up)
            if not np.isfinite(gw_df[col].to_numpy(float)).any():
                continue
            sel, form = pick_xi(gw_df, col)
            cap_idx = sel[col].idxmax()
            pts_sum = float(sel["total_points"].sum())
            cap_pts = float(sel.loc[cap_idx, "total_points"])
            pts_with_cap = pts_sum + cap_pts
            rows.append(
                {
                    "gw": int(gw),
                    "method": name,
                    "xi_points": pts_sum,
                    "xi_points_cap": pts_with_cap,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                    "n_players": int(len(sel)),
                    "mean_score": float(sel[col].mean()),
                }
            )

    return pd.DataFrame(rows)


def summarize(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in weekly.groupby("method"):
        g = g.sort_values("gw")
        cum = g["xi_points_cap"].cumsum()
        rows.append(
            {
                "method": method,
                "n_gw": int(len(g)),
                "total_points": float(g["xi_points_cap"].sum()),
                "mean_gw": float(g["xi_points_cap"].mean()),
                "total_no_cap": float(g["xi_points"].sum()),
                "final_cum": float(cum.iloc[-1]),
            }
        )
    out = pd.DataFrame(rows).sort_values("total_points", ascending=False)
    return out


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    focus = ["xp", "exp_points", "blend_xp_exp", "xmi", "price", "random"]
    colors = {
        "xp": "crimson",
        "exp_points": "steelblue",
        "blend_xp_exp": "darkorange",
        "xmi": "seagreen",
        "price": "gray",
        "random": "lightgray",
        "roll3_points": "purple",
    }
    for method in focus:
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        cum = g["xi_points_cap"].cumsum()
        ax.plot(
            g["gw"],
            cum,
            "-o",
            ms=3,
            lw=2.2 if method in ("xp", "blend_xp_exp") else 1.4,
            color=colors.get(method, "black"),
            label=method,
        )
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative XI points (captain ×2)")
    ax.set_title("Stripped season climb — no budget / chips / transfers")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, weekly: pd.DataFrame) -> None:
    base = float(summary.loc[summary["method"] == "exp_points", "total_points"].iloc[0])
    lines = [
        "# Stage 14 — Stripped season climb",
        "",
        "Each GW pick a **position-legal XI** by score from the full pool, "
        "bank actual points (captain = top score in XI, ×2).",
        "",
        "**Stripped out:** budget, chips, transfer continuity, bench.",
        "**Kept:** 1 GKP + legal formation (best of common shapes by score sum).",
        "",
        "Priors are leakage-free at GW t (`n_prior ≥ 3`).",
        "",
        "## Final standings (captain ×2)",
        "",
        "| method | total | mean/GW | vs exp_points |",
        "|---|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        delta = r.total_points - base
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | {delta:+.0f} |"
        )

    # Per-GW lead changes vs exp
    exp = weekly.loc[weekly["method"] == "exp_points"].set_index("gw")["xi_points_cap"]
    lines += [
        "",
        "## vs exp_points by GW (Δ XI pts with captain)",
        "",
        "| method | GWs ahead | GWs behind | mean Δ/GW |",
        "|---|---:|---:|---:|",
    ]
    for method in ("xp", "blend_xp_exp", "xmi", "price", "random"):
        m = weekly.loc[weekly["method"] == method].set_index("gw")["xi_points_cap"]
        both = pd.concat([m, exp], axis=1, keys=["m", "e"]).dropna()
        d = both["m"] - both["e"]
        lines.append(
            f"| {method} | {int((d > 0).sum())} | {int((d < 0).sum())} | {d.mean():+.2f} |"
        )

    xp_total = float(summary.loc[summary["method"] == "xp", "total_points"].iloc[0])
    blend_total = float(
        summary.loc[summary["method"] == "blend_xp_exp", "total_points"].iloc[0]
    )
    if blend_total > base + 10:
        verdict = f"PASS — blend beats exp_points by {blend_total - base:+.0f}"
    elif xp_total > base + 10:
        verdict = f"PASS — xP beats exp_points by {xp_total - base:+.0f}"
    elif max(xp_total, blend_total) >= base:
        verdict = (
            f"WEAK — best model {max(xp_total, blend_total):.0f} vs exp {base:.0f} "
            f"(Δ {max(xp_total, blend_total) - base:+.0f})"
        )
    else:
        verdict = (
            f"FAIL — models trail exp_points "
            f"(xP {xp_total - base:+.0f}, blend {blend_total - base:+.0f})"
        )

    lines += [
        "",
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "Pass bar: finish ≥ **+10** cumulative pts vs exp_points over the season "
        "(captain rule identical across methods).",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb.csv`",
        "",
        "## Read",
        "",
        "- This tests **ranking quality under XI constraints**, not transfer skill.",
        "- If xP / blend cannot beat exp_points here, MILP will not save it.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    feat = build_scores()
    weekly = run_season(feat)
    summary = summarize(weekly)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb.png")
    write_report(REPORTS / "stage_14_season_climb.md", summary, weekly)
    return {"weekly": weekly, "summary": summary}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_14_season_climb.md")
