"""Stage 17 — Starter-only, no-minutes, per-position Ridge (fresh seasons).

Changes vs stage 16 ``ridge_ms``:
  - Drop minutes / xMi / xp_appear (and xP terms that embed play_scale)
  - Train only on rows with minutes ≥ 60
  - Climb candidates: prior 3-GW max minutes ≥ 60 (leakage-free)
  - Separate Ridge per position (GKP / DEF / MID / FWD)
  - Drop stale 2022/23; recency sample weights on 23/24–25/26
  - No new captain model (same climb captain = top score)

Writes:
  data/processed/season_climb_ridge_v2.csv
  data/plots/season_climb_ridge_v2.png
  reports/stage_17_ridge_starters.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.ridge_multiseason import EVAL_SEASON, SEASONS as ALL_SEASONS, build_one_season
from src.models.season_climb import FORMATIONS, pick_xi, summarize
from src.models.season_climb_ml import MIN_TRAIN_ROWS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

# Drop stale 2022/23
SEASONS: list[tuple[str, str]] = [s for s in ALL_SEASONS if s[0] >= "2023-24"]
SEASON_WEIGHT = {"2023-24": 1.0, "2024-25": 2.0, "2025-26": 3.0}

# No minutes / xMi / appear; no xP play_scale terms
FEATURE_COLS = [
    "exp_points",
    "roll3_points",
    "exp_xG",
    "exp_xA",
    "share_xG",
    "share_xA",
    "lam_scored",
    "lam_assist",
    "p_cs_mkt",
    "p_not_lose",
    "attack_strength",
    "defend_threat",
    "exp_defcon_hit",
    "value",
    "e_total",
    "p_over",
    "p_under",
]
POSITIONS = ("GKP", "DEF", "MID", "FWD")
MIN_POS_TRAIN = 150


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    return x.reset_index(drop=True)


def _make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]
    )


def build_fresh_seasons() -> pd.DataFrame:
    frames = []
    for season, code in SEASONS:
        frames.append(build_one_season(season, code))
    df = pd.concat(frames, ignore_index=True)
    # Eligibility: max minutes over prior 3 GWs ≥ 60 (no current-GW minutes)
    df = df.sort_values(["player_id", "gw"], kind="mergesort")
    g = df.groupby("player_id", sort=False)
    df["prior_max3_minutes"] = g["minutes"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).max()
    )
    df["eligible"] = df["prior_max3_minutes"].fillna(0) >= 60.0
    df["sample_weight"] = df["season"].map(SEASON_WEIGHT).fillna(1.0)
    df = add_forward_target_h3(df)
    return df


def add_forward_target_h3(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Mean total_points over GW t..t+H-1 within (season, player); for Ridge y."""
    out = df.sort_values(["season", "player_id", "gw"], kind="mergesort").copy()
    g = out.groupby(["season", "player_id"], sort=False)["total_points"]
    acc = pd.to_numeric(out["total_points"], errors="coerce").fillna(0.0)
    cnt = pd.Series(1.0, index=out.index)
    for h in range(1, horizon):
        sh = g.shift(-h)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(float)
    out["target_fwd_h3"] = acc / cnt.replace(0, np.nan)
    return out


def walk_forward_pos_ridge(all_feat: pd.DataFrame) -> tuple[pd.Series, list[int]]:
    """Per-position Ridge; train on minutes≥60 only; predict eligible rows."""
    preds = pd.Series(np.nan, index=all_feat.index, dtype=float)
    scored: list[int] = []
    eval_ord = all_feat.loc[all_feat["season"] == EVAL_SEASON, "season_ord"].iloc[0]
    gws = sorted(all_feat.loc[all_feat["season"] == EVAL_SEASON, "gw"].unique())

    for gw in gws:
        train_all = all_feat.loc[
            (all_feat["season_ord"] < eval_ord)
            | ((all_feat["season"] == EVAL_SEASON) & (all_feat["gw"] < gw))
        ]
        # Starters only for training
        train_all = train_all.loc[train_all["minutes"] >= 60]
        test_idx = all_feat.index[
            (all_feat["season"] == EVAL_SEASON)
            & (all_feat["gw"] == gw)
            & (all_feat["eligible"])
        ]
        if len(train_all) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue

        ok = True
        for pos in POSITIONS:
            tr = train_all.loc[train_all["position"] == pos]
            te_idx = all_feat.index.intersection(test_idx).tolist()
            te_pos = all_feat.loc[te_idx]
            te_pos = te_pos.loc[te_pos["position"] == pos]
            if len(tr) < MIN_POS_TRAIN or len(te_pos) == 0:
                # fallback: skip GW if any position can't fit
                if len(te_pos) > 0 and len(tr) < MIN_POS_TRAIN:
                    ok = False
                    break
                continue
            x_tr = _design_matrix(tr)
            y_tr = tr["total_points"].to_numpy(float)
            w = tr["sample_weight"].to_numpy(float)
            x_te = _design_matrix(te_pos).reindex(columns=x_tr.columns, fill_value=0)
            model = _make_ridge()
            model.fit(x_tr, y_tr, model__sample_weight=w)
            preds.loc[te_pos.index] = model.predict(x_te)
        if ok and preds.loc[test_idx].notna().any():
            scored.append(int(gw))
    return preds, scored


def walk_forward_global_ridge_starters(all_feat: pd.DataFrame) -> tuple[pd.Series, list[int]]:
    """Comparator: one Ridge, still starter-only train / eligible predict."""
    preds = pd.Series(np.nan, index=all_feat.index, dtype=float)
    scored: list[int] = []
    eval_ord = all_feat.loc[all_feat["season"] == EVAL_SEASON, "season_ord"].iloc[0]
    gws = sorted(all_feat.loc[all_feat["season"] == EVAL_SEASON, "gw"].unique())

    for gw in gws:
        train = all_feat.loc[
            (
                (all_feat["season_ord"] < eval_ord)
                | ((all_feat["season"] == EVAL_SEASON) & (all_feat["gw"] < gw))
            )
            & (all_feat["minutes"] >= 60)
        ]
        test_idx = all_feat.index[
            (all_feat["season"] == EVAL_SEASON)
            & (all_feat["gw"] == gw)
            & (all_feat["eligible"])
        ]
        if len(train) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue
        x_tr = _design_matrix(train)
        y_tr = train["total_points"].to_numpy(float)
        w = train["sample_weight"].to_numpy(float)
        x_te = _design_matrix(all_feat.loc[test_idx]).reindex(
            columns=x_tr.columns, fill_value=0
        )
        model = _make_ridge()
        model.fit(x_tr, y_tr, model__sample_weight=w)
        preds.loc[test_idx] = model.predict(x_te)
        scored.append(int(gw))
    return preds, scored


def walk_forward_global_ridge_h3(
    all_feat: pd.DataFrame,
    *,
    target_col: str = "target_fwd_h3",
) -> tuple[pd.Series, list[int]]:
    """Global Ridge on forward H=3 mean points (starter train / eligible predict)."""
    preds = pd.Series(np.nan, index=all_feat.index, dtype=float)
    scored: list[int] = []
    eval_ord = all_feat.loc[all_feat["season"] == EVAL_SEASON, "season_ord"].iloc[0]
    gws = sorted(all_feat.loc[all_feat["season"] == EVAL_SEASON, "gw"].unique())

    for gw in gws:
        train = all_feat.loc[
            (
                (all_feat["season_ord"] < eval_ord)
                | ((all_feat["season"] == EVAL_SEASON) & (all_feat["gw"] < gw))
            )
            & (all_feat["minutes"] >= 60)
            & (all_feat[target_col].notna())
        ]
        test_idx = all_feat.index[
            (all_feat["season"] == EVAL_SEASON)
            & (all_feat["gw"] == gw)
            & (all_feat["eligible"])
        ]
        if len(train) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue
        x_tr = _design_matrix(train)
        y_tr = train[target_col].to_numpy(float)
        w = train["sample_weight"].to_numpy(float)
        x_te = _design_matrix(all_feat.loc[test_idx]).reindex(
            columns=x_tr.columns, fill_value=0
        )
        model = _make_ridge()
        model.fit(x_tr, y_tr, model__sample_weight=w)
        preds.loc[test_idx] = model.predict(x_te)
        scored.append(int(gw))
    return preds, scored


def run_season_eligible(
    feat: pd.DataFrame,
    extra_score_cols: dict[str, str],
    gws: list[int],
) -> pd.DataFrame:
    """Climb using only eligible (likely 60′) candidates for every method."""
    score_cols = {
        "xp": "score_xp",
        "exp_points": "score_exp_points",
        "price": "score_price",
        "random": "score_random",
    }
    score_cols.update(extra_score_cols)
    rows: list[dict[str, Any]] = []

    for gw in gws:
        gw_df = feat.loc[(feat["gw"] == gw) & (feat["eligible"])].copy()
        if gw_df["position"].nunique() < 4:
            continue
        z_xp = (gw_df["score_xp"] - gw_df["score_xp"].mean()) / (
            gw_df["score_xp"].std() + 1e-6
        )
        z_exp = (gw_df["score_exp_points"] - gw_df["score_exp_points"].mean()) / (
            gw_df["score_exp_points"].std() + 1e-6
        )
        gw_df["score_blend"] = 0.5 * z_xp + 0.5 * z_exp
        methods = dict(score_cols)
        methods["blend_xp_exp"] = "score_blend"

        for name, col in methods.items():
            if col not in gw_df.columns or not np.isfinite(gw_df[col]).any():
                continue
            # Methods that only scored eligible rows may still have NaNs — drop them
            sub = gw_df.loc[np.isfinite(gw_df[col])]
            if sub["position"].nunique() < 4:
                continue
            try:
                sel, form = pick_xi(sub, col)
            except RuntimeError:
                continue
            cap_idx = sel[col].idxmax()
            pts_sum = float(sel["total_points"].sum())
            cap_pts = float(sel.loc[cap_idx, "total_points"])
            rows.append(
                {
                    "gw": int(gw),
                    "method": name,
                    "xi_points": pts_sum,
                    "xi_points_cap": pts_sum + cap_pts,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                    "n_players": int(len(sel)),
                    "mean_score": float(sel[col].mean()),
                    "n_eligible": int(len(gw_df)),
                }
            )
    return pd.DataFrame(rows)


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    focus = [
        "ridge_pos",
        "ridge_global_starters",
        "xp",
        "exp_points",
        "blend_xp_exp",
        "price",
    ]
    colors = {
        "ridge_pos": "crimson",
        "ridge_global_starters": "tomato",
        "xp": "steelblue",
        "exp_points": "gray",
        "blend_xp_exp": "darkorange",
        "price": "seagreen",
    }
    for method in focus:
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        ax.plot(
            g["gw"],
            g["xi_points_cap"].cumsum(),
            "-o",
            ms=3,
            lw=2.4 if method.startswith("ridge") else 1.4,
            color=colors.get(method, "black"),
            label=method,
        )
    ax.set_xlabel("Gameweek (2025/26)")
    ax.set_ylabel("Cumulative XI points (captain ×2)")
    ax.set_title("Starter-only / no-xMi / per-position Ridge climb")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    scored_gws: list[int],
    n_by_season: dict[str, int],
) -> None:
    base = float(summary.loc[summary["method"] == "exp_points", "total_points"].iloc[0])
    xp_total = float(summary.loc[summary["method"] == "xp", "total_points"].iloc[0])
    pos_total = float(summary.loc[summary["method"] == "ridge_pos", "total_points"].iloc[0])
    glob = summary.loc[summary["method"] == "ridge_global_starters", "total_points"]
    glob_total = float(glob.iloc[0]) if len(glob) else float("nan")

    lines = [
        "# Stage 17 — Starter-only, no-minutes, per-position Ridge",
        "",
        "Highest-leverage follow-ups (no captain model yet):",
        "",
        "- **No minutes features** (dropped xMi / xp_appear / play-scaled xP terms)",
        "- **Train** only on player-GWs with **minutes ≥ 60**",
        "- **Climb pool** = players with prior-3 max minutes ≥ 60 (leakage-free)",
        "- **Per-position Ridge** (GKP/DEF/MID/FWD)",
        "- **Fresh data:** seasons 2023/24–2025/26 with recency sample weights "
        f"`{SEASON_WEIGHT}` (no 2022/23)",
        "",
        f"- Eval GWs: **{scored_gws[0]}–{scored_gws[-1]}** (n={len(scored_gws)})",
        f"- Rows by season: "
        + ", ".join(f"{k}={v}" for k, v in n_by_season.items()),
        "",
        "## Final standings (eligible pool, captain ×2)",
        "",
        "| method | total | mean/GW | vs exp | vs xP |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - base:+.0f} | {r.total_points - xp_total:+.0f} |"
        )

    if np.isfinite(glob_total) and glob_total > pos_total + 10:
        best, best_total = "ridge_global_starters", glob_total
    else:
        best, best_total = "ridge_pos", pos_total

    if best_total > xp_total + 10 and best_total > base + 10:
        verdict = (
            f"PASS — {best} leads "
            f"(vs xP {best_total - xp_total:+.0f}, vs exp {best_total - base:+.0f}; "
            f"pos vs global {pos_total - glob_total:+.0f})"
        )
    elif best_total >= xp_total:
        verdict = f"WEAK — {best} ≈/≥ xP ({best_total - xp_total:+.0f})"
    else:
        verdict = f"FAIL — best ridge trails xP ({best_total - xp_total:+.0f})"

    lines += [
        "",
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_ridge_v2.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_ridge_v2.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print("Building fresh-season features (no 2022/23)…")
    all_feat = build_fresh_seasons()
    n_by_season = all_feat.groupby("season").size().to_dict()
    print("rows", n_by_season)
    print(
        "eligible rate 2025/26",
        float(all_feat.loc[all_feat.season == EVAL_SEASON, "eligible"].mean()),
    )

    print("Per-position Ridge…")
    pred_pos, scored_pos = walk_forward_pos_ridge(all_feat)
    print("Global starter Ridge…")
    pred_glob, scored_glob = walk_forward_global_ridge_starters(all_feat)
    scored = sorted(set(scored_pos) & set(scored_glob))
    if not scored:
        raise RuntimeError("No scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat["score_ridge_pos"] = pred_pos.loc[eval_feat.index]
    eval_feat["score_ridge_global_starters"] = pred_glob.loc[eval_feat.index]

    weekly = run_season_eligible(
        eval_feat,
        {
            "ridge_pos": "score_ridge_pos",
            "ridge_global_starters": "score_ridge_global_starters",
        },
        scored,
    )
    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[weekly["method"].isin(counts[counts >= len(scored)].index)].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_ridge_v2.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_ridge_v2.png")
    write_report(
        REPORTS / "stage_17_ridge_starters.md", summary, scored, n_by_season
    )
    return {"summary": summary, "weekly": weekly, "scored_gws": scored, "n_by_season": n_by_season}


if __name__ == "__main__":
    out = run()
    print(f"\nGWs {out['scored_gws'][0]}–{out['scored_gws'][-1]} n={len(out['scored_gws'])}")
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_17_ridge_starters.md")
