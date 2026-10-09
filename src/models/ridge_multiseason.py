"""Stage 16 — Multi-season walk-forward Ridge on season climb.

Train Ridge on prior seasons + earlier GWs of 2025/26 (walk-forward).
Evaluate with the locked stripped XI climb on 2025/26 only.

Compares:
  ridge_ms   — multi-season walk-forward Ridge
  ridge_1s   — single-season (2025/26 only) walk-forward Ridge
  xp, exp_points, blend, price, random

Writes:
  data/processed/season_climb_ridge_ms.csv
  data/plots/season_climb_ridge_ms.png
  reports/stage_16_ridge_multiseason.md
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

from src.ingest.fpl_odds import join_players_to_fixtures, load_football_data, load_player_logs
from src.models.season_climb import pick_xi, run_season, summarize
from src.models.season_climb_ml import FEATURE_COLS, MIN_TRAIN_ROWS, _design_matrix, _make_ridge
from src.models.xp_engine import (
    MIN_HISTORY,
    add_market_pots,
    add_player_priors,
    compute_xp,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

# Train history + eval season
SEASONS: list[tuple[str, str]] = [
    ("2022-23", "2223"),
    ("2023-24", "2324"),
    ("2024-25", "2425"),
    ("2025-26", "2526"),
]
EVAL_SEASON = "2025-26"


def _attach_value_defcon(players: pd.DataFrame, season: str) -> pd.DataFrame:
    raw = pd.read_csv(CACHE / f"merged_gw_{season.replace('-', '_')}.csv")
    extra = pd.DataFrame(
        {
            "player_id": raw["element"].astype(str),
            "gw": pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce"),
            "value": pd.to_numeric(raw.get("value"), errors="coerce"),
            "transfers_balance": pd.to_numeric(
                raw.get("transfers_balance"), errors="coerce"
            ).fillna(0.0),
        }
    ).dropna(subset=["gw"])
    extra["gw"] = extra["gw"].astype(int)
    if "defensive_contribution" in raw.columns:
        extra["defcon_raw"] = pd.to_numeric(
            raw["defensive_contribution"], errors="coerce"
        ).fillna(0.0)
    else:
        extra["defcon_raw"] = 0.0
    extra = extra.drop_duplicates(["player_id", "gw"], keep="first")
    out = players.copy()
    out["player_id"] = out["player_id"].astype(str)
    out["gw"] = pd.to_numeric(out["gw"], errors="coerce").astype(int)
    # Player logs already carry value. The sheet join must not create value_x.
    out = out.merge(extra, on=["player_id", "gw"], how="left", suffixes=("", "_sheet"))
    if "value_sheet" in out.columns:
        if "value" not in out.columns:
            out["value"] = out["value_sheet"]
        else:
            out["value"] = out["value"].fillna(out["value_sheet"])
        out = out.drop(columns=["value_sheet"])
    out["value"] = out["value"].fillna(
        out.groupby("position")["value"].transform("median")
    )
    out["defcon_raw"] = out["defcon_raw"].fillna(0.0)
    out["transfers_balance"] = out["transfers_balance"].fillna(0.0)
    return out


def build_one_season(
    season: str, fd_code: str, *, early_buy: bool = False
) -> pd.DataFrame:
    """Joined odds + xP features for one season (within-season priors only).

    ``early_buy`` keeps rows with one or two prior appearances and caps
    their decision score. The default still drops those rows.
    """
    fixtures = load_football_data(code=fd_code)
    players = load_player_logs(season=season)
    players = _attach_value_defcon(players, season)
    joined, _, stats = join_players_to_fixtures(players, fixtures)
    print(
        f"  {season}: players={stats['n_player_appearances']} "
        f"joined={stats['n_player_joined']} rate={stats['join_rate']:.2f}"
    )

    # Season-scoped player key so expanding priors don't leak across years
    joined = joined.copy()
    joined["element"] = joined["player_id"].astype(str)
    joined["season"] = season
    joined["player_id"] = season + ":" + joined["element"]

    # xp_engine expects defcon_raw / columns like load_joined
    from src.models.xp_engine import (
        DEFCON_THRESH,
        MIN_MINUTES,
        add_team_prior_score,
    )

    joined["xG"] = pd.to_numeric(joined["xG"], errors="coerce").fillna(0.0)
    joined["xA"] = pd.to_numeric(joined["xA"], errors="coerce").fillna(0.0)
    joined["p_not_lose"] = pd.to_numeric(joined["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        joined["p_draw"], errors="coerce"
    )
    joined["p_over"] = pd.to_numeric(joined["p_over25"], errors="coerce")
    joined["p_under"] = pd.to_numeric(joined["p_under25"], errors="coerce")
    thr = joined["position"].map(DEFCON_THRESH)
    joined["defcon_hit"] = (
        joined["position"].isin(DEFCON_THRESH)
        & (joined["minutes"] >= MIN_MINUTES)
        & (joined["defcon_raw"] >= thr.fillna(999))
    ).astype(float)
    # Pre-2025/26: no DefCon → hit stays 0

    feat = add_market_pots(joined)
    feat = add_player_priors(feat)
    feat = compute_xp(feat)
    feat = add_team_prior_score(feat)
    from src.models.season_climb_ft import early_score_table

    early_scores = early_score_table(feat)
    if early_buy:
        feat = feat.loc[pd.to_numeric(feat["n_prior"], errors="coerce") >= 1].copy()
    else:
        feat = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()

    # Climb score aliases
    rng = np.random.default_rng(abs(hash(season)) % (2**32))
    feat["score_random"] = rng.random(len(feat))
    feat["score_xp"] = feat["xp"]
    if early_buy:
        from src.models.early_buy import cap_decision_score

        feat = cap_decision_score(feat)
    feat["score_exp_points"] = feat["exp_points"]
    feat["score_team_prior"] = feat["xp_team_prior"]
    feat["score_roll3_points"] = feat["roll3_points"]
    feat["score_xmi"] = feat["xmi"]
    feat["score_price"] = feat["value"]
    # Order key for walk-forward
    feat["season_ord"] = {s: i for i, (s, _) in enumerate(SEASONS)}[season]
    # A DataFrame in attrs breaks pandas ranking. A tuple compares cleanly.
    feat.attrs["early_scores"] = tuple(
        zip(
            early_scores["player_id"].astype(str),
            early_scores["gw"].astype(int),
            early_scores["score_xp"].astype(float),
            strict=False,
        )
    )
    return feat


def build_all_seasons() -> pd.DataFrame:
    frames = []
    for season, code in SEASONS:
        frames.append(build_one_season(season, code))
    return pd.concat(frames, ignore_index=True)


def walk_forward_ridge(
    all_feat: pd.DataFrame, *, multi_season: bool
) -> tuple[pd.Series, list[int]]:
    """Predict 2025/26 GWs. multi_season uses prior seasons in train set."""
    eval_mask = all_feat["season"] == EVAL_SEASON
    eval_df = all_feat.loc[eval_mask]
    gws = sorted(eval_df["gw"].unique())
    preds = pd.Series(np.nan, index=all_feat.index, dtype=float)
    scored: list[int] = []

    for gw in gws:
        if multi_season:
            train = all_feat.loc[
                (all_feat["season_ord"] < all_feat.loc[eval_mask, "season_ord"].iloc[0])
                | ((all_feat["season"] == EVAL_SEASON) & (all_feat["gw"] < gw))
            ]
        else:
            train = all_feat.loc[
                (all_feat["season"] == EVAL_SEASON) & (all_feat["gw"] < gw)
            ]
        test_idx = all_feat.index[
            (all_feat["season"] == EVAL_SEASON) & (all_feat["gw"] == gw)
        ]
        if len(train) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue
        x_train = _design_matrix(train)
        y_train = train["total_points"].to_numpy(float)
        x_test = _design_matrix(all_feat.loc[test_idx]).reindex(
            columns=x_train.columns, fill_value=0
        )
        model = _make_ridge()
        model.fit(x_train, y_train)
        preds.loc[test_idx] = model.predict(x_test)
        scored.append(int(gw))
    return preds, scored


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    focus = [
        "ridge_ms",
        "ridge_1s",
        "xp",
        "exp_points",
        "blend_xp_exp",
        "price",
    ]
    colors = {
        "ridge_ms": "crimson",
        "ridge_1s": "tomato",
        "xp": "steelblue",
        "exp_points": "gray",
        "blend_xp_exp": "darkorange",
        "price": "seagreen",
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
            lw=2.4 if method.startswith("ridge") else 1.4,
            color=colors.get(method, "black"),
            label=method,
        )
    ax.set_xlabel("Gameweek (2025/26)")
    ax.set_ylabel("Cumulative XI points (captain ×2)")
    ax.set_title("Multi-season Ridge vs single-season Ridge / xP / exp")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    scored_gws: list[int],
    n_by_season: dict[str, int],
) -> None:
    base = float(summary.loc[summary["method"] == "exp_points", "total_points"].iloc[0])
    xp_total = float(summary.loc[summary["method"] == "xp", "total_points"].iloc[0])
    ms = float(summary.loc[summary["method"] == "ridge_ms", "total_points"].iloc[0])
    s1 = float(summary.loc[summary["method"] == "ridge_1s", "total_points"].iloc[0])

    lines = [
        "# Stage 16 — Multi-season walk-forward Ridge (season climb)",
        "",
        "Natural next step after stage 15: train Ridge on **prior seasons + earlier "
        "2025/26 GWs**, predict each 2025/26 GW, run the locked stripped XI climb.",
        "",
        f"- Train seasons: {', '.join(s for s, _ in SEASONS)}",
        f"- Eval: **{EVAL_SEASON}** GWs **{scored_gws[0]}–{scored_gws[-1]}** (n={len(scored_gws)})",
        f"- Min train rows: **{MIN_TRAIN_ROWS}**",
        f"- Rows by season: "
        + ", ".join(f"{k}={v}" for k, v in n_by_season.items()),
        "",
        "Within-season expanding priors only (player keys are `season:element`). "
        "DefCon features are 0 before 2025/26.",
        "",
        "## Final standings (captain ×2, aligned GWs)",
        "",
        "| method | total | mean/GW | vs exp | vs xP | vs ridge_1s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - base:+.0f} | {r.total_points - xp_total:+.0f} | "
            f"{r.total_points - s1:+.0f} |"
        )

    if ms > max(xp_total, s1, base) + 10:
        verdict = (
            f"PASS — ridge_ms best "
            f"(vs 1-season {ms - s1:+.0f}, vs xP {ms - xp_total:+.0f}, vs exp {ms - base:+.0f})"
        )
    elif ms > s1 + 10:
        verdict = f"PARTIAL — multi-season helps vs 1-season ({ms - s1:+.0f}) but check xP/exp"
    elif ms >= s1:
        verdict = f"WEAK — ridge_ms ≈ ridge_1s ({ms - s1:+.0f})"
    else:
        verdict = f"FAIL — multi-season trails 1-season Ridge ({ms - s1:+.0f})"

    lines += [
        "",
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_ridge_ms.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_ridge_ms.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print("Building multi-season features…")
    all_feat = build_all_seasons()
    n_by_season = all_feat.groupby("season").size().to_dict()

    print("Walk-forward Ridge (multi-season)…")
    pred_ms, scored_ms = walk_forward_ridge(all_feat, multi_season=True)
    print("Walk-forward Ridge (2025/26 only)…")
    pred_1s, scored_1s = walk_forward_ridge(all_feat, multi_season=False)

    # Align to intersection of scored GWs
    scored = sorted(set(scored_ms) & set(scored_1s))
    if not scored:
        raise RuntimeError("No overlapping scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat["score_ridge_ms"] = pred_ms.loc[eval_feat.index]
    eval_feat["score_ridge_1s"] = pred_1s.loc[eval_feat.index]

    extra = {
        "ridge_ms": "score_ridge_ms",
        "ridge_1s": "score_ridge_1s",
    }
    weekly = run_season(eval_feat, extra_score_cols=extra, gws=scored)
    counts = weekly.groupby("method")["gw"].nunique()
    keep = counts[counts >= len(scored)].index.tolist()
    weekly = weekly.loc[weekly["method"].isin(keep)].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_ridge_ms.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_ridge_ms.png")
    write_report(
        REPORTS / "stage_16_ridge_multiseason.md",
        summary,
        weekly,
        scored,
        n_by_season,
    )
    return {
        "summary": summary,
        "weekly": weekly,
        "scored_gws": scored,
        "n_by_season": n_by_season,
    }


if __name__ == "__main__":
    out = run()
    print("\nRows by season:", out["n_by_season"])
    print(f"GWs: {out['scored_gws'][0]}–{out['scored_gws'][-1]} (n={len(out['scored_gws'])})")
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_16_ridge_multiseason.md")
