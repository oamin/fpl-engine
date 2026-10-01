"""Stage 11 — Trial: team λ × share / Poisson CS / DefCon P(hit).

Condition on starters (mins ≥ 60). Compare to roll3 player baselines.

Channels:
  G/A:  E[G] = λ_scored × share_xG ;  E[A] = λ_assist × share_xA
  CS:   P(CS) = exp(−λ_conceded)  (Poisson; team grain → GKP/DEF)
  DefCon: P(hit threshold | start) from player prior hit rate

λ from leakage-free team roll3 xG / xGC, opponent-adjusted.

Writes:
  data/processed/channel_trial.csv
  data/plots/channel_trial_ga.png
  data/plots/channel_trial_cs_defcon.png
  reports/stage_11_channel_trial.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

POS_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
ROLL = 3
MIN_HISTORY = 3
MIN_MINUTES = 60.0
DEFCON_THRESH = {"DEF": 10.0, "MID": 12.0, "FWD": 12.0}


def _roll_mean(s: pd.Series, window: int = ROLL) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    n = int(y.size)
    out = {
        "n": float(n),
        "mae": float("nan"),
        "r2": float("nan"),
        "corr": float("nan"),
        "decile_corr": float("nan"),
        "top_bottom": float("nan"),
        "brier": float("nan"),
        "mean_y": float("nan"),
        "mean_pred": float("nan"),
    }
    if n < 30:
        return out
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["mae"] = float(np.mean(np.abs(err)))
    out["r2"] = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    out["corr"] = (
        float(np.corrcoef(pred, y)[0, 1])
        if np.std(pred) > 0 and np.std(y) > 0
        else float("nan")
    )
    out["mean_y"] = float(y.mean())
    out["mean_pred"] = float(pred.mean())
    # Brier when y is 0/1
    if set(np.unique(y)).issubset({0.0, 1.0}):
        p = np.clip(pred, 0.0, 1.0)
        out["brier"] = float(np.mean((p - y) ** 2))
    tmp = pd.DataFrame({"x": pred, "y": y})
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        if len(g) >= 4 and g["x"].std() > 0 and g["y"].std() > 0:
            out["decile_corr"] = float(g["x"].corr(g["y"]))
        if len(g) >= 2:
            out["top_bottom"] = float(g["y"].iloc[-1] - g["y"].iloc[0])
    except ValueError:
        pass
    return out


def load_base(season: str = "2025_26") -> pd.DataFrame:
    """player_matches + xGC / defcon from Vaastav cache."""
    pm = pd.read_csv(PROCESSED / "player_matches.csv")
    raw = pd.read_csv(CACHE / f"merged_gw_{season}.csv")
    extra = pd.DataFrame(
        {
            "player_id": raw["element"].astype(str),
            "gw": pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce"),
            "xGC": pd.to_numeric(raw["expected_goals_conceded"], errors="coerce").fillna(
                0.0
            ),
            "defcon_raw": pd.to_numeric(
                raw["defensive_contribution"], errors="coerce"
            ).fillna(0.0),
        }
    ).dropna(subset=["gw"])
    extra["gw"] = extra["gw"].astype(int)
    extra = extra.drop_duplicates(["player_id", "gw"], keep="first")

    df = pm.copy()
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    df = df.dropna(subset=["gw", "fixture_id", "position"]).copy()
    df["gw"] = df["gw"].astype(int)
    df["player_id"] = df["player_id"].astype(str)
    df = df.merge(extra, on=["player_id", "gw"], how="left")
    df["xGC"] = df["xGC"].fillna(0.0)
    df["defcon_raw"] = df["defcon_raw"].fillna(0.0)
    df["xG"] = pd.to_numeric(df["xG"], errors="coerce").fillna(0.0)
    df["xA"] = pd.to_numeric(df["xA"], errors="coerce").fillna(0.0)
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0.0)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0.0)
    df["cs"] = pd.to_numeric(df["clean_sheets"], errors="coerce").fillna(0.0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    is_home = df["is_home"].astype(int) == 1
    df["team_cs"] = np.where(
        is_home,
        pd.to_numeric(df["away_goals"], errors="coerce") == 0,
        pd.to_numeric(df["home_goals"], errors="coerce") == 0,
    ).astype(float)
    df["p_not_lose"] = pd.to_numeric(df["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        df["p_draw"], errors="coerce"
    )
    thr = df["position"].map(DEFCON_THRESH)
    df["defcon_hit"] = (
        (df["position"].isin(DEFCON_THRESH))
        & (df["minutes"] >= MIN_MINUTES)
        & (df["defcon_raw"] >= thr)
    ).astype(float)
    return df


def build_team_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team×fixture with attack/defence proxies."""
    played = df.loc[df["minutes"] > 0].copy()
    g = played.groupby(
        ["team_norm", "fixture_id", "date", "gw", "is_home"], as_index=False
    ).agg(
        team_xg=("xG", "sum"),
        team_xa=("xA", "sum"),
        team_goals=("goals", "sum"),
        team_assists=("assists", "sum"),
        team_cs=("team_cs", "max"),
    )
    starters = df.loc[df["minutes"] >= MIN_MINUTES]
    xgc = (
        starters.groupby(["team_norm", "fixture_id"], as_index=False)["xGC"]
        .median()
        .rename(columns={"xGC": "team_xgc"})
    )
    g = g.merge(xgc, on=["team_norm", "fixture_id"], how="left")
    g["team_xgc"] = g["team_xgc"].fillna(g["team_xgc"].median())

    g = g.sort_values(["team_norm", "date", "gw"], kind="mergesort")
    tg = g.groupby("team_norm", sort=False)
    g["n_prior_team"] = tg.cumcount()
    g["roll_att"] = tg["team_xg"].transform(_roll_mean)
    g["roll_xa"] = tg["team_xa"].transform(_roll_mean)
    g["roll_def"] = tg["team_xgc"].transform(_roll_mean)
    g["roll_cs"] = tg["team_cs"].transform(_roll_mean)
    for col, src in (
        ("roll_att", "team_xg"),
        ("roll_xa", "team_xa"),
        ("roll_def", "team_xgc"),
        ("roll_cs", "team_cs"),
    ):
        g[col] = g[col].fillna(g[src].mean())
    return g


def add_match_lambda(team: pd.DataFrame, fixtures_players: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjusted λ_scored / λ_conceded; Poisson P(CS)."""
    # Map fixture → both sides' rolls
    sides = team[
        [
            "team_norm",
            "fixture_id",
            "is_home",
            "roll_att",
            "roll_xa",
            "roll_def",
            "roll_cs",
            "n_prior_team",
            "team_cs",
            "team_xg",
            "team_xa",
        ]
    ].copy()
    # Opponent = other team on same fixture
    opp = sides.rename(
        columns={
            "team_norm": "opp_norm",
            "roll_att": "opp_att",
            "roll_xa": "opp_xa",
            "roll_def": "opp_def",
            "is_home": "opp_is_home",
        }
    )[["fixture_id", "opp_norm", "opp_att", "opp_xa", "opp_def", "opp_is_home"]]
    # Each fixture has 2 teams; merge where opp is the other
    a = sides.merge(opp, on="fixture_id", how="left")
    a = a.loc[a["team_norm"] != a["opp_norm"]].copy()
    # If duplicate (shouldn't with 2 teams), keep matching is_home opposite
    a = a.loc[a["is_home"].astype(int) != a["opp_is_home"].astype(int)].copy()
    a = a.drop_duplicates(["team_norm", "fixture_id"], keep="first")

    league_att = float(a["roll_att"].mean())
    league_def = float(a["roll_def"].mean())
    league_att = league_att if league_att > 0 else 1.0
    league_def = league_def if league_def > 0 else 1.0

    # Dixon–Coles-ish: λ = att × opp_def / league_def  (scoring)
    #                 μ = def × opp_att / league_att  (conceding)
    home_boost = np.where(a["is_home"].astype(int) == 1, 1.08, 1.0)
    a["lam_scored"] = a["roll_att"] * (a["opp_def"] / league_def) * home_boost
    a["lam_conceded"] = a["roll_def"] * (a["opp_att"] / league_att) / np.where(
        a["is_home"].astype(int) == 1, 1.08, 1.0
    )
    a["lam_assist"] = a["roll_xa"] * (a["opp_def"] / league_def) * home_boost
    a["p_cs_pois"] = np.exp(-np.clip(a["lam_conceded"], 0.05, 5.0))
    return a


def add_player_shares(df: pd.DataFrame, team_lam: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw", "date"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    out["n_prior"] = g.cumcount()
    out["roll_xG"] = g["xG"].transform(_roll_mean)
    out["roll_xA"] = g["xA"].transform(_roll_mean)
    out["roll_goals"] = g["goals"].transform(_roll_mean)
    out["roll_assists"] = g["assists"].transform(_roll_mean)
    out["roll_cs"] = g["cs"].transform(_roll_mean)
    # DefCon hit rate among prior starts (mins≥60 only in rate — use all rows with hit 0 if <60)
    out["roll_defcon_hit"] = g["defcon_hit"].transform(_roll_mean)
    # Expanding start-conditioned hit rate
    def _expand_hit_given_start(s: pd.Series) -> pd.Series:
        # s is defcon_hit aligned; need minutes from group — handled below differently
        return s.shift(1).expanding().mean()

    out["prior_defcon_hit"] = g["defcon_hit"].transform(_expand_hit_given_start)

    for col, src in (
        ("roll_xG", "xG"),
        ("roll_xA", "xA"),
        ("roll_goals", "goals"),
        ("roll_assists", "assists"),
        ("roll_cs", "cs"),
        ("roll_defcon_hit", "defcon_hit"),
        ("prior_defcon_hit", "defcon_hit"),
    ):
        pos_mean = out.groupby("position")[src].transform("mean")
        out[col] = out[col].fillna(pos_mean)

    # Team roll att on same fixture for share denominator
    t = team_lam[
        [
            "team_norm",
            "fixture_id",
            "lam_scored",
            "lam_conceded",
            "lam_assist",
            "p_cs_pois",
            "roll_att",
            "roll_xa",
            "n_prior_team",
        ]
    ].drop_duplicates(["team_norm", "fixture_id"])
    out = out.merge(t, on=["team_norm", "fixture_id"], how="inner")

    # Share = player roll / team roll (attack)
    out["share_xG"] = out["roll_xG"] / out["roll_att"].replace(0, np.nan)
    out["share_xA"] = out["roll_xA"] / out["roll_xa"].replace(0, np.nan)
    # Clip shares — sum of rolls ≠ team roll exactly; allow >1 then renormalize soft
    out["share_xG"] = out["share_xG"].clip(0, 1.0).fillna(0.0)
    out["share_xA"] = out["share_xA"].clip(0, 1.0).fillna(0.0)

    out["pred_goals"] = out["lam_scored"] * out["share_xG"]
    out["pred_assists"] = out["lam_assist"] * out["share_xA"]
    out["pred_gi"] = out["pred_goals"] + out["pred_assists"]
    out["pred_cs"] = out["p_cs_pois"]  # team P; for GKP/DEF
    out["pred_defcon"] = out["prior_defcon_hit"].clip(0, 1)

    return out


def evaluate_fixed(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    df["gi"] = df["goals"] + df["assists"]
    starters = df.loc[
        (df["minutes"] >= MIN_MINUTES) & (df["n_prior"] >= MIN_HISTORY)
    ].copy()
    rows: list[dict[str, Any]] = []

    def add(chunk: pd.DataFrame, pred: str, y: str, label: str, scope: str) -> None:
        if len(chunk) < 30:
            return
        m = _metrics(chunk[y].to_numpy(float), chunk[pred].to_numpy(float))
        rows.append(
            {
                "label": label,
                "predictor": pred,
                "target": y,
                "scope": scope,
                "filter": "mins>=60",
                **m,
            }
        )

    scopes_ga = {
        "MID+FWD": starters["position"].isin(["MID", "FWD"]),
        "MID": starters["position"] == "MID",
        "FWD": starters["position"] == "FWD",
    }
    for scope, mask in scopes_ga.items():
        chunk = starters.loc[mask]
        add(chunk, "pred_goals", "goals", "λ×share_xG → goals", scope)
        add(chunk, "roll_xG", "goals", "roll3 xG → goals (base)", scope)
        add(chunk, "roll_goals", "goals", "roll3 goals → goals (base)", scope)
        add(chunk, "pred_assists", "assists", "λ×share_xA → assists", scope)
        add(chunk, "roll_xA", "assists", "roll3 xA → assists (base)", scope)
        add(chunk, "pred_gi", "gi", "λ×share → G+A", scope)
        add(chunk, "roll_xG", "gi", "roll3 xG → G+A (loose base)", scope)
        add(chunk, "roll_goals", "gi", "roll3 goals → G+A (base)", scope)

    # CS — GKP/DEF
    for scope, mask in (
        ("GKP+DEF", starters["position"].isin(["GKP", "DEF"])),
        ("GKP", starters["position"] == "GKP"),
        ("DEF", starters["position"] == "DEF"),
    ):
        chunk = starters.loc[mask]
        add(chunk, "pred_cs", "cs", "Poisson P(CS) → CS", scope)
        add(chunk, "p_not_lose", "cs", "p_not_lose → CS (base)", scope)
        add(chunk, "roll_cs", "cs", "roll3 player CS → CS (base)", scope)
        add(chunk, "roll_cs_team", "cs", "roll3 team CS → CS (base)", scope)

    # DefCon hit — DEF/MID (FWD rare)
    for scope, mask in (
        ("DEF", starters["position"] == "DEF"),
        ("MID", starters["position"] == "MID"),
        ("DEF+MID", starters["position"].isin(["DEF", "MID"])),
    ):
        chunk = starters.loc[mask]
        add(chunk, "pred_defcon", "defcon_hit", "prior P(DefCon hit) → hit", scope)
        add(chunk, "roll_defcon_hit", "defcon_hit", "roll3 DefCon hit → hit (base)", scope)

    return rows


def _panel(ax, x, y, title, xlabel, binary=False) -> None:
    m = _metrics(y, x)
    ax.scatter(x, y, s=10, alpha=0.15, edgecolors="none", color="steelblue")
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=6, label="decile")
    except ValueError:
        pass
    ax.axhline(float(np.nanmean(y)), color="gray", ls=":", lw=1)
    extra = f"Brier={m['brier']:.3f}" if binary and np.isfinite(m["brier"]) else f"Δ={m['top_bottom']:.2f}"
    ax.set_title(f"{title}\ncorr={m['corr']:.3f} dcorr={m['decile_corr']:.3f} {extra}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("realised")
    if binary:
        ax.set_ylim(-0.05, 1.05)


def plot_ga(starters: pd.DataFrame, out: Path) -> None:
    midfwd = starters.loc[starters["position"].isin(["MID", "FWD"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Trial: λ×share vs roll3 (MID+FWD, mins≥60)", fontsize=13)
    panels = [
        (0, 0, "pred_goals", "goals", "λ×share → goals", False),
        (0, 1, "roll_xG", "goals", "roll3 xG → goals", False),
        (1, 0, "pred_gi", "gi", "λ×share → G+A", False),
        (1, 1, "roll_xG", "gi", "roll3 xG → G+A", False),
    ]
    for r, c, px, py, title, _ in panels:
        _panel(
            axes[r][c],
            midfwd[px].to_numpy(float),
            midfwd[py].to_numpy(float),
            title,
            px,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_cs_defcon(starters: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Trial: Poisson CS + DefCon P(hit) (mins≥60)", fontsize=13)
    gkd = starters.loc[starters["position"].isin(["GKP", "DEF"])]
    defm = starters.loc[starters["position"].isin(["DEF", "MID"])]
    _panel(axes[0][0], gkd["pred_cs"].to_numpy(float), gkd["cs"].to_numpy(float),
           "Poisson P(CS) → CS", "p_cs_pois", binary=True)
    _panel(axes[0][1], gkd["p_not_lose"].to_numpy(float), gkd["cs"].to_numpy(float),
           "p_not_lose → CS", "p_not_lose", binary=True)
    _panel(axes[1][0], defm["pred_defcon"].to_numpy(float), defm["defcon_hit"].to_numpy(float),
           "prior P(DefCon hit)", "P(hit)", binary=True)
    _panel(axes[1][1], defm["roll_defcon_hit"].to_numpy(float), defm["defcon_hit"].to_numpy(float),
           "roll3 DefCon hit", "roll3 hit rate", binary=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage 11 — Trial: team λ × share / Poisson CS / DefCon",
        "",
        "Conditioned on **minutes ≥ 60**. Leakage-free team roll3 xG/xGC → "
        "opponent-adjusted λ; player share of team attack; Poisson P(CS); "
        "expanding DefCon threshold hit rate.",
        "",
        "## Goals / assists (MID+FWD)",
        "",
        "| pair | n | corr | decile_corr | Δ top−bot | MAE | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[df["scope"] == "MID+FWD"].itertuples():
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.corr:.3f} | {r.decile_corr:.3f} | "
            f"{r.top_bottom:.3f} | {r.mae:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )

    lines += [
        "",
        "## Clean sheets (GKP+DEF)",
        "",
        "| pair | n | corr | decile_corr | Brier | Δ top−bot | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[df["scope"] == "GKP+DEF"].itertuples():
        brier = f"{r.brier:.3f}" if np.isfinite(r.brier) else "nan"
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.corr:.3f} | {r.decile_corr:.3f} | "
            f"{brier} | {r.top_bottom:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )

    lines += [
        "",
        "## DefCon hit (DEF+MID)",
        "",
        "| pair | n | corr | decile_corr | Brier | Δ top−bot | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[df["scope"] == "DEF+MID"].itertuples():
        brier = f"{r.brier:.3f}" if np.isfinite(r.brier) else "nan"
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.corr:.3f} | {r.decile_corr:.3f} | "
            f"{brier} | {r.top_bottom:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )

    lines += [
        "",
        "### By position (headline predictors)",
        "",
        "| scope | pair | corr | decile_corr | Δ / Brier |",
        "|---|---|---:|---:|---:|",
    ]
    headlines = {
        "MID": "λ×share_xG → goals",
        "FWD": "λ×share_xG → goals",
        "GKP": "Poisson P(CS) → CS",
        "DEF": "Poisson P(CS) → CS",
        "DEF": "prior P(DefCon hit) → hit",  # duplicate key - fix
    }
    # manual rows
    for scope, label in [
        ("MID", "λ×share_xG → goals"),
        ("FWD", "λ×share_xG → goals"),
        ("GKP", "Poisson P(CS) → CS"),
        ("DEF", "Poisson P(CS) → CS"),
        ("DEF", "prior P(DefCon hit) → hit"),
        ("MID", "prior P(DefCon hit) → hit"),
    ]:
        hit = df.loc[(df["scope"] == scope) & (df["label"] == label)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        metric = f"Brier={float(r['brier']):.3f}" if np.isfinite(r["brier"]) else f"Δ={float(r['top_bottom']):.3f}"
        lines.append(
            f"| {scope} | {label} | {float(r['corr']):.3f} | {float(r['decile_corr']):.3f} | {metric} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/channel_trial_ga.png`",
        "- `data/plots/channel_trial_cs_defcon.png`",
        "",
        "## Output",
        "",
        "- `data/processed/channel_trial.csv`",
        "",
        "## Read",
        "",
        "- Win = trial corr/decile/Brier clearly beats roll3 / p_not_lose baselines.",
        "- Modest lift still useful for EV ranking among known starters.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    base = load_base()
    team = build_team_fixtures(base)
    team_lam = add_match_lambda(team, base)
    # team roll_cs onto players as roll_cs_team
    feat = add_player_shares(base, team_lam)
    feat = feat.merge(
        team_lam[["team_norm", "fixture_id", "roll_cs"]].rename(
            columns={"roll_cs": "roll_cs_team"}
        ),
        on=["team_norm", "fixture_id"],
        how="left",
    )
    feat["gi"] = feat["goals"] + feat["assists"]
    rows = evaluate_fixed(feat)
    starters = feat.loc[
        (feat["minutes"] >= MIN_MINUTES) & (feat["n_prior"] >= MIN_HISTORY)
    ].copy()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROCESSED / "channel_trial.csv", index=False)
    keep = [
        "player_id",
        "player_name",
        "team_norm",
        "position",
        "gw",
        "fixture_id",
        "minutes",
        "goals",
        "assists",
        "gi",
        "cs",
        "defcon_hit",
        "pred_goals",
        "pred_assists",
        "pred_gi",
        "pred_cs",
        "pred_defcon",
        "lam_scored",
        "lam_conceded",
        "share_xG",
        "share_xA",
        "roll_xG",
        "roll_xA",
        "p_not_lose",
    ]
    feat.loc[feat["minutes"] >= MIN_MINUTES, keep].to_csv(
        PROCESSED / "channel_trial_features.csv", index=False
    )
    plot_ga(starters, PLOTS / "channel_trial_ga.png")
    plot_cs_defcon(starters, PLOTS / "channel_trial_cs_defcon.png")
    write_report(REPORTS / "stage_11_channel_trial.md", rows)
    return {"rows": rows, "n_starters": len(starters)}


if __name__ == "__main__":
    out = run()
    df = pd.DataFrame(out["rows"])
    print(f"starters(backtest)={out['n_starters']}")
    for scope in ("MID+FWD", "GKP+DEF", "DEF+MID"):
        print(f"\n=== {scope} ===")
        for r in df.loc[df.scope == scope].itertuples():
            extra = f"Brier={r.brier:.3f}" if np.isfinite(r.brier) else f"Δ={r.top_bottom:.3f}"
            print(
                f"  {r.label:36s}  corr={r.corr:.3f}  dcorr={r.decile_corr:.3f}  {extra}"
            )
    print(f"\nWrote {REPORTS}/stage_11_channel_trial.md")
