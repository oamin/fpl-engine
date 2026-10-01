"""Stage 9 — Shin / OU priors + roll3 CS → next-match clean sheets.

Focus: GKP/DEF (CS points matter). Leakage-free roll3; fixture markets are
pre-match (football-data closing-ish).

Writes:
  data/processed/cs_shin_backtest.csv
  data/processed/cs_shin_features.csv
  data/plots/cs_shin_team.png
  data/plots/cs_shin_player.png
  reports/stage_9_cs_shin.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

ROLL = 3
MIN_HISTORY = 3
MIN_MINUTES = 60.0


def _roll_mean(s: pd.Series, window: int = ROLL) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    n = int(y.size)
    if n < 20:
        return {
            "n": float(n),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "corr": float("nan"),
            "decile_corr": float("nan"),
            "mean_y": float("nan"),
            "mean_pred": float("nan"),
        }
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = (
        float(np.corrcoef(pred, y)[0, 1])
        if np.std(pred) > 0 and np.std(y) > 0
        else float("nan")
    )
    decile_corr = float("nan")
    tmp = pd.DataFrame({"x": pred, "y": y})
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        if len(g) >= 4 and g["x"].std() > 0 and g["y"].std() > 0:
            decile_corr = float(g["x"].corr(g["y"]))
    except ValueError:
        pass
    return {
        "n": float(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "corr": corr,
        "decile_corr": decile_corr,
        "mean_y": float(y.mean()),
        "mean_pred": float(pred.mean()),
    }


def load_players() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "player_matches.csv")
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    df = df.dropna(subset=["gw", "position", "player_id", "fixture_id"]).copy()
    df["gw"] = df["gw"].astype(int)
    is_home = df["is_home"].astype(int) == 1
    # Team kept a clean sheet (fixture scoreline).
    df["team_cs"] = np.where(
        is_home,
        pd.to_numeric(df["away_goals"], errors="coerce") == 0,
        pd.to_numeric(df["home_goals"], errors="coerce") == 0,
    ).astype(float)
    df["cs"] = pd.to_numeric(df["clean_sheets"], errors="coerce").fillna(0.0)
    df["p_not_lose"] = pd.to_numeric(df["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        df["p_draw"], errors="coerce"
    )
    df["p_under"] = pd.to_numeric(df["p_under25"], errors="coerce")
    df["p_cs_proxy"] = 1.0 - pd.to_numeric(df["p_lose"], errors="coerce")  # 1 - P(lose)
    df["defend_ease"] = 1.0 - pd.to_numeric(df["defend_threat"], errors="coerce")
    return df


def add_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw", "date"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    out["n_prior"] = g.cumcount()
    out["roll3_cs"] = g["cs"].transform(_roll_mean)
    pos_mean = out.groupby("position")["cs"].transform("mean")
    out["roll3_cs"] = out["roll3_cs"].fillna(pos_mean)

    # Team CS rate (one value per team×fixture; roll over team history).
    team = (
        out.drop_duplicates(["team_norm", "fixture_id"])
        .sort_values(["team_norm", "date", "gw"], kind="mergesort")
        .copy()
    )
    tg = team.groupby("team_norm", sort=False)
    team["roll3_team_cs"] = tg["team_cs"].transform(_roll_mean)
    team_mean = team["team_cs"].mean()
    team["roll3_team_cs"] = team["roll3_team_cs"].fillna(team_mean)
    out = out.merge(
        team[["team_norm", "fixture_id", "roll3_team_cs"]],
        on=["team_norm", "fixture_id"],
        how="left",
    )
    return out


def team_grain(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team×fixture (market → team CS)."""
    cols = [
        "fixture_id",
        "team_norm",
        "date",
        "gw",
        "is_home",
        "team_cs",
        "p_not_lose",
        "p_under",
        "p_cs_proxy",
        "defend_ease",
        "p_lose",
        "roll3_team_cs",
    ]
    return df[cols].drop_duplicates(["team_norm", "fixture_id"]).copy()


def evaluate(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # --- Team grain ---
    team = team_grain(df)
    team_pairs = [
        ("p_under", "team_cs", "p_under → team CS"),
        ("p_not_lose", "team_cs", "p_not_lose → team CS"),
        ("p_cs_proxy", "team_cs", "1−p_lose → team CS"),
        ("defend_ease", "team_cs", "1−defend_threat → team CS"),
        ("roll3_team_cs", "team_cs", "roll3 team CS → team CS"),
    ]
    for pred, y, label in team_pairs:
        m = _metrics(team[y].to_numpy(float), team[pred].to_numpy(float))
        rows.append(
            {
                "grain": "team",
                "label": label,
                "predictor": pred,
                "target": y,
                "filter": "ALL",
                "scope": "TEAM",
                **m,
            }
        )

    # --- Player grain: GKP/DEF ---
    def_pos = df.loc[df["position"].isin(["GKP", "DEF"])].copy()
    hist = def_pos.loc[def_pos["n_prior"] >= MIN_HISTORY]
    starters = hist.loc[hist["minutes"] >= MIN_MINUTES]

    player_pairs = [
        ("p_under", "cs", "p_under → CS"),
        ("p_not_lose", "cs", "p_not_lose → CS"),
        ("p_cs_proxy", "cs", "1−p_lose → CS"),
        ("defend_ease", "cs", "1−defend_threat → CS"),
        ("roll3_cs", "cs", "roll3 player CS → CS"),
        ("roll3_team_cs", "cs", "roll3 team CS → CS"),
    ]
    for pred, y, label in player_pairs:
        for filt_name, chunk in (
            ("ALL_def", hist),
            ("mins>=60", starters),
        ):
            m = _metrics(chunk[y].to_numpy(float), chunk[pred].to_numpy(float))
            rows.append(
                {
                    "grain": "player",
                    "label": label,
                    "predictor": pred,
                    "target": y,
                    "filter": filt_name,
                    "scope": "GKP+DEF",
                    **m,
                }
            )
            for pos in ("GKP", "DEF"):
                pos_chunk = chunk.loc[chunk["position"] == pos]
                if len(pos_chunk) < 30:
                    continue
                mp = _metrics(
                    pos_chunk[y].to_numpy(float), pos_chunk[pred].to_numpy(float)
                )
                rows.append(
                    {
                        "grain": "player",
                        "label": label,
                        "predictor": pred,
                        "target": y,
                        "filter": filt_name,
                        "scope": pos,
                        **mp,
                    }
                )
    return rows


def _decile_plot(ax, x: np.ndarray, y: np.ndarray, title: str, xlabel: str) -> None:
    m = _metrics(y, x)
    ax.scatter(x, y, s=10, alpha=0.18, edgecolors="none", color="steelblue")
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=6, label="decile")
    except ValueError:
        pass
    ax.axhline(float(np.nanmean(y)), color="gray", ls=":", lw=1, label="mean y")
    ax.set_title(
        f"{title}\nR²={m['r2']:.3f} corr={m['corr']:.3f} dcorr={m['decile_corr']:.3f}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CS (0/1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper left")


def plot_team(team: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Market / roll3 → team clean sheet (fixture grain)", fontsize=13)
    panels = [
        (0, 0, "p_under", "p_under → team CS"),
        (0, 1, "p_not_lose", "p_not_lose → team CS"),
        (1, 0, "p_cs_proxy", "1−p_lose → team CS"),
        (1, 1, "roll3_team_cs", "roll3 team CS → team CS"),
    ]
    for r, c, col, title in panels:
        _decile_plot(
            axes[r][c],
            team[col].to_numpy(float),
            team["team_cs"].to_numpy(float),
            title,
            col,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_player(starters: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        "Priors → player CS (GKP+DEF, minutes ≥ 60)", fontsize=13
    )
    panels = [
        (0, 0, "p_under", "p_under → CS"),
        (0, 1, "p_not_lose", "p_not_lose → CS"),
        (1, 0, "roll3_cs", "roll3 player CS → CS"),
        (1, 1, "roll3_team_cs", "roll3 team CS → CS"),
    ]
    for r, c, col, title in panels:
        _decile_plot(
            axes[r][c],
            starters[col].to_numpy(float),
            starters["cs"].to_numpy(float),
            title,
            col,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], n_players: int, n_team: int) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage 9 — Shin / OU + roll3 → clean sheets",
        "",
        "Fixture markets (Shin 1X2 / under 2.5) and leakage-free roll-3 CS rates "
        "vs realised team / player clean sheets.",
        "",
        f"- Player-match rows (joined): **{n_players}**",
        f"- Team×fixture rows: **{n_team}**",
        f"- Player backtest: GKP+DEF, ≥ {MIN_HISTORY} prior apps; starters = minutes ≥ {MIN_MINUTES:.0f}",
        "",
        "## Team grain (market → team CS)",
        "",
        "| pair | n | MAE | R² | corr | decile_corr | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[df["grain"] == "team"].itertuples():
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | "
            f"{r.corr:.3f} | {r.decile_corr:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )

    lines += [
        "",
        "## Player grain — GKP+DEF, minutes ≥ 60",
        "",
        "| pair | n | MAE | R² | corr | decile_corr | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[
        (df["grain"] == "player")
        & (df["scope"] == "GKP+DEF")
        & (df["filter"] == "mins>=60")
    ].itertuples():
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | "
            f"{r.corr:.3f} | {r.decile_corr:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )

    lines += [
        "",
        "## By position (mins ≥ 60) — best market vs roll3",
        "",
        "| pos | pair | n | corr | decile_corr | R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pos in ("GKP", "DEF"):
        for label in ("p_under → CS", "p_not_lose → CS", "roll3 player CS → CS"):
            hit = df.loc[
                (df["scope"] == pos)
                & (df["filter"] == "mins>=60")
                & (df["label"] == label)
            ]
            if hit.empty:
                continue
            r = hit.iloc[0]
            lines.append(
                f"| {pos} | {label} | {int(r['n'])} | {float(r['corr']):.3f} | "
                f"{float(r['decile_corr']):.3f} | {float(r['r2']):.3f} |"
            )

    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/cs_shin_team.png`",
        "- `data/plots/cs_shin_player.png`",
        "",
        "## Output",
        "",
        "- `data/processed/cs_shin_backtest.csv`",
        "- `data/processed/cs_shin_features.csv`",
        "",
        "## Read",
        "",
        "- Markets are **not** on CS-probability scale → prefer **corr / decile_corr** over R²/MAE.",
        "- If market corr ≫ roll3 CS, keep Shin/OU as the CS channel prior.",
        "- If both weak (corr ≲ 0.15), CS is mostly noise given current features.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_players()
    feat = add_roll_features(raw)
    rows = evaluate(feat)
    team = team_grain(feat)
    starters = feat.loc[
        (feat["position"].isin(["GKP", "DEF"]))
        & (feat["n_prior"] >= MIN_HISTORY)
        & (feat["minutes"] >= MIN_MINUTES)
    ]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROCESSED / "cs_shin_backtest.csv", index=False)
    keep = [
        "player_id",
        "player_name",
        "team_norm",
        "position",
        "gw",
        "date",
        "fixture_id",
        "minutes",
        "cs",
        "team_cs",
        "n_prior",
        "p_under",
        "p_not_lose",
        "p_cs_proxy",
        "defend_ease",
        "p_lose",
        "roll3_cs",
        "roll3_team_cs",
        "total_points",
    ]
    feat[keep].to_csv(PROCESSED / "cs_shin_features.csv", index=False)

    plot_team(team, PLOTS / "cs_shin_team.png")
    plot_player(starters, PLOTS / "cs_shin_player.png")
    write_report(REPORTS / "stage_9_cs_shin.md", rows, len(raw), len(team))
    return {"rows": rows, "n": len(raw), "n_team": len(team), "n_starters": len(starters)}


if __name__ == "__main__":
    out = run()
    df = pd.DataFrame(out["rows"])
    print(f"players={out['n']} team_fixtures={out['n_team']} starters={out['n_starters']}")
    print("\n=== TEAM grain ===")
    for r in df.loc[df.grain == "team"].itertuples():
        print(
            f"  {r.label:32s}  R²={r.r2:.3f}  corr={r.corr:.3f}  "
            f"dcorr={r.decile_corr:.3f}  MAE={r.mae:.3f}  ȳ={r.mean_y:.3f}"
        )
    print("\n=== GKP+DEF mins≥60 ===")
    show = df.loc[(df.scope == "GKP+DEF") & (df["filter"] == "mins>=60")]
    for r in show.itertuples():
        print(
            f"  {r.label:32s}  R²={r.r2:.3f}  corr={r.corr:.3f}  "
            f"dcorr={r.decile_corr:.3f}  MAE={r.mae:.3f}  ȳ={r.mean_y:.3f}"
        )
    print(f"\nWrote {REPORTS}/stage_9_cs_shin.md")
