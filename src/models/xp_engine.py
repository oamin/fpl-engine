"""Stage 13 — Thin xP engine + long-horizon Spearman IC gate.

Deterministic component xP (no sims):

  xP = xp_appear + xp_goals + xp_assists + xp_cs + xp_defcon
       + xp_saves + xp_bps − xp_deductions

  appear     = P(play>0)·1 + P(play≥60)·1   (probabilistic step)
  goals/ast  = share × λ × pts              (no play_scale dampener)
  CS/DefCon  = P60 × rate × pts
  saves      = P60 · (λ_conceded · 2.0) / 3   (GKP only; 1pt / 3 saves)
  bps        ≈ 0.18·xp_goals + 0.12·xp_assists + 0.08·xp_cs
  deductions = P60·λ_conceded/2 (GKP/DEF) + (xMi/90)·0.15 (YC)

Horizon blend (multi-GW scorers only; not inside compute_xp):
  w_h = max(floor, γ^h);  score_h = w_h·xp + (1−w_h)·exp_points

λ_team / P(CS) from football-data Shin 1X2 + OU (no Odds API).
Shares / DefCon from expanding history (leakage-free).

Gate: Spearman IC of xP vs mean points over next H∈{1,3,5,8}
vs baselines: exp points, roll3 points, xMi, value (price).

Writes:
  data/processed/xp_engine.csv
  data/processed/xp_ic_gate.csv
  data/plots/xp_ic_gate.png
  reports/stage_13_xp_engine.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rules.fpl_2026 import CS_POINTS as CS_PTS
from src.rules.fpl_2026 import GOAL_POINTS as GOAL_PTS

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

POS_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
HORIZONS = (1, 3, 5, 8)
MIN_HISTORY = 3
ROLL_XMI = 3
MIN_MINUTES = 60.0
DEFCON_THRESH = {"DEF": 10.0, "MID": 12.0, "FWD": 12.0}
# Goal and clean-sheet awards are the rules-module maps. A goalkeeper goal is 6.
# Same-season team pots with no history use these fixed neutrals. They are not
# estimated from the season being scored.
NEUTRAL_TEAM_XG = 1.40
NEUTRAL_TEAM_XA = 1.05
# Empirical ~2.0 saves per GC for 60'+ GKs (25/26); 1 FPL pt per 3 saves.
SAVES_PER_GC = 2.0
# FWD goal under-projection: leakage-free position scale clipped to this band.
FWD_GOAL_SCALE_LO = 0.90
FWD_GOAL_SCALE_HI = 1.55
FWD_GOAL_SCALE_MIN_N = 25
# Additive FWD residual correction. Helps bias/stripped; hurts FT (stage 26).
# Default off — pure xP v2 + goal-scale only for transfer paths.
FWD_LEVEL_CAL = False
FWD_LEVEL_ADD_LO = -0.25
FWD_LEVEL_ADD_HI = 1.25
# Multi-GW score blend: w_h = max(BLEND_FLOOR, BLEND_GAMMA**h)
BLEND_GAMMA = 0.85
BLEND_FLOOR = 0.5
# Light fade toward team-strength prior (only inside multi-GW V).
TEAM_FADE_HOLD_H = 3          # w=1 for h <= 3
TEAM_FADE_END_H = 5           # reach end weight by h=5
TEAM_FADE_END_W = 0.60


def horizon_blend_weight(
    h: int, *, gamma: float = BLEND_GAMMA, floor: float = BLEND_FLOOR
) -> float:
    """Weight on xP at horizon step h (0 = current GW)."""
    return float(max(floor, gamma**h))


def team_prior_fade_weight(
    h: int,
    *,
    hold_h: int = TEAM_FADE_HOLD_H,
    end_h: int = TEAM_FADE_END_H,
    end_w: float = TEAM_FADE_END_W,
) -> float:
    """Weight on fixture xP vs team-strength prior; 1.0 through hold_h, then linear fade."""
    if h <= hold_h:
        return 1.0
    if h >= end_h:
        return float(end_w)
    return float(1.0 + (end_w - 1.0) * (h - hold_h) / max(1, end_h - hold_h))


def blend_xp_exp(
    xp: float,
    exp_points: float,
    h: int,
    *,
    gamma: float = BLEND_GAMMA,
    floor: float = BLEND_FLOOR,
    schedule: str = "gamma",
) -> float:
    """Leakage-free horizon blend of decision-time xP and a second score."""
    if schedule == "team_fade":
        w = team_prior_fade_weight(h)
    else:
        w = horizon_blend_weight(h, gamma=gamma, floor=floor)
    return float(w * xp + (1.0 - w) * exp_points)


def _roll_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _exp_mean(s: pd.Series) -> pd.Series:
    return s.shift(1).expanding(min_periods=1).mean()


def _spearman(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    n = int(y.size)
    out = {
        "n": float(n),
        "spearman": float("nan"),
        "pearson": float("nan"),
        "top_bottom": float("nan"),
        "mean_y": float("nan"),
        "mean_pred": float("nan"),
    }
    if n < 40:
        return out
    out["mean_y"] = float(y.mean())
    out["mean_pred"] = float(pred.mean())
    if np.std(pred) > 0 and np.std(y) > 0:
        out["pearson"] = float(np.corrcoef(pred, y)[0, 1])
        out["spearman"] = float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
    tmp = pd.DataFrame({"x": pred, "y": y})
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True)["y"].mean()
        if len(g) >= 2:
            out["top_bottom"] = float(g.iloc[-1] - g.iloc[0])
    except ValueError:
        pass
    return out


def load_joined(season: str = "2025_26") -> pd.DataFrame:
    """Appearances joined to odds + price/defcon from Vaastav."""
    pm = pd.read_csv(PROCESSED / "player_matches.csv")
    raw = pd.read_csv(CACHE / f"merged_gw_{season}.csv")
    extra = pd.DataFrame(
        {
            "player_id": raw["element"].astype(str),
            "gw": pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce"),
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            "defcon_raw": pd.to_numeric(
                raw["defensive_contribution"], errors="coerce"
            ).fillna(0.0),
            "xGC": pd.to_numeric(raw["expected_goals_conceded"], errors="coerce").fillna(
                0.0
            ),
        }
    ).dropna(subset=["gw"])
    extra["gw"] = extra["gw"].astype(int)
    extra = extra.drop_duplicates(["player_id", "gw"], keep="first")

    df = pm.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    df = df.dropna(subset=["gw", "fixture_id", "position"]).copy()
    df["gw"] = df["gw"].astype(int)
    df = df.merge(extra, on=["player_id", "gw"], how="left")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["defcon_raw"] = df["defcon_raw"].fillna(0.0)
    df["xG"] = pd.to_numeric(df["xG"], errors="coerce").fillna(0.0)
    df["xA"] = pd.to_numeric(df["xA"], errors="coerce").fillna(0.0)
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0.0)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0.0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0.0)
    df["p_not_lose"] = pd.to_numeric(df["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        df["p_draw"], errors="coerce"
    )
    df["attack_strength"] = pd.to_numeric(df["attack_strength"], errors="coerce")
    df["defend_threat"] = pd.to_numeric(df["defend_threat"], errors="coerce")
    df["p_over"] = pd.to_numeric(df["p_over25"], errors="coerce")
    df["p_under"] = pd.to_numeric(df["p_under25"], errors="coerce")
    thr = df["position"].map(DEFCON_THRESH)
    df["defcon_hit"] = (
        df["position"].isin(DEFCON_THRESH)
        & (df["minutes"] >= MIN_MINUTES)
        & (df["defcon_raw"] >= thr)
    ).astype(float)
    return df


def add_market_pots(df: pd.DataFrame) -> pd.DataFrame:
    """Implied team λ and market CS proxy from 1X2 + OU 2.5."""
    out = df.copy()
    # Expected match goals from OU (centered ~2.45 when over≈under).
    p_o = out["p_over"].fillna(0.5)
    p_u = out["p_under"].fillna(0.5)
    out["e_total"] = 2.45 + 1.10 * (p_o - p_u)

    att = out["attack_strength"].fillna(0.5)
    threat = out["defend_threat"].fillna(0.5)
    denom = (att + threat).replace(0, np.nan)
    out["lam_scored"] = out["e_total"] * att / denom
    out["lam_scored"] = out["lam_scored"].fillna(out["e_total"] * 0.5).clip(0.2, 3.5)

    # Assists pot: scale scored λ by typical xA/xG ratio (~0.7 of attack share channel)
    out["lam_assist"] = (out["lam_scored"] * 0.75).clip(0.1, 3.0)

    # Market CS proxy: favourite tilt + under (no direct CS odds on free feed).
    # Tuned to land near empirical CS base ~0.25; ranking-driven for IC.
    pnl = out["p_not_lose"].fillna(0.5)
    out["p_cs_mkt"] = np.clip(0.08 + 0.35 * pnl + 0.15 * p_u, 0.05, 0.55)
    return out


def add_player_priors(
    df: pd.DataFrame, fill_from: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Shift-1 priors. A missing prior stays missing unless ``fill_from`` supplies it.

    The fill frame is an earlier season. A position mean of the season being
    scored includes later gameweeks, so that fill is not used. A team with no
    shifted history takes ``NEUTRAL_TEAM_XG`` and ``NEUTRAL_TEAM_XA``.
    """
    out = df.sort_values(["player_id", "gw", "date"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    out["n_prior"] = g.cumcount()
    out["xmi"] = g["minutes"].transform(lambda s: _roll_mean(s, ROLL_XMI))
    out["exp_points"] = g["total_points"].transform(_exp_mean)
    out["roll3_points"] = g["total_points"].transform(lambda s: _roll_mean(s, 3))
    out["exp_xG"] = g["xG"].transform(_exp_mean)
    out["exp_xA"] = g["xA"].transform(_exp_mean)
    out["exp_defcon_hit"] = g["defcon_hit"].transform(_exp_mean)

    for col, src in (
        ("xmi", "minutes"),
        ("exp_points", "total_points"),
        ("roll3_points", "total_points"),
        ("exp_xG", "xG"),
        ("exp_xA", "xA"),
        ("exp_defcon_hit", "defcon_hit"),
    ):
        if fill_from is not None:
            pos_mean = fill_from.groupby("position")[src].mean()
            out[col] = out[col].fillna(out["position"].map(pos_mean))

    # Team expanding attack (sum of player xG per team-fixture, then team expanding mean)
    played = out.loc[out["minutes"] > 0]
    team_fix = played.groupby(
        ["team_norm", "fixture_id", "date", "gw"], as_index=False
    ).agg(team_xg=("xG", "sum"), team_xa=("xA", "sum"))
    team_fix = team_fix.sort_values(["team_norm", "date", "gw"], kind="mergesort")
    tg = team_fix.groupby("team_norm", sort=False)
    team_fix["exp_team_xg"] = tg["team_xg"].transform(_exp_mean)
    team_fix["exp_team_xa"] = tg["team_xa"].transform(_exp_mean)
    if fill_from is None:
        team_fix["exp_team_xg"] = team_fix["exp_team_xg"].fillna(NEUTRAL_TEAM_XG)
        team_fix["exp_team_xa"] = team_fix["exp_team_xa"].fillna(NEUTRAL_TEAM_XA)
    else:
        played_fill = fill_from.loc[
            pd.to_numeric(fill_from["minutes"], errors="coerce").fillna(0) > 0
        ]
        prior_teams = played_fill.groupby(
            ["team_norm", "fixture_id"], as_index=False
        ).agg(team_xg=("xG", "sum"), team_xa=("xA", "sum"))
        team_fix["exp_team_xg"] = team_fix["exp_team_xg"].fillna(
            float(prior_teams["team_xg"].mean()) if len(prior_teams) else np.nan
        )
        team_fix["exp_team_xa"] = team_fix["exp_team_xa"].fillna(
            float(prior_teams["team_xa"].mean()) if len(prior_teams) else np.nan
        )

    out = out.merge(
        team_fix[["team_norm", "fixture_id", "exp_team_xg", "exp_team_xa"]],
        on=["team_norm", "fixture_id"],
        how="left",
    )
    if fill_from is None:
        out["exp_team_xg"] = out["exp_team_xg"].fillna(NEUTRAL_TEAM_XG)
        out["exp_team_xa"] = out["exp_team_xa"].fillna(NEUTRAL_TEAM_XA)
    else:
        out["exp_team_xg"] = out["exp_team_xg"].fillna(float(fill_from["xG"].mean()) * 8)
        out["exp_team_xa"] = out["exp_team_xa"].fillna(float(fill_from["xA"].mean()) * 8)

    out["share_xG"] = (out["exp_xG"] / out["exp_team_xg"].replace(0, np.nan)).clip(0, 1)
    out["share_xA"] = (out["exp_xA"] / out["exp_team_xa"].replace(0, np.nan)).clip(0, 1)
    out["share_xG"] = out["share_xG"].fillna(0.0)
    out["share_xA"] = out["share_xA"].fillna(0.0)
    return out


def compute_xp(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized xP: FPL-aligned appearance, FWD goal cal, BPS + deductions."""
    out = df.copy()
    xmi = out["xmi"].to_numpy(float)
    pos = out["position"]

    # Probabilistic step appearance: 1pt for play>0, +1pt for ≥60'.
    p_play = np.clip(xmi / 15.0, 0.0, 1.0)
    p60 = np.where(xmi >= 30.0, np.clip((xmi - 30.0) / 30.0, 0.0, 1.0), 0.0)
    xp_appear = p_play * 1.0 + p60 * 1.0

    goal_pts = pos.map(GOAL_PTS).fillna(4.0).to_numpy(float)
    cs_pts = pos.map(CS_PTS).fillna(0.0).to_numpy(float)

    # Shares already minute-adjusted vs team; do not apply play_scale again.
    e_goals = out["share_xG"].to_numpy(float) * out["lam_scored"].to_numpy(float)
    e_assists = out["share_xA"].to_numpy(float) * out["lam_assist"].to_numpy(float)

    xp_goals = e_goals * goal_pts
    xp_assists = e_assists * 3.0
    xp_cs = p60 * out["p_cs_mkt"].to_numpy(float) * cs_pts
    xp_defcon = np.where(
        pos.isin(["DEF", "MID", "FWD"]),
        p60 * out["exp_defcon_hit"].to_numpy(float) * 2.0,
        0.0,
    )

    # Deductions + GKP saves share λ_conceded (no floor inflate).
    e_total = out["e_total"].to_numpy(float)
    lam_scored = out["lam_scored"].to_numpy(float)
    lam_conceded = np.clip(e_total - lam_scored, 0.0, 5.0)

    # GKP save points: E[saves] ≈ λ_c · 2.0; 1 pt per 3 saves; gated by p60.
    expected_saves = np.where(pos == "GKP", lam_conceded * SAVES_PER_GC, 0.0)
    xp_saves = np.where(pos == "GKP", p60 * (expected_saves / 3.0), 0.0)

    out["p_play"] = p_play
    out["p60"] = p60
    out["lam_conceded"] = lam_conceded
    out["expected_saves"] = expected_saves
    out["e_goals"] = e_goals
    out["xp_appear"] = xp_appear
    out["xp_goals"] = xp_goals
    out["xp_assists"] = xp_assists
    out["xp_cs"] = xp_cs
    out["xp_defcon"] = xp_defcon
    out["xp_saves"] = xp_saves

    # Leakage-free FWD goal scale, then rebuild BPS + total.
    out = _apply_fwd_goal_calibration(out)
    xp_goals = out["xp_goals"].to_numpy(float)
    xp_assists = out["xp_assists"].to_numpy(float)
    xp_cs = out["xp_cs"].to_numpy(float)

    xp_bps = 0.18 * xp_goals + 0.12 * xp_assists + 0.08 * xp_cs
    xp_gc_loss = np.where(pos.isin(["GKP", "DEF"]), p60 * (lam_conceded / 2.0), 0.0)
    xp_card_loss = (xmi / 90.0) * 0.15
    xp_deductions = xp_gc_loss + xp_card_loss

    out["xp_bps"] = xp_bps
    out["xp_gc_loss"] = xp_gc_loss
    out["xp_card_loss"] = xp_card_loss
    out["xp_deductions"] = xp_deductions
    out["xp"] = (
        out["xp_appear"].to_numpy(float)
        + xp_goals
        + xp_assists
        + xp_cs
        + out["xp_defcon"].to_numpy(float)
        + out["xp_saves"].to_numpy(float)
        + xp_bps
        - xp_deductions
    )
    out = _apply_fwd_level_calibration(out) if FWD_LEVEL_CAL else out
    if "fwd_level_add" not in out.columns:
        out["fwd_level_add"] = 0.0
    out["baseline_value"] = out["value"].astype(float)
    return out


def _apply_fwd_goal_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Scale FWD xp_goals by position expanding mean(goals)/mean(e_goals), shift-1.

    Preserves the input row order (required: callers hold aligned arrays).
    """
    out = df.copy()
    out["fwd_goal_scale"] = 1.0
    is_fwd = out["position"].to_numpy() == "FWD"
    if not is_fwd.any():
        return out

    # Chronological order for expanding stats, then map scale back to original index.
    order = out.sort_values(["gw", "date", "player_id"], kind="mergesort").index.to_numpy()
    if "goals" in out.columns:
        goals = pd.to_numeric(out.loc[order, "goals"], errors="coerce").fillna(0.0)
    else:
        goals = pd.Series(np.zeros(len(order)), index=order)
    e_goals = pd.to_numeric(out.loc[order, "e_goals"], errors="coerce").fillna(0.0)
    fwd_mask = out.loc[order, "position"].to_numpy() == "FWD"
    fwd_order = order[fwd_mask]

    g_c = np.cumsum(goals.loc[fwd_order].to_numpy(float))
    e_c = np.cumsum(e_goals.loc[fwd_order].to_numpy(float))
    g_prior = np.concatenate([[np.nan], g_c[:-1]])
    e_prior = np.concatenate([[np.nan], e_c[:-1]])
    n_prior = np.arange(len(fwd_order), dtype=float)
    raw = np.divide(g_prior, e_prior, out=np.ones_like(g_prior), where=(e_prior > 1e-6))
    raw = np.where(n_prior >= FWD_GOAL_SCALE_MIN_N, raw, 1.0)
    raw = np.clip(raw, FWD_GOAL_SCALE_LO, FWD_GOAL_SCALE_HI)
    raw = np.where(np.isfinite(raw), raw, 1.0)

    scale = pd.Series(1.0, index=out.index)
    scale.loc[fwd_order] = raw
    out["fwd_goal_scale"] = scale.to_numpy(float)
    out["xp_goals"] = out["xp_goals"].to_numpy(float) * out["fwd_goal_scale"].to_numpy(float)
    return out


def _apply_fwd_level_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free FWD level correction: expanding mean(points − xP), shift-1."""
    out = df.copy()
    out["fwd_level_add"] = 0.0
    is_fwd = out["position"].to_numpy() == "FWD"
    if not is_fwd.any() or "total_points" not in out.columns:
        return out

    order = out.sort_values(["gw", "date", "player_id"], kind="mergesort").index.to_numpy()
    fwd_order = order[out.loc[order, "position"].to_numpy() == "FWD"]
    y = pd.to_numeric(out.loc[fwd_order, "total_points"], errors="coerce").fillna(0.0).to_numpy(float)
    pred = pd.to_numeric(out.loc[fwd_order, "xp"], errors="coerce").fillna(0.0).to_numpy(float)
    resid = y - pred
    c = np.cumsum(resid)
    prior = np.concatenate([[np.nan], c[:-1]])
    n_prior = np.arange(len(fwd_order), dtype=float)
    level = np.divide(prior, np.maximum(n_prior, 1.0))
    level = np.where(n_prior >= FWD_GOAL_SCALE_MIN_N, level, 0.0)
    level = np.clip(np.where(np.isfinite(level), level, 0.0), FWD_LEVEL_ADD_LO, FWD_LEVEL_ADD_HI)

    add = pd.Series(0.0, index=out.index)
    add.loc[fwd_order] = level
    out["fwd_level_add"] = add.to_numpy(float)
    out["xp"] = out["xp"].to_numpy(float) + out["fwd_level_add"].to_numpy(float)
    return out


def add_team_prior_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fixture-agnostic team-strength xP using expanding team λ / CS priors.

    Used only as the outer-horizon blend target (not the primary score).
    """
    out = df.sort_values(["team_norm", "gw", "date"], kind="mergesort").copy()
    g = out.groupby("team_norm", sort=False)
    out["team_lam_scored"] = g["lam_scored"].transform(_exp_mean)
    out["team_lam_assist"] = g["lam_assist"].transform(_exp_mean)
    out["team_p_cs"] = g["p_cs_mkt"].transform(_exp_mean)
    out["team_e_total"] = g["e_total"].transform(_exp_mean)
    for col, src in (
        ("team_lam_scored", "lam_scored"),
        ("team_lam_assist", "lam_assist"),
        ("team_p_cs", "p_cs_mkt"),
        ("team_e_total", "e_total"),
    ):
        out[col] = out[col].fillna(out[src])

    pos = out["position"]
    p60 = out["p60"].to_numpy(float)
    goal_pts = pos.map(GOAL_PTS).fillna(4.0).to_numpy(float)
    cs_pts = pos.map(CS_PTS).fillna(0.0).to_numpy(float)

    e_goals = out["share_xG"].to_numpy(float) * out["team_lam_scored"].to_numpy(float)
    e_assists = out["share_xA"].to_numpy(float) * out["team_lam_assist"].to_numpy(float)
    xp_goals = e_goals * goal_pts * out["fwd_goal_scale"].to_numpy(float)
    xp_assists = e_assists * 3.0
    xp_cs = p60 * out["team_p_cs"].to_numpy(float) * cs_pts
    xp_defcon = out["xp_defcon"].to_numpy(float)
    lam_c = np.clip(
        out["team_e_total"].to_numpy(float) - out["team_lam_scored"].to_numpy(float),
        0.0,
        5.0,
    )
    xp_saves = np.where(pos == "GKP", p60 * (lam_c * SAVES_PER_GC / 3.0), 0.0)
    xp_bps = 0.18 * xp_goals + 0.12 * xp_assists + 0.08 * xp_cs
    xp_gc = np.where(pos.isin(["GKP", "DEF"]), p60 * (lam_c / 2.0), 0.0)
    xp_card = out["xp_card_loss"].to_numpy(float)
    out["xp_team_prior"] = (
        out["xp_appear"].to_numpy(float)
        + xp_goals
        + xp_assists
        + xp_cs
        + xp_defcon
        + xp_saves
        + xp_bps
        - xp_gc
        - xp_card
        + out.get("fwd_level_add", pd.Series(0.0, index=out.index)).to_numpy(float)
    )
    out["score_team_prior"] = out["xp_team_prior"]
    return out


def add_forward_means(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    for H in HORIZONS:
        fwd = g["total_points"].transform(
            lambda s, h=H: s.iloc[::-1].rolling(h, min_periods=h).mean().iloc[::-1]
        )
        out[f"fwd{H}_mean_points"] = fwd
    out["has_fwd8"] = out["fwd8_mean_points"].notna()
    return out


def ic_gate(df: pd.DataFrame) -> list[dict[str, Any]]:
    base = df.loc[(df["n_prior"] >= MIN_HISTORY) & df["has_fwd8"]].copy()
    universes = {
        "ALL": base,
        "regulars_xmi45": base.loc[base["xmi"] >= 45],
        "regulars_xmi60": base.loc[base["xmi"] >= 60],
    }
    predictors = [
        ("xp", "xP engine"),
        ("exp_points", "exp points"),
        ("roll3_points", "roll3 points"),
        ("xmi", "xMi"),
        ("baseline_value", "price (value)"),
        ("exp_xG", "exp xG only"),
        ("exp_defcon_hit", "exp DefCon hit only"),
        ("p_cs_mkt", "p_cs_mkt only"),
    ]
    rows: list[dict[str, Any]] = []
    for univ_name, univ in universes.items():
        for H in HORIZONS:
            y = univ[f"fwd{H}_mean_points"].to_numpy(float)
            for col, label in predictors:
                m = _spearman(y, univ[col].to_numpy(float))
                rows.append(
                    {
                        "universe": univ_name,
                        "horizon": H,
                        "label": label,
                        "predictor": col,
                        **m,
                    }
                )
            # Position slices for xP vs exp points
            for pos in ("GKP", "DEF", "MID", "FWD"):
                chunk = univ.loc[univ["position"] == pos]
                if len(chunk) < 40:
                    continue
                for col, label in (("xp", "xP engine"), ("exp_points", "exp points")):
                    m = _spearman(
                        chunk[f"fwd{H}_mean_points"].to_numpy(float),
                        chunk[col].to_numpy(float),
                    )
                    rows.append(
                        {
                            "universe": f"{univ_name}|{pos}",
                            "horizon": H,
                            "label": label,
                            "predictor": col,
                            **m,
                        }
                    )
    return rows


def plot_ic(rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle("xP IC gate — Spearman vs mean points over next H GWs", fontsize=12)

    for ax, univ, title in zip(
        axes,
        ("regulars_xmi45", "ALL"),
        ("Regulars (xMi≥45)", "ALL joined appearances"),
    ):
        sub = df.loc[df["universe"] == univ]
        for label, color in (
            ("xP engine", "crimson"),
            ("exp points", "steelblue"),
            ("xMi", "seagreen"),
            ("price (value)", "gray"),
            ("roll3 points", "orange"),
        ):
            s = sub.loc[sub["label"] == label].sort_values("horizon")
            if s.empty:
                continue
            ax.plot(
                s["horizon"],
                s["spearman"],
                "o-",
                color=color,
                lw=2 if label == "xP engine" else 1.4,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("horizon H")
        ax.set_ylabel("Spearman IC")
        ax.set_xticks(list(HORIZONS))
        ax.set_ylim(-0.05, 0.75)
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=7, loc="lower right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], n: int) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage 13 — Thin xP engine + Spearman IC gate",
        "",
        "Deterministic component xP from **xMi + market λ/CS proxy + expanding "
        "shares + DefCon hit rate**. No sims, no Odds API props.",
        "",
        f"- Joined player-match rows: **{n}**",
        "- Gate: Spearman IC vs **mean points over next H GWs**",
        "- Baselines: exp points, roll3 points, xMi, price",
        "",
        "## Formula (v2)",
        "",
        "```",
        "p_play   = clip(xMi/15, 0, 1)",
        "p60      = clip((xMi−30)/30, 0, 1) if xMi≥30 else 0",
        "appear   = p_play·1 + p60·1",
        "λ        ≈ e_total(OU) × attack_strength / (att+threat)",
        "P(CS)    ≈ clip(0.08 + 0.35·p_not_lose + 0.15·p_under)",
        "goal_pts = GOAL_POINTS from src/rules/fpl_2026.py ({GKP:6, DEF:6, MID:5, FWD:4})",
        "xP       = appear",
        "         + share_xG · λ · goal_pts          (no play_scale)",
        "         + share_xA · λ_a · 3",
        "         + p60 · P(CS) · cs_pts",
        "         + p60 · P(DefCon hit) · 2",
        "         + p60 · (λ_c · 2.0)/3              (GKP saves)",
        "         + 0.18·xp_goals + 0.12·xp_assists + 0.08·xp_cs   (BPS proxy)",
        "         − p60 · λ_conceded/2 (GKP/DEF) − (xMi/90)·0.15   (YC)",
        "```",
        "",
        "## IC — regulars xMi≥45",
        "",
        "| predictor | H=1 | H=3 | H=5 | H=8 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = [
        "xP engine",
        "exp points",
        "roll3 points",
        "xMi",
        "price (value)",
        "exp xG only",
        "exp DefCon hit only",
    ]
    for label in labels:
        vals = []
        for H in HORIZONS:
            hit = df.loc[
                (df["universe"] == "regulars_xmi45")
                & (df["label"] == label)
                & (df["horizon"] == H)
            ]
            vals.append(f"{float(hit.iloc[0]['spearman']):.3f}" if not hit.empty else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## IC — ALL appearances",
        "",
        "| predictor | H=1 | H=3 | H=5 | H=8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("xP engine", "exp points", "xMi", "price (value)"):
        vals = []
        for H in HORIZONS:
            hit = df.loc[
                (df["universe"] == "ALL") & (df["label"] == label) & (df["horizon"] == H)
            ]
            vals.append(f"{float(hit.iloc[0]['spearman']):.3f}" if not hit.empty else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## xP vs exp points by position (regulars xMi≥45, H=5)",
        "",
        "| pos | xP Spearman | exp pts Spearman | Δ |",
        "|---|---:|---:|---:|",
    ]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        a = df.loc[
            (df["universe"] == f"regulars_xmi45|{pos}")
            & (df["label"] == "xP engine")
            & (df["horizon"] == 5)
        ]
        b = df.loc[
            (df["universe"] == f"regulars_xmi45|{pos}")
            & (df["label"] == "exp points")
            & (df["horizon"] == 5)
        ]
        if a.empty or b.empty:
            continue
        sa, sb = float(a.iloc[0]["spearman"]), float(b.iloc[0]["spearman"])
        lines.append(f"| {pos} | {sa:.3f} | {sb:.3f} | {sa - sb:+.3f} |")

    # Gate verdict
    xp8 = df.loc[
        (df["universe"] == "regulars_xmi45")
        & (df["label"] == "xP engine")
        & (df["horizon"] == 8)
    ]
    exp8 = df.loc[
        (df["universe"] == "regulars_xmi45")
        & (df["label"] == "exp points")
        & (df["horizon"] == 8)
    ]
    verdict = "inconclusive"
    if not xp8.empty and not exp8.empty:
        dx = float(xp8.iloc[0]["spearman"]) - float(exp8.iloc[0]["spearman"])
        if dx >= 0.03:
            verdict = f"PASS — xP beats exp points by {dx:+.3f} IC at H=8"
        elif dx >= 0.0:
            verdict = f"WEAK PASS — xP edges exp points by {dx:+.3f} IC at H=8"
        else:
            verdict = f"FAIL — xP trails exp points by {dx:+.3f} IC at H=8"

    lines += [
        "",
        f"## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "Pass bar: xP Spearman IC ≥ exp points at H=5 and H=8 on regulars "
        "(prefer Δ ≥ 0.03 before investing in MILP).",
        "",
        "## Plots",
        "",
        "- `data/plots/xp_ic_gate.png`",
        "",
        "## Output",
        "",
        "- `data/processed/xp_engine.csv` — per player-match xP + components",
        "- `data/processed/xp_ic_gate.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_joined()
    feat = add_market_pots(raw)
    feat = add_player_priors(feat)
    feat = compute_xp(feat)
    feat = add_forward_means(feat)
    rows = ic_gate(feat)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    keep = [
        "player_id",
        "player_name",
        "team_norm",
        "position",
        "gw",
        "date",
        "fixture_id",
        "minutes",
        "total_points",
        "value",
        "xmi",
        "xp",
        "xp_appear",
        "xp_goals",
        "xp_assists",
        "xp_cs",
        "xp_defcon",
        "xp_saves",
        "xp_bps",
        "xp_deductions",
        "xp_gc_loss",
        "xp_card_loss",
        "lam_scored",
        "lam_conceded",
        "p_cs_mkt",
        "p_play",
        "p60",
        "share_xG",
        "share_xA",
        "exp_defcon_hit",
        "exp_points",
        "roll3_points",
        "n_prior",
    ]
    feat[keep].to_csv(PROCESSED / "xp_engine.csv", index=False)
    pd.DataFrame(rows).to_csv(PROCESSED / "xp_ic_gate.csv", index=False)
    plot_ic(rows, PLOTS / "xp_ic_gate.png")
    write_report(REPORTS / "stage_13_xp_engine.md", rows, len(feat))
    return {"rows": rows, "n": len(feat)}


if __name__ == "__main__":
    out = run()
    df = pd.DataFrame(out["rows"])
    print(f"n={out['n']}")
    print("\nSpearman IC — regulars xMi≥45")
    for label in (
        "xP engine",
        "exp points",
        "roll3 points",
        "xMi",
        "price (value)",
        "exp xG only",
    ):
        xs = []
        for H in HORIZONS:
            hit = df.loc[
                (df.universe == "regulars_xmi45")
                & (df.label == label)
                & (df.horizon == H)
            ]
            xs.append(
                f"H{H}={float(hit.iloc[0]['spearman']):.3f}" if not hit.empty else f"H{H}=—"
            )
        print(f"  {label:22s}  " + "  ".join(xs))
    print(f"\nWrote {REPORTS}/stage_13_xp_engine.md")
