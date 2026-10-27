"""``score_xp_v2`` — Numagent 2.0 primary player-GW score assembler.

Part of ``protocol.numagent_2_0`` (supersedes Numagent 1.0 and ``score_xp``).
Spec: ``reports/numagent_2_0_method.md`` · index: ``reports/LOCKED.md``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.markets import fpl_score_rates, scoreline
from src.markets.player_attack import player_attack_from_m
from src.rules.fpl_2026 import (
    APPEARANCE_60,
    APPEARANCE_UNDER_60,
    CS_POINTS,
    DEFCON_POINTS,
    DEFCON_THRESHOLD,
    GOALS_CONCEDED_DEDUCTION,
    MINUTES_FOR_APPEARANCE_BONUS,
    normalize_position,
    save_points,
)

# Canonical score name for comparisons / climbs (not ``numagent_xp``).
SCORE_NAME = "score_xp_v2"
SIG_FLOOR = 0.5
# Early-season rate n (protocol.score_xp_v2_rate_m0, promoted 2026-10-16).
RATE_M0 = 270.0
RATE_SE_FLOOR = 0.25


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def appearance_xp(m: float) -> float:
    if m <= 0:
        return 0.0
    if m < MINUTES_FOR_APPEARANCE_BONUS:
        return float(APPEARANCE_UNDER_60)
    return float(APPEARANCE_60)


def weight_and_delta(r: float, dr: float, m: float, sigma_m: float) -> tuple[float, float]:
    """w = r·m/90, Δw from independent rate and minutes errors.

    Used for assist shares and rate×minutes quantities. Goal shares use
    ``goal_share_weight`` (rate only; locked 2026-10-27).
    """
    r = float(max(r, 0.0))
    dr = float(max(dr, 0.0))
    m = float(m)
    sm = float(max(sigma_m, 0.0))
    w = r * m / 90.0
    dw = math.sqrt((m / 90.0 * dr) ** 2 + (r / 90.0 * sm) ** 2)
    return w, dw


def goal_share_weight(r: float, dr: float = 0.0) -> tuple[float, float]:
    """Goal-share weight: ``w = r`` (minutes gate the club, not the weight).

    Locked 2026-10-27 (``cascade_share_no_renorm_minutes``): compositional
    ``π ∝ r`` among ``m>0``. Minutes-weighted ``w = r·m/90`` over-concentrated
    π on high-m XI attackers (OL resid +10.2 → +0.2).
    """
    r = float(max(r, 0.0))
    dr = float(max(dr, 0.0))
    return r, dr


def shares_and_deltas(w: np.ndarray, dw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renormalised shares with dependent-sum SE (method §6)."""
    w = np.asarray(w, dtype=float)
    dw = np.asarray(dw, dtype=float)
    n = w.size
    W = float(w.sum())
    if W <= 0.0 or n == 0:
        return np.zeros(n), np.zeros(n)
    s = w / W
    ds = np.zeros(n)
    sum_dw2 = float(np.sum(dw**2))
    for i in range(n):
        others = sum_dw2 - float(dw[i] ** 2)
        ds[i] = (1.0 / W) * math.sqrt(
            (1.0 - s[i]) ** 2 * float(dw[i] ** 2) + (s[i] ** 2) * max(others, 0.0)
        )
    return s, ds


# Locked position×minutes prior weights (closed-season per-90 rates, MID=1).
# Pooled 2022-23…2025-26 appearance rows; not fit on the live holdout.
SHARE_POS_WEIGHT_G: Mapping[str, float] = {
    "GKP": 0.0,
    "DEF": 0.255,
    "MID": 1.0,
    "FWD": 2.467,
}
SHARE_POS_WEIGHT_A: Mapping[str, float] = {
    "GKP": 0.042,
    "DEF": 0.396,
    "MID": 1.0,
    "FWD": 0.954,
}


def shrink_shares_by_se(
    s: np.ndarray,
    ds: np.ndarray,
    m: np.ndarray,
    *,
    tau2: float | None = None,
    tau2_floor: float = 1e-4,
    prior_support: str = "positive_share",
    positions: Sequence[str] | None = None,
    channel: str = "G",
) -> tuple[np.ndarray, dict[str, float]]:
    """Shrink compositional shares toward a minutes prior using share SE.

    Pre-registered variants:

    - ``prior_support='position_minutes'``: ``s0 ∝ m · w(pos)`` with locked
      per-90 position weights (goals vs assists via ``channel``). Pulls noisy
      early-season CB goal mass toward a role baseline without inventing
      equal minutes share for every player.
    - ``prior_support='positive_share'``: prior only on raw ``s>0`` (parked).
    - ``prior_support='all_minutes'``: ``s0 = m/Σm`` on all ``m>0`` (parked;
      DEF/GK leak).

    ``τ²`` = median of ``Δs²`` on playing rows with ``Δs>0``, else ``tau2_floor``.
    ``K_i = τ² / (τ² + Δs_i²)``; ``s̃ = Ks + (1−K)s0``; renormalise on ``m>0``.
    """
    s = np.asarray(s, dtype=float).copy()
    ds = np.asarray(ds, dtype=float)
    m = np.asarray(m, dtype=float)
    n = s.size
    support = str(prior_support or "positive_share")
    if support not in {"positive_share", "all_minutes", "position_minutes"}:
        raise ValueError(
            "prior_support must be 'positive_share', 'all_minutes', or "
            f"'position_minutes', got {prior_support!r}"
        )
    ch = str(channel or "G").upper()
    if ch not in {"G", "A"}:
        raise ValueError(f"channel must be 'G' or 'A', got {channel!r}")
    diag: dict[str, float] = {
        "n": float(n),
        "tau2": float("nan"),
        "mean_k": float("nan"),
        "max_abs_delta": 0.0,
        "prior_support": {
            "positive_share": 1.0,
            "all_minutes": 0.0,
            "position_minutes": 2.0,
        }[support],
    }
    if n == 0:
        return s, diag
    playing = m > 0.0
    if not np.any(playing):
        return np.zeros(n), diag
    m_play = np.where(playing, m, 0.0)
    if support == "position_minutes":
        if positions is None or len(positions) != n:
            raise ValueError("position_minutes prior requires positions of length n")
        weights = SHARE_POS_WEIGHT_G if ch == "G" else SHARE_POS_WEIGHT_A
        mass = np.zeros(n)
        for i in range(n):
            if not playing[i]:
                continue
            pos = normalize_position(str(positions[i]))
            mass[i] = float(m[i]) * float(weights.get(pos, 0.0))
        m_sum = float(mass.sum())
        s0 = mass / m_sum if m_sum > 0 else np.zeros(n)
    elif support == "positive_share":
        prior_mask = playing & (s > 0.0)
        m_prior = np.where(prior_mask, m, 0.0)
        m_sum = float(m_prior.sum())
        s0 = m_prior / m_sum if m_sum > 0 else np.zeros(n)
    else:
        m_sum = float(m_play.sum())
        s0 = m_play / m_sum if m_sum > 0 else np.zeros(n)

    ds2 = np.where(playing, np.maximum(ds, 0.0) ** 2, np.nan)
    finite = ds2[np.isfinite(ds2) & (ds2 > 0)]
    if tau2 is None:
        tau2 = float(np.median(finite)) if finite.size else float(tau2_floor)
    tau2 = float(max(tau2, tau2_floor))
    diag["tau2"] = tau2

    k = np.ones(n)
    for i in range(n):
        if not playing[i]:
            k[i] = 0.0
            continue
        k[i] = tau2 / (tau2 + float(ds2[i]) if np.isfinite(ds2[i]) else tau2)
    s_hat = k * s + (1.0 - k) * s0
    if support == "positive_share":
        s_hat = np.where(s > 0.0, s_hat, 0.0)
    s_hat = np.where(playing, np.maximum(s_hat, 0.0), 0.0)
    z = float(s_hat.sum())
    if z > 0:
        s_hat = s_hat / z
    diag["mean_k"] = float(k[playing].mean()) if np.any(playing) else float("nan")
    diag["max_abs_delta"] = float(np.max(np.abs(s_hat - s))) if n else 0.0
    return s_hat, diag


def prior_mean_se(values: np.ndarray) -> tuple[float, float, int]:
    """Expanding-style prior μ and SE=σ/√n on a finished sample."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return 0.0, 0.0, 0
    mu = float(v.mean())
    if n == 1:
        return mu, 0.0, 1
    sig = float(v.std(ddof=1))
    se = sig / math.sqrt(n)
    return mu, se, n


def defcon_xp(
    mu_count: float,
    sig_count: float,
    n_prior: int,
    tau: float,
    m: float,
) -> tuple[float, float, float]:
    """Return (xp_defcon, se, P(hit)) from expanding count residual."""
    if m < MINUTES_FOR_APPEARANCE_BONUS or tau <= 0 or int(n_prior) <= 0:
        return 0.0, 0.0, 0.0
    sig = max(float(sig_count), SIG_FLOOR)
    delta = float(mu_count) - float(tau)
    z = (-0.5 - delta) / sig
    p_hit = float(np.clip(1.0 - _norm_cdf(z), 0.0, 1.0))
    xp = float(DEFCON_POINTS) * p_hit
    se = 2.0 * _norm_pdf(z) / math.sqrt(float(n_prior))
    return xp, se, p_hit


def saves_xp(
    saves_mu: float,
    saves_se: float,
    m: float,
    sigma_m: float = 0.0,
    *,
    position: str = "GKP",
) -> tuple[float, float]:
    """GKP save points from expanding mean of ⌊saves/3⌋ per appearance.

    ``saves_mu`` / ``saves_se`` are already in FPL save-point units (empirical
    multinomial expectation of the stepped award), not raw saves. Scale by
    ``m/90``; minutes uncertainty enters like ``weight_and_delta``.
    """
    pos = str(position)
    if pos == "GK":
        pos = "GKP"
    if m <= 0 or normalize_position(pos) != "GKP":
        return 0.0, 0.0
    # Prior is save *points* per appearance (~90′). weight_and_delta as per-90.
    return weight_and_delta(
        r=float(max(saves_mu, 0.0)),
        dr=float(max(saves_se, 0.0)),
        m=float(m),
        sigma_m=float(max(sigma_m, 0.0)),
    )


def load_alphas(path=None) -> tuple[float, float, float, float]:
    """(α_G, α_A, SE_α_G, SE_α_A) from pooled store."""
    doc = fpl_score_rates.load_store(path) if path is not None else fpl_score_rates.load_store()
    pooled = doc.get("pooled") or fpl_score_rates.pool_counts(doc["prior"], doc.get("live"))
    prior = doc["prior"]
    ag = float(pooled["player_g_per_match_g"])
    aa = float(pooled["a_per_match_g"])
    se_g = float(
        pooled.get("player_g_per_match_g_se_binomial")
        or prior.get("player_g_per_match_g_se")
        or 0.0
    )
    se_a = float(pooled.get("a_per_match_g_se_binomial") or prior.get("a_per_match_g_se") or 0.0)
    return ag, aa, se_g, se_a


def _player_key_series(frame: pd.DataFrame) -> pd.Series:
    """Stable id key for projecting full-club shares onto a selection board."""
    if "player_id" in frame.columns:
        return frame["player_id"].map(lambda x: str(x).split(":")[-1])
    if "element" in frame.columns:
        return frame["element"].map(lambda x: str(x).split(":")[-1])
    if "name" in frame.columns:
        return frame["name"].astype(str)
    raise ValueError("share projection needs player_id, element, or name")


def club_shares_on_frame(
    club: pd.DataFrame,
    *,
    share_se_shrink: bool = True,
    share_se_tau2: float | None = None,
    share_se_prior: str = "position_minutes",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float] | None]:
    """Compositional ``sG``/``sA`` on a club frame (full m>0 side).

    Goal shares: ``π ∝ rG`` (minutes gate only). Assist shares: ``∝ rA·m/90``.
    Returns ``(sG, sA, dsG, dsA, shrink_diag)``. Shares sum to 1 on ``club``.
    """
    if club.empty:
        z = np.zeros(0, dtype=float)
        return z, z, z, z, None

    def _pos(p: object) -> str:
        s = str(p)
        if s == "GK":
            s = "GKP"
        return normalize_position(s)

    pos = club["position"].map(_pos)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name in club.columns:
            return pd.to_numeric(club[name], errors="coerce").fillna(default).to_numpy(float)
        return np.full(len(club), float(default), dtype=float)

    m = col("m")
    sm = col("sigma_m")
    rG = col("rG")
    drG = col("drG")
    rA = col("rA")
    drA = col("drA")
    n = len(club)
    wG = np.zeros(n)
    dwG = np.zeros(n)
    wA = np.zeros(n)
    dwA = np.zeros(n)
    for i in range(n):
        wG[i], dwG[i] = goal_share_weight(rG[i], drG[i])
        wA[i], dwA[i] = weight_and_delta(rA[i], drA[i], m[i], sm[i])
    sG, dsG = shares_and_deltas(wG, dwG)
    sA, dsA = shares_and_deltas(wA, dwA)
    shrink_diag: dict[str, float] | None = None
    if share_se_shrink:
        sG_raw, sA_raw = sG.copy(), sA.copy()
        pos_list = [str(p) for p in pos.tolist()]
        sG, g_diag = shrink_shares_by_se(
            sG,
            dsG,
            m,
            tau2=share_se_tau2,
            prior_support=share_se_prior,
            positions=pos_list,
            channel="G",
        )
        sA, a_diag = shrink_shares_by_se(
            sA,
            dsA,
            m,
            tau2=share_se_tau2,
            prior_support=share_se_prior,
            positions=pos_list,
            channel="A",
        )
        shrink_diag = {
            "tau2_g": g_diag["tau2"],
            "tau2_a": a_diag["tau2"],
            "mean_k_g": g_diag["mean_k"],
            "mean_k_a": a_diag["mean_k"],
            "max_abs_delta_g": g_diag["max_abs_delta"],
            "max_abs_delta_a": a_diag["max_abs_delta"],
            "max_abs_delta_sG": float(np.max(np.abs(sG - sG_raw))) if n else 0.0,
            "max_abs_delta_sA": float(np.max(np.abs(sA - sA_raw))) if n else 0.0,
            "prior_support": g_diag.get("prior_support", float("nan")),
            "n_share_club": float(n),
        }
    return sG, sA, dsG, dsA, shrink_diag


def project_club_shares(
    club: pd.DataFrame,
    players: pd.DataFrame,
    *,
    share_se_shrink: bool = True,
    share_se_tau2: float | None = None,
    share_se_prior: str = "position_minutes",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float] | None]:
    """Map full-club shares onto a (possibly thin) selection board.

    Shares are **not** re-normalised on ``players`` — missing mass stays with
    impact subs / non-selected teammates (the G overprint fix).
    """
    sG_c, sA_c, dsG_c, dsA_c, diag = club_shares_on_frame(
        club,
        share_se_shrink=share_se_shrink,
        share_se_tau2=share_se_tau2,
        share_se_prior=share_se_prior,
    )
    n = len(players)
    sG = np.zeros(n, dtype=float)
    sA = np.zeros(n, dtype=float)
    dsG = np.zeros(n, dtype=float)
    dsA = np.zeros(n, dtype=float)
    if club.empty or players.empty:
        return sG, sA, dsG, dsA, diag
    club_keys = _player_key_series(club).tolist()
    by_key = {k: i for i, k in enumerate(club_keys)}
    for j, key in enumerate(_player_key_series(players).tolist()):
        i = by_key.get(key)
        if i is None:
            continue
        sG[j] = float(sG_c[i])
        sA[j] = float(sA_c[i])
        dsG[j] = float(dsG_c[i])
        dsA[j] = float(dsA_c[i])
    if diag is not None:
        diag = {
            **diag,
            "n_selection": float(n),
            "sel_sG_mass": float(sG.sum()),
            "sel_sA_mass": float(sA.sum()),
        }
    return sG, sA, dsG, dsA, diag


def score_team_side(
    players: pd.DataFrame,
    *,
    xg: float,
    xcs: float,
    xgc: float,
    alpha_g: float,
    alpha_a: float,
    d_alpha_g: float = 0.0,
    d_alpha_a: float = 0.0,
    d_xcs: float = 0.0,
    d_xgc: float = 0.0,
    sG_override: np.ndarray | None = None,
    sA_override: np.ndarray | None = None,
    share_players: pd.DataFrame | None = None,
    prop_p_by_player: Mapping[str | int, float] | None = None,
    share_se_shrink: bool = True,
    share_se_tau2: float | None = None,
    share_se_prior: str = "position_minutes",
) -> pd.DataFrame:
    """Score one team's players for one fixture side.

    Required columns: ``player_id`` (or ``name``), ``position``,
    ``rG``, ``drG``, ``rA``, ``drA``, ``m``, ``sigma_m``.

    Optional priors (default 0): ``defcon_mu``, ``defcon_sig``, ``defcon_n``,
    ``bonus_mu``, ``bonus_se``, ``card_mu``, ``card_se``,
    ``saves_mu``, ``saves_se`` (GKP expanding mean of ⌊saves/3⌋ per appearance).

    Optional ``sG_override`` / ``sA_override``: compositional shares from a
    correlated-draw sampler (must be length n and sum ≈ 1 on the playing set).

    Goal shares use ``goal_share_weight`` (``π ∝ rG``; minutes gate only).
    Assist shares keep ``weight_and_delta`` (``∝ rA·m/90``).

    Optional ``share_players``: full-club frame (``m>0``, impact subs) used to
    build ``sG``/``sA``. Selection boards stay thin; shares are projected and
    **not** re-normalised on ``players``. Default for cascade xG when the
    selection pool is ≥60′.

    Optional ``prop_p_by_player``: FPL id (or ``player_id`` string) → fair
    P(anytime scorer). When set, goal shares are **shaped** by props
    (``prop_shares.prop_goal_pi`` / row-1); ``M`` still sets ``G_team``.
    Assists never take a prop overlay.

    Optional ``share_se_shrink``: shrink ``sG``/``sA`` toward a share prior using
    share SE (``shrink_shares_by_se``) before props / expectations.
    ``share_se_prior``: ``position_minutes`` (role×minutes; trial default when
    shrink on), ``positive_share``, or ``all_minutes`` (legacy).
    """
    if players.empty:
        return players.copy()

    out = players.copy()

    def _pos(p: object) -> str:
        s = str(p)
        if s == "GK":
            s = "GKP"
        return normalize_position(s)

    pos = out["position"].map(_pos)
    out["position"] = pos

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default).to_numpy(float)
        return np.full(len(out), float(default), dtype=float)

    m = col("m")
    sm = col("sigma_m")
    rG = col("rG")
    drG = col("drG")
    rA = col("rA")
    drA = col("drA")

    n = len(out)
    wG = np.zeros(n)
    dwG = np.zeros(n)
    wA = np.zeros(n)
    dwA = np.zeros(n)
    for i in range(n):
        wG[i], dwG[i] = goal_share_weight(rG[i], drG[i])
        wA[i], dwA[i] = weight_and_delta(rA[i], drA[i], m[i], sm[i])

    share_shrink_diag: dict[str, float] | None = None
    used_share_club = (
        share_players is not None
        and not share_players.empty
        and sG_override is None
        and sA_override is None
    )
    if used_share_club:
        sG, sA, dsG, dsA, share_shrink_diag = project_club_shares(
            share_players,
            out,
            share_se_shrink=share_se_shrink,
            share_se_tau2=share_se_tau2,
            share_se_prior=share_se_prior,
        )
    else:
        sG, dsG = shares_and_deltas(wG, dwG)
        sA, dsA = shares_and_deltas(wA, dwA)
    if sG_override is not None:
        sG = np.asarray(sG_override, dtype=float)
        if sG.shape != (n,):
            raise ValueError(f"sG_override length {sG.size} != n={n}")
        dsG = np.zeros(n)
    if sA_override is not None:
        sA = np.asarray(sA_override, dtype=float)
        if sA.shape != (n,):
            raise ValueError(f"sA_override length {sA.size} != n={n}")
        dsA = np.zeros(n)

    if (
        share_se_shrink
        and sG_override is None
        and sA_override is None
        and not used_share_club
    ):
        sG_raw, sA_raw = sG.copy(), sA.copy()
        pos_list = [str(p) for p in pos.tolist()]
        sG, g_diag = shrink_shares_by_se(
            sG,
            dsG,
            m,
            tau2=share_se_tau2,
            prior_support=share_se_prior,
            positions=pos_list,
            channel="G",
        )
        sA, a_diag = shrink_shares_by_se(
            sA,
            dsA,
            m,
            tau2=share_se_tau2,
            prior_support=share_se_prior,
            positions=pos_list,
            channel="A",
        )
        share_shrink_diag = {
            "tau2_g": g_diag["tau2"],
            "tau2_a": a_diag["tau2"],
            "mean_k_g": g_diag["mean_k"],
            "mean_k_a": a_diag["mean_k"],
            "max_abs_delta_g": g_diag["max_abs_delta"],
            "max_abs_delta_a": a_diag["max_abs_delta"],
            "max_abs_delta_sG": float(np.max(np.abs(sG - sG_raw))) if n else 0.0,
            "max_abs_delta_sA": float(np.max(np.abs(sA - sA_raw))) if n else 0.0,
            "prior_support": g_diag.get("prior_support", float("nan")),
        }

    # Optional anytime props → row-index map (m>0 only).
    prop_p: dict[int, float] | None = None
    if prop_p_by_player:
        id_col = "player_id" if "player_id" in out.columns else "element"
        if id_col not in out.columns:
            raise ValueError("prop_p_by_player needs player_id or element column")
        lookup = {str(k).split(":")[-1]: float(v) for k, v in prop_p_by_player.items()}
        prop_p = {}
        for i, raw in enumerate(out[id_col].tolist()):
            if m[i] <= 0:
                continue  # do not put prop mass on unavailable players
            key = str(raw).split(":")[-1]
            if key in lookup:
                prop_p[i] = lookup[key]
        if not prop_p:
            prop_p = None

    # M-dependent attack (pots × shares; props shape π when present).
    pots, attack, prop_diag = player_attack_from_m(
        sG,
        sA,
        xg=float(xg),
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        d_alpha_g=float(d_alpha_g),
        d_alpha_a=float(d_alpha_a),
        dsG=dsG,
        dsA=dsA,
        positions=pos.tolist(),
        prop_p=prop_p,
    )
    # Reflect props-shaped goal shares on the output board.
    if prop_diag is not None:
        sG = np.asarray(attack.E_G, dtype=float)
        g_tmp = float(pots.G_team)
        if g_tmp > 0:
            sG = sG / g_tmp
        dsG = np.zeros(n)
    g_team = float(pots.G_team)
    a_team = float(pots.A_team)
    eG = np.asarray(attack.E_G, dtype=float)
    eA = np.asarray(attack.E_A, dtype=float)
    dE_G = np.asarray(attack.dE_G, dtype=float)
    dE_A = np.asarray(attack.dE_A, dtype=float)
    assert attack.xp_goals is not None and attack.xp_assists is not None
    assert attack.d_xp_goals is not None and attack.d_xp_assists is not None
    xp_goals = np.asarray(attack.xp_goals, dtype=float)
    xp_assists = np.asarray(attack.xp_assists, dtype=float)
    d_xp_goals = np.asarray(attack.d_xp_goals, dtype=float)
    d_xp_assists = np.asarray(attack.d_xp_assists, dtype=float)

    xp_appear = np.array([appearance_xp(mi) for mi in m], dtype=float)

    xp_cs = np.zeros(n)
    d_xp_cs = np.zeros(n)
    xp_gc = np.zeros(n)
    d_xp_gc = np.zeros(n)
    for i, p in enumerate(pos):
        if m[i] <= 0:
            continue
        cs_pts = float(CS_POINTS[normalize_position(p)])
        if m[i] >= MINUTES_FOR_APPEARANCE_BONUS and cs_pts:
            xp_cs[i] = float(xcs) * cs_pts
            d_xp_cs[i] = float(d_xcs) * cs_pts
        if normalize_position(p) in ("GKP", "DEF"):
            xp_gc[i] = float(GOALS_CONCEDED_DEDUCTION) * float(xgc)  # -1 * xGC
            d_xp_gc[i] = abs(float(GOALS_CONCEDED_DEDUCTION)) * float(d_xgc)

    # DefCon
    dc_mu = col("defcon_mu")
    dc_sig = col("defcon_sig", SIG_FLOOR)
    dc_n = col("defcon_n").astype(int)
    xp_defcon = np.zeros(n)
    d_xp_defcon = np.zeros(n)
    p_defcon = np.zeros(n)
    for i, p in enumerate(pos):
        key = normalize_position(p)
        tau = float(DEFCON_THRESHOLD.get(key, 0) or 0)
        xp_defcon[i], d_xp_defcon[i], p_defcon[i] = defcon_xp(
            dc_mu[i], dc_sig[i], int(dc_n[i]), tau, m[i]
        )

    # Bonus / cards — prior μ; zero if not appearing
    bonus_mu = col("bonus_mu")
    bonus_se = col("bonus_se")
    card_mu = col("card_mu")
    card_se = col("card_se")
    playing = m > 0
    xp_bonus = np.where(playing, bonus_mu, 0.0)
    d_xp_bonus = np.where(playing, bonus_se, 0.0)
    xp_cards = np.where(playing, card_mu, 0.0)
    d_xp_cards = np.where(playing, card_se, 0.0)

    # GKP saves — expanding mean ± SE of stepped save points (⌊saves/3⌋)
    saves_mu = col("saves_mu")
    saves_se = col("saves_se")
    xp_saves = np.zeros(n)
    d_xp_saves = np.zeros(n)
    for i, p in enumerate(pos):
        xp_saves[i], d_xp_saves[i] = saves_xp(
            saves_mu[i], saves_se[i], m[i], sm[i], position=str(p)
        )

    # Out players: zero everything
    out_mask = m <= 0
    for arr in (
        xp_appear,
        xp_goals,
        d_xp_goals,
        xp_assists,
        d_xp_assists,
        xp_cs,
        d_xp_cs,
        xp_gc,
        d_xp_gc,
        xp_defcon,
        d_xp_defcon,
        xp_bonus,
        d_xp_bonus,
        xp_cards,
        d_xp_cards,
        xp_saves,
        d_xp_saves,
        eG,
        eA,
        sG,
        sA,
    ):
        arr[out_mask] = 0.0

    xp = (
        xp_appear
        + xp_goals
        + xp_assists
        + xp_cs
        + xp_gc
        + xp_defcon
        + xp_bonus
        + xp_cards
        + xp_saves
    )
    # Independent component SEs in quadrature (conservative)
    d_xp = np.sqrt(
        d_xp_goals**2
        + d_xp_assists**2
        + d_xp_cs**2
        + d_xp_gc**2
        + d_xp_defcon**2
        + d_xp_bonus**2
        + d_xp_cards**2
        + d_xp_saves**2
    )

    out["wG"] = wG
    out["wA"] = wA
    out["sG"] = sG
    out["sA"] = sA
    out["dsG"] = dsG
    out["dsA"] = dsA
    out["share_se_shrink"] = bool(share_se_shrink)
    if share_shrink_diag is not None:
        for key, val in share_shrink_diag.items():
            out[f"share_shrink_{key}"] = float(val)
    out["E_G"] = eG
    out["E_A"] = eA
    out["xp_appear"] = xp_appear
    out["xp_goals"] = xp_goals
    out["xp_assists"] = xp_assists
    out["xp_cs"] = xp_cs
    out["xp_gc"] = xp_gc
    out["xp_defcon"] = xp_defcon
    out["p_defcon"] = p_defcon
    out["xp_bonus"] = xp_bonus
    out["xp_cards"] = xp_cards
    out["xp_saves"] = xp_saves
    out["xp"] = xp
    out[SCORE_NAME] = xp
    out["xp_se"] = d_xp
    out[f"{SCORE_NAME}_se"] = d_xp
    out["xp_lo"] = xp - d_xp
    out["xp_hi"] = xp + d_xp
    out["G_team"] = g_team
    out["A_team"] = a_team
    out["xg"] = float(xg)
    out["xcs"] = float(xcs)
    out["xgc"] = float(xgc)
    if prop_diag is not None:
        out["prop_n"] = int(prop_diag.get("n_props") or 0)
        out["prop_covered"] = float(prop_diag.get("covered") or 0.0)
        out["prop_residual"] = float(prop_diag.get("residual") or 0.0)
        out["prop_scale"] = float(prop_diag.get("prop_scale") or 1.0)
        out["prop_capped"] = bool(prop_diag.get("capped"))
    return out


def score_fixture(
    matrix: np.ndarray,
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    *,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
    d_alpha_g: float = 0.0,
    d_alpha_a: float = 0.0,
) -> pd.DataFrame:
    """Score both sides from a Dixon–Coles matrix and player tables."""
    if alpha_g is None or alpha_a is None:
        ag, aa, se_g, se_a = load_alphas()
        alpha_g = ag if alpha_g is None else alpha_g
        alpha_a = aa if alpha_a is None else alpha_a
        if d_alpha_g == 0.0:
            d_alpha_g = se_g
        if d_alpha_a == 0.0:
            d_alpha_a = se_a

    mom = scoreline.team_moments(matrix)
    home = score_team_side(
        home_players,
        xg=mom["xg_home"],
        xcs=mom["xcs_home"],
        xgc=mom["xgc_home"],
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        d_alpha_g=d_alpha_g,
        d_alpha_a=d_alpha_a,
    )
    home["side"] = "home"
    away = score_team_side(
        away_players,
        xg=mom["xg_away"],
        xcs=mom["xcs_away"],
        xgc=mom["xgc_away"],
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        d_alpha_g=d_alpha_g,
        d_alpha_a=d_alpha_a,
    )
    away["side"] = "away"
    return pd.concat([home, away], ignore_index=True)


def asof_share_history(
    history: pd.DataFrame,
    *,
    before_gw: int,
    season: str | None = None,
    gw_col: str = "GW",
    season_col: str = "season",
) -> pd.DataFrame:
    """Rows for player rates / shares: **this season only**, GW ``< before_gw``.

    PI 2026-10-16: no prior-season fill for ``rG``/``rA`` → ``sG``/``sA``.
    Early-season empty history is allowed (rates stay 0 until appearances).
    α_G / α_A may still pool a prior (team pot scale); that is not share mass.
    """
    if history is None or len(history) == 0:
        return history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
    h = history.copy()
    # Accept common GW column spellings from merged / live logs.
    if gw_col not in h.columns:
        for alt in ("gw", "event", "round"):
            if alt in h.columns:
                gw_col = alt
                break
    if gw_col not in h.columns:
        raise ValueError("asof_share_history needs a GW column")
    gw = pd.to_numeric(h[gw_col], errors="coerce")
    h = h.loc[gw.notna() & (gw < int(before_gw))].copy()
    if season is not None and season_col in h.columns:
        want = str(season).replace("_", "-")
        col = h[season_col].astype(str).str.replace("_", "-", regex=False)
        h = h.loc[col == want].copy()
    assert_single_season_rate_history(h, season_col=season_col)
    return h.reset_index(drop=True)


def assert_single_season_rate_history(
    history: pd.DataFrame,
    *,
    season_col: str = "season",
) -> None:
    """Raise if rate/share history mixes seasons (prior-season leak)."""
    if history is None or len(history) == 0:
        return
    if season_col not in history.columns:
        return
    seasons = (
        history[season_col]
        .dropna()
        .astype(str)
        .str.replace("_", "-", regex=False)
        .unique()
        .tolist()
    )
    if len(seasons) > 1:
        raise ValueError(
            f"player share/rate history spans seasons {seasons}; "
            "same-season only (no prior-season fill)"
        )


def player_rates_from_history(
    history: pd.DataFrame,
    *,
    goals_col: str = "goals_scored",
    assists_col: str = "assists",
    minutes_col: str = "minutes",
    id_col: str = "name",
    m0: float | None = None,
    rate_se_floor: float | None = None,
) -> pd.DataFrame:
    """Appearance-only rG/rA (minutes > 0).

    Caller must pass **same-season** history with GW ``< t`` only
    (``asof_share_history``). Never concatenate a prior season for shares.

    Early-season n (``protocol.score_xp_v2_rate_m0``, promoted 2026-10-16):

    - ``r = count / (M + M0)`` — minutes pseudo-count (default ``M0=270``).
    - ``Δr = max(√count, c) / (M + M0)`` — zero-event SE floor ``c``
      (default ``0.25``). Pass ``m0=0``, ``rate_se_floor=0`` for legacy MLE.
    """
    assert_single_season_rate_history(history)
    m0 = float(RATE_M0 if m0 is None else max(m0, 0.0))
    c_floor = float(RATE_SE_FLOOR if rate_se_floor is None else max(rate_se_floor, 0.0))
    h = history.copy()
    h[minutes_col] = pd.to_numeric(h[minutes_col], errors="coerce").fillna(0.0)
    h = h[h[minutes_col] > 0]
    if goals_col not in h.columns and "goals" in h.columns:
        goals_col = "goals"
    h[goals_col] = pd.to_numeric(h[goals_col], errors="coerce").fillna(0.0)
    h[assists_col] = pd.to_numeric(h[assists_col], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for pid, g in h.groupby(id_col, sort=False):
        M = float(g[minutes_col].sum())
        G = float(g[goals_col].sum())
        A = float(g[assists_col].sum())
        if M <= 0:
            continue
        denom = M + m0
        rows.append(
            {
                id_col: pid,
                "rG": G / denom,
                "drG": max(math.sqrt(G) if G > 0 else 0.0, c_floor) / denom,
                "rA": A / denom,
                "drA": max(math.sqrt(A) if A > 0 else 0.0, c_floor) / denom,
                "rate_minutes": M,
                "rate_n": int(len(g)),
                "rate_goals": G,
                "rate_assists": A,
                "rate_m0": m0,
                "rate_se_floor": c_floor,
            }
        )
    return pd.DataFrame(rows)


def history_priors_for_xp(
    history: pd.DataFrame,
    *,
    id_col: str = "name",
    minutes_col: str = "minutes",
    bonus_col: str = "bonus",
    defcon_col: str = "defensive_contribution",
    yellow_col: str = "yellow_cards",
    red_col: str = "red_cards",
    saves_col: str = "saves",
    position_col: str = "position",
) -> pd.DataFrame:
    """Prior μ±SE for bonus/cards/saves and expanding DefCon count stats.

    Saves prior is the expanding mean of stepped FPL save points
    ``⌊saves / SAVES_PER_POINT⌋`` on appearance rows (minutes > 0) — the
    empirical multinomial expectation of the discrete award.
    Call on history **strictly before** the forecast GW, **same season only**
    (no prior-season fill — same rule as player goal/assist shares).
    """
    assert_single_season_rate_history(history)
    h = history.copy()
    h[minutes_col] = pd.to_numeric(h[minutes_col], errors="coerce").fillna(0.0)
    h[bonus_col] = pd.to_numeric(h.get(bonus_col, 0), errors="coerce").fillna(0.0)
    h[defcon_col] = pd.to_numeric(h.get(defcon_col, 0), errors="coerce").fillna(0.0)
    if saves_col in h.columns:
        h[saves_col] = pd.to_numeric(h[saves_col], errors="coerce").fillna(0.0)
    else:
        h[saves_col] = 0.0
    h["save_pts"] = h[saves_col].map(
        lambda s: float(save_points(int(max(0, round(float(s))))))
    )
    y = pd.to_numeric(h.get(yellow_col, 0), errors="coerce").fillna(0.0)
    r = pd.to_numeric(h.get(red_col, 0), errors="coerce").fillna(0.0)
    h["card_pts"] = -1.0 * y + -3.0 * r

    rows: list[dict[str, Any]] = []
    for pid, g in h.groupby(id_col, sort=False):
        apps = g[g[minutes_col] > 0]
        starts60 = g[g[minutes_col] >= MINUTES_FOR_APPEARANCE_BONUS]
        b_mu, b_se, b_n = prior_mean_se(apps[bonus_col].to_numpy(float))
        c_mu, c_se, c_n = prior_mean_se(apps["card_pts"].to_numpy(float))
        s_mu, s_se, s_n = prior_mean_se(apps["save_pts"].to_numpy(float))
        if len(starts60):
            dc = starts60[defcon_col].to_numpy(float)
            dc_mu, _, dc_n = prior_mean_se(dc)
            dc_sig = float(np.std(dc, ddof=1)) if len(dc) >= 2 else SIG_FLOOR
        else:
            dc_mu, dc_sig, dc_n = 0.0, SIG_FLOOR, 0
        pos = str(g[position_col].iloc[0]) if position_col in g.columns else "MID"
        rows.append(
            {
                id_col: pid,
                "position": pos,
                "bonus_mu": b_mu,
                "bonus_se": b_se,
                "bonus_n": b_n,
                "card_mu": c_mu,
                "card_se": c_se,
                "card_n": c_n,
                "saves_mu": s_mu,
                "saves_se": s_se,
                "saves_n": s_n,
                "defcon_mu": dc_mu,
                "defcon_sig": max(dc_sig, SIG_FLOOR),
                "defcon_n": dc_n,
            }
        )
    return pd.DataFrame(rows)
