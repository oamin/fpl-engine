"""Risk-seeking decision μ on top of μ_post (does not replace the mean).

    mu_dec = mu_post + λ · (w_se · z(se_post) + w_cov · z(sd_cov))

``se_post`` is posterior SE under constant-K shrink (K · xp_se).
``sd_cov`` is fixture-covariance draw SD (RSS over DGW fixtures).
z(·) is within-position standardised. Missing cov → se-only (weights renormalise).

λ > 0 is risk-seeking (chase / diverge). λ = 0 recovers μ_post.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategy.shrinkage import CohortMoments

DEFAULT_LAMBDA = 0.35
DEFAULT_W_SE = 0.5
DEFAULT_W_COV = 0.5


def _k_constant(mom: CohortMoments) -> float:
    den = mom.tau2 + mom.mean_se2
    if den <= 0 or mom.tau2 <= 0:
        return 0.0
    return float(mom.tau2 / den)


def posterior_se(
    se: np.ndarray,
    pos: np.ndarray,
    moments_by_pos: dict[str, CohortMoments] | None,
) -> np.ndarray:
    """Constant-K posterior SE ≈ K · xp_se (estimation variance of μ_post)."""
    se = np.asarray(se, dtype=float)
    se = np.where(np.isfinite(se) & (se >= 0), se, 0.0)
    out = se.copy()
    if not moments_by_pos:
        return out
    for p, mom in moments_by_pos.items():
        msk = pos == p
        out[msk] = _k_constant(mom) * se[msk]
    return out


def z_by_position(
    values: np.ndarray,
    pos: np.ndarray,
) -> np.ndarray:
    """Within-position z-score; constant columns → 0."""
    values = np.asarray(values, dtype=float)
    pos = np.asarray(pos, dtype=str)
    out = np.zeros(len(values), dtype=float)
    for p in sorted(set(pos.tolist())):
        msk = pos == p
        v = values[msk]
        finite = np.isfinite(v)
        if finite.sum() < 2:
            continue
        mu = float(np.mean(v[finite]))
        sd = float(np.std(v[finite], ddof=1))
        if sd < 1e-9:
            continue
        z = np.zeros(v.size, dtype=float)
        z[finite] = (v[finite] - mu) / sd
        out[msk] = z
    return out


def attach_risk_mu(
    pool: pd.DataFrame,
    *,
    mu_col: str = "mu_post",
    se_col: str = "xp_se",
    cov_sd_col: str = "sd_cov",
    pos_col: str = "position",
    out_col: str = "mu_dec",
    risk_col: str = "risk_unit",
    lambda_risk: float = DEFAULT_LAMBDA,
    w_se: float = DEFAULT_W_SE,
    w_cov: float = DEFAULT_W_COV,
    moments_by_pos: dict[str, CohortMoments] | None = None,
) -> pd.DataFrame:
    """Attach ``mu_dec`` / ``risk_unit`` / ``se_post``. Leaves ``mu_post`` intact."""
    out = pool.copy()
    if mu_col not in out.columns:
        raise ValueError(f"pool needs {mu_col}")
    n = len(out)
    pos = out[pos_col].astype(str).to_numpy() if pos_col in out.columns else np.array(["?"] * n)

    se_raw = (
        pd.to_numeric(out[se_col], errors="coerce").fillna(0.0).to_numpy(float)
        if se_col in out.columns
        else np.zeros(n, dtype=float)
    )
    se_post = posterior_se(se_raw, pos, moments_by_pos)
    out["se_post"] = se_post

    if cov_sd_col in out.columns:
        sd_cov = pd.to_numeric(out[cov_sd_col], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        sd_cov = np.zeros(n, dtype=float)
    out["sd_cov"] = sd_cov

    z_se = z_by_position(se_post, pos)
    z_cov = z_by_position(sd_cov, pos)
    has_cov = np.isfinite(sd_cov) & (sd_cov > 1e-9)
    # Renormalise weights per row when cov is missing.
    w_s = float(w_se)
    w_c = float(w_cov)
    risk = np.zeros(n, dtype=float)
    for i in range(n):
        if has_cov[i] and (w_s + w_c) > 0:
            ws, wc = w_s / (w_s + w_c), w_c / (w_s + w_c)
            risk[i] = ws * z_se[i] + wc * z_cov[i]
        elif w_s > 0:
            risk[i] = z_se[i]
        else:
            risk[i] = 0.0
    out[risk_col] = risk
    mu = pd.to_numeric(out[mu_col], errors="coerce").fillna(0.0).to_numpy(float)
    out[out_col] = mu + float(lambda_risk) * risk
    return out


DEFAULT_CHASE_LAMBDA = 0.35
DEFAULT_LAMBDA_MAX = 0.35
DEFAULT_GAP_SCALE = 30.0  # points behind/ahead → strong tilt


def gap_to_lambdas(
    gap: float,
    *,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
    scale: float = DEFAULT_GAP_SCALE,
    chase_only: bool = False,
) -> tuple[float, float, str]:
    """Map as-of cumulative gap (L3 − L4) → (λ_chase, λ_cons, posture).

    ``gap > 0`` ⇒ L4 behind L3 ⇒ chase. ``gap < 0`` ⇒ ahead ⇒ consolidate
    (unless ``chase_only``, then stay neutral). Near zero ⇒ neutral.
    Uses ``tanh`` so |λ| ≤ λ_max.
    """
    g = float(gap)
    s = float(max(scale, 1e-6))
    lam_max = float(max(lambda_max, 0.0))
    lam_chase = lam_max * float(np.tanh(max(g, 0.0) / s))
    lam_cons = (
        0.0
        if chase_only
        else lam_max * float(np.tanh(max(-g, 0.0) / s))
    )
    if lam_chase > 0.05 * lam_max:
        posture = "chase"
    elif lam_cons > 0.05 * lam_max:
        posture = "consolidate"
    else:
        posture = "neutral"
    return lam_chase, lam_cons, posture


def attach_posture_mu(
    pool: pd.DataFrame,
    *,
    lambda_chase: float = 0.0,
    lambda_cons: float = 0.0,
    mu_col: str = "mu_post",
    upside_col: str = "upside_semi_sd",
    downside_col: str = "downside_semi_sd",
    pos_col: str = "position",
    out_col: str = "mu_dec",
) -> pd.DataFrame:
    """``mu_dec = mu_post + λ_c·z(up) − λ_k·z(down)``. Leaves ``mu_post`` intact."""
    out = pool.copy()
    if mu_col not in out.columns:
        raise ValueError(f"pool needs {mu_col}")
    n = len(out)
    pos = out[pos_col].astype(str).to_numpy() if pos_col in out.columns else np.array(["?"] * n)
    u = (
        pd.to_numeric(out[upside_col], errors="coerce").fillna(0.0).to_numpy(float)
        if upside_col in out.columns
        else np.zeros(n, dtype=float)
    )
    d = (
        pd.to_numeric(out[downside_col], errors="coerce").fillna(0.0).to_numpy(float)
        if downside_col in out.columns
        else np.zeros(n, dtype=float)
    )
    out["upside_semi_sd"] = u
    out["downside_semi_sd"] = d
    z_u = z_by_position(u, pos)
    z_d = z_by_position(d, pos)
    out["chase_unit"] = z_u
    out["cons_unit"] = z_d
    mu = pd.to_numeric(out[mu_col], errors="coerce").fillna(0.0).to_numpy(float)
    out[out_col] = mu + float(lambda_chase) * z_u - float(lambda_cons) * z_d
    out["lambda_chase"] = float(lambda_chase)
    out["lambda_cons"] = float(lambda_cons)
    return out


def attach_chase_mu(
    pool: pd.DataFrame,
    *,
    mu_col: str = "mu_post",
    upside_col: str = "upside_semi_sd",
    pos_col: str = "position",
    out_col: str = "mu_dec",
    lambda_chase: float = DEFAULT_CHASE_LAMBDA,
) -> pd.DataFrame:
    """Chase posture: ``mu_dec = mu_post + λ · z(upside_semi_sd)``.

    Upside semi-SD from fixture-cov draws. Leaves ``mu_post`` intact. No SE blend.
    """
    out = pool.copy()
    if mu_col not in out.columns:
        raise ValueError(f"pool needs {mu_col}")
    n = len(out)
    pos = out[pos_col].astype(str).to_numpy() if pos_col in out.columns else np.array(["?"] * n)
    if upside_col in out.columns:
        u = pd.to_numeric(out[upside_col], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        u = np.zeros(n, dtype=float)
    out["upside_semi_sd"] = u
    z_u = z_by_position(u, pos)
    out["chase_unit"] = z_u
    mu = pd.to_numeric(out[mu_col], errors="coerce").fillna(0.0).to_numpy(float)
    out[out_col] = mu + float(lambda_chase) * z_u
    return out


def attach_mu_pm_se(
    pool: pd.DataFrame,
    *,
    sign: float = 1.0,
    mu_col: str = "mu_post",
    se_col: str = "xp_se",
    pos_col: str = "position",
    out_col: str = "mu_dec",
    moments_by_pos: dict[str, CohortMoments] | None = None,
) -> pd.DataFrame:
    """``mu_dec = mu_post + sign · se_post`` (raw posterior SE; no z, no cov)."""
    out = pool.copy()
    if mu_col not in out.columns:
        raise ValueError(f"pool needs {mu_col}")
    n = len(out)
    pos = out[pos_col].astype(str).to_numpy() if pos_col in out.columns else np.array(["?"] * n)
    se_raw = (
        pd.to_numeric(out[se_col], errors="coerce").fillna(0.0).to_numpy(float)
        if se_col in out.columns
        else np.zeros(n, dtype=float)
    )
    se_post = posterior_se(se_raw, pos, moments_by_pos)
    out["se_post"] = se_post
    mu = pd.to_numeric(out[mu_col], errors="coerce").fillna(0.0).to_numpy(float)
    out[out_col] = mu + float(sign) * se_post
    return out


def risk_defs(
    *,
    lambda_risk: float = DEFAULT_LAMBDA,
    w_se: float = DEFAULT_W_SE,
    w_cov: float = DEFAULT_W_COV,
    n_scenarios: int | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "lambda_risk": float(lambda_risk),
        "w_se": float(w_se),
        "w_cov": float(w_cov),
        "form": "mu_dec = mu_post + λ (w_se z(se_post) + w_cov z(sd_cov))",
        "posture": "risk_seeking",
    }
    if n_scenarios is not None:
        d["n_scenarios"] = int(n_scenarios)
    return d
