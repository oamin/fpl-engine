"""Empirical-Bayes shrinkage of absolute μ using estimation SE.

Claude/PI lock: do **not** penalise the objective with SE. Shrink the mean:

    K = τ² / (τ² + SE_total²)
    μ_post = prior + K · (μ − prior)

``prior`` and ``τ²`` are cohort (position) moments. ``SE_total² = SE² + σ_model²``.
``η·SE`` penalties are retired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SIG_MODEL_FLOOR = 0.0


@dataclass(frozen=True)
class CohortMoments:
    """As-of cohort prior and between-player variance for one position."""

    prior: float
    tau2: float
    sigma_model2: float
    n: int
    mean_se2: float

    def k_constant(self) -> float:
        """SE-independent shrink factor using cohort mean SE² (+ model error)."""
        den = self.tau2 + self.mean_se2 + self.sigma_model2
        if den <= 0 or self.tau2 <= 0:
            return 0.0
        return float(self.tau2 / den)


def _finite(mu: np.ndarray, se: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    m = np.isfinite(mu) & np.isfinite(se) & (se >= 0)
    if y is not None:
        m &= np.isfinite(y)
    return m


def estimate_cohort_moments(
    mu: np.ndarray,
    se: np.ndarray,
    y: np.ndarray | None = None,
    *,
    sigma_model2: float | None = None,
) -> CohortMoments:
    """Method-of-moments EB.

    τ² = max(0, Var(μ) − mean(SE²)) — true spread of means among the cohort.
    σ_model² = max(0, Var(y−μ) − mean(SE²) − E[outcome var]) when y is given;
    outcome var proxy is max(|μ|, 1). K uses SE_total² = SE² + σ_model².
    """
    mu = np.asarray(mu, dtype=float)
    se = np.asarray(se, dtype=float)
    mask = _finite(mu, se, y)
    mu_m = mu[mask]
    se_m = se[mask]
    n = int(mu_m.size)
    if n == 0:
        return CohortMoments(0.0, 0.0, 0.0, 0, 0.0)
    prior = float(mu_m.mean())
    mean_se2 = float(np.mean(se_m**2)) if n else 0.0
    if n < 2:
        sm2 = float(max(0.0, sigma_model2)) if sigma_model2 is not None else DEFAULT_SIG_MODEL_FLOOR
        return CohortMoments(prior, 0.0, sm2, n, mean_se2)
    var_mu = float(np.var(mu_m, ddof=1))
    tau2 = float(max(0.0, var_mu - mean_se2))
    if sigma_model2 is None:
        if y is not None:
            resid = np.asarray(y, dtype=float)[mask] - mu_m
            outcome = np.maximum(np.abs(mu_m), 1.0)
            sigma_model2 = float(
                max(0.0, np.var(resid, ddof=1) - mean_se2 - float(np.mean(outcome)))
            )
        else:
            sigma_model2 = DEFAULT_SIG_MODEL_FLOOR
    else:
        sigma_model2 = float(max(0.0, sigma_model2))
    return CohortMoments(prior, tau2, sigma_model2, n, mean_se2)


def shrink_mu(
    mu: np.ndarray | pd.Series,
    se: np.ndarray | pd.Series,
    *,
    prior: float,
    tau2: float,
    sigma_model2: float = 0.0,
    mode: str = "se",
    k_constant: float | None = None,
) -> np.ndarray:
    """Return μ_post.

    mode:
      - ``none``: μ unchanged
      - ``constant``: K = k_constant (or τ²/(τ²+mean-style) if provided via k_constant)
      - ``se``: K_i = τ² / (τ² + SE_i² + σ_model²)
    """
    m = np.asarray(mu, dtype=float)
    s = np.asarray(se, dtype=float)
    s = np.where(np.isfinite(s) & (s >= 0), s, 0.0)
    prior = float(prior)
    tau2 = float(max(0.0, tau2))
    sm2 = float(max(0.0, sigma_model2))
    mode = str(mode)
    if mode == "none":
        return m.copy()
    if mode == "constant":
        if k_constant is None:
            raise ValueError("k_constant required for mode='constant'")
        k = float(np.clip(k_constant, 0.0, 1.0))
        return prior + k * (m - prior)
    if mode != "se":
        raise ValueError(f"unknown mode {mode!r}")
    if tau2 <= 0:
        return np.full_like(m, prior, dtype=float)
    se_tot2 = s**2 + sm2
    k = tau2 / (tau2 + se_tot2)
    k = np.where(np.isfinite(m), k, 0.0)
    return prior + k * (m - prior)


def shrink_frame(
    df: pd.DataFrame,
    *,
    mu_col: str = "score_xp_v2",
    se_col: str = "xp_se",
    pos_col: str = "position",
    y_col: str | None = None,
    out_col: str = "mu_post",
    mode: str = "se",
    moments_by_pos: dict[str, CohortMoments] | None = None,
    use_sigma_model_in_k: bool = False,
) -> tuple[pd.DataFrame, dict[str, CohortMoments]]:
    """Attach μ_post using per-position cohort moments (passed or estimated on df).

    ``use_sigma_model_in_k`` defaults False: single-GW residual σ_model² is still
    stored on the moments for calibration, but is **not** folded into K until a
    multi-week (outcome-noise-stripped) estimate is locked. Folding the raw
    residual leftover over-shrinks (K≈0.12).
    """
    out = df.copy()
    mu = pd.to_numeric(out[mu_col], errors="coerce").to_numpy(float)
    se = pd.to_numeric(out[se_col], errors="coerce").fillna(0.0).to_numpy(float)
    pos = out[pos_col].astype(str).to_numpy()
    y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(float) if y_col else None

    if moments_by_pos is None:
        moments_by_pos = {}
        for p in sorted(set(pos.tolist())):
            msk = pos == p
            moments_by_pos[p] = estimate_cohort_moments(
                mu[msk], se[msk], y[msk] if y is not None else None
            )

    post = np.zeros(len(out), dtype=float)
    for p, mom in moments_by_pos.items():
        msk = pos == p
        sm2 = mom.sigma_model2 if use_sigma_model_in_k else 0.0
        if mode == "constant":
            if use_sigma_model_in_k:
                k_c = mom.k_constant()
            else:
                den = mom.tau2 + mom.mean_se2
                k_c = float(mom.tau2 / den) if den > 0 and mom.tau2 > 0 else 0.0
            post[msk] = shrink_mu(
                mu[msk],
                se[msk],
                prior=mom.prior,
                tau2=mom.tau2,
                sigma_model2=sm2,
                mode="constant",
                k_constant=k_c,
            )
        else:
            post[msk] = shrink_mu(
                mu[msk],
                se[msk],
                prior=mom.prior,
                tau2=mom.tau2,
                sigma_model2=sm2,
                mode=mode,
            )
    out[out_col] = post
    return out, moments_by_pos


def calibration_z(
    y: np.ndarray,
    mu: np.ndarray,
    se: np.ndarray,
    *,
    sigma_model2: float = 0.0,
    outcome_var: np.ndarray | None = None,
) -> dict[str, float]:
    """z = (y−μ)/sqrt(SE_total² + outcome_var); report SD and coverages."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    se = np.asarray(se, dtype=float)
    if outcome_var is None:
        # Points are overdispersed; use residual-free proxy max(μ, 1) as rough scale
        outcome_var = np.maximum(np.abs(mu), 1.0)
    else:
        outcome_var = np.asarray(outcome_var, dtype=float)
    se_tot2 = np.maximum(se, 0.0) ** 2 + float(max(0.0, sigma_model2))
    denom = np.sqrt(se_tot2 + np.maximum(outcome_var, 0.0))
    mask = np.isfinite(y) & np.isfinite(mu) & (denom > 1e-9)
    z = (y[mask] - mu[mask]) / denom[mask]
    if z.size == 0:
        return {"n": 0, "z_sd": float("nan"), "cover_1": float("nan"), "cover_2": float("nan")}
    return {
        "n": int(z.size),
        "z_sd": float(z.std(ddof=1)) if z.size > 1 else 0.0,
        "cover_1": float(np.mean(np.abs(z) <= 1.0)),
        "cover_2": float(np.mean(np.abs(z) <= 2.0)),
        "mae": float(np.mean(np.abs(y[mask] - mu[mask]))),
        "mse": float(np.mean((y[mask] - mu[mask]) ** 2)),
    }


def gaussian_log_score(y: np.ndarray, mu: np.ndarray, *, sigma: float = 3.0) -> float:
    """Mean Gaussian log score (higher better); matches engine σ=3 convention."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mask = np.isfinite(y) & np.isfinite(mu)
    if not mask.any():
        return float("nan")
    r = y[mask] - mu[mask]
    s2 = float(sigma) ** 2
    return float(-0.5 * np.mean(np.log(2 * np.pi * s2) + (r**2) / s2))


def moments_to_dict(moments: dict[str, CohortMoments]) -> dict[str, Any]:
    return {
        p: {
            "prior": m.prior,
            "tau2": m.tau2,
            "sigma_model2": m.sigma_model2,
            "n": m.n,
            "mean_se2": m.mean_se2,
            "k_constant": m.k_constant(),
        }
        for p, m in moments.items()
    }
