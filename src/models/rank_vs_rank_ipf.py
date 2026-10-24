"""Margin-consistent rank×rank λ via iterative proportional fitting (RAS).

Given place season totals GF[r], GA[r], find λ_h(r_h, r_a), λ_a(r_h, r_a) so
that a full double round-robin recovers those margins:

    GF_i = Σ_{j≠i} [λ_h(i,j) + λ_a(j,i)]
    GA_i = Σ_{j≠i} [λ_a(i,j) + λ_h(j,i)]

Prior kernel is the cascade_long product map
``λ_h = GF_i·GA_j / (38·L)``. IPF reweights until margins match.
Diagnostic / trial only — not the locked cascade_long path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.models.opta_gf_ga_horizon import N_MATCHES, gf_ga_from_rank, league_avg_gf

N_PLACES = 20


@dataclass(frozen=True)
class RankVsRankLam:
    """0-based matrices; ``lam_home[i, j]`` = place (i+1) hosts place (j+1)."""

    lam_home: np.ndarray  # (20, 20), diag 0
    lam_away: np.ndarray
    gf: np.ndarray
    ga: np.ndarray
    league_gf: float
    n_iter: int
    max_margin_err: float

    def lambdas(self, rank_h: int, rank_a: int) -> tuple[float, float]:
        i, j = int(rank_h) - 1, int(rank_a) - 1
        if not (0 <= i < N_PLACES and 0 <= j < N_PLACES) or i == j:
            raise ValueError(f"bad ranks ({rank_h}, {rank_a})")
        return float(self.lam_home[i, j]), float(self.lam_away[i, j])


def _gf_ga_vectors(
    profile: Mapping[int, Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    gf = np.array(
        [
            float(gf_ga_from_rank(r, profile=profile, method="rankmean")["gf"])
            for r in range(1, 21)
        ],
        dtype=float,
    )
    ga = np.array(
        [
            float(gf_ga_from_rank(r, profile=profile, method="rankmean")["ga"])
            for r in range(1, 21)
        ],
        dtype=float,
    )
    L = float(league_avg_gf(profile))
    return gf, ga, L


def product_prior(gf: np.ndarray, ga: np.ndarray, league_gf: float) -> tuple[np.ndarray, np.ndarray]:
    """Product kernel used by ``lambdas_from_gf_ga`` (HOME_ADV=1)."""
    L = max(float(league_gf), 1e-9)
    H = np.zeros((N_PLACES, N_PLACES), dtype=float)
    A = np.zeros((N_PLACES, N_PLACES), dtype=float)
    for i in range(N_PLACES):
        for j in range(N_PLACES):
            if i == j:
                continue
            H[i, j] = (gf[i] * ga[j]) / (N_MATCHES * L)
            A[i, j] = (gf[j] * ga[i]) / (N_MATCHES * L)
    return H, A


def _margins(H: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gf_rec = H.sum(axis=1) + A.sum(axis=0)
    ga_rec = A.sum(axis=1) + H.sum(axis=0)
    return gf_rec, ga_rec


def fit_rank_vs_rank_ipf(
    profile: Mapping[int, Mapping[str, float]],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> RankVsRankLam:
    """IPF so recovered round-robin GF/GA match place means in ``profile``."""
    gf, ga, L = _gf_ga_vectors(profile)
    H, A = product_prior(gf, ga, L)

    n_iter = 0
    err = float("inf")
    for n_iter in range(1, max_iter + 1):
        # Match GF (scoring) margins.
        gf_rec, _ = _margins(H, A)
        for i in range(N_PLACES):
            if gf_rec[i] <= 1e-12:
                continue
            f = gf[i] / gf_rec[i]
            H[i, :] *= f
            A[:, i] *= f
        # Match GA (conceding) margins.
        _, ga_rec = _margins(H, A)
        for i in range(N_PLACES):
            if ga_rec[i] <= 1e-12:
                continue
            f = ga[i] / ga_rec[i]
            A[i, :] *= f
            H[:, i] *= f

        gf_rec, ga_rec = _margins(H, A)
        err = float(max(np.max(np.abs(gf_rec - gf)), np.max(np.abs(ga_rec - ga))))
        if err < tol:
            break

    # Keep diagonal exactly zero (no self-fixtures).
    np.fill_diagonal(H, 0.0)
    np.fill_diagonal(A, 0.0)
    gf_rec, ga_rec = _margins(H, A)
    err = float(max(np.max(np.abs(gf_rec - gf)), np.max(np.abs(ga_rec - ga))))
    return RankVsRankLam(
        lam_home=H,
        lam_away=A,
        gf=gf,
        ga=ga,
        league_gf=L,
        n_iter=n_iter,
        max_margin_err=err,
    )


def product_lambdas(
    rank_h: int,
    rank_a: int,
    profile: Mapping[int, Mapping[str, float]],
) -> tuple[float, float]:
    """Current cascade_long product λ (raw, unclipped) for comparison."""
    h = gf_ga_from_rank(rank_h, profile=profile, method="rankmean")
    a = gf_ga_from_rank(rank_a, profile=profile, method="rankmean")
    L = max(float(league_avg_gf(profile)), 1e-9)
    lam_h = (float(h["gf"]) * float(a["ga"])) / (N_MATCHES * L)
    lam_a = (float(a["gf"]) * float(h["ga"])) / (N_MATCHES * L)
    return lam_h, lam_a
