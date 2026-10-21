"""Unit tests for LOSO xG-quartile pot scales (no season totals)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.xg_strata_recal_screen import apply_scales, fit_quartile_scales


def test_fit_scales_pull_high_xg_down():
    # Synthetic: low xG over-delivers, high xG under-delivers.
    rng = np.random.default_rng(0)
    low = pd.DataFrame({"xg": rng.uniform(0.5, 1.0, 200), "goals": rng.uniform(0.9, 1.4, 200)})
    high = pd.DataFrame({"xg": rng.uniform(2.0, 2.8, 200), "goals": rng.uniform(1.4, 2.0, 200)})
    train = pd.concat([low, high], ignore_index=True)
    fit = fit_quartile_scales(train, n_bins=4)
    assert len(fit["scales"]) >= 2
    assert fit["scales"][0] > fit["scales"][-1]
    assert all(0.5 <= s <= 1.5 for s in fit["scales"])


def test_apply_scales_identity_when_scale_one():
    fit = {"edges": [0.0, 1.0, 3.0], "scales": [1.0, 1.0]}
    x = np.array([0.5, 1.5, 2.5])
    y = apply_scales(x, fit)
    np.testing.assert_allclose(y, x)


def test_apply_scales_multiplies_bins():
    fit = {"edges": [0.0, 1.5, 4.0], "scales": [1.2, 0.8]}
    x = np.array([1.0, 2.0])
    y = apply_scales(x, fit)
    np.testing.assert_allclose(y, np.array([1.2, 1.6]))
