"""Proposition 1: expected AUROC of a nearest-centroid probe after leaky per-class residualization
under the null (no signal), K-fold CV, n samples per class."""
import numpy as np
from scipy.stats import norm


def leaky_residualization_auroc(n, K, eigvals):
    """eigvals: covariance spectrum. Returns the predicted (below-chance) AUROC.
    Accurate for near-isotropic spectra; conservative (understates the bias) for steep ones."""
    lam = np.asarray(eigvals, float)
    d_eff = lam.sum() ** 2 / (lam ** 2).sum()
    t = n / K
    return float(norm.cdf(-(1 - t / n) * np.sqrt(2 * d_eff / (2 + (2 * t - 1) * (1 - t / n)))))
