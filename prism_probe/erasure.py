"""Linear concept erasure: INLP (Ravfogel et al. 2020) and closed-form LEACE (Belrose et al. 2023)."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression


def inlp_basis(X, y, k, C=0.1):
    """Orthonormal basis (d, <=k) of iteratively removed logistic directions. Stops early if
    the concept is fully erased (weight vanishes)."""
    Q = np.zeros((X.shape[1], 0))
    Xc = X.copy()
    for _ in range(k):
        w = LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(Xc, y).coef_.ravel()
        w -= Q @ (Q.T @ w)
        if np.linalg.norm(w) < 1e-10:
            break
        w /= np.linalg.norm(w)
        Q = np.column_stack([Q, w])
        Xc = Xc - np.outer(Xc @ w, w)
    return Q


def leace_eraser(X, y):
    """Return f(Z) erasing the binary concept y from linear reach (whitened projection)."""
    mu = X.mean(0)
    lam, V = np.linalg.eigh(LedoitWolf().fit(X).covariance_)
    lam = np.clip(lam, 1e-8, None)
    W, Winv = (V / np.sqrt(lam)) @ V.T, (V * np.sqrt(lam)) @ V.T
    u = W @ ((X - mu) * (y - y.mean())[:, None]).mean(0)
    u /= np.linalg.norm(u) + 1e-12
    A = Winv @ np.outer(u, u) @ W
    return lambda Z: Z - (Z - mu) @ A.T
