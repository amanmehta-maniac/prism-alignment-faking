"""Gate 0b (prereg: prereg/gate0b_prereg.md).

Erase condition identity learned from training-fold refusals (INLP-k, LEACE),
then compare B-vs-A AUROC on compliant answers and on refusals under the same
erasure. Out-of-fold scores saved to outputs/scores_0b/{model}.npz.
"""
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common

KS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
LEVELS = [f"inlp{k}" for k in KS] + ["leace"]
OUT = common.OUT / "scores_0b"


def inlp_basis(X, y, kmax):
    """Orthonormal basis (d, kmax) of INLP directions fit on (X, y)."""
    Q = np.zeros((X.shape[1], 0))
    Xc = X.copy()
    for _ in range(kmax):
        w = LogisticRegression(C=common.CFG["mask_logreg_C"], max_iter=1000,
                               class_weight="balanced").fit(Xc, y).coef_.ravel()
        w -= Q @ (Q.T @ w)                      # re-orthogonalize
        w /= np.linalg.norm(w) + 1e-12
        Q = np.column_stack([Q, w])
        Xc = Xc - np.outer(Xc @ w, w)
    return Q


def leace_fn(X, y):
    """Closed-form LEACE eraser for a binary concept (Belrose et al. 2023)."""
    mu = X.mean(0)
    lam, V = np.linalg.eigh(LedoitWolf().fit(X).covariance_)
    lam = np.clip(lam, 1e-8, None)
    W = (V / np.sqrt(lam)) @ V.T
    Winv = (V * np.sqrt(lam)) @ V.T
    sxz = ((X - mu) * (y - y.mean())[:, None]).mean(0)
    u = W @ sxz
    u /= np.linalg.norm(u) + 1e-12
    A = Winv @ np.outer(u, u) @ W              # x' = x - A (x - mu)
    return lambda Z: Z - (Z - mu) @ A.T


def fold(held, Xc, yc, qc, Xr, yr, qr, n_seeds):
    ctr, cte, rtr, rte = qc != held, qc == held, qr != held, qr == held
    sc = StandardScaler().fit(Xr[rtr])
    C, R = sc.transform(Xc), sc.transform(Xr)
    Q = inlp_basis(R[rtr], yr[rtr], max(KS))
    erasers = {f"inlp{k}": (lambda Z, B=Q[:, :k]: Z - (Z @ B) @ B.T) for k in KS}
    erasers["leace"] = leace_fn(R[rtr], yr[rtr])
    out = {}
    for lvl, f in erasers.items():
        Ce, Re = f(C), f(R)
        for task, X, y, tr, te in [("comply", Ce, yc, ctr, cte), ("refuse", Re, yr, rtr, rte)]:
            if not te.any():
                continue
            mlp = np.mean([common.mlp(s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                           for s in range(n_seeds)], axis=0)
            lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
                X[tr], y[tr]).predict_proba(X[te])[:, 1]
            out[(lvl, task)] = (np.where(te)[0], mlp, lr)
    return out


def main(models):
    OUT.mkdir(parents=True, exist_ok=True)
    for key in models:
        t0 = time.time()
        d = common.load_model(key)
        L = common.CFG["models"][key]["peak_layer"]
        Xc, yc, qc, _, _ = common.task(d, L, "COMPLY")
        Xr, yr, qr, _, _ = common.task(d, L, "REFUSE")
        held = np.union1d(np.unique(qc), np.unique(qr))
        res = Parallel(n_jobs=common.CFG["n_jobs"])(
            delayed(fold)(h, Xc, yc, qc, Xr, yr, qr, common.CFG["probe"]["n_seeds"]) for h in held)
        arrays = {"y_comply": yc, "q_comply": qc, "y_refuse": yr, "q_refuse": qr}
        for lvl in LEVELS:
            for task, n in [("comply", len(yc)), ("refuse", len(yr))]:
                m, l = np.full(n, np.nan), np.full(n, np.nan)
                for r in res:
                    if (lvl, task) in r:
                        idx, a, b = r[(lvl, task)]
                        m[idx], l[idx] = a, b
                arrays[f"{lvl}_{task}_mlp"], arrays[f"{lvl}_{task}_lr"] = m, l
        np.savez(OUT / f"{key}.npz", **arrays, commit=common.git_commit())
        for lvl in LEVELS:
            print(f"{key} {lvl:8s} comply {common.auroc(yc, arrays[f'{lvl}_comply_mlp']):.3f} "
                  f"refuse {common.auroc(yr, arrays[f'{lvl}_refuse_mlp']):.3f}", flush=True)
        print(f"{key} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["llama", "qwen"])
