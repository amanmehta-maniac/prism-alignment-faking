"""Gate 0c (prereg: prereg/gate0c_prereg.md): split-erasure rerun of 0b.

Outer LOQO. Inside each fold, training queries are split into halves E/T:
scaler + erasure fit on E-refusals; comply and R-control probes both trained on
T; tested on the held-out query. 3 split seeds, scores averaged.
"""
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common
from gate0b_erasure import leace_fn

KS = [0, 16, 64, 128]
LEVELS = [f"inlp{k}" for k in KS] + ["leace"]
SPLIT_SEEDS = [0, 1, 2]
OUT = common.OUT / "scores_0c"


def inlp_basis(X, y, kmax):
    Q = np.zeros((X.shape[1], 0))
    Xc = X.copy()
    for _ in range(kmax):
        w = LogisticRegression(C=common.CFG["mask_logreg_C"], max_iter=1000,
                               class_weight="balanced").fit(Xc, y).coef_.ravel()
        w -= Q @ (Q.T @ w)
        if np.linalg.norm(w) < 1e-10:
            break
        w /= np.linalg.norm(w)
        Q = np.column_stack([Q, w])
        Xc = Xc - np.outer(Xc @ w, w)
    return Q


def fold(held, Xc, yc, qc, Xr, yr, qr, n_seeds):
    allq = np.union1d(qc, qr)
    trq = allq[allq != held]
    out = {}
    for ss in SPLIT_SEEDS:
        rng = np.random.default_rng(1000 * ss + int(np.where(allq == held)[0][0]))
        E = set(rng.permutation(trq)[: len(trq) // 2])
        inE_c, inE_r = np.isin(qc, list(E)), np.isin(qr, list(E))
        te_c, te_r = qc == held, qr == held
        T_c, T_r = ~te_c & ~inE_c, ~te_r & ~inE_r
        sc = StandardScaler().fit(Xr[inE_r])
        C, R = sc.transform(Xc), sc.transform(Xr)
        Q = inlp_basis(R[inE_r], yr[inE_r], max(KS))
        erasers = {f"inlp{k}": (lambda Z, B=Q[:, :min(k, Q.shape[1])]: Z - (Z @ B) @ B.T) for k in KS}
        erasers["leace"] = leace_fn(R[inE_r], yr[inE_r])
        for lvl, f in erasers.items():
            Ce, Re = f(C), f(R)
            for task, X, y, tr, te in [("comply", Ce, yc, T_c, te_c), ("refuse", Re, yr, T_r, te_r)]:
                if not te.any() or len(np.unique(y[tr])) < 2:
                    continue
                mlp = np.mean([common.mlp(s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                               for s in range(n_seeds)], axis=0)
                lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
                    X[tr], y[tr]).predict_proba(X[te])[:, 1]
                out.setdefault((lvl, task), []).append((np.where(te)[0], mlp, lr))
    # average over split seeds
    avg = {}
    for key, lst in out.items():
        avg[key] = (lst[0][0], np.mean([m for _, m, _ in lst], 0), np.mean([l for _, _, l in lst], 0))
    return avg


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
