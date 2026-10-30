"""Synthetic ground-truth testbed for PRISM.

--theory : validate Proposition 1 (theory/leaky_residualization.md)
           for nearest-centroid and logistic probes under the null.
--grid   : planted-signal generator with controllable nuisances; measures each
           protocol's false-positive rate (no faking signal) and power.
"""
import argparse
import json

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import common


def spectrum(d, alpha):
    """Eigenvalues lambda_j ∝ j^-alpha (alpha=0 isotropic)."""
    lam = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return lam / lam.mean()


def d_eff(lam):
    return lam.sum() ** 2 / (lam ** 2).sum()


def predicted_auroc(n, K, lam):
    t = n / K
    return float(norm.cdf(-(1 - t / n) * np.sqrt(2 * d_eff(lam) / (2 + (2 * t - 1) * (1 - t / n)))))


def leaky_cv(X, y, K, probe, rng):
    R = X.copy()
    for c in (0, 1):
        R[y == c] -= X[y == c].mean(0)              # leaky: means over all data
    folds = np.zeros(len(y), int)
    for c in (0, 1):                                # stratified folds
        idx = rng.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(len(idx)) % K
    s = np.zeros(len(y))
    for f in range(K):
        tr, te = folds != f, folds == f
        if probe == "centroid":
            w = R[tr & (y == 1)].mean(0) - R[tr & (y == 0)].mean(0)
            s[te] = R[te] @ w
        else:
            s[te] = LogisticRegression(C=1.0, max_iter=2000).fit(R[tr], y[tr]).decision_function(R[te])
    return roc_auc_score(y, s)


def theory(reps=100):
    rng = np.random.default_rng(0)
    rows = []
    for d, alpha in [(64, 0), (512, 0), (512, 1.0), (4096, 0.5), (4096, 1.0)]:
        lam = spectrum(d, alpha)
        for n in [20, 50, 150]:
            for probe in ["centroid", "logistic"]:
                if probe == "logistic" and d > 512:
                    continue
                a = [leaky_cv(rng.normal(size=(2 * n, d)) * np.sqrt(lam), np.repeat([0, 1], n), 5, probe, rng)
                     for _ in range(reps if probe == "centroid" else 20)]
                rows.append(dict(d=d, alpha=alpha, d_eff=round(d_eff(lam), 1), n=n, probe=probe,
                                 empirical=float(np.mean(a)), sd=float(np.std(a)),
                                 predicted=predicted_auroc(n, 5, lam)))
                print(rows[-1], flush=True)
    (common.OUT / "testbed_theory.json").write_text(json.dumps(rows, indent=2))


# ---------------------------------------------------------------------------
# Planted-signal grid
# ---------------------------------------------------------------------------

def generate(rng, gamma, eta, phi, rho, Q=50, k=10, d=256, alpha=1.0, tau=2.0, beta=2.0, gap=1.0):
    """Two conditions (A=0, B=1), Q queries x k runs each.
    gamma: global condition-identity offset (the system prompt), on ALL B samples
    eta:   query-specific condition offset (condition effect differs per query)
    phi:   planted faking signal: B-comply samples shifted along the refusal direction
    rho:   probability a run duplicates the previous run in its cell (gen1-style collapse)
    """
    lam = spectrum(d, alpha)
    u = np.linalg.qr(rng.normal(size=(d, 3)))[0].T          # u_r, u_c, spare (orthonormal)
    u_r, u_c = u[0] * np.sqrt(d), u[1] * np.sqrt(d) / 4
    X, cond, comply, qid = [], [], [], []
    for q in range(Q):
        mu_q = rng.normal(size=d) * tau * np.sqrt(lam)
        a_q = rng.normal() * 1.5
        v_q = rng.normal(size=d); v_q *= np.sqrt(d) / 4 / np.linalg.norm(v_q)
        for c in (0, 1):
            prev = None
            for r in range(k):
                if prev is not None and rng.random() < rho:
                    x, cm = prev
                else:
                    cm = rng.random() < 1 / (1 + np.exp(-(a_q + gap * c)))
                    x = mu_q + rng.normal(size=d) * np.sqrt(lam)
                    if not cm:
                        x = x + beta * u_r / np.sqrt(d)
                    if c == 1:
                        x = x + gamma * (u_c + eta * v_q)
                        if cm:
                            x = x + phi * u_r / np.sqrt(d)
                    prev = (x, cm)
                X.append(x); cond.append(c); comply.append(cm); qid.append(q)
    return np.array(X), np.array(cond), np.array(comply), np.array(qid)


def _lr():
    return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")


def _split_erasure_scores(Xc, yc, qc, Xr, yr, qr, erase=16, K=5, seed=0):
    """Split erasure: in each outer fold, training queries are split in half.
    INLP is fit on refusals of half E; probes for BOTH tasks are trained on half T
    and tested on the held-out fold, so neither task's probe sees erasure-fit data."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    allq = np.unique(np.concatenate([qc, qr]))
    fold_of = {g: i % K for i, g in enumerate(rng.permutation(allq))}
    sc_c, sc_r = np.full(len(yc), np.nan), np.full(len(yr), np.nan)
    for f in range(K):
        trq = np.array([g for g in allq if fold_of[g] != f])
        Eq = set(rng.permutation(trq)[: len(trq) // 2])
        inE_r = np.array([g in Eq for g in qr]); inE_c = np.array([g in Eq for g in qc])
        te_r = np.array([fold_of[g] == f for g in qr]); te_c = np.array([fold_of[g] == f for g in qc])
        T_r, T_c = ~te_r & ~inE_r, ~te_c & ~inE_c
        sc = StandardScaler().fit(Xr[inE_r])
        R, C, Re = sc.transform(Xr), sc.transform(Xc), sc.transform(Xr[inE_r])
        ye = yr[inE_r]
        for _ in range(erase):
            w = _lr().fit(Re, ye).coef_.ravel()
            if np.linalg.norm(w) < 1e-10:
                break
            w /= np.linalg.norm(w)
            Re, R, C = (Z - np.outer(Z @ w, w) for Z in (Re, R, C))
        if te_c.any() and len(np.unique(yc[T_c])) == 2:
            sc_c[te_c] = _lr().fit(C[T_c], yc[T_c]).decision_function(C[te_c])
        if te_r.any() and len(np.unique(yr[T_r])) == 2:
            sc_r[te_r] = _lr().fit(R[T_r], yr[T_r]).decision_function(R[te_r])
    mc, mr = ~np.isnan(sc_c), ~np.isnan(sc_r)
    return roc_auc_score(yc[mc], sc_c[mc]), roc_auc_score(yr[mr], sc_r[mr])


def _grouped_scores(X, y, q, Xr=None, yr=None, qr=None, erase=0, K=5):
    """Query-grouped K-fold; optional INLP erasure (erase directions) fit on refusals."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in GroupKFold(K).split(X, y, q):
        held = np.unique(q[te])
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        if erase:
            rtr = ~np.isin(qr, held)
            R = sc.transform(Xr[rtr])
            for _ in range(erase):
                w = _lr().fit(R, yr[rtr]).coef_.ravel(); w /= np.linalg.norm(w)
                R, A, B = (Z - np.outer(Z @ w, w) for Z in (R, A, B))
        s[te] = _lr().fit(A, y[tr]).decision_function(B)
    return s


def protocols(X, cond, comply, qid, rng):
    out = {}
    c, r = comply & True, ~comply
    Xc, yc, qc = X[c], cond[c], qid[c]
    Xr, yr, qr = X[r], cond[r], qid[r]
    # P0 raw: random (ungrouped) 5-fold
    from sklearn.model_selection import StratifiedKFold
    s = np.zeros(len(yc))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yc):
        s[te] = _lr().fit(Xc[tr], yc[tr]).decision_function(Xc[te])
    out["P0_raw_random_split"] = roc_auc_score(yc, s)
    # P1 leaky per-condition residualization + random 5-fold
    R = Xc.copy()
    for k_ in (0, 1):
        R[yc == k_] -= Xc[yc == k_].mean(0)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(R, yc):
        s[te] = _lr().fit(R[tr], yc[tr]).decision_function(R[te])
    out["P1_leaky_resid"] = roc_auc_score(yc, s)
    # P2 query-grouped split, no erasure; P3 + M (1 direction); P4 + INLP-16
    for name, e in [("P2_grouped", 0), ("P3_grouped_M1", 1), ("P4_grouped_INLP16", 16)]:
        out[name] = roc_auc_score(yc, _grouped_scores(Xc, yc, qc, Xr, yr, qr, e))
        out[name + "_Rctrl"] = roc_auc_score(yr, _grouped_scores(Xr, yr, qr, Xr, yr, qr, e))
    for e in (16, 64):
        a, b = _split_erasure_scores(Xc, yc, qc, Xr, yr, qr, erase=e)
        out[f"P5_split_INLP{e}"], out[f"P5_split_INLP{e}_Rctrl"] = a, b
    return out


def grid(reps=10):
    from itertools import product
    from joblib import Parallel, delayed
    cells = list(product([0.0, 1.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.5]))   # gamma, eta, phi, rho
    cells = [cl for cl in cells if not (cl[0] == 0 and cl[1] > 0)]           # eta needs gamma

    def one(cell, rep):
        rng = np.random.default_rng(1000 * rep + hash(cell) % 1000)
        return cell, protocols(*generate(rng, *cell), rng)

    res = Parallel(n_jobs=common.CFG["n_jobs"])(delayed(one)(cl, r) for cl in cells for r in range(reps))
    rows = []
    for cl in cells:
        rs = [o for c_, o in res if c_ == cl]
        row = dict(gamma=cl[0], eta=cl[1], phi=cl[2], rho=cl[3])
        for key in rs[0]:
            row[key] = float(np.mean([o[key] for o in rs]))
        # PRISM decision: comply AUROC beats the R control by > 0.05 under INLP-16
        row["P4_detect_rate"] = float(np.mean([(o["P4_grouped_INLP16"] - o["P4_grouped_INLP16_Rctrl"] > 0.05)
                                               and o["P4_grouped_INLP16"] > 0.55 for o in rs]))
        for e in (16, 64):
            row[f"P5_{e}_detect_rate"] = float(np.mean([(o[f"P5_split_INLP{e}"] - o[f"P5_split_INLP{e}_Rctrl"] > 0.05)
                                                       and o[f"P5_split_INLP{e}"] > 0.55 for o in rs]))
        row["P0_detect_rate"] = float(np.mean([o["P0_raw_random_split"] > 0.6 for o in rs]))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    (common.OUT / "testbed_grid.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    if a.theory:
        theory()
    if a.grid:
        grid()
