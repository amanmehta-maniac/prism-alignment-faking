"""Label-free leave-one-group-out probing, query-bootstrap CIs, and the split-erasure R control."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .erasure import inlp_basis, leace_eraser


def _probe(C=1.0):
    return LogisticRegression(C=C, max_iter=2000, class_weight="balanced")


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def loqo_scores(X, y, g, probe=_probe):
    """Out-of-fold scores, holding out one group (query) at a time. Every transform is fit on
    training groups only and never sees a test label."""
    s = np.full(len(y), np.nan)
    for held in np.unique(g):
        tr, te = g != held, g == held
        sc = StandardScaler().fit(X[tr])
        if len(np.unique(y[tr])) < 2:
            continue
        s[te] = probe().fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
    return s


def query_bootstrap_auroc(y, s, g, reps=2000, seed=0):
    """Pooled AUROC with a 95% CI from resampling whole groups."""
    m = ~np.isnan(s)
    y, s, g = y[m], s[m], g[m]
    names = np.unique(g); idx = [np.where(g == n)[0] for n in names]
    rng = np.random.default_rng(seed)
    vals = [_auroc(y[i], s[i]) for i in (np.concatenate([idx[j] for j in rng.integers(0, len(names), len(names))])
                                         for _ in range(reps))]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return _auroc(y, s), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_ci(y1, s1, g1, y2, s2, g2, reps=2000, seed=0):
    """AUROC(task 1) - AUROC(task 2) with a joint group bootstrap (same groups resampled for both)."""
    names = np.union1d(g1, g2)
    i1 = {n: np.where(g1 == n)[0] for n in names}; i2 = {n: np.where(g2 == n)[0] for n in names}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(names, len(names))
        a = np.concatenate([i1[n] for n in pick]); b = np.concatenate([i2[n] for n in pick])
        v = _auroc(y1[a], s1[a]) - _auroc(y2[b], s2[b])
        if np.isfinite(v):
            vals.append(v)
    return _auroc(y1, s1) - _auroc(y2, s2), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def split_erasure_scores(Xt, yt, gt, Xc, yc, gc, k=16, method="inlp", seed=0):
    """Split-erasure LOQO. In each fold, training groups are split into halves E and T: the eraser is
    fit on E-control samples; target and control probes are both trained on T. This avoids the bias
    of fitting the eraser on the control's own training data."""
    allg = np.union1d(gt, gc)
    st, sc_ = np.full(len(yt), np.nan), np.full(len(yc), np.nan)
    for held in allg:
        rng = np.random.default_rng(seed + int(np.where(allg == held)[0][0]))
        trg = allg[allg != held]
        E = set(rng.permutation(trg)[: len(trg) // 2])
        eT, eC = np.isin(gt, list(E)), np.isin(gc, list(E))
        teT, teC = gt == held, gc == held
        sc = StandardScaler().fit(Xc[eC])
        T, Cc = sc.transform(Xt), sc.transform(Xc)
        if method == "inlp":
            Q = inlp_basis(Cc[eC], yc[eC], k)
            f = lambda Z: Z - (Z @ Q) @ Q.T
        else:
            f = leace_eraser(Cc[eC], yc[eC])
        T, Cc = f(T), f(Cc)
        for X, y, tr, te, out in [(T, yt, ~teT & ~eT, teT, st), (Cc, yc, ~teC & ~eC, teC, sc_)]:
            if te.any() and len(np.unique(y[tr])) == 2:
                out[te] = _probe().fit(X[tr], y[tr]).decision_function(X[te])
    return st, sc_


def prism_report(Xt, yt, gt, Xc, yc, gc, erase_k=16, reps=2000):
    """Target task vs matched control under (i) label-free LOQO and (ii) split erasure.
    A behavior-specific signal requires the target to beat the control (delta CI > 0)."""
    out = {}
    s_t, s_c = loqo_scores(Xt, yt, gt), loqo_scores(Xc, yc, gc)
    out["loqo"] = dict(target=query_bootstrap_auroc(yt, s_t, gt, reps), control=query_bootstrap_auroc(yc, s_c, gc, reps),
                       delta=paired_delta_ci(yt, s_t, gt, yc, s_c, gc, reps))
    e_t, e_c = split_erasure_scores(Xt, yt, gt, Xc, yc, gc, k=erase_k)
    mt, mc = ~np.isnan(e_t), ~np.isnan(e_c)
    out["split_erasure"] = dict(target=query_bootstrap_auroc(yt[mt], e_t[mt], gt[mt], reps),
                                control=query_bootstrap_auroc(yc[mc], e_c[mc], gc[mc], reps),
                                delta=paired_delta_ci(yt[mt], e_t[mt], gt[mt], yc[mc], e_c[mc], gc[mc], reps))
    out["behavior_specific"] = bool(out["split_erasure"]["delta"][1] > 0)
    return out
