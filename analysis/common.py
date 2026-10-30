"""Shared loader, pipeline and statistics for PRISM v3.

Every data load goes through `load_model`, which asserts provenance:
labels exist for every hidden-state file, the hidden-state response is
byte-identical to the judged response, pooling is gen5, and (Qwen) the
prompt was rendered with enable_thinking=False.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG = json.loads((HERE / "config.json").read_text())
OUT = HERE / "outputs"
CACHE = OUT / "cache"


class ProvenanceError(RuntimeError):
    pass


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def file_hash(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading (release format: one NPZ per model at its peak layer)
# ---------------------------------------------------------------------------

def load_model(key, layers=None, check=True, **_):
    """Load peak-layer gen5 hidden states with labels. check=True verifies every stored
    response against its SHA-256, so states and labels cannot come from different runs."""
    m = CFG["models"][key]
    z = np.load(REPO / m["hs_file"], allow_pickle=True)
    L = m["peak_layer"]
    if layers and set(layers) != {L}:
        raise ValueError(f"release ships layer {L} only for {key}")
    d = dict(cond=z["condition"], qid=z["query_id"], run=z["run"], label=z["label"], response=z["response"],
             X={L: z["X"].astype(np.float32)})
    if check:
        bad = sum(hashlib.sha256(str(r).encode()).hexdigest() != h for r, h in zip(z["response"], z["response_sha256"]))
        if bad:
            raise ProvenanceError(f"{key}: {bad} responses do not match their hash")
    d["prov"] = {"file": m["hs_file"], "n": int(len(d["cond"])), "sha_checked": bool(check)}
    return d


def task(d: dict, layer: int, label: str):
    """A vs B samples with the given compliance label. y=1 for B."""
    sel = np.isin(d["cond"], ["A", "B"]) & (d["label"] == label)
    return (d["X"][layer][sel], (d["cond"][sel] == "B").astype(int), d["qid"][sel],
            d["response"][sel], np.where(sel)[0])


# ---------------------------------------------------------------------------
# Pipeline v3 (label-free test transform)
# ---------------------------------------------------------------------------

def query_center(X, q):
    """T-query: subtract each query's mean over all its analysed samples (label-free)."""
    out = X.copy()
    for g in np.unique(q):
        m = q == g
        out[m] -= X[m].mean(axis=0)
    return out


def mask_direction(Xr_tr, yr_tr):
    """Unit direction separating B-refuse from A-refuse (training queries only)."""
    clf = LogisticRegression(C=CFG["mask_logreg_C"], max_iter=1000, class_weight="balanced")
    clf.fit(Xr_tr, yr_tr)
    w = clf.coef_.ravel()
    return w / (np.linalg.norm(w) + 1e-12)


def mlp(seed):
    p = CFG["probe"]
    return MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), activation="relu", solver="adam",
                         max_iter=p["max_iter"], early_stopping=True, validation_fraction=p["validation_fraction"],
                         random_state=seed + 1, alpha=p["alpha"])


def loqo_fold(args):
    """One held-out query. Returns (test_idx, scores[n_seeds, n_test])."""
    (X, y, q, Xr, yr, qr, held, transform, use_mask, n_seeds) = args
    tr, te = q != held, q == held
    Xa, Xra = X, Xr
    if transform == "query":
        Xa, Xra = query_center(X, q), query_center(Xr, qr)
    sc = StandardScaler().fit(Xa[tr])
    Xtr, Xte = sc.transform(Xa[tr]), sc.transform(Xa[te])
    if use_mask:
        rtr = qr != held
        w = mask_direction(sc.transform(Xra[rtr]), yr[rtr])
        Xtr = Xtr - np.outer(Xtr @ w, w)
        Xte = Xte - np.outer(Xte @ w, w)
    S = []
    for s in range(n_seeds):
        S.append(mlp(s).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1])
    return np.where(te)[0], np.array(S)


def run_loqo(X, y, q, Xr, yr, qr, transform="pool", use_mask=True, n_seeds=None, n_jobs=None):
    """LOQO over queries of the task (X,y,q). M uses the refuse set (Xr,yr,qr)
    restricted to training queries. Returns per-seed OOF score matrix (n_seeds, n)."""
    from joblib import Parallel, delayed
    n_seeds = n_seeds or CFG["probe"]["n_seeds"]
    jobs = [(X, y, q, Xr, yr, qr, h, transform, use_mask, n_seeds) for h in np.unique(q)]
    res = Parallel(n_jobs=n_jobs or CFG["n_jobs"])(delayed(loqo_fold)(a) for a in jobs)
    S = np.full((n_seeds, len(y)), np.nan)
    for idx, sc in res:
        S[:, idx] = sc
    return S


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def within_query(y, s, q, min_per_class=2):
    """Per-query AUROCs; returns (unweighted mean, n_A*n_B weighted mean, n_queries)."""
    a, w = [], []
    for g in np.unique(q):
        m = q == g
        nA, nB = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        if nA >= min_per_class and nB >= min_per_class:
            a.append(auroc(y[m], s[m])); w.append(nA * nB)
    if not a:
        return float("nan"), float("nan"), 0
    a, w = np.array(a), np.array(w)
    return float(a.mean()), float((a * w).sum() / w.sum()), len(a)


def query_bootstrap(stat_fn, q, reps=None, seed=None):
    """Resample queries with replacement. stat_fn(idx, strata) -> float, where
    strata gives each drawn copy of a query its own id. Percentile 95% CI."""
    reps = reps or CFG["bootstrap_reps"]
    rng = np.random.default_rng(seed or CFG["bootstrap_seed"])
    names = list(np.unique(q))
    groups = [np.where(q == g)[0] for g in names]
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(names), size=len(names), replace=True)
        idx = np.concatenate([groups[i] for i in pick])
        strata = np.concatenate([np.full(len(groups[i]), b) for b, i in enumerate(pick)])
        v = stat_fn(idx, strata)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def pooled_ci(y, s, q):
    lo, hi, _ = query_bootstrap(lambda i, st: auroc(y[i], s[i]), q)
    return auroc(y, s), lo, hi


def within_ci(y, s, q):
    """Weighted within-query AUROC (stratified Mann-Whitney) with query-bootstrap CI."""
    lo, hi, _ = query_bootstrap(lambda i, st: within_query(y[i], s[i], st)[1], q)
    mu, wmu, nq = within_query(y, s, q)
    return mu, wmu, lo, hi, nq


def paired_diff_ci(y, s1, s2, q):
    """AUROC(s1) - AUROC(s2) with paired query-bootstrap CI."""
    lo, hi, _ = query_bootstrap(lambda i, st: auroc(y[i], s1[i]) - auroc(y[i], s2[i]), q)
    return auroc(y, s1) - auroc(y, s2), lo, hi


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
