"""Refusal residue (paper §4.2) recomputed from the provenance-checked loader,
with query-clustered inference. v_refusal = mean(A-refuse) - mean(A-comply).

Reports Cohen's d of the projection for B-comply vs A-comply and B-refuse vs
A-refuse, run-level query-bootstrap CIs, a query-level sign-flip permutation
test on per-query mean shifts, and the within-query (query fixed-effect) d.
v_refusal is fit on Condition A only, so it never sees B data. (The paper's
v_refusal also uses all A queries; A-only fitting avoids using B labels.)
"""
import json

import numpy as np

import common


def cohens_d(a, b):
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s


def analyse(key):
    d = common.load_model(key, check=False)
    L = common.CFG["models"][key]["peak_layer"]
    X, cond, lab, q = d["X"][L], d["cond"], d["label"], d["qid"]
    v = X[(cond == "A") & (lab == "REFUSE")].mean(0) - X[(cond == "A") & (lab == "COMPLY")].mean(0)
    v /= np.linalg.norm(v)
    p = X @ v
    out = {}
    for task in ["COMPLY", "REFUSE"]:
        a = (cond == "A") & (lab == task)
        b = (cond == "B") & (lab == task)
        sel = a | b
        y, qq, pp = b[sel].astype(int), q[sel], p[sel]
        dd = cohens_d(pp[y == 1], pp[y == 0])
        lo, hi, _ = common.query_bootstrap(lambda i, st: cohens_d(pp[i][y[i] == 1], pp[i][y[i] == 0]), qq)
        # within-query: subtract each query's A-mean, pooled sd
        shifts = []
        for g in np.unique(qq):
            m = qq == g
            if (y[m] == 1).sum() and (y[m] == 0).sum():
                shifts.append(pp[m][y[m] == 1].mean() - pp[m][y[m] == 0].mean())
        shifts = np.array(shifts)
        sd = pp.std(ddof=1)
        rng = np.random.default_rng(common.CFG["bootstrap_seed"])
        flips = np.array([(shifts * rng.choice([-1, 1], len(shifts))).mean() for _ in range(10000)])
        p_perm = (np.sum(np.abs(flips) >= abs(shifts.mean())) + 1) / (len(flips) + 1)
        out[task] = dict(d=float(dd), ci=[lo, hi], n_A=int(a.sum()), n_B=int(b.sum()),
                         within_query_d=float(shifts.mean() / sd), n_queries_both=int(len(shifts)),
                         n_pos=int((shifts > 0).sum()), p_signflip_query=float(p_perm))
    # within-query 2x2 interaction: per query with both comply and refuse in both conditions
    diffs = []
    for g in np.unique(q):
        sh = {}
        for task in ["COMPLY", "REFUSE"]:
            a = (cond == "A") & (lab == task) & (q == g)
            b = (cond == "B") & (lab == task) & (q == g)
            if a.any() and b.any():
                sh[task] = p[b].mean() - p[a].mean()
        if len(sh) == 2:
            diffs.append(sh["COMPLY"] - sh["REFUSE"])
    diffs = np.array(diffs)
    rng = np.random.default_rng(common.CFG["bootstrap_seed"] + 1)
    flips = np.array([(diffs * rng.choice([-1, 1], len(diffs))).mean() for _ in range(10000)])
    out["interaction_within_query"] = dict(
        mean_diff_d=float(diffs.mean() / p.std(ddof=1)), n_queries=int(len(diffs)), n_pos=int((diffs > 0).sum()),
        p_signflip=float((np.sum(np.abs(flips) >= abs(diffs.mean())) + 1) / 10001))
    return out


if __name__ == "__main__":
    res = {k: analyse(k) for k in ["qwen", "llama"]}
    res["commit"] = common.git_commit()
    (common.OUT / "residue_check.json").write_text(json.dumps(res, indent=2))
    for k in ["qwen", "llama"]:
        print(k, "interaction", res[k]["interaction_within_query"])
        for t, r in res[k].items():
            if t == "interaction_within_query":
                continue
            print(f"{k} {t}: d={r['d']:+.3f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] within-q d={r['within_query_d']:+.3f} "
                  f"({r['n_pos']}/{r['n_queries_both']} queries >0, sign-flip p={r['p_signflip_query']:.4f})")
