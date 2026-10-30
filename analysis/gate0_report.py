"""Gate 0 report: applies the pre-registered pass rules to saved OOF scores.

Writes outputs/gate0_report.md and results/claims.json entries (gate0.*).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

import common

SC = common.OUT / "scores"
TEXT = ["tfidf_full", "tfidf_first5", "minilm", "handfeat"]
CLAIMS = common.OUT / "claims.json"


def load(name):
    z = np.load(SC / f"{name}.npz", allow_pickle=True)
    S = z["S"]
    return z["y"], z["q"], S.mean(0), (S if S.shape[0] > 1 else None)


def seed_std(y, S):
    return float(np.std([common.auroc(y, s) for s in S])) if S is not None else float("nan")


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def model_report(key, claims):
    name = common.CFG["models"][key]["name"]
    L = []
    r = {}
    rows = []
    for task in ["comply", "refuse"]:
        for tr in ["pool", "query"]:
            for m in ["M", "noM"]:
                y, q, s, S = load(f"{key}_{task}_{tr}_{m}")
                p, lo, hi = common.pooled_ci(y, s, q)
                wmu_u, wmu, wlo, whi, nq = common.within_ci(y, s, q)
                r[(task, tr, m)] = dict(p=p, lo=lo, hi=hi, w=wmu, wlo=wlo, whi=whi, nq=nq, s=s, y=y, q=q)
                rows.append(f"| {task} | {tr} | {m} | {len(y)} ({int((y==0).sum())}/{int((y==1).sum())}) | "
                            f"{len(np.unique(q))} | {fmt(p, lo, hi)} | {fmt(wmu, wlo, whi)} (n_q={nq}) | "
                            f"{seed_std(y, S):.3f} |")
    L += [f"## {name}", "",
          "| Task | Transform | Mask | n (A/B) | queries | Pooled AUROC [95% query-bootstrap CI] | "
          "Within-query weighted AUROC [CI] | seed std |", "|---|---|---|---|---|---|---|---|"] + rows + [""]

    C, R = r[("comply", "pool", "M")], r[("refuse", "pool", "M")]
    g01 = C["lo"] > R["hi"]
    g02 = (R["lo"] <= 0.5 <= R["hi"]) and (0.40 <= R["p"] <= 0.60)
    g03 = C["w"] > 0.60 and C["wlo"] > 0.5

    # text baselines (comply task, and on the R control for information)
    trows, tscore = [], {}
    for task in ["comply", "refuse"]:
        for t in TEXT:
            y, q, s, _ = load(f"{key}_{task}_text_{t}")
            p, lo, hi = common.pooled_ci(y, s, q)
            if task == "comply":
                tscore[t] = (p, s)
                assert np.array_equal(y, C["y"]) and np.array_equal(q, C["q"]), "text/probe sample mismatch"
            trows.append(f"| {task} | {t} | {fmt(p, lo, hi)} |")
    best = max(tscore, key=lambda t: tscore[t][0])
    sb = tscore[best][1]
    dp, dlo, dhi = common.paired_diff_ci(C["y"], C["s"], sb, C["q"])
    g05 = dlo > 0
    # G0.6: residualize probe logit on best-text logit (label-free OLS), AUROC of residual
    a, b = common.logit(C["s"]), common.logit(sb)
    beta = np.polyfit(b, a, 1)
    resid = a - np.polyval(beta, b)
    rp, rlo, rhi = common.pooled_ci(C["y"], resid, C["q"])
    g06 = rlo > 0.5
    rho = stats.spearmanr(a, b).correlation

    L += ["### Text-only baselines (same samples, same LOQO folds)", "",
          "| Task | Baseline | Pooled AUROC [CI] |", "|---|---|---|"] + trows + [""]
    L += ["### Gate 0 rules (primary = T-pool + M)", "",
          f"- **G0.1** comply {fmt(C['p'], C['lo'], C['hi'])} vs R control {fmt(R['p'], R['lo'], R['hi'])}: "
          f"comply lower bound > control upper bound → **{'PASS' if g01 else 'FAIL'}**",
          f"- **G0.2** R control CI contains 0.5 and point in [0.40, 0.60] → **{'PASS' if g02 else 'FAIL'}**",
          f"- **G0.3** within-query weighted AUROC {fmt(C['w'], C['wlo'], C['whi'])} > 0.60, CI excl. 0.5 → "
          f"**{'PASS' if g03 else 'FAIL'}**",
          f"- **G0.5** best text baseline = `{best}` ({tscore[best][0]:.3f}); probe − text = {fmt(dp, dlo, dhi)} → "
          f"**{'PASS' if g05 else 'FAIL'}**",
          f"- **G0.6** probe logit residualized on `{best}` logit (Spearman ρ={rho:.2f}): AUROC {fmt(rp, rlo, rhi)} → "
          f"**{'PASS' if g06 else 'FAIL'}**", ""]
    gates = dict(G0_1=g01, G0_2=g02, G0_3=g03, G0_5=g05, G0_6=g06)
    for k, v in [("comply_auroc", C), ("rcontrol_auroc", R)]:
        claims[f"gate0.{key}.{k}"] = {"value": v["p"], "ci": [v["lo"], v["hi"]]}
    claims[f"gate0.{key}.within_query_weighted"] = {"value": C["w"], "ci": [C["wlo"], C["whi"]]}
    claims[f"gate0.{key}.best_text"] = {"name": best, "value": tscore[best][0]}
    claims[f"gate0.{key}.probe_minus_text"] = {"value": dp, "ci": [dlo, dhi]}
    claims[f"gate0.{key}.residual_probe_auroc"] = {"value": rp, "ci": [rlo, rhi]}
    claims[f"gate0.{key}.gates"] = gates
    return L, gates


def decision(g):
    if not (g["G0_1"] and g["G0_2"]):
        return "F2 (detector was a leak)"
    if not (g["G0_5"] and g["G0_6"]):
        return "F1 (text matches the probe)"
    if not g["G0_3"]:
        return "G0.3 failed (not covered by the plan's decision rule; report to user)"
    return "PASS → continue to sections 3-6"


def main():
    claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}
    L = ["# Gate 0 report", "", f"Commit `{common.git_commit()}`. Prereg: `prereg/gate0_prereg.md`. "
         "CIs: 2,000 query-level bootstrap replicates. Scores: mean of 5 MLP seeds' out-of-fold probabilities.", ""]
    dec = {}
    for key in ["llama", "qwen"]:
        d = common.load_model(key, check=False)
        L += [f"Provenance ({key}): `{d['prov']}`", ""]
        part, g = model_report(key, claims)
        L += part
        dec[key] = decision(g)
    L += ["## Decision", "", f"- Llama-3.1-8B (headline): **{dec['llama']}**",
          f"- Qwen3-32B (same rules): **{dec['qwen']}**", ""]
    sp = common.OUT / "step_p_provenance.json"
    if sp.exists():
        j = json.loads(sp.read_text())
        L += ["## Step P: provenance of the published Qwen Table 3 numbers", "",
              f"Unmodified original functions, {j['n_seeds']} seed(s).", "",
              "| Quantity | Published | Last-token states (wrong path) | gen5 states (matched) |", "|---|---|---|---|"]
        for k, v in j["published"].items():
            a, b = j["legacy_last_token"][k], j["gen5"][k]
            L.append(f"| {k} | {v:.3f} | {a['mean']:.3f} ± {a['std']:.3f} | {b['mean']:.3f} ± {b['std']:.3f} |")
        L.append("")
    (common.OUT / "gate0_report.md").write_text("\n".join(L))
    for k in claims:
        claims[k].setdefault("script", "analysis/prism_v3/gate0_report.py")
        claims[k]["commit"] = common.git_commit()
    CLAIMS.parent.mkdir(parents=True, exist_ok=True)
    CLAIMS.write_text(json.dumps(claims, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
    print("\n".join(L))


if __name__ == "__main__":
    main()
