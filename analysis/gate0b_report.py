"""Gate 0b report: Δ(k) = AUROC_comply − AUROC_refuse under identical erasure,
joint query bootstrap; applies the pre-registered k* rule."""
import json
import sys

import numpy as np

import common

GATE = sys.argv[1] if len(sys.argv) > 1 else "0b"
if GATE == "0c":
    from gate0c_split_erasure import KS, LEVELS, OUT as SC
else:
    from gate0b_erasure import KS, LEVELS, OUT as SC

CLAIMS = common.OUT / "claims.json"


def joint_ci(yc, sc, qc, yr, sr, qr, reps=None):
    reps = reps or common.CFG["bootstrap_reps"]
    rng = np.random.default_rng(common.CFG["bootstrap_seed"])
    names = np.union1d(qc, qr)
    gc = {g: np.where(qc == g)[0] for g in names}
    gr = {g: np.where(qr == g)[0] for g in names}
    vals = []
    for _ in range(reps):
        pick = rng.choice(names, len(names))
        ic = np.concatenate([gc[g] for g in pick]); ir = np.concatenate([gr[g] for g in pick])
        a, b = common.auroc(yc[ic], sc[ic]), common.auroc(yr[ir], sr[ir])
        vals.append((a, b, a - b))
    v = np.array(vals)
    v = v[np.isfinite(v).all(1)]
    return np.percentile(v, [2.5, 97.5], axis=0)


def main():
    claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}
    L = [f"# Gate {GATE} report: condition-identity erasure", "",
         f"Commit `{common.git_commit()}`. Prereg: `prereg/gate{GATE}_prereg.md`. "
         + ("Split erasure: fit on E-half refusals, probes trained on T-half; " if GATE == "0c" else "Erasure fit on training-fold refusals only; ")
         + "LOQO; 2,000 joint query-bootstrap replicates.", ""]
    for key in ["llama", "qwen"]:
        z = np.load(SC / f"{key}.npz", allow_pickle=True)
        yc, qc, yr, qr = z["y_comply"], z["q_comply"], z["y_refuse"], z["q_refuse"]
        L += [f"## {common.CFG['models'][key]['name']}", "",
              "| Erasure | Probe | Comply AUROC [CI] | R control AUROC [CI] | Δ comply − refuse [CI] |",
              "|---|---|---|---|---|"]
        res = {}
        for lvl in LEVELS:
            for probe in ["mlp", "lr"]:
                sc, sr = z[f"{lvl}_comply_{probe}"], z[f"{lvl}_refuse_{probe}"]
                a, b = common.auroc(yc, sc), common.auroc(yr, sr)
                (alo, blo, dlo), (ahi, bhi, dhi) = joint_ci(yc, sc, qc, yr, sr, qr)
                res[(lvl, probe)] = dict(a=a, alo=alo, ahi=ahi, b=b, blo=blo, bhi=bhi, d=a - b, dlo=dlo, dhi=dhi)
                L.append(f"| {lvl} | {probe} | {a:.3f} [{alo:.3f}, {ahi:.3f}] | {b:.3f} [{blo:.3f}, {bhi:.3f}] | "
                         f"{a - b:+.3f} [{dlo:+.3f}, {dhi:+.3f}] |")
        reached = [k for k in KS if res[(f"inlp{k}", "mlp")]["b"] <= 0.60]
        kstar = (reached[0] if reached else 128) if GATE == "0b" else 128
        r = res[(f"inlp{kstar}", "mlp")]
        passed = (r["alo"] > 0.5 and r["dlo"] > 0) if GATE == "0b" else r["dlo"] > 0
        lr_ = res[("leace", "mlp")]
        L += ["", f"- k\\* = {kstar} " + ("" if reached else "(**erasure failed to reach chance**: no k gives R control ≤ 0.60)"),
              f"- At k\\*: comply {r['a']:.3f} [{r['alo']:.3f}, {r['ahi']:.3f}], Δ {r['d']:+.3f} [{r['dlo']:+.3f}, {r['dhi']:+.3f}] "
              f"→ pre-registered rule: **{'PASS' if passed else 'FAIL (clean null)'}**",
              f"- LEACE robustness: R control {lr_['b']:.3f}, Δ {lr_['d']:+.3f} [{lr_['dlo']:+.3f}, {lr_['dhi']:+.3f}]", ""]
        claims[f"gate{GATE}.{key}"] = {"kstar": kstar, "erasure_reached_chance": bool(reached), "pass_rule": bool(passed),
                                   "comply": [r["a"], r["alo"], r["ahi"]], "rcontrol": [r["b"], r["blo"], r["bhi"]],
                                   "delta": [r["d"], r["dlo"], r["dhi"]], "script": f"analysis/prism_v3/gate0b_report.py {GATE}",
                                   "commit": common.git_commit()}
    (common.OUT / f"gate{GATE}_report.md").write_text("\n".join(L))
    CLAIMS.write_text(json.dumps(claims, indent=2, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
