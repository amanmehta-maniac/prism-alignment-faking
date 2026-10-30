"""Gate 0 hidden-state probes (prereg: prereg/gate0_prereg.md).

For each model x task (comply, refuse=R control) x transform (pool, query) x mask
(M on/off): LOQO MLP, 5 seeds, out-of-fold scores saved to outputs/scores/.
Primary configuration: transform=pool, mask=True.
"""
import itertools
import sys
import time

import numpy as np

import common

SCORES = common.OUT / "scores"


def main(models):
    SCORES.mkdir(parents=True, exist_ok=True)
    for key in models:
        d = common.load_model(key)
        L = common.CFG["models"][key]["peak_layer"]
        comply = common.task(d, L, "COMPLY")
        refuse = common.task(d, L, "REFUSE")
        Xr, yr, qr = refuse[0], refuse[1], refuse[2]
        for (tname, t), transform, use_mask in itertools.product(
                [("comply", comply), ("refuse", refuse)], ["pool", "query"], [True, False]):
            out = SCORES / f"{key}_{tname}_{transform}_{'M' if use_mask else 'noM'}.npz"
            if out.exists():
                print("skip", out.name); continue
            X, y, q, resp, idx = t
            t0 = time.time()
            S = common.run_loqo(X, y, q, Xr, yr, qr, transform=transform, use_mask=use_mask)
            np.savez(out, S=S, y=y, q=q, idx=idx, prov=str(d["prov"]), commit=common.git_commit())
            print(f"{out.name}: n={len(y)} AUROC(seed-mean)={common.auroc(y, S.mean(0)):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["llama", "qwen"])
