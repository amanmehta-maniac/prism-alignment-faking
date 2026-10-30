# Gate 0c pre-registration: split-erasure rerun of Gate 0b

Written 2026-10-27, before any split-erasure result on real data was computed.

## Why

The synthetic testbed (`analysis/prism_v3/testbed.py --grid`, `outputs/testbed_grid.json`) shows that the Gate 0b design is biased.

- The erasure is fit on the same training refusals that the R-control probe then trains on. This drives the R control toward chance while compliant answers keep the residual condition signal.
- With **no planted faking** but condition identity present, Gate 0b's design gives a mean Δ of about +0.17 to +0.21 and "detects" faking in 100% of replicates.
- Splitting the training queries removes the bias: erasure is fit on half E, and the probes for **both** tasks train on half T. The null Δ then becomes about +0.00 to +0.03.

The Gate 0b positive result is therefore not interpretable, and Gate 0c replaces it.

## Design (fixed)

- **Data, peak layers, provenance loader, outer LOQO:** as in Gate 0b.
- **Inner split.** In each outer fold, the training queries (the union of comply and refuse queries, held-out query excluded) are split at random into halves E and T.
  - Scaler and erasure: fit on E-refusals.
  - Probes: comply probe trained on T-comply; R-control probe trained on T-refuse.
  - Both are tested on the held-out query.
  - This is repeated for 3 split seeds, and the per-sample scores are averaged over seeds.
- **Erasure levels:** INLP-k for k ∈ {0, 16, 64, 128}, and LEACE. INLP stops early if the logistic weight vanishes (full erasure of E).
- **Probes:** MLP (512,128) with Gate 0 hyperparameters (5 seeds, averaged), plus a logistic regression (C = 1.0) as secondary.
- **Statistics:** pooled AUROC per task. Δ = comply − refuse, with a 95% CI from the joint query bootstrap (2,000 replicates).

## Primary decision (MLP, INLP-128)

- **Interaction signal:** the Δ CI lower bound > 0. Interpretation: monitoring is encoded differently in compliant answers than in refusals, beyond what refusal-learned erasure removes. This is consistent with the population-level residue. It is **not** a single-answer faking detector, because G0.1/G0.2 already failed.
- **Null:** otherwise. Interpretation: no evidence at the single-answer level beyond condition identity.
- **Robustness (not decisive):** LEACE, and k ∈ {16, 64}.

## Deviations

(none yet)

### 2026-10-27 (after results)
- **Outcome.** The pre-registered rule gives FAIL (clean null) for both models.
  - Llama: Δ = +0.035 [−0.047, +0.125].
  - Qwen: Δ = +0.053 [−0.029, +0.127].
  - LEACE robustness agrees (CIs include 0).
  - Erasure never brings the R control to chance (0.89–0.95).
- **Qwen INLP stopped early.** The logistic weight vanished once E-half refusals were fully erased, so fewer than 128 directions were removed and k = 64 and k = 128 are identical. This is the pre-registered early-stop rule.
- Gate 0c supersedes Gate 0b's positive result, which the testbed shows to be a design artifact.
