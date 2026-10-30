# Leaky residualization: closed-form bias under the null

Validated numerically by `analysis/prism_v3/testbed.py --theory` (results in `outputs/testbed_theory.json`).

## Setup

- Two classes (conditions) with n samples each. Under the null, x_i ~ N(μ, Σ) i.i.d., independent of the label y_i.
- **Leaky residualization** subtracts class means computed on *all* data, test points included: r_i = x_i − x̄_{y_i}.
- K-fold cross-validation. Each test fold holds t = n/K points per class.
- **Probe**: nearest centroid, s(x) = xᵀ(m̂₁ − m̂₀), where m̂_c is the mean training residual of class c.

## Proposition 1

Residuals of a class sum to zero. So m̂_c = −S_c / (n − t), where S_c is the sum of the class-c test residuals in the current fold. For a test point i of class 1:

  s_i = −(r_iᵀS₁ − r_iᵀS₀) / (n − t).

The second moments are:
- E[r_iᵀr_i] = (1 − 1/n) tr Σ
- E[r_iᵀr_j] = −tr Σ / n for j ≠ i in the same class, and 0 across classes.

This gives:

  E[s | y = 1] = −tr Σ / n,  E[s | y = 0] = +tr Σ / n.

For the variance, take Gaussian x:
- Var(‖r‖²) ≈ 2 tr Σ².
- Residuals within a class are negatively correlated (Cov(r_i, r_j) = −Σ/n), so a sum of m same-class residuals has covariance m(1 − m/n)Σ.
- The t − 1 same-class and t other-class cross terms therefore contribute ≈ (2t − 1)(1 − t/n) tr Σ².

So Var(s) ≈ [2 + (2t − 1)(1 − t/n)] tr Σ² / (n − t)². Treating the class-1 and class-0 scores as approximately independent normals:

  **AUROC ≈ Φ( −(1 − t/n) · √( 2 d_eff / (2 + (2t − 1)(1 − t/n)) ) )**,  where d_eff = (tr Σ)² / tr Σ²

is the participation ratio of Σ. For K-fold CV (t = n/K, t ≫ 1), this is approximately Φ(−√((1 − 1/K) K d_eff / n)).

(A first version omitted the (1 − t/n) factor in the variance. That version overstated the variance by about 20%, and the testbed caught it.)

## Consequences

1. **The bias is systematically below chance, and it grows with d_eff / n.** In the paper's setting (d = 4096–5120, a few hundred samples per class, d_eff ≫ 1), the nearest-centroid AUROC is pushed far below 0.5. More samples help only through √(d_eff/n).
2. **A true mean signal δ cannot survive.** x̄_y absorbs it exactly, so leaky per-condition residualization removes every class-mean difference. Any above-chance AUROC under this protocol must come from non-mean structure, such as covariance differences or nonlinear features. This is how a flexible probe can sit *above* chance on a no-faking control (Llama, 0.63), while a nearest-centroid probe sits below it.
3. **Per-(condition, query) cells** with k runs behave the same way, cell by cell. The relevant per-cell n is k (≈ 3–10). The leak is therefore much stronger, unless the whole cell is held out together, which is what LOQO does.
4. **Label-dependent test transforms** (the bug in `phase4_gen5_loqo_and_ortho.py`, which subtracts m̂_{y_test}) make the test score an explicit function of the test label. The resulting AUROC can take any value in [0, 1] under the null, so the transform is invalid regardless of the direction of the bias.

## Numerical validation (testbed, K = 5, nearest centroid, 100 replicates)

- **Isotropic Σ:** accurate. For example, d = 64 gives n = 150: 0.095 observed vs 0.099 predicted, and n = 50: 0.011 vs 0.015.
- **Power-law spectra (d_eff ≈ 28–48):** the formula *under-predicts* the bias, for example 0.170 observed vs 0.196 predicted, and 0.110 vs 0.132. Treat it as a conservative bound there. The cause is the Gaussian approximation to ‖r‖² when a few eigenvalues dominate.
- **Logistic probes (C = 1):** close to the formula on isotropic data. They are much *more* biased than nearest centroid on anisotropic data (for example 0.006 vs 0.170). Regularized linear probes are therefore not protected.

## What the proposition does not cover

- Logistic and MLP probes. The testbed checks them empirically: they show the same sign of bias, and its magnitude depends on regularization.
- Dependence induced by duplicate states (gen1 collapse). Duplicates reduce the effective t and n.
