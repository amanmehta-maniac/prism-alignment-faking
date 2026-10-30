import numpy as np

from prism_probe import generate, leaky_residualization_auroc, prism_report


def _split(X, cond, comply, q):
    return X[comply], cond[comply], q[comply], X[~comply], cond[~comply], q[~comply]


def test_null_with_condition_identity_is_not_behavior_specific():
    rng = np.random.default_rng(1)
    rep = prism_report(*_split(*generate(rng, gamma=1.0, eta=2.0, phi=0.0, rho=0.0, Q=30, d=64)), erase_k=8, reps=300)
    assert rep["loqo"]["control"][0] > 0.6          # the control exposes condition identity
    assert not rep["behavior_specific"]


def test_planted_signal_without_condition_identity_is_detected():
    rng = np.random.default_rng(2)
    rep = prism_report(*_split(*generate(rng, gamma=0.0, eta=0.0, phi=2.0, rho=0.0, Q=30, d=64)), erase_k=8, reps=300)
    assert rep["loqo"]["delta"][1] > 0


def test_theory_isotropic():
    assert 0.05 < leaky_residualization_auroc(150, 5, np.ones(64)) < 0.15
