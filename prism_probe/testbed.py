"""Synthetic ground-truth generator: condition identity, query offsets, planted faking signal, duplicates."""
import numpy as np


def spectrum(d, alpha):
    """Eigenvalues lambda_j ∝ j^-alpha (alpha=0 isotropic)."""
    lam = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return lam / lam.mean()


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
