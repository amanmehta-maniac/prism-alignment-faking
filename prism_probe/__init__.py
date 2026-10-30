"""prism-probe: controls for hidden-state probes of condition-dependent behavior.

The core idea: a probe that separates condition B from condition A on the behavior of
interest (e.g. monitored vs unmonitored *compliant* answers) is only evidence of a
behavior-specific signal if it beats the same pipeline on a matched control where the
behavior cannot occur (e.g. monitored vs unmonitored *refusals*).

    from prism_probe import prism_report
    rep = prism_report(X_target, y_target, groups_target, X_control, y_control, groups_control)
"""
from .core import loqo_scores, prism_report, query_bootstrap_auroc, paired_delta_ci
from .erasure import inlp_basis, leace_eraser
from .theory import leaky_residualization_auroc
from .testbed import generate

__all__ = ["loqo_scores", "prism_report", "query_bootstrap_auroc", "paired_delta_ci", "inlp_basis",
           "leace_eraser", "leaky_residualization_auroc", "generate"]
__version__ = "0.1.0"
