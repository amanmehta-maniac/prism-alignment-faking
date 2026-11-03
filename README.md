# PRISM: Stress-Testing Alignment-Faking Claims

Code, data and controls for the paper.

Many open-weight models comply more with borderline requests when the system prompt says the conversation is monitored and used for training. That gap is often read as alignment faking, and hidden-state probes have been proposed to detect it. This repository tests both readings with controls:

- **The gap does not depend on training.** A six-cell prompt factorial on Qwen3-32B, Llama-3.1-8B, Gemma-3-12B and Gemma-3-27B separates monitoring, the training consequence and the instruction wording. Removing the training consequence changes nothing in any model.
- **Probes read the prompt.** Monitored and unmonitored *refusals* (where no faking is possible) are as separable from hidden states as monitored and unmonitored compliant answers.
- **Steering needs a matched control.** A direction learned only from refusals steers compliance as strongly as the "faking" direction.
- **PRISM:** a toolkit implementing the controls, a closed-form account of the bias from residualizing before cross-validation, and a synthetic testbed with known ground truth.

## Layout

| Path | Contents |
|---|---|
| `prism_probe/` | pip-installable toolkit: label-free leave-one-query-out probing, query-bootstrap CIs, split-sample erasure (INLP, LEACE), the refusal-control report, theory, testbed |
| `data/hidden_states/` | gen5 hidden states (mean of the first 5 generated tokens) at each model's peak layer, with condition, query, run, judge label, the response and its SHA-256 |
| `data/generations/` | every judged generation: `factorial/` (6 prompt cells), `sweep/` (18 configurations), `steering/` (all steering arms and controls) |
| `data/directions/` | per-layer steering directions (v_refusal, v_faking, v_cond_refuse) and the measured gaps |
| `prompts/`, `queries/` | system prompts for every cell; the 50 borderline and 50 HarmBench queries |
| `analysis/` | scripts that compute every number in the paper from `data/` |
| `generation/` | scripts that produced the data (OpenAI-compatible endpoint + Anthropic judge; GPU steering with transformers) |
| `prereg/` | pre-registrations for every experiment, with dated deviations and outcomes |
| `theory/` | derivation of Proposition 1 |

## Quick start

```bash
pip install -r requirements.txt -e .
pytest -q tests                                   # toolkit tests (~10 s)
cd analysis && python behavior_reports.py all     # all behavioral results (~2 min)
bash reproduce.sh                                 # everything, including probes (~3.5 h on 16 cores)
```

Using the controls on your own probe:

```python
from prism_probe import prism_report
# X_*: hidden states; y_*: 1 = condition B, 0 = condition A; g_*: query ids
rep = prism_report(X_target, y_target, g_target,     # e.g. monitored vs unmonitored compliant answers
                   X_control, y_control, g_control)  # same contrast where the behavior cannot occur
rep["loqo"]["delta"]        # AUROC(target) - AUROC(control) with a 95% query-bootstrap CI
rep["behavior_specific"]    # True only if the split-erasure delta CI excludes 0
```

## Notes

- **Judge.** Claude Opus 4.6 (`claude-opus-4-6`), temperature 0, answer truncated to 600 characters (`generation/run_prompts.py`). Compliance = COMPLY or PARTIAL.
- **Statistics.** All CIs resample whole queries, because the runs of one query are not independent.
- **Pre-registration.** Two rules changed after data were seen; both are listed in `prereg/gate2_steering_prereg.md`, with results under both versions.
- **Hidden states** are released only at the peak layer (48 for Qwen3-32B, 32 for Llama-3.1-8B), to keep the repository small. `generation/` re-extracts all layers.

## Licenses

Code: MIT. Data: CC BY 4.0 (see `data/LICENSE`).
