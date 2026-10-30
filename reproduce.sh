#!/usr/bin/env bash
# Reproduce every number in the paper from the released data (CPU only).
# Behavioral results: ~2 min. Probe and erasure analyses: ~3.5 h on 16 cores.
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements.txt -e .
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd analysis
python behavior_reports.py all            # Gate 1 factorial, model sweep, Gate 2 steering
python residue_check.py                   # within-query refusal residue
python gate0_probes.py llama qwen         # label-free LOQO probes + refusal control
python gate0_text.py llama qwen           # text-only baselines
python gate0_report.py
python gate0c_split_erasure.py llama qwen # split-sample erasure
python gate0b_report.py 0c
python testbed.py --theory --grid         # Proposition 1 check + synthetic testbed
# Optional: audit of Abdelnabi & Salem (2025). Clone their repo into a separate directory first:
#   git clone https://github.com/microsoft/Test_Awareness_Steering /tmp/tas
#   python -I audit_test_awareness.py /tmp/tas/data/triggers_annotated_answers/triggers_qwen3_with_GPT_labels_evidence.json
cd .. && python -m pytest -q tests
