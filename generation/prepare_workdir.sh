#!/usr/bin/env bash
# Assemble the input layout the GPU scripts expect: WORK/in/{prompts,queries,*.py,<key>_directions.npz}.
# Usage: bash generation/prepare_workdir.sh /path/to/work   then e.g.
#   python WORK/in/steer_job.py --model meta-llama/Llama-3.1-8B-Instruct --key llama --work WORK
#   python WORK/in/arditi_job.py --model meta-llama/Llama-3.1-8B-Instruct --key llama --work WORK
#   python WORK/in/sweep_job.py --work WORK --runs 5                       (model sweep)
#   python WORK/in/sweep_job.py --work WORK --runs 10 --factorial gemma3-12b gemma3-27b
# Tested with transformers==4.51.0, accelerate==0.34.2, torch 2.x, one H200 per model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; W="${1:?work dir}"
mkdir -p "$W/in" "$W/out"
cp -R "$ROOT/prompts" "$ROOT/queries" "$W/in/"
cp "$ROOT"/generation/{steer_job,arditi_job,sweep_job}.py "$W/in/"
cp "$ROOT/data/directions/llama-3.1-8b_directions.npz" "$W/in/llama_directions.npz"
cp "$ROOT/data/directions/qwen3-32b_directions.npz" "$W/in/qwen_directions.npz"
cp "$ROOT/data/directions/directions_meta.json" "$W/in/"
echo "work dir ready: $W"
