#!/usr/bin/env bash
set -euo pipefail

cd /tmp/jt_run.step5_flores2k
export HF_HOME=/tmp/jt_run.step5_flores2k/hf_home
export HF_DATASETS_CACHE=/tmp/jt_run.step5_flores2k/datasets
export TRANSFORMERS_CACHE=/tmp/jt_run.step5_flores2k/transformers

exec /opt/miniconda/envs/SFT/bin/python -u plora_step5_language_routed_training.py \
  --rank-json plora_step4_rank_budgets.json \
  --output-dir output_flores2k \
  --budget-key fair_budget \
  --dataset-kind flores200_mirror \
  --max-train-samples 2000 \
  --max-eval-samples 9 \
  --max-steps 2000 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --use-bf16 \
  --trust-remote-code
