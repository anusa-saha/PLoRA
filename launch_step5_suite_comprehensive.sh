#!/usr/bin/env bash
set -euo pipefail

cd /root/PLoRA

export HF_HOME=/root/PLoRA/.hf_home
export HF_DATASETS_CACHE=/root/PLoRA/.hf_datasets
export TRANSFORMERS_CACHE=/root/PLoRA/.hf_transformers

exec /opt/miniconda/envs/SFT/bin/python -u /root/PLoRA/plora_step5_experiment_suite.py \
  --python-exe /opt/miniconda/envs/SFT/bin/python \
  --train-script /root/PLoRA/plora_step5_language_routed_training.py \
  --rank-json /root/PLoRA/plora_step4_rank_budgets.json \
  --output-root /root/PLoRA/outputs/step5_suite_comprehensive \
  --suite-name flores2k_paper_grid \
  --dataset-kind flores200_mirror \
  --budget-keys fair_budget,equal_budget \
  --lambda-values 0.0,0.0005,0.001 \
  --sample-sizes 250,500,1000,2000 \
  --seeds 11,22,33 \
  --max-eval-samples 9 \
  --max-steps 2000 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --max-length 512 \
  --eval-every 250 \
  --save-every 500 \
  --log-every 25 \
  --use-bf16 \
  --trust-remote-code
