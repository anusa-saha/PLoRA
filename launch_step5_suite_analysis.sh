#!/usr/bin/env bash
set -euo pipefail

cd /root/PLoRA

exec /opt/miniconda/envs/SFT/bin/python -u /root/PLoRA/analyze_step5_suite.py \
  --output-root /root/PLoRA/outputs/step5_suite_comprehensive \
  --analysis-dir /root/PLoRA/outputs/step5_suite_comprehensive/analysis
