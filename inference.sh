#!/bin/bash
# Example inference launch. Fill in --run-dir with your fine-tuned experiment
# directory (containing config.json, dataset_statistics.json and checkpoints/).

CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --run-dir /path/to/experiments/my_finetune_run \
  --cases-json assets/inference_cases/example_prompts.json \
  --output-root outputs \
  --gather-root gather_results \
  --action-model-type DiT-B
