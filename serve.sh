#!/usr/bin/env bash
# Start the vLLM server for one model on port 8011. Usage: serve.sh qwen|llama
# GPU numbering follows nvidia-smi (PCI bus order). GPU1 runs cooler, so it carries more of the load.
set -euo pipefail
W=/data3/yashdewangan/tradingagents
. $W/env.sh; . $W/venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
COMMON="--port 8011 --max-model-len 16384 --gpu-memory-utilization 0.90"

case "$1" in
  qwen)
    # pipeline-parallel so the split can be uneven: first stage (listed first) on GPU1 gets 26 of 48 layers
    export CUDA_VISIBLE_DEVICES=1,0 VLLM_PP_LAYER_PARTITION=26,22
    exec vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --served-model-name qwen \
      --pipeline-parallel-size 2 $COMMON ;;
  llama)
    # 8B fits on one card: GPU1 only
    export CUDA_VISIBLE_DEVICES=1
    exec vllm serve meta-llama/Llama-3.1-8B-Instruct --served-model-name llama $COMMON ;;
  *) echo "usage: $0 qwen|llama"; exit 1 ;;
esac
