#!/usr/bin/env bash
# train.sh — compute action stats then launch π-CLIP-DiT training.
#
# Usage:
#   ./train.sh                               # train on libero_spatial, no wandb
#   ./train.sh libero_object                 # different benchmark
#   ./train.sh libero_spatial use_wandb=true # enable wandb explicitly
#
# All Hydra overrides can be appended after the benchmark name:
#   ./train.sh libero_spatial train.n_epochs=100 policy.freeze_clip=true
#
# Multi-GPU: batch size is auto-scaled by GPU count (32 × n_gpus).
# Override with: ./train.sh libero_spatial train.batch_size=64

set -euo pipefail

BENCHMARK="${1:-libero_spatial}"
shift 2>/dev/null || true   # consume benchmark arg; remaining args passed to Hydra

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS="$REPO_ROOT/action_stats.npz"
PYTHON="${PYTHON:-python}"   # override with PYTHON=/path/to/python if needed

# ---------------------------------------------------------------------------
# Step 0 — detect GPU count and auto-scale batch size
# ---------------------------------------------------------------------------
N_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
N_GPUS="${N_GPUS:-1}"
BASE_BATCH=32
BATCH_SIZE=$(( BASE_BATCH * N_GPUS ))
echo "[train.sh] Detected ${N_GPUS} GPU(s) — batch_size=${BATCH_SIZE}"

# ---------------------------------------------------------------------------
# Step 1 — compute action stats if not already done
# ---------------------------------------------------------------------------
if [[ ! -f "$STATS" ]]; then
    echo "[train.sh] action_stats.npz not found — computing now..."
    "$PYTHON" "$REPO_ROOT/scripts/compute_action_stats.py" \
        --suites libero_spatial libero_object libero_goal \
        --n_train_tasks 8 \
        --out "$STATS"
    echo "[train.sh] action_stats.npz saved to $STATS"
else
    echo "[train.sh] Using existing action_stats.npz: $STATS"
fi

# ---------------------------------------------------------------------------
# Step 2 — launch training
# ---------------------------------------------------------------------------
echo "[train.sh] Launching training on benchmark=$BENCHMARK ..."
echo ""

# PYTHONUNBUFFERED=1 forces line-by-line stdout so `tail -f train.log` works.
# TRANSFORMERS_VERBOSITY=warning suppresses verbose HuggingFace HTTP logs.
PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=warning HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    "$PYTHON" -u -m libero.lifelong.main \
    benchmark_name="$BENCHMARK" \
    lifelong=clip_multitask \
    policy=bc_clip_flow_policy \
    "policy.action_stats_path=$STATS" \
    "train.batch_size=$BATCH_SIZE" \
    device=cuda \
    use_wandb=false \
    "$@" 2>&1 | tee "$REPO_ROOT/train.log"
# Pass use_wandb=true on the CLI to enable: ./train.sh libero_spatial use_wandb=true
