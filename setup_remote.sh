#!/usr/bin/env bash
# setup_remote.sh — one-command environment setup for remote/cluster machines.
#
# Usage (run once after cloning):
#   git clone https://github.com/Neopoleon/LIBERO.git && cd LIBERO
#   bash setup_remote.sh
#
# Then to train:
#   conda activate libero
#   ./train.sh libero_spatial use_wandb=true
#   tail -f train.log
#
# Requirements: conda (miniforge/miniconda), CUDA 12.8 drivers.
# For different CUDA: edit torch/torchvision lines in environment.yml first.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="libero"

echo "========================================"
echo " π-CLIP-DiT remote setup"
echo " repo: $REPO_ROOT"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Conda env from environment.yml
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Creating conda env '${ENV_NAME}'..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Already exists — updating (--prune removes stale packages)..."
    conda env update -f "$REPO_ROOT/environment.yml" --name "$ENV_NAME" --prune -y
else
    conda env create -f "$REPO_ROOT/environment.yml" --name "$ENV_NAME" -y
fi

# Resolve paths via conda info (works on any cluster layout)
CONDA_BASE="$(conda info --base)"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

# ---------------------------------------------------------------------------
# 2. robosuite / robomimic — must skip egl_probe (headless GPU nodes)
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Installing robosuite + robomimic (headless-safe, --no-deps)..."
"$PIP" install robomimic==0.2.0 --no-deps -q
"$PIP" install robosuite==1.4.0 --no-deps -q

# ---------------------------------------------------------------------------
# 3. Install this repo in editable mode
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing LIBERO package (editable)..."
"$PIP" install -e "$REPO_ROOT" -q

# ---------------------------------------------------------------------------
# 4. ~/.libero/config.yaml — prevents interactive input() call on import
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Writing ~/.libero/config.yaml..."
"$PYTHON" - <<PYEOF
import pathlib, yaml
cfg_path = pathlib.Path.home() / ".libero" / "config.yaml"
cfg_path.parent.mkdir(parents=True, exist_ok=True)
repo = "$REPO_ROOT"
if not cfg_path.exists():
    cfg_path.write_text(yaml.dump({
        "benchmark_root": f"{repo}/libero/libero",
        "bddl_files":     f"{repo}/libero/libero/bddl_files",
        "init_states":    f"{repo}/libero/libero/init_files",
        "datasets":       f"{repo}/datasets",
        "assets":         f"{repo}/libero/libero/assets",
    }))
    print(f"  Written: {cfg_path}")
else:
    print(f"  Already exists: {cfg_path}")
PYEOF

# ---------------------------------------------------------------------------
# 5. Download LIBERO datasets (~3 GB total)
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Downloading LIBERO datasets (libero_spatial / object / goal)..."
echo "  Tip: set HF_TOKEN env var for higher HF Hub rate limits."
"$PYTHON" "$REPO_ROOT/benchmark_scripts/download_libero_datasets.py" \
    --datasets libero_spatial libero_object libero_goal || \
    echo "  [warn] Download failed/interrupted — rerun manually if needed."

# ---------------------------------------------------------------------------
# 6. Precompute action stats (required for BCClipFlowPolicy normalisation)
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Precomputing action_stats.npz..."
STATS="$REPO_ROOT/action_stats.npz"
if [[ -f "$STATS" ]]; then
    echo "  Already exists: $STATS"
else
    "$PYTHON" "$REPO_ROOT/scripts/compute_action_stats.py" \
        --suites libero_spatial libero_object libero_goal \
        --n_train_tasks 8 \
        --out "$STATS"
fi

# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------
echo ""
echo "  Smoke-checking imports..."
"$PYTHON" -c "
from libero.lifelong.models.bc_clip_flow_policy import BCClipFlowPolicy
from libero.lifelong.algos.clip_multitask import CLIPMultitask
import torch
print(f'  OK | torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')
"

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "  # Train (batch size auto-scales for n GPUs):"
echo "  ./train.sh libero_spatial  use_wandb=true"
echo "  ./train.sh libero_object   use_wandb=true"
echo "  ./train.sh libero_goal     use_wandb=true"
echo ""
echo "  # Watch live:"
echo "  tail -f train.log"
echo ""
echo "  # Quick smoke-test (10 epochs):"
echo "  ./train.sh libero_spatial train.n_epochs=10 n_train_tasks=2 train.batch_size=8 train.num_workers=0"
echo ""
