#!/usr/bin/env bash
# setup_remote.sh — one-command environment setup for remote GPU machines.
#
# Usage (on remote):
#   git clone https://github.com/Neopoleon/LIBERO.git && cd LIBERO
#   bash setup_remote.sh
#
# After setup, activate the env and train:
#   conda activate libero
#   ./train.sh libero_spatial use_wandb=true

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="libero"
PIP="$HOME/miniforge3/envs/${ENV_NAME}/bin/pip"
PYTHON="$HOME/miniforge3/envs/${ENV_NAME}/bin/python"

# Fall back to mamba/conda-provided pip if miniforge path differs
if [[ ! -f "$PIP" ]]; then
    PIP="$(conda run -n $ENV_NAME which pip 2>/dev/null || echo pip)"
    PYTHON="$(conda run -n $ENV_NAME which python 2>/dev/null || echo python)"
fi

echo "========================================"
echo " π-CLIP-DiT remote setup"
echo " conda env: $ENV_NAME"
echo " repo:      $REPO_ROOT"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1 — create conda env from environment.yml
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Conda env '${ENV_NAME}' already exists — skipping create."
    echo "        To recreate: conda env remove -n ${ENV_NAME} && bash setup_remote.sh"
else
    echo "[setup] Creating conda env '${ENV_NAME}' from environment.yml..."
    conda env create -f "$REPO_ROOT/environment.yml" -n "$ENV_NAME"
fi

# ---------------------------------------------------------------------------
# Step 2 — install extra pip packages not in environment.yml conda section
# ---------------------------------------------------------------------------
echo "[setup] Installing pip packages..."
"$PIP" install -q \
    hydra-core omegaconf easydict wandb einops transformers \
    flow_matching torchdiffeq bddl h5py tensorboard tensorboardX \
    termcolor psutil tqdm matplotlib "opencv-python" imageio imageio-ffmpeg \
    packaging pyyaml thop cloudpickle

echo "[setup] Installing robosuite + robomimic (no-deps to skip egl_probe)..."
"$PIP" install -q robomimic==0.2.0 --no-deps
"$PIP" install -q robosuite==1.4.0 --no-deps
"$PIP" install -q mujoco numba pyopengl glfw

# ---------------------------------------------------------------------------
# Step 3 — install this repo in editable mode
# ---------------------------------------------------------------------------
echo "[setup] Installing LIBERO package in editable mode..."
"$PIP" install -q -e "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Step 4 — create ~/.libero/config.yaml (required to suppress interactive prompt)
# ---------------------------------------------------------------------------
"$PYTHON" - <<'EOF'
import os, yaml, pathlib
cfg_path = pathlib.Path.home() / ".libero" / "config.yaml"
if not cfg_path.exists():
    cfg_path.parent.mkdir(exist_ok=True)
    root = str(pathlib.Path.home() / "LIBERO")
    cfg = {
        "benchmark_root": f"{root}/libero/benchmark",
        "bddl_files":     f"{root}/libero/bddl_files",
        "init_states":    f"{root}/libero/init_task_states",
        "datasets":       f"{root}/datasets",
        "assets":         f"{root}/libero/envs/assets",
    }
    cfg_path.write_text(yaml.dump(cfg))
    print(f"[setup] Created {cfg_path}")
else:
    print(f"[setup] {cfg_path} already exists — skipping.")
EOF

# ---------------------------------------------------------------------------
# Step 5 — download LIBERO datasets (HF Hub)
# ---------------------------------------------------------------------------
echo "[setup] Downloading LIBERO datasets from HuggingFace..."
echo "        (set HF_TOKEN env var first if using a private token)"
"$PYTHON" "$REPO_ROOT/scripts/download_libero_datasets.py" \
    --datasets libero_spatial libero_object libero_goal 2>/dev/null || \
"$PYTHON" - <<'EOF'
# Fallback: use huggingface_hub directly
from huggingface_hub import snapshot_download
import os
root = os.path.expanduser("~/LIBERO/datasets")
os.makedirs(root, exist_ok=True)
for suite in ["libero_spatial", "libero_object", "libero_goal"]:
    out = os.path.join(root, suite)
    if os.path.isdir(out) and os.listdir(out):
        print(f"[setup] {suite} already downloaded — skipping.")
    else:
        print(f"[setup] Downloading {suite}...")
        snapshot_download(
            repo_id=f"libero-bench/libero",
            repo_type="dataset",
            local_dir=out,
            allow_patterns=f"{suite}/*",
        )
print("[setup] Datasets ready.")
EOF

# ---------------------------------------------------------------------------
# Step 6 — smoke import check
# ---------------------------------------------------------------------------
echo "[setup] Smoke-testing imports..."
"$PYTHON" -c "
from libero.lifelong.models.bc_clip_flow_policy import BCClipFlowPolicy
from libero.lifelong.algos.clip_multitask import CLIPMultitask
import torch
print(f'[setup] OK — torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
"

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   conda activate $ENV_NAME"
echo "   ./train.sh libero_spatial use_wandb=true"
echo ""
echo " For 2-GPU training batch size is auto-scaled."
echo " Monitor: tail -f train.log"
echo "========================================"
