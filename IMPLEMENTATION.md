# π-CLIP-DiT: Implementation Guide

> Language-conditioned flow matching policy for LIBERO robot manipulation.
> This document explains how the new π-CLIP-DiT module fits into the existing LIBERO codebase and how to configure / run training and evaluation.

---

## 1. Architecture Overview

```
Observation (128×128 RGB)
         │
   CLIP ViT-B/32 (frozen)
         │ patch tokens (B, 49, 768)
   visual_proj (Linear 768→512)
         │ (B, 49, 512)  ──────────────────┐
                                           │
Task description (string)               DiT Cross-Attention Backbone
         │                                 │  (6 × DiTCrossAttnBlock)
   CLIPTextModel (fine-tuned)            action_tokens ← x_t (B, T, 512)
         │ hidden states (B, 77, 512)      │
         └──────────────────────────────►  │  CrossAttn(action ← visual)
                                           │  CrossAttn(action ← lang, masked)
Flow timestep t ~ U(0,1)                  │  adaLN-Zero SelfAttn
         │                                 │  adaLN-Zero FFN
   TimestepEmbedder                        │
         │ (B, 512) ─── adaLN-Zero cond ───┘
                                           │
                                    action_head (Linear 512→7)
                                           │
                              v_pred (B, T, 7) = predicted velocity
```

**Training objective (Conditional Flow Matching):**
```
x_t = (1-t)*x0 + t*x1         # straight-line interpolant
u_t = x1 - x0                  # constant conditional velocity
loss = MSE(v_pred(x_t, t, obs, lang), u_t)
```
where `x0` = normalised data action, `x1 ~ N(0,I)`.

**Inference:** Euler ODE from `t=1` (noise) → `t=0` (data) in 20 steps.

---

## 2. New Files

| File | Purpose |
|------|---------|
| `libero/lifelong/models/modules/dit_modules.py` | `TimestepEmbedder`, `CrossAttention`, `DiTCrossAttnBlock`, `DiTCrossAttnBackbone` |
| `libero/lifelong/models/bc_clip_flow_policy.py` | `BCClipFlowPolicy` — main policy class |
| `libero/lifelong/algos/clip_multitask.py` | `CLIPMultitask` — dual-LR algo with val split and checkpointing |
| `libero/configs/policy/bc_clip_flow_policy.yaml` | Policy hyperparameters |
| `libero/configs/lifelong/clip_multitask.yaml` | Algo config |
| `scripts/compute_action_stats.py` | Compute per-dim action mean/std from HDF5 files |
| `scripts/eval_policy.py` | Standalone headless eval on any checkpoint |
| `tests/test_bc_clip_flow_policy.py` | 6 unit tests (all pass) |

## 3. Modified Files

| File | What changed |
|------|-------------|
| `libero/lifelong/models/__init__.py` | Added `BCClipFlowPolicy` import (auto-registers via metaclass) |
| `libero/lifelong/algos/__init__.py` | Added `CLIPMultitask` import |
| `libero/lifelong/utils.py` | Added `clip_live` branch in `get_task_embs()` |
| `libero/lifelong/main.py` | Added `n_train_tasks` OOD filter; added `CLIPMultitask` to multitask dispatch branch |
| `libero/configs/config.yaml` | Added `n_train_tasks: null` top-level field |

---

## 4. Config Hierarchy

LIBERO uses Hydra. The relevant config hierarchy for training π-CLIP-DiT is:

```
libero/configs/config.yaml          ← root config (device, seed, wandb, n_train_tasks)
  ├─ defaults:
  │   ├─ lifelong: clip_multitask   ← libero/configs/lifelong/clip_multitask.yaml
  │   └─ policy: bc_clip_flow_policy ← libero/configs/policy/bc_clip_flow_policy.yaml
  └─ train:
      ├─ n_epochs: 500
      ├─ batch_size: 32
      └─ ...
```

### `libero/configs/policy/bc_clip_flow_policy.yaml`

```yaml
policy_type: BCClipFlowPolicy
freeze_clip: false          # set true to freeze CLIP text encoder (faster)
action_chunk_size: 10       # number of future actions predicted
embed_size: 512             # DiT hidden size
num_heads: 8                # attention heads
depth: 6                    # number of DiTCrossAttnBlocks
ff_hidden: 2048             # FFN hidden dim
action_stats_path: "action_stats.npz"  # output of compute_action_stats.py
```

Override individual fields from the command line:
```bash
policy.depth=4 policy.action_chunk_size=16 policy.freeze_clip=true
```

### `libero/configs/lifelong/clip_multitask.yaml`

```yaml
algo: CLIPMultitask
eval_in_train: false   # set true to run sim evaluation during training
```

### Root config fields relevant to π-CLIP-DiT

| Field | Default | Description |
|-------|---------|-------------|
| `task_embedding_format` | `clip_live` | Must be `clip_live` for BCClipFlowPolicy |
| `n_train_tasks` | `null` (all tasks) | Set to `8` for OOD split (tasks 0-7 train, 8-9 OOD) |
| `use_wandb` | `false` | Set `true` to log metrics to Weights & Biases |
| `train.n_epochs` | — | Set to `500` for full training |
| `device` | `cuda` | Training device |

---

## 5. Step-by-Step: First Run

### 5a. Compute action statistics
```bash
# Must be run once before training (requires downloaded datasets)
conda run -n libero python scripts/compute_action_stats.py \
    --suites libero_spatial libero_object libero_goal \
    --n_train_tasks 8 \
    --out action_stats.npz
```

### 5b. Download datasets
```bash
conda run -n libero python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial --use-huggingface --download-dir datasets/
conda run -n libero python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_object --use-huggingface --download-dir datasets/
conda run -n libero python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_goal   --use-huggingface --download-dir datasets/
```

### 5c. Train
```bash
conda run -n libero python libero/lifelong/main.py \
    benchmark_name=libero_spatial \
    lifelong=clip_multitask \
    policy=bc_clip_flow_policy \
    policy.action_stats_path=/abs/path/to/action_stats.npz \
    train.n_epochs=500 \
    n_train_tasks=8 \
    device=cuda \
    use_wandb=true
```

### 5d. Evaluate a checkpoint
```bash
conda run -n libero python scripts/eval_policy.py \
    --checkpoint experiments/<run>/checkpoint_latest.pth \
    --benchmark libero_spatial \
    --n_tasks 8 \
    --device cuda \
    --n_eval 20

# OOD evaluation (tasks 8-9 only):
conda run -n libero python scripts/eval_policy.py \
    --checkpoint experiments/<run>/checkpoint_epoch_0500.pth \
    --benchmark libero_spatial \
    --task_ids 8 9 \
    --device cuda
```

---

## 6. Checkpointing

`CLIPMultitask` saves three kinds of checkpoints:

| File | Frequency | Description |
|------|-----------|-------------|
| `checkpoint_latest.pth` | Every epoch | Rolling latest — safe to keep training from |
| `checkpoint_epoch_XXXX.pth` | Every 100 epochs | Periodic snapshots |
| `multitask_model.pth` | When best sim success | Best policy by success rate (if `eval_in_train=true`) |

---

## 7. Weights & Biases Logging

Set `use_wandb=true` in the training command (or `libero/configs/config.yaml`).

Logged every 10 epochs:
- `train/flow_loss` — training CFM loss
- `val/flow_loss` — held-out 10% validation CFM loss
- `epoch`

To also log simulation success during training, set `lifelong.eval_in_train=true`
(expensive: runs N rollouts per task every `eval.eval_every` epochs).

Initialize W&B project/entity via the normal CLI:
```bash
wandb login
# then add to training command:
# wandb.project=pi-clip-dit wandb.entity=your_entity use_wandb=true
```

---

## 8. OOD Evaluation Split

`n_train_tasks=8` means tasks `0..7` are used for training and `8..9` are
held out as OOD test tasks. This is enforced in `main.py`:

```python
n_train = cfg.n_train_tasks if cfg.n_train_tasks is not None else n_tasks
train_datasets = datasets[:n_train]
algo.learn_all_tasks(train_datasets, benchmark, result_summary)
```

The `action_stats.npz` should also be computed only on train tasks:
```bash
python scripts/compute_action_stats.py --n_train_tasks 8 ...
```

---

## 9. Unit Tests

```bash
conda run -n libero python -m pytest tests/test_bc_clip_flow_policy.py -v
```

| Test | What it checks |
|------|---------------|
| `test_dit_modules_shapes` | Output shapes for all DiT building blocks |
| `test_adaln_zero_init` | adaLN-Zero final linear is zero-initialised |
| `test_cfm_loss` | Loss is scalar ≥ 0; gradient reaches `action_head` and then backbone |
| `test_get_action_shape` | `get_action` returns `(B, 7)` numpy, all finite |
| `test_encode_lang_mask` | `lang_key_padding_mask` matches zero-attention-mask positions |
| `test_cfm_interpolation` | CFM interpolant endpoint correctness (t=0 → data, t=1 → noise) |

---

## 10. Key Design Decisions

**Why CLIP patch tokens (49) instead of a single CLS embedding?**
Spatial patch tokens preserve spatial structure needed for manipulation (e.g. which drawer, which object). CLS collapses this into a single vector.

**Why per-token CLIP text features instead of a single sentence embedding?**
Cross-attention over the full 77-token sequence lets the DiT attend to individual words (e.g. "pick up the **red** mug" vs "pick up the **blue** mug").

**Why Conditional Flow Matching instead of DDPM/DDIM?**
CFM uses straight-line ODE paths (`x_t = (1-t)*x0 + t*x1`) which converge faster, use a simpler loss, and can be sampled with very few Euler steps (20 here vs 50-100 for diffusion).

**Why adaLN-Zero?**
Zero-init gates ensure each DiT block starts as the identity function, so training loss at epoch 0 is the same as training a linear head. This makes the early training signal clean and avoids instability.

**Why a zero-init action head?**
Same reasoning: at initialisation the policy outputs zero velocity everywhere, so the initial CFM loss is simply `E[||x1 - x0||^2]` — a constant. This prevents the first backward pass from producing huge gradients. Note: backbone parameters get zero gradient on the very first pass (since `W_head = 0`), but `action_head` itself receives gradient and becomes non-zero after the first update, unblocking backbone training from step 2 onwards.
