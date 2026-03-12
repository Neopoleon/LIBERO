# π-CLIP-DiT: Language-Conditioned Flow Matching Policy for Robot Manipulation

**CS224N Final Project — Implementation Plan & Design Reference**

---

## Project Hypothesis

Language-conditioned task representations improve robot policy robustness under distribution shift. Specifically:
- Language encodes task semantics that remain invariant when visual conditions change.
- Finetuning the language encoder end-to-end (rather than using frozen embeddings) allows representations to adapt from general vision-language pretraining to manipulation-domain semantics.
- Attending to the full per-token language sequence (vs. a single pooled vector) lets the policy ground individual words — object names, spatial relations, action verbs — into action predictions.

---

## Benchmarks & Evaluation

### Suite Selection

Three LIBERO controlled-transfer suites, 30 tasks total:

| Suite | Train | OOD Eval | What varies | Language role |
|---|---|---|---|---|
| LIBERO-Spatial | 8 tasks | 2 tasks | Object positions / spatial relations | "left bowl" vs "right bowl" |
| LIBERO-Object | 8 tasks | 2 tasks | Object identity | Which object to manipulate |
| LIBERO-Goal | 8 tasks | 2 tasks | Goal/instruction only; scene is identical | **Only** discriminating signal |

**Why these suites, not LIBERO-90→LIBERO-10:**
LIBERO-90/10 is designed for *lifelong learning* (sequential fine-tuning). The three controlled suites each isolate one type of knowledge transfer, making it possible to cleanly measure when language conditioning helps (Goal) vs. when vision alone should suffice (Spatial, Object). This gives more interpretable ablation results for a language-focused paper.

**Why 80/20 OOD split within each suite:**
Holding out 2/10 tasks per suite creates a true OOD generalization test (policy has never seen those task descriptions at train time) while keeping training data large enough for reasonable BC performance. Using the same suite for train/eval but on unseen tasks is a standard setup in generalization studies.

**Why LIBERO-Goal is the primary benchmark:**
The 10 LIBERO-Goal tasks share an identical visual scene — only the language instruction differs. A vision-only policy cannot distinguish tasks; language is the *only* signal. This is the strongest possible test of the hypothesis that language conditioning improves generalization.

---

## Architecture Design

### Overview

```
image (B,3,128,128)
  → upsample to 224×224
  → CLIP vision transformer (frozen) → last_hidden_state[:, 1:, :]
  → visual_tokens (B, 49, 768)
  → visual_proj Linear(768, 512)
  → (B, 49, 512)

task description string
  → CLIP tokenizer → input_ids (B,77), attn_mask (B,77)
  → packed as task_emb (B, 154) LongTensor through dataloader
  → unpack in policy → CLIPTextModel (unfrozen, LR 5e-6)
  → lang_tokens (B, 77, 512)  +  lang_key_padding_mask (B, 77)

noisy action chunk x_t (B, T, 7)
  → action_proj MLP(7→512, GELU) + learned positional embedding
  → action_tokens (B, T, 512)

flow timestep t (B,)
  → TimestepEmbedder → t_emb (B, 512)

── DiTCrossAttnBackbone (6 blocks) ──────────────────────────────
  Each DiTCrossAttnBlock:
    adaLN-Zero(action_tokens, t_emb)           ← modulate with flow t
    CrossAttn(action_tokens ← visual_tokens)   ← attend to scene patches
    CrossAttn(action_tokens ← lang_tokens,     ← attend to task words
              key_padding_mask=lang_key_mask)
    SelfAttn(action_tokens)                    ← temporal coherence
    FFN(action_tokens)
─────────────────────────────────────────────────────────────────

→ action_head Linear(512, 7)  [zero-initialized]
→ v_pred (B, T, 7)

Training loss:  MSE(v_pred,  x_1 - x_0)
                where x_0 = clean action, x_1 ~ N(0,I), x_t = (1-t)x_0 + t*x_1

Inference: 20-step Euler ODE from x_1 ~ N(0,I) → x_0 (clean action chunk)
           → denormalize → return chunk[0] as executed action
```

---

## Design Decisions & Rationale

### 1. Visual Encoder: CLIP ViT-B/32 Patch Tokens (Frozen)

**Decision:** Use `clip.vision_model(pixel_values).last_hidden_state[:, 1:, :]` — the 49 spatial patch tokens of dimension 768 — rather than the pooled CLS image feature or DINOv2.

**Why not CLIP CLS (`get_image_features()`):**
The CLS vector is a single 512-dim global summary. It discards all spatial structure. For manipulation tasks — "pick the left bowl not the right one" — the policy needs to know *where* objects are in the image. Cross-attention over 49 spatial patches gives the policy the ability to attend to specific image regions.

**Why not DINOv2:**
DINOv2 produces rich spatial features (trained purely for self-supervised visual understanding), but its embedding space is unrelated to CLIP's text encoder space. If we use CLIP for language, using CLIP for vision keeps both modalities in the same jointly-trained embedding space. Cross-attention between action tokens and CLIP visual patches is more semantically coherent than cross-attending to DINOv2 features.

**Why frozen:**
CLIP's visual encoder has 86M+ parameters. Finetuning it alongside the policy on ~400 robot demonstrations would severely overfit. The visual features are rich enough frozen; gradient budget is better spent on the language encoder and the policy backbone. The text encoder is finetuned because language-manipulation grounding requires domain adaptation that visual features do not.

**Image resizing (128→224):**
LIBERO images are 128×128. CLIP ViT-B/32 expects 224×224 (7×7=49 patches of size 32×32). We bilinearly upsample to 224×224 before passing to CLIP. This is the same strategy used in the π-CLIP report and is standard practice.

---

### 2. Language Encoder: CLIPTextModel, Per-Token Sequence, Unfrozen

**Decision:** Use `CLIPTextModel.last_hidden_state` — all 77 token positions, each 512-dim — and finetune the text encoder at a low learning rate (5e-6).

**Why per-token sequence (77×512) instead of pooled CLS (512):**
A pooled embedding compresses the entire sentence into one vector. The policy gets no information about which *part* of the description is relevant at a given moment. With 77 per-token representations and cross-attention, the policy can learn to attend to "left" when deciding lateral positioning, "bowl" when identifying the target object, and "plate" when identifying the destination. This is the central NLP contribution of the project.

**Why finetune (unfrozen) CLIP text weights:**
CLIP was trained on internet image-text pairs, not robot manipulation data. The semantic distinctions that matter in manipulation (e.g., "wooden plate" vs. "white plate" vs. "black plate" — which are identical in natural image datasets) are underrepresented in CLIP's training distribution. Finetuning allows the text encoder to learn manipulation-domain semantics. Prior work (π-CLIP report) hypothesizes that finetuned embeddings will cluster more tightly by task semantics (validated by t-SNE and cosine similarity analysis).

**Why low LR (5e-6, 20× smaller than backbone):**
CLIP text encoder has 63M pretrained parameters that encode general language understanding. A full LR would destroy these representations in a few hundred gradient steps on a small robotics dataset. The 20× reduction (from 1e-4 to 5e-6) is a standard recipe for careful finetuning of pretrained transformers (analogous to BERT finetuning practices). If `freeze_clip=true` in the ablation, this parameter group is simply absent.

---

### 3. `clip_live` Task Embedding Format

**Decision:** Pack tokenized task descriptions as a `(N, 154)` LongTensor — `[:, :77]` = input_ids, `[:, 77:]` = attention_mask — and pass it through the existing LIBERO dataloader as `task_emb`.

**Why not precompute CLIP text embeddings:**
Precomputed embeddings are fixed tensors — gradients cannot flow back to the text encoder. To finetune CLIP text weights, the raw token IDs must be available at training time so the forward pass through CLIPTextModel happens inside the policy's computation graph.

**Why this packing trick:**
LIBERO's `SequenceVLDataset` expects `task_emb` to be a tensor of shape `(embedding_dim,)` per task. Packing token IDs and attention masks as a 154-dim integer tensor reuses this exact interface with zero changes to `datasets.py` or `main.py`'s dataset construction logic. The policy unpacks them internally. This minimizes changes to the existing codebase.

**Why max_length=77:**
CLIP's text transformer has a fixed positional encoding for 77 tokens (including BOS/EOS). This is the standard CLIP tokenization limit.

---

### 4. Denoising Backbone: DiT Blocks with Cross-Attention

**Decision:** Each block applies adaLN-Zero (for flow timestep conditioning), then CrossAttn(action→visual), then CrossAttn(action→language, masked), then SelfAttn, then FFN.

**Why adaLN-Zero for timestep conditioning (DiT component):**
Flow matching requires the denoising network to know *where in the flow trajectory* it is (i.e., the interpolation time `t ∈ [0,1]`). Standard layer normalization has fixed scale/shift; adaLN-Zero (from Peebles & Xie 2023) uses the timestep embedding to predict per-layer scale, shift, and gating parameters, initialized to produce identity transforms. This is more expressive than concatenating `t` as a token (which requires the model to learn to use it via attention). adaLN-Zero is the standard choice in DiT-based diffusion/flow models.

**Why cross-attention for visual and language (not concatenation):**
Concatenating visual (49 tokens) + language (77 tokens) + action (10 tokens) = 136 tokens would make self-attention O(136²) per layer, and the action tokens would have to compete with 126 conditioning tokens in the attention matrix. Cross-attention separates the roles: action tokens are always queries, visual/language are keys/values. The number of action tokens (10) controls computational cost; adding more conditioning tokens doesn't change the forward pass size.

**Why two separate cross-attention operations (visual then language) rather than one joint cross-attention:**
Visual and language provide fundamentally different information — spatial/perceptual vs. semantic/symbolic. Separate cross-attention layers let the model develop distinct attention patterns for each modality. Joint cross-attention over concatenated [visual | language] tokens would dilute this, as the model would need to learn to separate them implicitly.

**Why keep existing `Attention` and `TransformerFeedForwardNN` from LIBERO's `transformer_modules.py`:**
Reusing existing, tested modules reduces code and risk. The LIBERO `Attention` module already handles multi-head attention with optional masking. We wrap it rather than rewrite it.

**Why SelfAttn after cross-attention (not before):**
Action tokens first gather conditioning information from visual and language contexts, then refine their mutual temporal relationships. This ordering mirrors standard practice in cross-attention decoders (e.g., Transformer decoder: self-attn → cross-attn → FFN, except we separate the two cross-attn streams).

**Depth = 6, heads = 8, embed_size = 512:**
These are conservative hyperparameters that keep the model trainable on limited robotics data (~400 demos). The π-CLIP report uses 4+4 layers; 6 blocks gives more capacity for the hybrid architecture while staying well below the compute budget.

---

### 5. Conditional Flow Matching (CFM) Loss

**Decision:** Train with CFM using straight-line ODE paths: `x_t = (1-t)x_0 + t*x_1`, target velocity `u_t = x_1 - x_0`, loss `MSE(v_pred, u_t)`.

**Why flow matching instead of DDPM diffusion:**
- No noise schedule hyperparameters (β schedule, etc.) — straight-line paths between clean data and noise are parameter-free.
- Fewer inference steps needed: 20 Euler steps vs. 100–1000 for DDPM, because the learned vector field is simpler (straight lines, not curved diffusion paths).
- Empirically comparable or better sample quality on robotics tasks (Lipman et al. 2022).
- Facebook's `flow-matching` package provides `ConditionalFlowMatcher.sample_location_and_conditional_flow()` which handles the sampling in one call.

**Why action chunking (T=10):**
Predicting the next 10 actions at once (rather than one at a time) provides temporal consistency — the policy plans a short horizon rather than reacting myopically. Chunk size 10 at 20Hz = 0.5 seconds of planned motion, appropriate for short manipulation primitives. At inference, only the first predicted action is executed (the rest are discarded or used for smoothing).

**Why x_0 = clean action, x_1 = noise (not the reverse):**
Following the π-CLIP report convention: clean data is `x_0`, noise is `x_1`. Inference integrates from noise (`t=1`) to clean action (`t=0`). The velocity field points from noise toward data. This matches the `flow_matching` package's API.

**Why normalize actions (per-dim mean/std):**
Flow matching assumes the data distribution is reasonably well-conditioned. Raw LIBERO actions span [-1, 1] for most dimensions but can have very different scales across DOF. Per-dimension normalization ensures unit-variance inputs to the flow, making the Gaussian noise prior `N(0,I)` a good starting distribution and stabilizing training.

---

### 6. Dual-Optimizer Training

**Decision:** Two AdamW parameter groups: backbone+head at LR=1e-4, CLIP text encoder at LR=5e-6. Cosine LR schedule with 5-epoch linear warmup for both.

**Why separate LRs:**
The randomly-initialized backbone needs a large LR to train quickly. The pretrained CLIP text encoder needs a small LR to adapt without forgetting. Using a single LR for both either trains the backbone too slowly or destroys CLIP's pretrained representations.

**Why AdamW (not Adam):**
Weight decay (1e-4) regularizes both the backbone and the text encoder, reducing overfitting on the small LIBERO dataset. AdamW applies weight decay correctly (decoupled from gradient scaling), unlike Adam with L2 regularization.

**Why cosine schedule + warmup:**
Warmup (5 epochs linear ramp) prevents large early gradient steps from destabilizing the pretrained CLIP weights. Cosine decay reduces the LR smoothly as training converges, avoiding oscillation around minima late in training.

**Ablation condition (`freeze_clip=true`):**
When CLIP is frozen, the CLIP parameter group is simply omitted from the optimizer. Same code path, no architectural change. This isolates the effect of CLIP finetuning from the effect of per-token cross-attention.

---

### 7. Checkpoint Strategy & Validation Loss

**Decision:** Save `checkpoint_epoch_{N:04d}.pth` every 100 training epochs; also save `checkpoint_latest.pth` every epoch. Compute flow matching validation loss (CFM MSE) every 10 epochs on a held-out 10% demo split. Log to wandb as `val/flow_loss`.

**Why checkpoint every 100 epochs:**
Allows post-hoc evaluation at multiple training stages (e.g., epoch 200 vs. 500 vs. 1000) to understand when language conditioning begins to help. Enables recovery if a run diverges.

**Why val flow loss (not val success rate) at each checkpoint:**
Running environment rollouts every epoch is prohibitively slow (each rollout takes ~10 seconds × 20 rollouts × 10 tasks = 33 minutes per epoch eval). Flow matching loss on held-out demos is cheap, computed in one forward pass over the val set. It serves as a proxy for overfitting: if train loss decreases but val loss rises, the model is memorizing demonstrations.

**Why 10% val split:**
With ~50 demos × 8 tasks = 400 training trajectories, 10% = 40 held-out trajectories gives a stable estimate of generalization without sacrificing too much training data.

**Why eval rollouts every 10 epochs:**
Task success rate is the primary metric, but it's expensive. Every 10 epochs is frequent enough to catch training problems early without dominating wall-clock time.

---

### 8. OOD Evaluation Setup

**Decision:** Hold out the last 2 task IDs per suite (20%) as OOD test tasks; train on tasks 0–7.

**Why not use LIBERO-10 (standard OOD benchmark):**
LIBERO-10 contains long-horizon tasks that require lifelong learning (sequential fine-tuning), making it a poor match for our multitask BC setting. The controlled suites (Spatial/Object/Goal) have comparable difficulty across tasks and controlled distribution shifts, making them better for isolating the effect of language conditioning.

**Why deterministic task ID split (not random):**
Reproducibility. Any collaborator running the same experiment gets the same train/test split. The last 2 tasks by ID is a simple, unambiguous rule.

---

### 9. Baseline: Existing `BCTransformerPolicy` with CLIP Pooled Embeddings

**Decision:** Use LIBERO's existing `BCTransformerPolicy` (CLIP pooled CLS + BERT, frozen) as the baseline, trained with `task_embedding_format=clip`.

**Why this baseline:**
It represents the prior-work approach: frozen, pooled language embeddings fused via FiLM and MLP into a temporal transformer. Using it requires no new code and ensures the comparison is against a well-tuned existing system rather than a hastily implemented alternative.

**Three-way comparison:**
- `BCTransformerPolicy` (CLIP CLS, frozen, pooled): prior-work baseline
- `BCClipFlowPolicy` with `freeze_clip=true` (CLIP per-token, frozen): ablation — isolates per-token cross-attention from CLIP finetuning
- `BCClipFlowPolicy` with `freeze_clip=false` (CLIP per-token, finetuned): proposed method

This lets us separately attribute improvements to (a) per-token cross-attention architecture and (b) CLIP finetuning.

---

### 10. `clip_multitask.yaml` — New Algo Config

**Decision:** Create a new `CLIPMultitask` algorithm class extending the existing `Multitask` base, with overridden `setup_optimizer()` for dual-optimizer and added `clip_lr` config field.

**Why a separate algo class:**
The existing `Multitask` algo creates a single optimizer over all policy parameters. We need two separate LRs without changing the existing algo's behavior for other policies.

---

### 11. Action Statistics Preprocessing

**Decision:** Precompute per-dimension mean and std over all training demo actions; save to `action_stats.npz`; load at policy init time.

**Why precompute rather than compute online:**
Computing stats over all HDF5 demos at policy init time would require reading the entire dataset before training starts. Precomputing once and loading a small `.npz` is fast and clean.

---

### 12. Analysis Scripts (NLP Contribution)

Beyond task success rate, two analysis scripts provide interpretable evidence for the paper:

**`analyze_embeddings.py`:**
- t-SNE visualization of CLIP [EOS] token embeddings for all 30 tasks, comparing pretrained vs. finetuned distributions. Expected: finetuned clusters tighter by task semantics.
- Intra-cluster cosine similarity (mean pairwise similarity within each suite) before/after finetuning.
- Nearest-neighbor retrieval: for each OOD task, find the 3 most similar train tasks by embedding distance. Finetuned embeddings should retrieve more semantically relevant neighbors.

**`visualize_attention.py`:**
- For each action timestep in an eval rollout, extract cross-attention weights from the last language cross-attention layer.
- Plot heatmap over 77 token positions: which words does each action step attend to?
- Expected: "pick" / "grasp" attended during approach; object names attended during contact; "place" / "put" attended during release.

These scripts directly support the paper's NLP framing: *does finetuning improve semantic grounding, and can we see where the policy grounds language into actions?*

---

## Files to Create

| File | Purpose |
|---|---|
| `libero/lifelong/models/modules/dit_modules.py` | `TimestepEmbedder`, `CrossAttentionBlock`, `DiTCrossAttnBlock`, `DiTCrossAttnBackbone` |
| `libero/lifelong/models/bc_clip_flow_policy.py` | `BCClipFlowPolicy` — main policy class |
| `libero/lifelong/algos/clip_multitask.py` | `CLIPMultitask` — dual-optimizer training algo |
| `libero/configs/policy/bc_clip_flow_policy.yaml` | Policy hyperparameters |
| `libero/configs/lifelong/clip_multitask.yaml` | Training schedule, LRs, checkpoint/eval intervals |
| `scripts/compute_action_stats.py` | Precompute action mean/std over training demos |
| `scripts/eval_checkpoints.py` | Batch checkpoint evaluation + multi-policy comparison table |
| `scripts/analyze_embeddings.py` | t-SNE, cosine similarity, nearest-neighbor analysis |
| `scripts/visualize_attention.py` | Cross-attention heatmaps over token positions |
| `tests/test_bc_clip_flow_policy.py` | Unit tests for all new components |

## Files to Modify

| File | Change |
|---|---|
| `libero/lifelong/utils.py` | Add `clip_live` branch to `get_task_embs()` |
| `libero/lifelong/models/__init__.py` | Import `BCClipFlowPolicy` |
| `libero/lifelong/algos/__init__.py` | Import `CLIPMultitask` |
| `libero/lifelong/main.py` | Add `n_train_tasks` OOD split filter |
| `libero/configs/config.yaml` | Add `n_train_tasks: null` field |

---

## Implementation Order (with dependencies)

```
[1] dit_modules.py          ← no deps (pure PyTorch)
      ↓
[2] bc_clip_flow_policy.py  ← depends on [1] + existing Attention/FFN
      ↓
[3] utils.py patch          ← add clip_live (no deps)
[4] compute_action_stats.py ← requires datasets downloaded first
      ↓
[5] clip_multitask.py       ← depends on [2]
[6] Configs (yamls)         ← depends on [2],[5]
[7] Register imports        ← depends on [2],[5]
[8] main.py patch           ← depends on [6],[7]
      ↓
[9] Unit tests              ← run after each step above
      ↓
[10] Dataset download + action stats
[11] Training runs
[12] Analysis scripts       ← after training completes
```

---

## Hydra Config Hierarchy

```
libero/configs/
├── config.yaml             MODIFY: add n_train_tasks: null
├── policy/
│   └── bc_clip_flow_policy.yaml   NEW
│       ├── policy_type: BCClipFlowPolicy
│       ├── freeze_clip: false
│       ├── action_chunk_size: 10
│       ├── embed_size: 512 / num_heads: 8 / depth: 6 / ff_hidden: 2048
│       ├── action_stats_path: "action_stats.npz"
│       └── defaults: color_aug + translation_aug (reuse existing)
└── lifelong/
    └── clip_multitask.yaml  NEW: algo: CLIPMultitask, eval_in_train: false
```

---

## Experiment Commands

```bash
# 0. Install + download
pip install flow-matching
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial libero_object libero_goal

# 1. Precompute action normalization stats (train tasks 0-7 only)
python scripts/compute_action_stats.py \
  --suites libero_spatial libero_object libero_goal \
  --n_train_tasks 8 --out action_stats.npz

# 2a. Proposed: finetuned CLIP text
for SUITE in LIBERO_SPATIAL LIBERO_OBJECT LIBERO_GOAL; do
  python libero/lifelong/main.py benchmark_name=$SUITE \
    policy=bc_clip_flow_policy lifelong=clip_multitask \
    task_embedding_format=clip_live train.n_epochs=500 \
    n_train_tasks=8 use_wandb=true
done

# 2b. Ablation: frozen CLIP text
for SUITE in LIBERO_SPATIAL LIBERO_OBJECT LIBERO_GOAL; do
  python libero/lifelong/main.py benchmark_name=$SUITE \
    policy=bc_clip_flow_policy policy.freeze_clip=true \
    lifelong=clip_multitask task_embedding_format=clip_live \
    train.n_epochs=500 n_train_tasks=8 use_wandb=true
done

# 2c. Baseline: existing BCTransformerPolicy
for SUITE in LIBERO_SPATIAL LIBERO_OBJECT LIBERO_GOAL; do
  python libero/lifelong/main.py benchmark_name=$SUITE \
    policy=bc_transformer_policy task_embedding_format=clip \
    lifelong=multitask train.n_epochs=500 n_train_tasks=8 use_wandb=true
done

# 3. OOD evaluation (tasks 8-9, held out)
python scripts/eval_checkpoints.py \
  --experiment_dirs experiments/LIBERO_GOAL/clip_multitask/proposed \
                    experiments/LIBERO_GOAL/clip_multitask/frozen_ablation \
                    experiments/LIBERO_GOAL/multitask/bc_transformer \
  --benchmark LIBERO_GOAL --ood_task_ids 8 9 --device_id 0

# 4. Embedding analysis
python scripts/analyze_embeddings.py --suite LIBERO_GOAL \
  --proposed experiments/LIBERO_GOAL/clip_multitask/proposed/checkpoint_latest.pth

# 5. Attention visualization
python scripts/visualize_attention.py --suite LIBERO_GOAL --task_id 0
```

---

## Verification Checklist

1. `python -c "from libero.lifelong.models.bc_clip_flow_policy import BCClipFlowPolicy; print('OK')"`
2. `pytest tests/test_bc_clip_flow_policy.py -v` — all 6 unit tests pass
3. Forward pass: dummy `{obs.agentview_rgb (B,1,3,128,128), actions (B,10,7), task_emb (B,154)}` → `compute_loss` returns scalar
4. `get_action(data)` returns shape `(B, 7)`
5. Dual optimizer: `[g['lr'] for g in optimizer.param_groups]` shows `[1e-4, 5e-6]`
6. `clip_live` roundtrip: pack 154-dim → unpack → CLIPTextModel → `(B, 77, 512)`
7. After 100 training epochs: `checkpoint_epoch_0100.pth` exists
8. Wandb shows `val/flow_loss` logged every 10 epochs
9. `config.json` in experiment dir confirms `n_train_tasks=8` (OOD integrity)

---

## Implementation Log

> Running log of progress, issues encountered, and resolutions. Updated continuously.

### 2026-03-11 — Session 1 & 2: Full implementation

**Confirmed decisions:**
- `flow-matching` package (pip) for CFM correctness
- `use_wandb=true` (not `wandb.en=true`)
- OOD split via `n_train_tasks=8` in `main.py`
- `random_split` (seeded) for 10% val split in `CLIPMultitask`
- `n_epochs=500` in experiment commands
- Custom `CrossAttention` (not timm/transformers) — cleaner integration with existing `Attention` class

**All files implemented:**
- [x] `libero/lifelong/models/modules/dit_modules.py` — TimestepEmbedder, CrossAttention, DiTCrossAttnBlock, DiTCrossAttnBackbone
- [x] `libero/lifelong/models/bc_clip_flow_policy.py` — BCClipFlowPolicy (167M params, 15.9 GFLOPs)
- [x] `libero/lifelong/utils.py` (patch) — added `clip_live` branch in `get_task_embs()`
- [x] `libero/lifelong/algos/clip_multitask.py` — CLIPMultitask with dual AdamW, cosine+warmup LR, val split, checkpointing
- [x] `libero/configs/policy/bc_clip_flow_policy.yaml`
- [x] `libero/configs/lifelong/clip_multitask.yaml` — uses `# @package _global_` to set `task_embedding_format: clip_live`
- [x] `libero/lifelong/models/__init__.py` (patch)
- [x] `libero/lifelong/algos/__init__.py` (patch)
- [x] `libero/lifelong/main.py` (patch) — `n_train_tasks` filter, CLIPMultitask dispatch
- [x] `libero/configs/config.yaml` (patch) — added `n_train_tasks: null`
- [x] `scripts/compute_action_stats.py`
- [x] `scripts/eval_policy.py` — standalone headless eval wrapper
- [x] `tests/test_bc_clip_flow_policy.py` — 6/6 tests passing
- [x] `IMPLEMENTATION.md` — full docs

**Issues & resolutions:**
- `robomimic` install blocked by `egl_probe` wheel build → installed `robomimic==0.2.0 --no-deps` + missing deps individually
- `~/.libero/config.yaml` didn't exist → created it with Python defaults to suppress interactive `input()` prompt
- `task_embedding_format` defaults to `"bert"` in root config → set `task_embedding_format: clip_live` via `# @package _global_` in `clip_multitask.yaml`
- `persistent_workers=True` requires `num_workers>0` → guarded with `num_workers > 0`
- `CLIPMultitask` fell into `else` branch (per-task sequential) in `main.py` → added `"CLIPMultitask"` to the `if` condition
- `adaLN-Zero` zero-init action_head blocks backbone gradients on first pass → documented in tests; backbone gets grad from step 2 onwards

**Test training:**
- 10-epoch smoke test on RTX 4080 Laptop GPU: **exit code 0 ✓** (2 tasks, batch=8, num_workers=0)
- All 3 datasets downloaded: `libero_spatial` (10 tasks), `libero_object` (10 tasks), `libero_goal` (10 tasks)
- `action_stats.npz` computed from libero_spatial tasks 0-7 (400 demos, 48K timesteps)

**Additional issues resolved:**
- `conda run -n libero` in dev shell secretly ran `ribozyme` env (Python 3.12) — all packages are in `ribozyme`, not the `libero` env (Python 3.8)
- `train.sh` updated to drop `conda run` wrapper; uses active env's `python` directly
- stdout buffering swallowed epoch logs when piped → fixed with `python -u` + `PYTHONUNBUFFERED=1`

---

### DONE ✅

| Item | Status |
|------|--------|
| `dit_modules.py` | ✅ |
| `bc_clip_flow_policy.py` (167M params) | ✅ |
| `clip_multitask.py` (dual AdamW, cosine LR, val split) | ✅ |
| Config files + all patches | ✅ |
| 6/6 unit tests passing | ✅ |
| `compute_action_stats.py` | ✅ ran on libero_spatial |
| `eval_policy.py` headless eval wrapper | ✅ |
| `IMPLEMENTATION.md` | ✅ |
| `train.sh` one-command launcher | ✅ |
| All 3 datasets downloaded | ✅ |
| 10-epoch GPU smoke test | ❌ NOT YET — stdout buffering hid logs; re-run with fix |
| `environment.yml` for cluster transfer | ✅ written (ribozyme env, clean minimal) |

### NOT YET DONE ❌

| Item | Notes |
|------|-------|
| **10-epoch smoke test (confirmed visible output)** | Run `./train.sh libero_spatial train.n_epochs=10 n_train_tasks=2` |
| Recompute `action_stats.npz` across all 3 suites | Currently spatial-only; train.sh auto-computes on fresh cluster |
| **Full 500-epoch training** | `./train.sh` (wandb off by default; add `use_wandb=true` for cluster) |
| wandb verified live | Run with `use_wandb=true` and confirm metrics appear in dashboard |
| OOD eval (tasks 8-9) | `eval_policy.py --task_ids 8 9` after training |
| Headless parallel eval on all 3 suites | `eval_policy.py` on trained checkpoint |

### To run smoke test now

```bash
conda activate ribozyme
cd /home/jeff/LIBERO
./train.sh libero_spatial train.n_epochs=10 n_train_tasks=2 train.batch_size=8 train.num_workers=0
# watch: tail -f train.log | grep -E 'Epoch|val|Error'
```

### To launch full training on cluster

```bash
# 1. Setup env
conda env create -f environment.yml
conda activate libero
pip install -e .

# 2. Run (action_stats auto-computed if missing)
cd /path/to/LIBERO
./train.sh                        # libero_spatial, no wandb
./train.sh libero_spatial use_wandb=true  # with wandb

# Optional: recompute stats manually
python scripts/compute_action_stats.py \
    --suites libero_spatial libero_object libero_goal \
    --n_train_tasks 8 --out action_stats.npz

# Launch — output streams live to terminal AND train.log
./train.sh

# Watch in another terminal:
tail -f train.log | grep -E 'Epoch|val|succ|Error'
```
