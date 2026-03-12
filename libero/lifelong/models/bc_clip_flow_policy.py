"""BCClipFlowPolicy: π-CLIP-DiT flow matching policy for LIBERO.

Implements Conditional Flow Matching (CFM) over action chunks, conditioned on
CLIP visual patch features and CLIP text embeddings via a DiT cross-attention
backbone.

Training objective: regress the velocity field u_t = x1 - x0, where
  x_t = (1 - t) * x0 + t * x1,  x0 ~ data,  x1 ~ N(0, I).

Inference: Euler integration from t=1 (noise) to t=0 (data) in 20 steps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import robomimic.utils.tensor_utils as TensorUtils
from transformers import CLIPModel

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.data_augmentation import DataAugGroup
from libero.lifelong.models.modules.dit_modules import (
    DiTCrossAttnBackbone,
    TimestepEmbedder,
)


class BCClipFlowPolicy(BasePolicy):
    """Flow matching policy using CLIP encoders and a DiT cross-attention backbone.

    Args:
        cfg:        Hydra config. Relevant fields under cfg.policy:
                      freeze_clip, action_chunk_size, embed_size, num_heads,
                      depth, ff_hidden, action_stats_path.
        shape_meta: Dict with key "all_shapes" mapping obs names → shape tuples.
    """

    def __init__(self, cfg, shape_meta: dict) -> None:
        super().__init__(cfg, shape_meta)

        policy_cfg = cfg.policy
        embed_size: int = policy_cfg.embed_size
        action_chunk_size: int = policy_cfg.action_chunk_size

        self.rgb_key: str = cfg.data.obs.modality.rgb[0]

        # ------------------------------------------------------------------
        # CLIP encoders
        # ------------------------------------------------------------------
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Vision encoder is always frozen — we treat it as a fixed feature extractor.
        for p in self.clip.vision_model.parameters():
            p.requires_grad = False

        if policy_cfg.freeze_clip:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False

        # Project ViT-B/32 patch dim 768 → embed_size
        self.visual_proj = nn.Linear(768, embed_size)

        # ------------------------------------------------------------------
        # Action token components
        # ------------------------------------------------------------------
        self.action_proj = nn.Sequential(nn.Linear(7, embed_size), nn.GELU())
        self.action_pos_emb = nn.Parameter(torch.zeros(action_chunk_size, embed_size))

        # ------------------------------------------------------------------
        # DiT backbone
        # ------------------------------------------------------------------
        self.t_embedder = TimestepEmbedder(embed_size)
        self.backbone = DiTCrossAttnBackbone(
            depth=policy_cfg.depth,
            embed_size=embed_size,
            num_heads=policy_cfg.num_heads,
            ff_hidden=policy_cfg.ff_hidden,
            visual_dim=embed_size,   # after visual_proj: 768 → embed_size
            lang_dim=512,            # CLIPTextModel hidden size for ViT-B/32
        )

        # Zero-init action head so the policy outputs zero velocity at init.
        self.action_head = nn.Linear(embed_size, 7)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

        # ------------------------------------------------------------------
        # Action normalisation statistics (registered as buffers → saved with
        # model checkpoint and moved to device automatically)
        # ------------------------------------------------------------------
        stats = np.load(policy_cfg.action_stats_path)
        self.register_buffer("action_mean", torch.from_numpy(stats["mean"]).float())
        self.register_buffer("action_std", torch.from_numpy(stats["std"]).float())

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def encode_visual(self, obs_rgb: torch.Tensor) -> torch.Tensor:
        """Encode a single observation frame with the frozen CLIP vision encoder.

        Args:
            obs_rgb: (B, T_obs, 3, H, W) raw RGB observations. Only the first
                     temporal frame is used.

        Returns:
            (B, 49, embed_size) projected patch features (CLS token dropped).
        """
        x = obs_rgb[:, 0]  # (B, 3, H, W)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # last_hidden_state: (B, 50, 768) — index 0 is CLS, 1:50 are 7×7 patches
        patches = self.clip.vision_model(pixel_values=x).last_hidden_state[:, 1:]
        return self.visual_proj(patches)  # (B, 49, embed_size)

    def encode_lang(
        self, task_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode packed CLIP token ids + attention mask.

        Args:
            task_emb: (B, 154) LongTensor, first 77 cols are input_ids and
                      last 77 cols are attention_mask values (0 or 1).

        Returns:
            lang_tokens:          (B, 77, 512) CLIP text hidden states.
            lang_key_padding_mask: (B, 77) bool, True = padding position (ignored
                                   by cross-attention).
        """
        input_ids = task_emb[:, :77].long()
        attention_mask = task_emb[:, 77:].long()
        lang_tokens = self.clip.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, 77, 512)
        lang_key_padding_mask = attention_mask == 0  # True = padding
        return lang_tokens, lang_key_padding_mask

    # ------------------------------------------------------------------
    # Action token builder
    # ------------------------------------------------------------------

    def _build_action_tokens(self, actions_normalized: torch.Tensor) -> torch.Tensor:
        """Project normalised actions to token space and add positional embedding.

        Args:
            actions_normalized: (B, T, 7) normalised action chunk.

        Returns:
            (B, T, embed_size) action tokens.
        """
        return self.action_proj(actions_normalized) + self.action_pos_emb.unsqueeze(0)

    # ------------------------------------------------------------------
    # Core forward (training)
    # ------------------------------------------------------------------

    def forward(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity prediction and target for CFM loss.

        Expects ``data`` to have already been preprocessed by
        ``preprocess_input(train_mode=True)``.

        Args:
            data: Dict with keys "actions" (B, T, 7), "obs" containing the
                  RGB key (B, T_obs, 3, H, W), and "task_emb" (B, 154).

        Returns:
            v_pred: (B, T, 7) predicted velocity field.
            u_t:    (B, T, 7) ground-truth conditional velocity (x1 - x0).
        """
        actions = data["actions"]  # (B, T, 7)
        B = actions.shape[0]

        # Normalise ground-truth actions to zero-mean / unit-variance
        x0 = (actions - self.action_mean) / self.action_std.clamp(min=1e-8)

        # Sample noise endpoint and uniform flow timestep
        x1 = torch.randn_like(x0)
        t = torch.rand(B, device=x0.device)  # (B,)

        # Linear interpolation between data and noise
        t_bc = t[:, None, None]           # (B, 1, 1) for broadcasting
        x_t = (1.0 - t_bc) * x0 + t_bc * x1
        u_t = x1 - x0                     # target velocity (constant along path)

        # Build all conditioning signals
        action_tokens = self._build_action_tokens(x_t)
        visual_tokens = self.encode_visual(data["obs"][self.rgb_key])
        lang_tokens, lang_mask = self.encode_lang(data["task_emb"])
        t_emb = self.t_embedder(t)

        out = self.backbone(
            action_tokens, visual_tokens, lang_tokens, t_emb, lang_mask=lang_mask
        )
        v_pred = self.action_head(out)  # (B, T, 7)
        return v_pred, u_t

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, data: dict) -> torch.Tensor:
        """MSE between predicted and ground-truth velocity fields.

        Overrides BasePolicy.compute_loss (which assumes a policy_head API
        not used here).

        Args:
            data: Raw batch dict straight from the dataloader.

        Returns:
            Scalar MSE loss.
        """
        data = self.preprocess_input(data, train_mode=True)
        v_pred, u_t = self.forward(data)
        return F.mse_loss(v_pred, u_t)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_input(self, data: dict, train_mode: bool = True) -> dict:
        """Apply augmentation (train) or add temporal dim (inference).

        DataAugGroup.forward expects a tuple of tensors each shaped
        (B, T_obs, C, H, W).  We pass a single-element tuple containing the
        RGB observation, then unpack the result.

        Args:
            data:       Batch dict.
            train_mode: True during training, False during inference.

        Returns:
            Modified batch dict.
        """
        if train_mode:
            if self.cfg.train.use_augmentation:
                img = data["obs"][self.rgb_key]   # (B, T_obs, 3, H, W)
                (aug_img,) = self.img_aug((img,))
                data["obs"][self.rgb_key] = aug_img
            return data
        else:
            # Add a singleton time dimension to all tensors for inference
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}
            )
            data["task_emb"] = data["task_emb"].squeeze(1)
            return data

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, data: dict) -> np.ndarray:
        """Run Euler ODE integration from noise to action.

        Uses 20-step Euler integration stepping t from 1 → 0 (noise → data).

        Args:
            data: Observation batch dict (no temporal dim yet; preprocess_input
                  adds it).

        Returns:
            (B, 7) numpy array — the first action in the predicted chunk,
            denormalised to the original action scale.
        """
        data = self.preprocess_input(data, train_mode=False)
        B = data["task_emb"].shape[0]
        T = self.cfg.policy.action_chunk_size

        # Start from pure noise at t=1
        x = torch.randn(B, T, 7, device=self.device)

        # Pre-compute conditioning once (shared across all ODE steps)
        visual_tokens = self.encode_visual(data["obs"][self.rgb_key])
        lang_tokens, lang_mask = self.encode_lang(data["task_emb"])

        # Euler integration: t = 1 → 0, step size dt = 1/n_steps
        n_steps = 20
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = 1.0 - i * dt   # decreasing: 1.0, 0.95, …, 0.05
            t_tensor = torch.full((B,), t_val, device=self.device)
            t_emb = self.t_embedder(t_tensor)
            action_tokens = self._build_action_tokens(x)
            out = self.backbone(
                action_tokens, visual_tokens, lang_tokens, t_emb, lang_mask=lang_mask
            )
            v = self.action_head(out)   # (B, T, 7) predicted dx/dt
            x = x - dt * v             # step backward along flow (t → t - dt)

        # Denormalise and return first action in chunk
        x = x * self.action_std + self.action_mean
        return x[:, 0, :].cpu().numpy()  # (B, 7)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear any stateful history (none for this policy)."""
        super().reset()
