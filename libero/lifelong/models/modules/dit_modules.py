"""DiT-style modules for language-conditioned flow matching policy.

Implements four building blocks used by BCClipFlowPolicy:

  TimestepEmbedder    -- sinusoidal + MLP encoding of flow timestep t ∈ [0,1]
  CrossAttention      -- multi-head cross-attention (Q from x, K/V from context)
  DiTCrossAttnBlock   -- single transformer block with dual cross-attn + adaLN-Zero
  DiTCrossAttnBackbone -- stack of DiTCrossAttnBlocks

Per-block processing order (see DiTCrossAttnBlock.forward):
  1. CrossAttn(x ← visual_tokens)          visual scene grounding
  2. CrossAttn(x ← lang_tokens, mask)      task instruction grounding
  3. adaLN-Zero SelfAttn(x)               temporal coherence, flow-t conditioned
  4. adaLN-Zero FFN(x)                    feature mixing, flow-t conditioned

adaLN-Zero (Peebles & Xie 2023, DiT): the timestep embedding predicts per-layer
shift/scale/gate parameters via a zero-initialized linear, so every block starts
as an identity transform — critical for stable early training.

Reuses from transformer_modules.py:
  Attention                -- self-attention (used in step 3)
  TransformerFeedForwardNN -- FFN (used in step 4)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from libero.lifelong.models.modules.transformer_modules import (  # noqa: E402
    Attention,
    TransformerFeedForwardNN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply adaptive layer-norm modulation: x * (1 + scale) + shift.

    Args:
        x:     (B, T, D) token sequence.
        shift: (B, D)    per-sample shift predicted from timestep embedding.
        scale: (B, D)    per-sample scale predicted from timestep embedding.

    Returns:
        (B, T, D) modulated sequence.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar flow timesteps t ∈ [0, 1] to a dense vector.

    Uses sinusoidal frequency features followed by a 2-layer SiLU MLP,
    matching the standard DiT timestep embedding (Peebles & Xie 2023).

    Args:
        embed_size: Output embedding dimension.
        freq_dim:   Intermediate sinusoidal feature dimension (default 256).
    """

    def __init__(self, embed_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute sinusoidal features for timesteps.

        Args:
            t:   (B,) float timesteps in [0, 1].
            dim: Number of frequency channels (must be even).

        Returns:
            (B, dim) sinusoidal feature matrix.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )  # (half,)
        args = t[:, None].float() * freqs[None]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float timesteps in [0, 1].

        Returns:
            (B, embed_size) timestep embedding.
        """
        return self.mlp(self._sinusoidal(t, self.freq_dim))


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from x, keys/values from context.

    Separate Q and KV projections let the two token sequences live in
    different spaces.  Attention weights are stored in ``self.att_weights``
    after each forward call for downstream visualization.

    Args:
        embed_dim:  Query (action token) dimension.
        num_heads:  Number of attention heads.
        head_dim:   Dimension per head (default 64).
        kv_dim:     Key/value (context) dimension; defaults to embed_dim.
        dropout:    Attention dropout (default 0.0).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int = 64,
        kv_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kv_dim = kv_dim or embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(kv_dim, num_heads * head_dim * 2, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Stored after forward for attention visualization scripts.
        self.att_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, T, embed_dim) -- action tokens (queries).
            context:          (B, S, kv_dim)    -- visual/lang tokens (keys+values).
            key_padding_mask: (B, S) bool       -- True where context positions
                                                   should be ignored (padding).

        Returns:
            (B, T, embed_dim) attended output.
        """
        B, T, _ = x.shape

        # Q from action tokens; K, V from context
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, T, head_dim)

        kv = (
            self.kv_proj(context)
            .reshape(B, -1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (2, B, H, S, head_dim)
        k, v = kv[0], kv[1]  # each (B, H, S, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, S)

        if key_padding_mask is not None:
            # Broadcast (B, S) → (B, 1, 1, S); True = ignore → -inf
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = self.attn_drop(attn.softmax(dim=-1))
        self.att_weights = attn.detach()  # (B, H, T, S) — saved for visualization

        out = rearrange(attn @ v, "b h t d -> b t (h d)")
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# DiTCrossAttnBlock
# ---------------------------------------------------------------------------


class DiTCrossAttnBlock(nn.Module):
    """Single DiT block: dual cross-attention + adaLN-Zero self-attention + FFN.

    Processing order:
      1. pre-norm CrossAttn(x ← visual_tokens)
      2. pre-norm CrossAttn(x ← lang_tokens, key_padding_mask)
      3. adaLN-Zero SelfAttn(x)    (Attention from transformer_modules.py)
      4. adaLN-Zero FFN(x)         (TransformerFeedForwardNN)

    adaLN-Zero initialization: the final Linear of ``adaLN_modulation`` is
    zero-initialized so all gates start at 0 and each block is an identity
    transform at the start of training (Peebles & Xie 2023).

    Args:
        embed_size: Token dimension (must equal CLIP text dim = 512).
        num_heads:  Attention heads for all sub-layers.
        ff_hidden:  FFN hidden dimension.
        head_dim:   Per-head dimension for attention (default 64).
        visual_dim: CLIP visual patch dimension (default 512).
        lang_dim:   CLIP text token dimension (default 512).
        dropout:    Dropout for self-attention and FFN (default 0.0).
    """

    def __init__(
        self,
        embed_size: int = 512,
        num_heads: int = 8,
        ff_hidden: int = 2048,
        head_dim: int = 64,
        visual_dim: int = 512,
        lang_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Layer norms (one per sub-operation for independent normalisation)
        self.ln_visual = nn.LayerNorm(embed_size)
        self.ln_lang = nn.LayerNorm(embed_size)
        self.ln_self = nn.LayerNorm(embed_size)
        self.ln_ffn = nn.LayerNorm(embed_size)

        # Cross-attention: action → CLIP visual patches
        self.visual_cross_attn = CrossAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_dim=visual_dim,
        )

        # Cross-attention: action → CLIP text tokens (supports padding mask)
        self.lang_cross_attn = CrossAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_dim=lang_dim,
        )

        # Self-attention over action chunk — reuse existing tested module
        self.self_attn = Attention(
            dim=embed_size,
            num_heads=num_heads,
            head_output_size=head_dim,
            dropout=dropout,
        )

        # FFN — reuse existing tested module
        self.ffn = TransformerFeedForwardNN(
            dim=embed_size, hidden_dim=ff_hidden, dropout=dropout
        )

        # adaLN-Zero: predicts 6×embed_size params from timestep embedding.
        # Output order: shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn.
        # Final Linear zero-initialized → gates=0 → block is identity at init.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_size, 6 * embed_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        visual_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:             (B, T, embed_size)  -- noisy action chunk tokens.
            visual_tokens: (B, 49, visual_dim) -- frozen CLIP ViT patch features.
            lang_tokens:   (B, 77, lang_dim)   -- CLIP text tokens (may be finetuned).
            t_emb:         (B, embed_size)      -- flow timestep embedding.
            lang_mask:     (B, 77) bool         -- True = padding; None = no mask.

        Returns:
            (B, T, embed_size) updated action tokens.
        """
        # Predict adaLN parameters from timestep (gates zero at init)
        shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = (
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        )  # each (B, embed_size)

        # 1. Cross-attend to visual patches (pre-norm, no adaLN conditioning)
        x = x + self.visual_cross_attn(self.ln_visual(x), visual_tokens)

        # 2. Cross-attend to language tokens (pre-norm, padding positions masked)
        x = x + self.lang_cross_attn(
            self.ln_lang(x), lang_tokens, key_padding_mask=lang_mask
        )

        # 3. adaLN-Zero self-attention (temporal coherence across action chunk)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            _modulate(self.ln_self(x), shift_sa, scale_sa)
        )

        # 4. adaLN-Zero feed-forward network
        x = x + gate_ffn.unsqueeze(1) * self.ffn(
            _modulate(self.ln_ffn(x), shift_ffn, scale_ffn)
        )

        return x


# ---------------------------------------------------------------------------
# DiTCrossAttnBackbone
# ---------------------------------------------------------------------------


class DiTCrossAttnBackbone(nn.Module):
    """Stack of DiTCrossAttnBlocks forming the denoising transformer backbone.

    Args:
        depth:      Number of stacked DiTCrossAttnBlocks (default 6).
        embed_size: Token embedding dimension (default 512).
        num_heads:  Attention heads per block (default 8).
        ff_hidden:  FFN hidden size (default 2048).
        head_dim:   Per-head dimension (default 64).
        visual_dim: CLIP visual patch feature dimension (default 512).
        lang_dim:   CLIP text token dimension (default 512).
        dropout:    Dropout for attention and FFN layers (default 0.0).
    """

    def __init__(
        self,
        depth: int = 6,
        embed_size: int = 512,
        num_heads: int = 8,
        ff_hidden: int = 2048,
        head_dim: int = 64,
        visual_dim: int = 512,
        lang_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTCrossAttnBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    ff_hidden=ff_hidden,
                    head_dim=head_dim,
                    visual_dim=visual_dim,
                    lang_dim=lang_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_size)

    def forward(
        self,
        action_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_tokens: (B, T, embed_size)  -- noisy action chunk tokens.
            visual_tokens: (B, 49, visual_dim) -- CLIP ViT patch features (frozen).
            lang_tokens:   (B, 77, lang_dim)   -- CLIP text tokens (may be finetuned).
            t_emb:         (B, embed_size)      -- flow timestep embedding.
            lang_mask:     (B, 77) bool         -- True = padding position; None = no mask.

        Returns:
            (B, T, embed_size) refined action tokens, ready for action_head Linear.
        """
        x = action_tokens
        for block in self.blocks:
            x = block(x, visual_tokens, lang_tokens, t_emb, lang_mask=lang_mask)
        return self.final_norm(x)
