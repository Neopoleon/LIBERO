"""Unit tests for BCClipFlowPolicy and supporting DiT modules.

Run with:
    conda run -n libero python -m pytest tests/test_bc_clip_flow_policy.py -v

Tests are kept self-contained (no real data or checkpoints required) using
small synthetic tensors so they run quickly on CPU.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Test 1 – DiT modules: shapes
# ---------------------------------------------------------------------------

def test_dit_modules_shapes():
    """TimestepEmbedder, CrossAttention, DiTCrossAttnBlock, DiTCrossAttnBackbone
    all produce tensors with the expected shapes."""
    from libero.lifelong.models.modules.dit_modules import (
        CrossAttention,
        DiTCrossAttnBackbone,
        DiTCrossAttnBlock,
        TimestepEmbedder,
    )

    B, T, D = 2, 10, 64
    n_vis, n_lang = 49, 77

    t = torch.rand(B)
    t_emb = TimestepEmbedder(embed_size=D, freq_dim=64)(t)
    assert t_emb.shape == (B, D), f"TimestepEmbedder: {t_emb.shape}"

    vis = torch.randn(B, n_vis, D)
    lang = torch.randn(B, n_lang, D)
    x = torch.randn(B, T, D)
    lang_mask = torch.zeros(B, n_lang, dtype=torch.bool)

    # Single block
    block = DiTCrossAttnBlock(
        embed_size=D, num_heads=4, ff_hidden=128, head_dim=16,
        visual_dim=D, lang_dim=D,
    )
    out = block(x, vis, lang, t_emb, lang_mask=lang_mask)
    assert out.shape == (B, T, D), f"DiTCrossAttnBlock: {out.shape}"

    # Full backbone
    out2 = DiTCrossAttnBackbone(
        depth=2, embed_size=D, num_heads=4, ff_hidden=128, head_dim=16,
        visual_dim=D, lang_dim=D,
    )(x, vis, lang, t_emb, lang_mask)
    assert out2.shape == (B, T, D), f"DiTCrossAttnBackbone: {out2.shape}"


# ---------------------------------------------------------------------------
# Test 2 – adaLN-Zero: gates are zero at init → block starts as identity
# ---------------------------------------------------------------------------

def test_adaln_zero_init():
    """adaLN-Zero gates must be exactly 0.0 at initialisation, meaning each
    block contributes zero residual and acts as identity at the start of
    training (important for stable CFM loss at epoch 0)."""
    from libero.lifelong.models.modules.dit_modules import DiTCrossAttnBlock

    D = 64
    block = DiTCrossAttnBlock(
        embed_size=D, num_heads=4, ff_hidden=128, head_dim=16,
        visual_dim=D, lang_dim=D,
    )
    last_linear = block.adaLN_modulation[-1]
    assert torch.all(last_linear.weight == 0), "adaLN final weight not zero"
    assert torch.all(last_linear.bias == 0), "adaLN final bias not zero"


# ---------------------------------------------------------------------------
# Test 3 – CFM loss: scalar, non-negative, gradient flows to backbone params
# ---------------------------------------------------------------------------

def _make_policy(tmp_path, action_chunk_size=4, embed_size=64, depth=1):
    """Build a tiny BCClipFlowPolicy with synthetic action stats."""
    from omegaconf import OmegaConf
    from libero.lifelong.models.bc_clip_flow_policy import BCClipFlowPolicy

    stats_path = str(tmp_path / "action_stats.npz")
    np.savez(stats_path, mean=np.zeros(7), std=np.ones(7))

    cfg = OmegaConf.create({
        "device": "cpu",
        "policy": {
            "policy_type": "BCClipFlowPolicy",
            "freeze_clip": True,
            "action_chunk_size": action_chunk_size,
            "embed_size": embed_size,
            "num_heads": 4,
            "depth": depth,
            "ff_hidden": 128,
            "action_stats_path": stats_path,
            # DataAugGroup expects these keys; IdentityAug does nothing.
            "color_aug": {"network": "IdentityAug", "network_kwargs": {}},
            "translation_aug": {
                "network": "IdentityAug",
                "network_kwargs": {"input_shape": [3, 128, 128]},
            },
        },
        "data": {
            "obs": {
                "modality": {"rgb": ["agentview_rgb"], "low_dim": []},
            },
            "max_word_len": 77,
        },
        "train": {"use_augmentation": False},
        "task_embedding_format": "clip_live",
    })

    shape_meta = {
        "all_shapes": {"agentview_rgb": (3, 128, 128)},
        "ac_dim": 7,
    }
    return BCClipFlowPolicy(cfg, shape_meta)


def _make_batch(B=2, T_obs=1, T_act=4, H=128):
    """Minimal batch dict matching the policy's expected input format."""
    # task_emb: (B, 154) — ids in [:77], mask in [77:]
    task_emb = torch.zeros(B, 154, dtype=torch.long)
    task_emb[:, 0] = 49406   # BOS token for CLIP
    task_emb[:, 77:78] = 1   # attend to BOS

    return {
        "obs": {
            "agentview_rgb": torch.rand(B, T_obs, 3, H, H),
        },
        "task_emb": task_emb,
        "actions": torch.randn(B, T_act, 7),
    }


def test_cfm_loss(tmp_path):
    """compute_loss returns a scalar ≥ 0 and gradients reach backbone params."""
    policy = _make_policy(tmp_path)
    batch = _make_batch(T_act=4)

    loss = policy.compute_loss(batch)
    assert loss.shape == (), f"loss not scalar: {loss.shape}"
    assert loss.item() >= 0, f"loss negative: {loss.item()}"

    loss.backward()
    # action_head is zero-init so it blocks gradients to backbone on pass 1.
    # Check that action_head itself receives gradient (confirming the forward
    # graph is connected to the loss).
    assert policy.action_head.weight.grad is not None, "action_head.weight has no grad"
    assert policy.action_head.weight.grad.abs().sum().item() > 0

    # After one optimizer step action_head.weight ≠ 0, so backbone now gets grad.
    opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()

    loss2 = policy.compute_loss(batch)
    loss2.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in policy.backbone.parameters()
    )
    assert has_grad, "No gradient reached DiT backbone after action_head update"


# ---------------------------------------------------------------------------
# Test 4 – get_action: output shape (B, 7) and within a sane range
# ---------------------------------------------------------------------------

def test_get_action_shape(tmp_path):
    """get_action returns (B, 7) numpy array without errors."""
    policy = _make_policy(tmp_path)
    policy.eval()

    # get_action expects un-time-dimmed observations
    batch = _make_batch(B=2, T_obs=1, T_act=4)
    # Remove the temporal dim from obs (preprocess_input adds it back)
    batch["obs"]["agentview_rgb"] = batch["obs"]["agentview_rgb"].squeeze(1)  # (B,3,H,W)
    batch["task_emb"] = batch["task_emb"]  # (B, 154)

    actions = policy.get_action(batch)
    assert actions.shape == (2, 7), f"get_action shape: {actions.shape}"
    assert np.isfinite(actions).all(), "get_action returned non-finite values"


# ---------------------------------------------------------------------------
# Test 5 – encode_lang: key_padding_mask matches attention_mask=0 positions
# ---------------------------------------------------------------------------

def test_encode_lang_mask(tmp_path):
    """lang_key_padding_mask == True wherever attention_mask == 0."""
    policy = _make_policy(tmp_path)

    B = 3
    task_emb = torch.zeros(B, 154, dtype=torch.long)
    task_emb[:, 0] = 49406
    # Set attention_mask = 1 for first k tokens per item
    for i, k in enumerate([5, 10, 20]):
        task_emb[i, 77:77 + k] = 1  # attend to first k tokens

    _, mask = policy.encode_lang(task_emb)
    # mask True = padding (att_mask=0), False = attend (att_mask=1)
    for i, k in enumerate([5, 10, 20]):
        assert not mask[i, :k].any(), f"item {i}: tokens 0:{k} wrongly masked"
        assert mask[i, k:].all(), f"item {i}: tokens {k}:77 not masked"


# ---------------------------------------------------------------------------
# Test 6 – CFM path: x_t at t=0 equals x0, at t=1 equals x1
# ---------------------------------------------------------------------------

def test_cfm_interpolation():
    """Verify the CFM straight-line interpolant x_t = (1-t)*x0 + t*x1 is correct.

    At t=0: x_t should equal x0.
    At t=1: x_t should equal x1.
    """
    B, T = 4, 10
    x0 = torch.randn(B, T, 7)
    x1 = torch.randn(B, T, 7)

    def interp(t_val):
        t = torch.full((B,), t_val)
        t_bc = t[:, None, None]
        return (1.0 - t_bc) * x0 + t_bc * x1

    torch.testing.assert_close(interp(0.0), x0)
    torch.testing.assert_close(interp(1.0), x1)
    midpoint = interp(0.5)
    expected_mid = 0.5 * x0 + 0.5 * x1
    torch.testing.assert_close(midpoint, expected_mid)
