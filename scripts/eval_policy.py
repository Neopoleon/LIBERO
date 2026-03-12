"""Standalone headless evaluation script for BCClipFlowPolicy checkpoints.

Evaluates a saved checkpoint on one or more LIBERO task suites and reports
per-task and aggregate success rates.  All rendering is off-screen (EGL/OSMesa)
and tasks are evaluated in parallel using SubprocVectorEnv.

Usage::

    # Evaluate latest checkpoint on libero_spatial (train tasks 0-7)
    conda run -n libero python scripts/eval_policy.py \\
        --checkpoint experiments/.../checkpoint_latest.pth \\
        --benchmark libero_spatial \\
        --n_tasks 8 \\
        --device cuda

    # Evaluate all three suites including OOD tasks 8-9
    conda run -n libero python scripts/eval_policy.py \\
        --checkpoint experiments/.../checkpoint_epoch_0500.pth \\
        --benchmark libero_spatial libero_object libero_goal \\
        --n_tasks 10 \\
        --n_eval 20 \\
        --device cuda

    # Evaluate OOD tasks only (8-9) from libero_spatial
    conda run -n libero python scripts/eval_policy.py \\
        --checkpoint experiments/.../checkpoint_latest.pth \\
        --benchmark libero_spatial \\
        --task_ids 8 9 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force off-screen rendering before any MuJoCo/robosuite imports.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import yaml
from easydict import EasyDict
from omegaconf import OmegaConf

from libero.libero.benchmark import get_benchmark
from libero.lifelong.metric import evaluate_one_task_success
from libero.lifelong.utils import torch_load_model, get_task_embs


# ---------------------------------------------------------------------------
# Minimal config scaffolding
# ---------------------------------------------------------------------------

_DEFAULT_OBS_CFG = {
    "modality": {
        "rgb": ["agentview_rgb"],
        "low_dim": ["ee_pos", "ee_ori", "gripper_states", "joint_states"],
    },
    "obs_key_mapping": {
        "agentview_rgb": "agentview_image",
        "ee_pos": "robot0_eef_pos",
        "ee_ori": "robot0_eef_quat",
        "gripper_states": "robot0_gripper_qpos",
        "joint_states": "robot0_joint_pos",
    },
}


def _build_eval_cfg(args) -> EasyDict:
    """Construct a minimal EasyDict config sufficient for evaluate_one_task_success."""
    libero_cfg_path = os.path.expanduser("~/.libero/config.yaml")
    with open(libero_cfg_path) as f:
        libero_paths = yaml.safe_load(f)

    return EasyDict({
        "device": args.device,
        "bddl_folder": libero_paths["bddl_files"],
        "init_states_folder": libero_paths["init_states"],
        "data": {
            "img_h": 128,
            "img_w": 128,
            "obs": _DEFAULT_OBS_CFG,
            "obs_key_mapping": _DEFAULT_OBS_CFG["obs_key_mapping"],
            "max_word_len": 77,
        },
        "task_embedding_format": "clip_live",
        "eval": {
            "n_eval": args.n_eval,
            "num_procs": args.num_procs,
            "use_mp": args.num_procs > 1,
            "max_steps": args.max_steps,
            "save_sim_states": False,
        },
        "lifelong": {
            "algo": "CLIPMultitask",
            "eval_in_train": False,
        },
        "pretrain": False,
    })


# ---------------------------------------------------------------------------
# Algo shim — wraps just the policy for evaluate_one_task_success
# ---------------------------------------------------------------------------

class _AlgoShim:
    """Minimal algo shim exposing the interface expected by evaluate_one_task_success."""

    def __init__(self, policy):
        self.policy = policy

    def eval(self):
        self.policy.eval()

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint saved by CLIPMultitask.")
    p.add_argument("--benchmark", nargs="+",
                   default=["libero_spatial"],
                   help="One or more benchmark suite names.")
    p.add_argument("--n_tasks", type=int, default=None,
                   help="Evaluate the first N tasks of each suite. Defaults to all.")
    p.add_argument("--task_ids", type=int, nargs="+", default=None,
                   help="Explicit task indices to evaluate (overrides --n_tasks).")
    p.add_argument("--n_eval", type=int, default=20,
                   help="Rollouts per task (default: 20).")
    p.add_argument("--num_procs", type=int, default=4,
                   help="Parallel envs per task via SubprocVectorEnv (default: 4).")
    p.add_argument("--max_steps", type=int, default=600,
                   help="Max steps per episode (default: 600).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device (default: cuda if available).")
    p.add_argument("--output", default=None,
                   help="Optional path to save results as .npz.")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = _build_eval_cfg(args)

    # ------------------------------------------------------------------
    # Load policy from checkpoint
    # ------------------------------------------------------------------
    print(f"\n[eval] Loading checkpoint: {args.checkpoint}")
    state_dict, ckpt_cfg = torch_load_model(args.checkpoint)

    # Reconstruct policy using the saved config
    from libero.lifelong.models.bc_clip_flow_policy import BCClipFlowPolicy
    import libero.lifelong.benchmark as bm_api

    # Grab shape_meta from the first benchmark we can load
    probe_bench = get_benchmark(args.benchmark[0])()
    probe_data = probe_bench.get_dataset(0)
    shape_meta = probe_data.shape_meta

    policy = BCClipFlowPolicy(ckpt_cfg, shape_meta)
    policy.load_state_dict(state_dict)
    policy = policy.to(args.device)
    policy.eval()

    algo = _AlgoShim(policy)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    all_results = {}  # suite -> list of (task_name, success_rate)

    for suite_name in args.benchmark:
        print(f"\n{'='*60}")
        print(f" Suite: {suite_name}")
        print(f"{'='*60}")

        benchmark = get_benchmark(suite_name)()
        n_tasks_total = benchmark.n_tasks

        if args.task_ids is not None:
            task_ids = args.task_ids
        elif args.n_tasks is not None:
            task_ids = list(range(min(args.n_tasks, n_tasks_total)))
        else:
            task_ids = list(range(n_tasks_total))

        # Pre-compute task embeddings using clip_live format
        descriptions = [benchmark.get_task(i).language for i in task_ids]
        task_embs_all = get_task_embs(cfg, descriptions)   # (N, 154)

        suite_results = []
        for local_idx, task_id in enumerate(task_ids):
            task = benchmark.get_task(task_id)
            task_emb = task_embs_all[local_idx : local_idx + 1]  # (1, 154)

            t0 = time.time()
            success = evaluate_one_task_success(
                cfg, algo, task, task_emb, task_id,
                sim_states=None, task_str="",
            )
            elapsed = time.time() - t0
            print(
                f"  Task {task_id:2d} | {task.language[:55]:<55} | "
                f"success={success:.2f}  ({elapsed:.0f}s)"
            )
            suite_results.append((task.language, success))

        mean_success = np.mean([r[1] for r in suite_results])
        print(f"\n  Mean success ({suite_name}): {mean_success:.3f}")
        all_results[suite_name] = suite_results

    # ------------------------------------------------------------------
    # Print aggregate summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" AGGREGATE SUMMARY")
    print(f"{'='*60}")
    for suite_name, results in all_results.items():
        rates = [r[1] for r in results]
        print(f"  {suite_name:<22}  n={len(rates):2d}  mean={np.mean(rates):.3f}  "
              f"min={np.min(rates):.3f}  max={np.max(rates):.3f}")

    # ------------------------------------------------------------------
    # Optional save
    # ------------------------------------------------------------------
    if args.output:
        save_data = {}
        for suite_name, results in all_results.items():
            save_data[f"{suite_name}_tasks"] = [r[0] for r in results]
            save_data[f"{suite_name}_success"] = [r[1] for r in results]
        np.savez(args.output, **save_data)
        print(f"\n[eval] Results saved → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
