import os
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from libero.lifelong.algos.multitask import Multitask
from libero.lifelong.metric import *
from libero.lifelong.utils import *


class _PolicyLossModule(nn.Module):
    """Thin wrapper so nn.DataParallel can scatter batches across GPUs.

    DataParallel wraps ``forward``, not arbitrary methods.  This module exposes
    ``policy.compute_loss`` as ``forward`` so that multi-GPU scatter/gather
    works transparently.  The loss is unsqueezed to (1,) so DataParallel can
    cat replicas into (n_gpus,) and the caller does ``.mean()``.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, data: dict) -> torch.Tensor:
        return self.policy.compute_loss(data).unsqueeze(0)


class CLIPMultitask(Multitask):
    """Multitask learning with BCClipFlowPolicy: dual LR groups + val-loss tracking."""

    def start_task(self, task):
        """Override to create dual AdamW optimizer with separate LR for CLIP text encoder.

        Args:
            task: Task index, or -1 for multitask (all tasks).
        """
        self.current_task = task
        n_epochs = self.cfg.train.n_epochs

        clip_text_params = list(self.policy.clip.text_model.parameters())
        clip_text_ids = {id(p) for p in clip_text_params}
        other_params = [p for p in self.policy.parameters() if id(p) not in clip_text_ids]

        if getattr(self.cfg.policy, "freeze_clip", False):
            self.optimizer = torch.optim.AdamW(other_params, lr=1e-4, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": 1e-4},
                    {"params": clip_text_params, "lr": 5e-6},
                ],
                weight_decay=1e-4,
            )

        warmup_epochs = 5
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=n_epochs - warmup_epochs
                ),
            ],
            milestones=[warmup_epochs],
        )

    def observe(self, data):
        """Override to use DataParallel when multiple GPUs are available."""
        if self._dp is None:
            return super().observe(data)
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self._dp(data).mean()
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        return loss.item()

    def eval_observe(self, data):
        """Override to use DataParallel when multiple GPUs are available."""
        if self._dp is None:
            return super().eval_observe(data)
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            loss = self._dp(data).mean()
        return loss.item()

    def learn_all_tasks(self, datasets, benchmark, result_summary):
        """Override Multitask.learn_all_tasks with val split, periodic checkpoints, and val-loss logging.

        Args:
            datasets: List of task datasets.
            benchmark: Benchmark object with n_tasks and evaluation utilities.
            result_summary: Dict for recording result metrics.

        Returns:
            Tuple of (mean_success_auc, mean_loss_auc) floats.
        """
        self.start_task(-1)

        # --- Multi-GPU setup (DataParallel) ---
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"[info] DataParallel: using {n_gpus} GPUs")
            self._dp = nn.DataParallel(_PolicyLossModule(self.policy))
        else:
            self._dp = None

        concat_dataset = ConcatDataset(datasets)

        n_total = len(concat_dataset)
        n_train = int(0.9 * n_total)
        n_val = n_total - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            concat_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )

        num_workers = self.cfg.train.num_workers
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=num_workers,
            sampler=RandomSampler(train_dataset),
            persistent_workers=num_workers > 0,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=num_workers,
            shuffle=False,
            persistent_workers=num_workers > 0,
        )

        all_tasks = list(range(benchmark.n_tasks))
        best_checkpoint_name = os.path.join(self.experiment_dir, "multitask_model.pth")
        latest_checkpoint_name = os.path.join(self.experiment_dir, "checkpoint_latest.pth")

        prev_success_rate = -1.0
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        for epoch in range(0, self.cfg.train.n_epochs + 1):
            t0 = time.time()

            if epoch > 0:
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.4f} | time: {(t1-t0)/60:4.2f}"
            )

            if epoch > 0:
                torch_save_model(self.policy, latest_checkpoint_name, cfg=self.cfg)

            if epoch > 0 and epoch % 100 == 0:
                ep_ckpt = os.path.join(
                    self.experiment_dir, f"checkpoint_epoch_{epoch:04d}.pth"
                )
                torch_save_model(self.policy, ep_ckpt, cfg=self.cfg)

            if epoch % 10 == 0:
                self.policy.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for (idx, data) in enumerate(val_dataloader):
                        val_loss += self.eval_observe(data)
                val_loss /= len(val_dataloader)
                print(f"[info] Epoch: {epoch:3d} | val loss: {val_loss:5.4f}")
                if self.cfg.use_wandb:
                    wandb.log(
                        {
                            "val/flow_loss": val_loss,
                            "train/flow_loss": training_loss,
                            "epoch": epoch,
                        }
                    )

            if epoch % self.cfg.eval.eval_every == 0:
                self.policy.eval()
                losses.append(training_loss)

                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rate = 0.0
                successes.append(success_rate)

                if prev_success_rate < success_rate and not self.cfg.pretrain:
                    torch_save_model(self.policy, best_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                if self.cfg.lifelong.eval_in_train:
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                        + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f}",
                        flush=True,
                    )

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        if self.cfg.lifelong.eval_in_train:
            self.policy.load_state_dict(torch_load_model(best_checkpoint_name)[0])
        self.end_task(concat_dataset, -1, benchmark)

        losses = np.array(losses)
        successes = np.array(successes)
        torch.save(
            {"success": successes, "loss": losses},
            os.path.join(self.experiment_dir, "multitask_auc.log"),
        )

        if self.cfg.lifelong.eval_in_train:
            loss_at_best = losses[idx_at_best_succ]
            succ_at_best = successes[idx_at_best_succ]
            losses[idx_at_best_succ:] = loss_at_best
            successes[idx_at_best_succ:] = succ_at_best
        return successes.sum() / max(cumulated_counter, 1), losses.sum() / max(cumulated_counter, 1)
