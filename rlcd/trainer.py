"""Training / evaluation loop for RLCD Stage 1."""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, config_fingerprint
from .data import AuditDataset, collate_single, load_samples, split_indices
from .losses import readout_loss, readout_probs
from .metrics import summarize
from .prompts import PromptBuilder
from .sampling import sample_cot_full_ids

logger = logging.getLogger(__name__)


def _format_metrics(metrics: Dict[str, float]) -> str:
    order = [
        "acc", "f1", "precision", "recall", "auroc", "pr_auc",
        "brier", "brier_baseline", "ece", "aurc", "pos_rate", "acc_majority",
    ]
    parts = []
    for key in order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(k for k in metrics if k.startswith("cov@")):
        parts.append(f"{key}={metrics[key]:.3f}")
    return " ".join(parts)


class RLCDTrainer:
    def __init__(self, cfg: Config, accelerator, model, tokenizer, prompt_builder: PromptBuilder):
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.builder = prompt_builder
        self.device = accelerator.device

        self.optimizer = None
        self.scheduler = None
        self.train_dataset: Optional[AuditDataset] = None
        self.val_dataset: Optional[AuditDataset] = None
        self.global_step = 0
        self.epoch = 0
        self.sampled_rationales = 0
        self._debug_done = False

    # ------------------------------------------------------------------ setup
    def build_optimizer(self):
        head_modules = self.cfg.model.modules_to_save
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found.")

        head = [(n, p) for n, p in params if any(m in n for m in head_modules)]
        other = [(n, p) for n, p in params if not any(m in n for m in head_modules)]

        lr = self.cfg.train.lr
        wd = self.cfg.train.weight_decay
        groups = [{"params": [p for _, p in other], "lr": lr, "weight_decay": wd}]
        if head:
            head_lr = self.cfg.train.head_lr if self.cfg.train.head_lr is not None else lr
            groups.append(
                {"params": [p for _, p in head], "lr": head_lr, "weight_decay": wd}
            )
        self.optimizer = self._make_optimizer(groups)

    def _make_optimizer(self, groups):
        optim = self.cfg.train.optim
        if optim == "auto":
            optim = "paged_adamw8bit" if self.cfg.model.quantization != "none" else "adamw"
        if optim == "paged_adamw8bit":
            try:
                import bitsandbytes as bnb

                logger.info("Using bitsandbytes PagedAdamW8bit")
                return bnb.optim.PagedAdamW8bit(groups, lr=self.cfg.train.lr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PagedAdamW8bit unavailable (%s); falling back to AdamW", exc)
        return torch.optim.AdamW(groups, lr=self.cfg.train.lr)

    def build_scheduler(self, steps_per_epoch: int):
        from transformers.optimization import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_linear_schedule_with_warmup,
        )

        max_steps = self.cfg.train.max_steps
        if max_steps and max_steps > 0:
            total_steps = max_steps
        else:
            total_steps = max(1, steps_per_epoch * self.cfg.train.num_epochs)
        warmup = max(1, int(total_steps * self.cfg.train.warmup_ratio))
        kind = self.cfg.train.lr_scheduler
        if kind == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup, total_steps)
        if kind == "constant":
            return get_constant_schedule_with_warmup(self.optimizer, warmup)
        return get_cosine_schedule_with_warmup(self.optimizer, warmup, total_steps)

    # ------------------------------------------------------------------- data
    def build_datasets(self):
        samples = load_samples(self.cfg.data.path, self.cfg.data.format)
        logger.info("Loaded %d raw samples", len(samples))

        probe = AuditDataset(samples, self.cfg, self.builder, cache_dir=self.cfg.train.output_dir)
        n = len(probe)
        train_idx, val_idx = split_indices(
            n, self.cfg.data.val_fraction, self.cfg.data.seed, self.cfg.data.max_val_samples
        )
        if self.cfg.data.max_samples:
            train_idx = train_idx[: self.cfg.data.max_samples]
        valid = probe.valid_indices
        self.train_dataset = AuditDataset(
            samples, self.cfg, self.builder, indices=[valid[i] for i in train_idx]
        )
        self.val_dataset = AuditDataset(
            samples, self.cfg, self.builder, indices=[valid[i] for i in val_idx]
        )
        logger.info(
            "Train samples: %d | Val samples: %d", len(self.train_dataset), len(self.val_dataset)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.micro_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_single,
            drop_last=False,
        )

    # ------------------------------------------------------------------ steps
    def _base_ids(self, sample: Dict) -> Optional[torch.Tensor]:
        base = self.builder.base_ids(sample["prompt"], sample["response"])
        limit = self.cfg.data.max_seq_len
        if base.numel() > limit:
            if self.cfg.data.oversize_policy == "truncate":
                base = self.builder.truncate_base_ids(base, limit, self.cfg.data.truncate_side)
            else:
                return None
        return base.to(self.device)

    def _forward_last_logits(self, full_ids: torch.Tensor) -> torch.Tensor:
        """Forward keeping logits only for the last position (avoids a
        [seq_len, vocab] tensor, which OOMs for 260k-vocab models)."""
        input_ids = full_ids.unsqueeze(0)
        attention_mask = torch.ones_like(full_ids).unsqueeze(0)
        for extra in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **extra,
                )
                return outputs.logits[0, -1, :].float()
            except TypeError:
                continue
        raise RuntimeError("model forward failed with/without logits_to_keep")

    def _readout_forward(self, full_ids: torch.Tensor, label: int, M: int) -> torch.Tensor:
        logits_last = self._forward_last_logits(full_ids)
        return readout_loss(
            logits_last, self.builder.class_token_ids, self.builder.class_order, label, M
        )

    def train_epoch(self, dataloader, max_steps: int):
        cfg = self.cfg.train
        accum = cfg.gradient_accumulation_steps
        self.model.train()

        pbar = tqdm(dataloader, desc=f"epoch {self.epoch}")
        group_ok = True
        skipped = 0
        seen = 0
        epoch_start = time.time()
        for batch in pbar:
            sample = batch[0]
            seen += 1
            if cfg.progress_every and seen % cfg.progress_every == 0:
                elapsed = time.time() - epoch_start
                rate = seen / max(elapsed, 1e-6)
                cut = self.builder.marker_missing / max(1, self.sampled_rationales)
                logger.info(
                    "progress: %d samples | %.1f s/sample | %.1f samples/min | cut=%.1f%% | skip=%d",
                    seen, elapsed / seen, rate * 60.0, cut * 100.0, skipped,
                )

            with self.accelerator.accumulate(self.model):
                oom = False
                loss_value = float("nan")
                try:
                    base_ids = self._base_ids(sample)
                    if base_ids is None:
                        skipped += 1
                    else:
                        with self.accelerator.autocast():
                            full_list = sample_cot_full_ids(
                                self.model, self.builder, base_ids, cfg.M, cfg
                            )
                            self.sampled_rationales += len(full_list)
                            if cfg.debug_first_cot and not self._debug_done:
                                self._debug_done = True
                                tail = self.tokenizer.decode(
                                    full_list[0][-400:].tolist(), skip_special_tokens=False
                                )
                                logger.info("[debug] first trimmed sequence (tail):\n%s", tail)
                            for full_ids in full_list:
                                loss = self._readout_forward(full_ids, sample["label"], cfg.M)
                                self.accelerator.backward(loss)
                                loss_value = float(loss.detach().item())
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    skipped += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    if not cfg.skip_oom:
                        raise
                    logger.warning("OOM on step %d — skipping sample", self.global_step)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        oom = True
                        skipped += 1
                        gc.collect()
                        torch.cuda.empty_cache()
                        if not cfg.skip_oom:
                            raise
                        logger.warning("OOM (RuntimeError) — skipping sample")
                    else:
                        raise

                if oom:
                    group_ok = False
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    if group_ok and not oom:
                        self.accelerator.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            cfg.max_grad_norm,
                        )
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.global_step += 1
                    else:
                        group_ok = True
                    self.optimizer.zero_grad(set_to_none=True)

            if self.accelerator.sync_gradients:
                cut = self.builder.marker_missing / max(1, self.sampled_rationales)
                pbar.set_postfix(
                    step=self.global_step,
                    loss=f"{loss_value:.4f}",
                    skip=skipped,
                    cut=f"{cut:.2%}",
                )
                if self.global_step % cfg.log_every == 0:
                    self.accelerator.log(
                        {
                            "train/loss": loss_value,
                            "train/skipped": skipped,
                            "train/marker_cut_rate": cut,
                        },
                        self.global_step,
                    )
                if cfg.eval_every and self.global_step % cfg.eval_every == 0 and self.val_dataset is not None:
                    self.evaluate()
                if cfg.save_every and self.global_step % cfg.save_every == 0:
                    self.save_checkpoint()
                if max_steps and max_steps > 0 and self.global_step >= max_steps:
                    return

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        cfg = self.cfg.train
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return {}
        self.model.eval()
        self.model.config.use_cache = True

        all_probs: List[torch.Tensor] = []
        labels: List[int] = []
        sample_indices: List[int] = []
        limit = min(len(self.val_dataset), cfg.max_eval_batches)
        for i in tqdm(range(limit), desc="eval", leave=False):
            try:
                sample = self.val_dataset[i]
                base_ids = self._base_ids(sample)
                if base_ids is None:
                    continue
                with self.accelerator.autocast():
                    fulls = sample_cot_full_ids(self.model, self.builder, base_ids, cfg.eval_M, cfg)
                    probs = []
                    for full_ids in fulls:
                        probs.append(
                            readout_probs(
                                self._forward_last_logits(full_ids),
                                self.builder.class_token_ids,
                                self.builder.class_order,
                            )
                        )
                    all_probs.append(torch.stack(probs).mean(0).cpu())
                    labels.append(sample["label"])
                    sample_indices.append(sample.get("index", i))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower() and not isinstance(
                    exc, torch.cuda.OutOfMemoryError
                ):
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                logger.warning("OOM during eval sample %d — skipping", i)

        self.model.config.use_cache = False
        self.model.train()
        if not all_probs:
            return {}
        probs = torch.stack(all_probs)
        metrics, predictions = summarize(
            probs,
            labels,
            self.builder.class_order,
            self.cfg.label.positive_label,
            error_budgets=cfg.error_budgets,
        )
        logged = {f"eval/{k}": float(v) for k, v in metrics.items()}
        self.accelerator.log(logged, self.global_step)
        logger.info("Eval @ step %d | %s", self.global_step, _format_metrics(metrics))
        if self.accelerator.is_main_process and cfg.save_eval_predictions:
            self._dump_predictions(metrics, predictions, sample_indices)
        return metrics

    def _dump_predictions(self, metrics, predictions, sample_indices) -> None:
        out_dir = os.path.join(self.cfg.train.output_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"step_{self.global_step}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for idx, pred in zip(sample_indices, predictions):
                record = {"index": idx, **pred}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"step_{self.global_step}_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        logger.info("Wrote eval predictions to %s", path)

    # -------------------------------------------------------------- checkpoint
    def save_checkpoint(self, tag: Optional[str] = None):
        if not self.accelerator.is_main_process:
            return
        name = tag or f"step_{self.global_step}"
        save_dir = os.path.join(self.cfg.train.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "save_pretrained"):
            try:
                unwrapped.save_pretrained(save_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("save_pretrained failed (%s); saving raw trainable state", exc)
                self._save_trainable_state(unwrapped, save_dir)
        else:
            self._save_trainable_state(unwrapped, save_dir)

        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "rlcd_config.json"), "w", encoding="utf-8") as handle:
            json.dump(self.cfg.to_dict(), handle, indent=2, default=str)
        logger.info("Saved checkpoint to %s", save_dir)
        logger.info("Checkpoint files: %s", sorted(os.listdir(save_dir)))
        self.accelerator.wait_for_everyone()

    def _save_trainable_state(self, model, save_dir: str):
        state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(state, os.path.join(save_dir, "trainable_state.pt"))

    def run(self):
        cfg = self.cfg.train
        self.build_optimizer()
        dataloader = self.train_dataloader()
        steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
        self.scheduler = self.build_scheduler(steps_per_epoch)
        self.model, self.optimizer, dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, dataloader, self.scheduler
        )
        logger.info(
            "Fingerprint %s | accum=%d | steps/epoch=%d",
            config_fingerprint(self.cfg),
            cfg.gradient_accumulation_steps,
            steps_per_epoch,
        )
        try:
            for epoch in range(cfg.num_epochs):
                self.epoch = epoch
                self.train_epoch(dataloader, cfg.max_steps)
                if cfg.max_steps and cfg.max_steps > 0 and self.global_step >= cfg.max_steps:
                    break
            self.evaluate()
            self.save_checkpoint("final")
        except KeyboardInterrupt:
            logger.warning("Interrupted — saving a checkpoint before exit")
            self._safe_save("interrupted")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Run failed (%s) — saving a checkpoint before exit", exc)
            self._safe_save("crashed")
            raise
        self.accelerator.wait_for_everyone()

    def _safe_save(self, tag: str) -> None:
        try:
            self.save_checkpoint(tag)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not save checkpoint on %s: %s", tag, exc)
