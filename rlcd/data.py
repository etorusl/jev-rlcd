"""Dataset loading, oversize filtering and batching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .config import Config
from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


def load_samples(path: str, fmt: str = "json") -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path!r}. samples.json is intentionally gitignored; "
            f"place it on the cluster or pass --data.path."
        )
    if fmt == "jsonl":
        samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array or a {'data': [...]} object.")
    return data


class AuditDataset(Dataset):
    """Returns raw text samples; tokenization happens in the training loop.

    Oversize samples (base token length > max_seq_len) are dropped by default so
    no truncated / broken sample ever enters the loss.
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        cfg: Config,
        prompt_builder: PromptBuilder,
        indices: Optional[Sequence[int]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.builder = prompt_builder
        self.samples = samples
        self.dc = cfg.data

        self.valid_indices = list(indices) if indices is not None else self._filter_indices(cache_dir)

    # ------------------------------------------------------------------ filter
    def _filter_indices(self, cache_dir: Optional[str]) -> List[int]:
        cache_path = None
        cache_key = None
        if cache_dir and self.dc.filter_cache:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = self._cache_key()
            cache_path = os.path.join(cache_dir, f"valid_indices_{cache_key}.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                logger.info("Loaded %d valid indices from cache", len(cached))
                return list(cached)

        from tqdm import tqdm

        valid: List[int] = []
        dropped_base = 0
        dropped_prompt = 0
        prompt_lens: List[int] = []
        max_prompt = self.dc.max_prompt_tokens
        for idx in tqdm(range(len(self.samples)), desc="filtering by length", leave=False):
            sample = self.samples[idx]
            prompt_text = sample[self.dc.prompt_field]
            response_text = sample[self.dc.response_field]

            n_prompt = len(self.builder._encode(prompt_text))
            prompt_lens.append(n_prompt)
            if max_prompt is not None and n_prompt > max_prompt:
                dropped_prompt += 1
                continue

            base_text = self.builder.base_text(prompt_text, response_text)
            if len(self.builder._encode(base_text)) > self.dc.max_seq_len:
                dropped_base += 1
                continue
            valid.append(idx)

        logger.info(
            "Length filter: kept %d/%d | dropped %d (prompt>%s) | dropped %d (base>%d)",
            len(valid),
            len(self.samples),
            dropped_prompt,
            max_prompt,
            dropped_base,
            self.dc.max_seq_len,
        )
        self._log_length_stats(prompt_lens)
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(valid, handle)
        return valid

    @staticmethod
    def _log_length_stats(lengths: List[int]) -> None:
        if not lengths:
            return
        ordered = sorted(lengths)

        def pct(p: float) -> int:
            return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

        logger.info(
            "Prompt tokens (all samples): p50=%d p75=%d p90=%d p95=%d p99=%d max=%d",
            pct(0.50), pct(0.75), pct(0.90), pct(0.95), pct(0.99), ordered[-1],
        )

    def _cache_key(self) -> str:
        tok_name = getattr(self.builder.tokenizer, "name_or_path", "tokenizer")
        payload = json.dumps(
            {
                "path": self.dc.path,
                "max_seq_len": self.dc.max_seq_len,
                "max_prompt_tokens": self.dc.max_prompt_tokens,
                "policy": self.dc.oversize_policy,
                "prompt_mode": self.cfg.prompt.mode,
                "marker": self.cfg.prompt.marker,
                "thinking": self.cfg.prompt.disable_thinking,
                "tokenizer": tok_name,
                "n": len(self.samples),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    # ------------------------------------------------------------------- item
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.samples[self.valid_indices[i]]
        return {
            "prompt": sample[self.dc.prompt_field],
            "response": sample[self.dc.response_field],
            "label": int(sample[self.dc.label_field]),
            "index": self.valid_indices[i],
        }


def collate_single(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Micro batch is 1; keep raw python objects (tokenization is deferred)."""
    return batch


def split_indices(
    n: int, val_fraction: float, seed: int, max_val: int
) -> Tuple[List[int], List[int]]:
    if val_fraction <= 0:
        return list(range(n)), []
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = min(max_val, max(1, int(n * val_fraction)))
    val = perm[:n_val]
    train = perm[n_val:]
    return train, val
