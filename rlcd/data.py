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
        skipped = 0
        for idx in tqdm(range(len(self.samples)), desc="filtering by length"):
            sample = self.samples[idx]
            text = self.builder.base_text(
                sample[self.dc.prompt_field], sample[self.dc.response_field]
            )
            n_tokens = len(self.builder._encode(text))
            if n_tokens <= self.dc.max_seq_len:
                valid.append(idx)
            else:
                skipped += 1

        logger.info(
            "Kept %d/%d samples (dropped %d over max_seq_len=%d)",
            len(valid),
            len(self.samples),
            skipped,
            self.dc.max_seq_len,
        )
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(valid, handle)
        return valid

    def _cache_key(self) -> str:
        tok_name = getattr(self.builder.tokenizer, "name_or_path", "tokenizer")
        payload = json.dumps(
            {
                "path": self.dc.path,
                "max_seq_len": self.dc.max_seq_len,
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
