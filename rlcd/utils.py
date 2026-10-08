"""Small runtime helpers."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def count_parameters(model) -> int:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %s / %s", f"{trainable:,}", f"{total:,}")
    return trainable


def log_memory(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(
        "[mem %s] allocated=%.2fGB reserved=%.2fGB peak=%.2fGB",
        tag,
        allocated,
        reserved,
        peak,
    )
