"""Evaluation metrics for the binary audit readout."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def accuracy(probs: torch.Tensor, labels: Sequence[int], class_order: Sequence[int]) -> float:
    if probs.numel() == 0:
        return float("nan")
    preds = probs.argmax(dim=-1)
    target = torch.tensor(
        [list(class_order).index(int(y)) for y in labels], device=probs.device
    )
    return (preds == target).float().mean().item()


def brier_score(probs: torch.Tensor, labels: Sequence[int], class_order: Sequence[int]) -> float:
    if probs.numel() == 0:
        return float("nan")
    n_classes = probs.shape[-1]
    one_hot = torch.zeros_like(probs)
    for row, y in enumerate(labels):
        one_hot[row, list(class_order).index(int(y))] = 1.0
    return ((probs - one_hot) ** 2).sum(dim=-1).mean().item()


def auroc(scores: torch.Tensor, labels: Sequence[int], pos_label: int) -> float:
    """Rank-based AUROC of P(positive) against binary labels."""
    if scores.numel() == 0:
        return float("nan")
    y = torch.tensor([1 if int(v) == pos_label else 0 for v in labels], dtype=torch.float)
    n_pos = int(y.sum().item())
    n_neg = int(y.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)

    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.numel():
        j = i
        while j + 1 < sorted_scores.numel() and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # ranks are 1-based
            ranks[order[i : j + 1]] = avg
        i = j + 1

    pos_rank_sum = ranks[y == 1].sum().item()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize(
    probs: torch.Tensor, labels: Sequence[int], class_order: Sequence[int], pos_label: int
) -> Tuple[dict, List[float]]:
    if probs.numel() == 0:
        return {"acc": float("nan"), "brier": float("nan"), "auroc": float("nan"), "n": 0}, []
    pos_idx = list(class_order).index(pos_label)
    metrics = {
        "acc": accuracy(probs, labels, class_order),
        "brier": brier_score(probs, labels, class_order),
        "auroc": auroc(probs[:, pos_idx], labels, pos_label),
        "n": int(probs.shape[0]),
    }
    mean_conf = probs.max(dim=-1).values.tolist()
    return metrics, mean_conf
