"""Evaluation metrics for the binary audit readout.

Covers the RLCD-relevant calibration/selective-prediction metrics
(Brier, ECE, AURC, Cov@epsilon) plus standard classification metrics
(accuracy, precision/recall/F1, ROC-AUC, PR-AUC) and constant-predictor
baselines so you can tell whether training is learning or collapsing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def accuracy(probs: torch.Tensor, labels: Sequence[int], class_order: Sequence[int]) -> float:
    if probs.numel() == 0:
        return float("nan")
    preds = probs.argmax(dim=-1)
    target = torch.tensor(
        [list(class_order).index(int(y)) for y in labels], device=probs.device
    )
    return _as_float((preds == target).float().mean())


def brier_score(probs: torch.Tensor, labels: Sequence[int], class_order: Sequence[int]) -> float:
    if probs.numel() == 0:
        return float("nan")
    one_hot = torch.zeros_like(probs)
    for row, y in enumerate(labels):
        one_hot[row, list(class_order).index(int(y))] = 1.0
    return _as_float(((probs - one_hot) ** 2).sum(dim=-1).mean())


def auroc(scores: torch.Tensor, labels: Sequence[int], pos_label: int) -> float:
    """Rank-based ROC-AUC of P(positive) against binary labels (tie-aware)."""
    if scores.numel() == 0:
        return float("nan")
    y = torch.tensor([1.0 if int(v) == pos_label else 0.0 for v in labels])
    n_pos = int(y.sum().item())
    n_neg = int(y.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.numel():
        j = i
        while j + 1 < sorted_scores.numel() and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos_rank_sum = _as_float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: torch.Tensor, labels: Sequence[int], pos_label: int) -> float:
    """PR-AUC (average precision), matching sklearn.average_precision_score."""
    if scores.numel() == 0:
        return float("nan")
    y = torch.tensor([1.0 if int(v) == pos_label else 0.0 for v in labels])
    if y.sum().item() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    y = y[order]
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1.0 - y, dim=0)
    precision = tp / (tp + fp).clamp(min=1e-12)
    return _as_float(precision[y == 1].mean())


def f1_precision_recall(
    probs: torch.Tensor, labels: Sequence[int], class_order: Sequence[int], pos_label: int
) -> Tuple[float, float, float]:
    if probs.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    pos_idx = list(class_order).index(pos_label)
    pred_pos = probs.argmax(dim=-1) == pos_idx
    true_pos = torch.tensor([int(v) == pos_label for v in labels])
    tp = _as_float((pred_pos & true_pos).sum())
    fp = _as_float((pred_pos & ~true_pos).sum())
    fn = _as_float((~pred_pos & true_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def expected_calibration_error(
    conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15
) -> float:
    if conf.numel() == 0:
        return float("nan")
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum().item() > 0:
            acc = _as_float(correct[mask].float().mean())
            avg_conf = _as_float(conf[mask].mean())
            ece += _as_float(mask.float().mean()) * abs(acc - avg_conf)
    return ece


def aurc(conf: torch.Tensor, correct: torch.Tensor) -> float:
    """Area under the risk-coverage curve (lower is better).

    Sort by decreasing confidence; at coverage c the selective risk is the
    error rate among the c most confident predictions.
    """
    if conf.numel() == 0:
        return float("nan")
    order = torch.argsort(conf, descending=True)
    errors = (~correct[order]).float()
    cum = torch.cumsum(errors, dim=0)
    coverage = torch.arange(1, errors.numel() + 1, dtype=torch.float)
    risk = cum / coverage
    return _as_float(risk.mean())


def coverage_at_risk(conf: torch.Tensor, correct: torch.Tensor, budget: float) -> float:
    """Cov@eps: largest coverage whose selective risk stays <= budget."""
    if conf.numel() == 0:
        return float("nan")
    order = torch.argsort(conf, descending=True)
    errors = (~correct[order]).float()
    cum = torch.cumsum(errors, dim=0)
    coverage = torch.arange(1, errors.numel() + 1, dtype=torch.float)
    risk = cum / coverage
    ok = (risk <= budget).nonzero(as_tuple=False)
    if ok.numel() == 0:
        return 0.0
    return _as_float((int(ok[-1].item()) + 1) / errors.numel())


def _constant_probs(probs: torch.Tensor, pos_idx: int, pos_rate: float) -> torch.Tensor:
    """Constant predictor that always outputs the positive-class base rate."""
    baseline = torch.zeros_like(probs)
    baseline[:, pos_idx] = pos_rate
    neg_idx = 0 if pos_idx != 0 else min(1, probs.shape[1] - 1)
    baseline[:, neg_idx] = 1.0 - pos_rate
    return baseline


def summarize(
    probs: torch.Tensor,
    labels: Sequence[int],
    class_order: Sequence[int],
    pos_label: int,
    error_budgets: Sequence[float] = (0.05, 0.10),
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Returns (metrics, per-sample predictions) — both for logging/dumping."""
    if probs.numel() == 0:
        return {}, []

    class_order = list(class_order)
    pos_idx = class_order.index(pos_label)
    pred_idx = probs.argmax(dim=-1)
    pred_labels = [class_order[int(i)] for i in pred_idx]

    p_pos = probs[:, pos_idx]
    conf = probs.max(dim=-1).values
    correct = torch.tensor(
        [pl == int(y) for pl, y in zip(pred_labels, labels)], dtype=torch.bool
    )

    f1, precision, recall = f1_precision_recall(probs, labels, class_order, pos_label)
    pos_rate = sum(1 for y in labels if int(y) == pos_label) / max(1, len(labels))

    metrics: Dict[str, float] = {
        "n": float(probs.shape[0]),
        "acc": accuracy(probs, labels, class_order),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc(p_pos, labels, pos_label),
        "pr_auc": average_precision(p_pos, labels, pos_label),
        "brier": brier_score(probs, labels, class_order),
        "ece": expected_calibration_error(conf, correct),
        "aurc": aurc(conf, correct),
        "pos_rate": pos_rate,
        # constant-predictor references: if the model can't beat these it has
        # not learned to associate its rationales with the answer.
        "acc_majority": max(pos_rate, 1.0 - pos_rate),
        "brier_baseline": brier_score(_constant_probs(probs, pos_idx, pos_rate), labels, class_order),
    }
    for budget in error_budgets:
        metrics[f"cov@{budget:g}"] = coverage_at_risk(conf, correct, budget)

    predictions: List[Dict[str, object]] = []
    for row in range(probs.shape[0]):
        predictions.append(
            {
                "label": int(labels[row]),
                "pred": int(pred_labels[row]),
                "p_pos": _as_float(p_pos[row]),
                "conf": _as_float(conf[row]),
                "correct": bool(correct[row].item()),
            }
        )
    return metrics, predictions
