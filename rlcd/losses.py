"""RLCD Stage 1 readout: class scores, Brier reward and loss."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def class_log_scores(
    logits_last: torch.Tensor,
    class_token_ids: Dict[int, List[int]],
    class_order: List[int],
) -> torch.Tensor:
    """log p(class) = logsumexp over all surface-token ids of that class.

    Summing (in log space) over surface variants such as "1" and " 1" makes the
    readout robust to whether the model emits a leading space.
    """
    scores = []
    for cls in class_order:
        ids = class_token_ids[cls]
        index = torch.tensor(ids, dtype=torch.long, device=logits_last.device)
        scores.append(torch.logsumexp(logits_last.index_select(0, index), dim=0))
    return torch.stack(scores, dim=0)


def brier_reward(u: torch.Tensor, true_idx: int) -> torch.Tensor:
    """J(u, Y) = 2 u_Y - ||u||^2  (maximised when u is one-hot on Y)."""
    return 2.0 * u[true_idx] - (u * u).sum()


def readout_loss(
    logits_last: torch.Tensor,
    class_token_ids: Dict[int, List[int]],
    class_order: List[int],
    label: int,
    M: int,
) -> torch.Tensor:
    """Returns -J/M so that summing over the M samples gives -mean(J)."""
    scores = class_log_scores(logits_last, class_token_ids, class_order)
    u = F.softmax(scores, dim=0)
    true_idx = class_order.index(label)
    J = brier_reward(u, true_idx)
    return -J / M


@torch.no_grad()
def readout_probs(
    logits_last: torch.Tensor,
    class_token_ids: Dict[int, List[int]],
    class_order: List[int],
) -> torch.Tensor:
    scores = class_log_scores(logits_last, class_token_ids, class_order)
    return F.softmax(scores, dim=0)
