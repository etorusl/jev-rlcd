"""Monte-Carlo reasoning sampling with marker-based stopping."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import StoppingCriteria

from .config import TrainConfig
from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


class MarkerStoppingCriteria(StoppingCriteria):
    """Stops generation once every sequence ends with the marker or with EOS."""

    def __init__(self, prompt_builder: PromptBuilder, prompt_len: int) -> None:
        self.marker_variants = prompt_builder.marker_variants
        self.eos_token_id = prompt_builder.eos_token_id
        self.prompt_len = prompt_len

    def _sequence_done(self, gen) -> bool:
        if self.eos_token_id is not None and bool((gen == self.eos_token_id).any().item()):
            return True
        for variant in self.marker_variants:
            n = len(variant)
            if gen.numel() >= n and gen[-n:].tolist() == list(variant):
                return True
        return False

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D401
        gen = input_ids[:, self.prompt_len :]
        for row in gen:
            if not self._sequence_done(row):
                return False
        return True


@torch.no_grad()
def sample_cot_full_ids(
    model,
    prompt_builder: PromptBuilder,
    base_ids: torch.Tensor,
    M: int,
    cfg: TrainConfig,
) -> List[torch.Tensor]:
    """Generate M reasoning traces and return base+CoT token id tensors."""
    was_training = model.training
    model.eval()
    model.config.use_cache = True

    device = base_ids.device
    input_ids = base_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    stopping = MarkerStoppingCriteria(prompt_builder, prompt_len)
    gen_kwargs = dict(
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        min_new_tokens=cfg.min_new_tokens,
        stopping_criteria=[stopping],
        pad_token_id=prompt_builder.tokenizer.pad_token_id,
        eos_token_id=prompt_builder.tokenizer.eos_token_id,
        use_cache=True,
    )
    if cfg.top_k and cfg.top_k > 0:
        gen_kwargs["top_k"] = cfg.top_k

    sequences: List[torch.Tensor] = []
    if cfg.sampling_mode == "batched" and M > 1:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=M,
            **gen_kwargs,
        )
        sequences = [out[i] for i in range(out.shape[0])]
    else:
        for _ in range(M):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=1,
                **gen_kwargs,
            )
            sequences.append(out[0])

    results: List[torch.Tensor] = []
    for seq in sequences:
        cot = seq[prompt_len:].detach().to(base_ids.device)
        cot = prompt_builder.trim_at_eos(cot)
        results.append(prompt_builder.build_full_ids(base_ids, cot))

    model.config.use_cache = False
    if was_training:
        model.train()
    return results


def masked_marker_token_ids(variant: List[int]) -> Optional[torch.Tensor]:
    if not variant:
        return None
    return torch.tensor(variant, dtype=torch.long)
