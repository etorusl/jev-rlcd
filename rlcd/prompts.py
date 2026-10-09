"""Base-prompt construction, marker handling and target-token resolution."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch

from .config import LabelConfig, PromptConfig

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Turns (prompt, response) pairs into token id tensors.

    The builder is model/tokenizer aware because the readout needs the exact
    token ids that represent the answer classes *as a continuation of the
    marker*.  It auto-detects those ids and validates them at startup.
    """

    def __init__(
        self,
        tokenizer,
        prompt_cfg: PromptConfig,
        label_cfg: LabelConfig,
        chat_template_fn=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pc = prompt_cfg
        self.lc = label_cfg
        self.chat_template_fn = chat_template_fn

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id

        self.marker_ids = self._encode(prompt_cfg.marker)
        self.marker_variants = self._marker_variants()
        self.class_order = list(label_cfg.class_order)
        self.class_token_ids = self._resolve_class_token_ids()
        # Number of generated rationales that ended without the marker (i.e. were
        # cut off by max_new_tokens). Compared against total rationales sampled.
        self.marker_missing = 0

    # ------------------------------------------------------------------ utils
    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _marker_variants(self) -> List[List[int]]:
        candidates = [self.pc.marker, self.pc.marker + "\n", " " + self.pc.marker]
        variants: List[List[int]] = []
        for text in candidates:
            ids = self._encode(text)
            if ids and ids not in variants:
                variants.append(ids)
        return variants

    def _resolve_class_token_ids(self) -> Dict[int, List[int]]:
        override = self.lc.label_token_ids
        if override:
            resolved = {int(k): list(v) for k, v in override.items()}
            self._log_resolution(resolved)
            return resolved

        n_marker = len(self.marker_ids)
        resolved: Dict[int, List[int]] = {}
        for cls in self.class_order:
            strings = self.lc.label_strings.get(cls)
            if strings is None:
                strings = self.lc.label_strings.get(str(cls))  # type: ignore[arg-type]
            if not strings:
                raise ValueError(f"No answer strings configured for class {cls!r}")

            ids: List[int] = []
            for surface in strings:
                cand = self._encode(self.pc.marker + surface)
                if len(cand) == n_marker:
                    continue
                tid = cand[-1]
                if tid not in ids:
                    ids.append(tid)
            if not ids:
                raise ValueError(
                    f"Could not resolve a single answer token for class {cls!r}; "
                    f"check label_strings / marker for this tokenizer."
                )
            resolved[cls] = ids

        self._log_resolution(resolved)
        return resolved

    def _log_resolution(self, resolved: Dict[int, List[int]]) -> None:
        for cls in self.class_order:
            ids = resolved[cls]
            pieces = [
                (i, self.tokenizer.convert_ids_to_tokens(i)) for i in ids
            ]
            logger.info("Answer token(s) for class %s: %s", cls, pieces)

    # --------------------------------------------------------------- building
    def _system_text(self) -> str:
        system = self.pc.system.replace("{marker}", self.pc.marker)
        if self.pc.disable_thinking and self.pc.disable_thinking_phrase:
            system = system.rstrip() + self.pc.disable_thinking_phrase
        return system

    def base_text(self, prompt: str, response: str) -> str:
        if self.pc.mode == "chat_template":
            return self._base_text_chat_template(prompt, response)
        user = self.pc.user_template.format(prompt=prompt, response=response)
        return (
            self.pc.system_prefix
            + self._system_text()
            + self.pc.system_suffix
            + self.pc.user_prefix
            + user
            + self.pc.user_suffix
            + self.pc.assistant_prefix
        )

    def _base_text_chat_template(self, prompt: str, response: str) -> str:
        user = self.pc.user_template.format(prompt=prompt, response=response)
        messages = [
            {"role": "system", "content": self._system_text()},
            {"role": "user", "content": user},
        ]
        kwargs = dict(self.pc.chat_template_kwargs)
        if self.pc.disable_thinking:
            kwargs.setdefault("enable_thinking", False)
        apply_fn = self.chat_template_fn or self.tokenizer.apply_chat_template
        try:
            return apply_fn(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return apply_fn(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )

    def base_ids(self, prompt: str, response: str) -> torch.Tensor:
        ids = self._encode(self.base_text(prompt, response))
        return torch.tensor(ids, dtype=torch.long)

    def truncate_base_ids(self, ids: torch.Tensor, max_len: int, side: str) -> torch.Tensor:
        if ids.numel() <= max_len:
            return ids
        if max_len <= 0:
            return ids[:0]
        if side == "tail":
            keep = ids[-max_len:]
        elif side == "head":
            keep = ids[:max_len]
        else:  # middle: keep head and tail, drop the middle
            head = max_len // 2
            tail = max_len - head
            keep = torch.cat([ids[:head], ids[-tail:]])
        return keep.contiguous()

    # ------------------------------------------------------------- completion
    def trim_at_eos(self, cot_ids: torch.Tensor) -> torch.Tensor:
        if self.eos_token_id is None:
            return cot_ids
        matches = (cot_ids == self.eos_token_id).nonzero(as_tuple=False)
        if matches.numel() > 0:
            cot_ids = cot_ids[: int(matches[0].item())]
        return cot_ids

    def finish_with_marker(self, cot_ids: torch.Tensor) -> torch.Tensor:
        """Cut the CoT right after the first marker occurrence (inclusive)."""
        best_end: Optional[int] = None
        for variant in self.marker_variants:
            end = _find_subsequence(cot_ids, variant)
            if end is not None and (best_end is None or end < best_end):
                best_end = end
        if best_end is None:
            self.marker_missing += 1
            marker = torch.tensor(
                self.marker_variants[0], dtype=cot_ids.dtype, device=cot_ids.device
            )
            return torch.cat([cot_ids, marker])
        return cot_ids[:best_end]

    def build_full_ids(self, base_ids: torch.Tensor, cot_ids: torch.Tensor) -> torch.Tensor:
        cot_ids = cot_ids.to(base_ids.dtype)
        return torch.cat([base_ids, self.finish_with_marker(cot_ids)])


def _find_subsequence(haystack: torch.Tensor, needle: Sequence[int]) -> Optional[int]:
    """Return the index just past the first occurrence of needle, else None."""
    n = len(needle)
    if n == 0 or haystack.numel() < n:
        return None
    seq = haystack.tolist()
    for start in range(0, len(seq) - n + 1):
        if seq[start : start + n] == list(needle):
            return start + n
    return None
