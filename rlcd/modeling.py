"""Model / tokenizer loading, quantization and LoRA setup."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

logger = logging.getLogger(__name__)

_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
        local_files_only=cfg.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: ModelConfig, device_map=None):
    dtype = _DTYPE.get(cfg.torch_dtype, torch.bfloat16)
    kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        local_files_only=cfg.local_files_only,
    )
    if cfg.quantization in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        if cfg.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=dtype,
            )
        kwargs["device_map"] = device_map if device_map is not None else "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **kwargs)
    model.config.use_cache = False
    return model


def _linear_module_names(model: nn.Module, include_lm_head: bool) -> List[str]:
    from bitsandbytes.nn import Linear4bit, Linear8bitLt

    linear_types = (nn.Linear, Linear8bitLt, Linear4bit)
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_types) and not name.endswith("lm_head"):
            leaf = name.split(".")[-1]
            names.add(leaf)
    if include_lm_head:
        names.add("lm_head")
    if not names:
        raise ValueError("No linear modules found for LoRA; check model architecture.")
    return sorted(names)


def prepare_model(model, cfg: ModelConfig):
    if cfg.quantization in ("8bit", "4bit") and cfg.prepare_kbit_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
    return model


def apply_peft(model, cfg: ModelConfig):
    if cfg.train_scope == "head_only":
        for param in model.parameters():
            param.requires_grad_(False)
        enabled = []
        for name, param in model.named_parameters():
            if any(mod in name for mod in cfg.modules_to_save):
                param.requires_grad_(True)
                enabled.append(name)
        if not enabled:
            raise ValueError(
                f"head_only scope found no trainable params matching {cfg.modules_to_save}"
            )
        logger.info("head_only: trainable params: %s", enabled)
        return model

    from peft import LoraConfig, get_peft_model

    target_modules = cfg.lora_target_modules
    if target_modules is None:
        target_modules = _linear_module_names(model, include_lm_head=cfg.lora_includes_lm_head)

    modules_to_save = None
    if cfg.train_scope == "lora_plus_head" and cfg.modules_to_save:
        modules_to_save = [
            m for m in cfg.modules_to_save if not (cfg.lora_includes_lm_head and m == "lm_head")
        ] or None

    logger.info("LoRA target modules: %s", target_modules)
    logger.info("LoRA modules_to_save: %s", modules_to_save)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def configure_training_mode(model, cfg: ModelConfig):
    if cfg.use_gradient_checkpointing:
        kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": cfg.gradient_checkpointing_use_reentrant}}
        for target in _gradient_targets(model):
            try:
                target.gradient_checkpointing_enable(**kwargs)
                break
            except TypeError:
                try:
                    target.gradient_checkpointing_enable()
                    break
                except Exception:  # noqa: BLE001
                    continue
            except Exception:  # noqa: BLE001
                continue
    if cfg.train_scope != "head_only":
        for target in _gradient_targets(model):
            if hasattr(target, "enable_input_require_grads"):
                try:
                    target.enable_input_require_grads()
                    break
                except Exception:  # noqa: BLE001
                    continue
    model.config.use_cache = False
    return model


def _gradient_targets(model):
    targets = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        targets.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            targets.append(inner)
    return targets


def build_model(cfg: ModelConfig, device_map=None):
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, device_map=device_map)
    model = prepare_model(model, cfg)
    model = apply_peft(model, cfg)
    model = configure_training_mode(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(1, total),
    )
    return model, tokenizer


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def named_trainable_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]
