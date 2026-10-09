#!/usr/bin/env python
"""Entry point for RLCD Stage 1 training.

Example:
    python train.py --config configs/audit_gemma.yaml
    python train.py --config configs/audit_gemma.yaml --set train.M=2 --set model.quantization=none
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

import torch

from rlcd.config import Config, apply_overrides, config_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLCD Stage 1 training")
    parser.add_argument("--config", type=str, default="configs/audit_gemma.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override, repeatable, e.g. --set train.lr=1e-5",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Override model name or path")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def coerce(value: str) -> Any:
    import yaml

    try:
        return yaml.safe_load(value)
    except Exception:  # noqa: BLE001
        return value


def build_config(args: argparse.Namespace) -> Config:
    if os.path.exists(args.config):
        cfg = Config.from_yaml(args.config)
    else:
        logging.warning("Config %s not found; using defaults", args.config)
        cfg = Config()

    overrides: Dict[str, Any] = {}
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key.strip()] = coerce(value.strip())
    if args.output_dir:
        overrides["train.output_dir"] = args.output_dir
    if args.model:
        overrides["model.model_name_or_path"] = args.model
    if args.data_path:
        overrides["data.path"] = args.data_path
    if args.resume_from_checkpoint:
        overrides["train.resume_from_checkpoint"] = args.resume_from_checkpoint
    return apply_overrides(cfg, overrides)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # bitsandbytes spams "MatMul8bitLt: inputs will be cast ..." for every layer.
    class _DropCastNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "will be cast" not in record.getMessage()

    for name in (
        "bitsandbytes",
        "bitsandbytes.autograd",
        "bitsandbytes.autograd._functions",
        "bitsandbytes.nn.modules",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.addFilter(_DropCastNoise())
    for name in ("transformers", "peft", "huggingface_hub", "httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)

    import warnings

    warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*")


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)

    if not hasattr(torch, "accelerator"):
        raise SystemExit(
            f"torch {torch.__version__} is too old: recent transformers builds require "
            "the `torch.accelerator` namespace (torch>=2.6). Reinstall with "
            "`pip install 'torch>=2.6.0'` (see scripts/setup_cluster.sh)."
        )

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    from rlcd.modeling import build_model
    from rlcd.prompts import PromptBuilder
    from rlcd.trainer import RLCDTrainer
    from rlcd.utils import count_parameters, log_memory

    set_seed(cfg.train.seed)
    if cfg.train.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.train.bf16 else "no",
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.train.output_dir, exist_ok=True)
        logging.info("Config fingerprint: %s", config_fingerprint(cfg))
        logging.info("Effective config:\n%s", _pretty(cfg.to_dict()))

    device_map = None
    if cfg.model.quantization in ("8bit", "4bit"):
        device_map = {"": accelerator.local_process_index}

    model, tokenizer, processor = build_model(cfg.model, device_map=device_map)
    prompt_builder = PromptBuilder(
        tokenizer,
        cfg.prompt,
        cfg.label,
        chat_template_fn=(processor.apply_chat_template if processor is not None else None),
    )

    if cfg.train.resume_from_checkpoint:
        model = _maybe_resume(model, cfg.train.resume_from_checkpoint)

    if accelerator.is_main_process:
        count_parameters(model)
        log_memory("after model load")

    trainer = RLCDTrainer(cfg, accelerator, model, tokenizer, prompt_builder)
    trainer.build_datasets()

    if cfg.train.wandb:
        try:
            accelerator.init_trackers(
                cfg.train.run_name or "rlcd-stage1",
                config=cfg.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("W&B init failed: %s", exc)

    trainer.run()

    if cfg.train.wandb:
        accelerator.end_training()


def _pretty(data: Dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, default=str)


def _maybe_resume(model, path: str):
    import logging
    import os

    if not os.path.isdir(path):
        logging.warning("resume_from_checkpoint %s is not a directory; ignoring", path)
        return model
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, path, is_trainable=True)
        logging.info("Resumed LoRA adapter from %s", path)
        return model
    state_file = os.path.join(path, "trainable_state.pt")
    if os.path.exists(state_file):
        import torch

        state = torch.load(state_file, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info(
            "Resumed trainable state from %s (missing=%d unexpected=%d)",
            path,
            len(missing),
            len(unexpected),
        )
    return model


if __name__ == "__main__":
    main()
