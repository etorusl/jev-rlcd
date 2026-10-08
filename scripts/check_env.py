#!/usr/bin/env python
"""Print the installed stack and flag the common version incompatibilities."""

from __future__ import annotations

import sys


def version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return f"NOT INSTALLED ({exc})"


def main() -> int:
    ok = True
    print("python        :", sys.version.split()[0])

    import torch

    print("torch         :", torch.__version__)
    if not hasattr(torch, "accelerator"):
        print("  !! torch.accelerator missing -> transformers will fail to import.")
        print("     Install torch>=2.6 (see scripts/setup_cluster.sh).")
        ok = False

    print("transformers  :", version("transformers"))
    print("accelerate    :", version("accelerate"))
    print("peft          :", version("peft"))
    print("bitsandbytes  :", version("bitsandbytes"))
    print("safetensors   :", version("safetensors"))
    print("pyyaml        :", version("yaml"))
    print("numpy         :", version("numpy"))
    print("pillow        :", version("PIL"))

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda build    :", torch.version.cuda)
        print("device        :", torch.cuda.get_device_name(0))
        print("capability    :", torch.cuda.get_device_capability(0))

    try:
        import transformers  # noqa: F401

        from transformers import AutoModelForCausalLM  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print("  !! transformers import failed:", exc)
        ok = False

    print("RESULT        :", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
