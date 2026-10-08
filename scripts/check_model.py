#!/usr/bin/env python
"""Locate an HF model in the local cache and check the files the loader needs.

Usage:
    python scripts/check_model.py
    python scripts/check_model.py --model google/gemma-4-12B-it
    python scripts/check_model.py --list        # just list all cached repos
"""

from __future__ import annotations

import argparse
import os


def cache_dirs() -> list[str]:
    dirs = []
    for var in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(var):
            dirs.append(os.environ[var])
    if os.environ.get("HF_HOME"):
        dirs.append(os.path.join(os.environ["HF_HOME"], "hub"))
    dirs.append(os.path.expanduser("~/.cache/huggingface/hub"))
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_repo_dirs(model_id: str) -> list[str]:
    folder = "models--" + model_id.replace("/", "--")
    found = []
    for base in cache_dirs():
        cand = os.path.join(base, folder)
        if os.path.isdir(cand):
            found.append(cand)
    # Fallback: brute force a few likely roots.
    for root in ("/workspace", "/data", os.path.expanduser("~")):
        for dirpath, dirnames, _ in os.walk(root):
            if ".locks" in dirpath.split(os.sep):
                continue
            if folder in dirnames:
                found.append(os.path.join(dirpath, folder))
            if dirpath.count(os.sep) > 6:
                dirnames[:] = []
    return sorted(set(found))


def list_snapshot_files(repo_dir: str) -> list[str]:
    snaps = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snaps):
        return []
    files = []
    for rev in os.listdir(snaps):
        for name in os.listdir(os.path.join(snaps, rev)):
            files.append(name)
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--list", action="store_true", help="list every cached repo")
    args = parser.parse_args()

    print("HF_HOME        :", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE   :", os.environ.get("HF_HUB_CACHE"))
    print("searched dirs  :", cache_dirs())
    try:
        from huggingface_hub import constants

        active = constants.HF_HUB_CACHE
        print("active hub     :", active)
        if not os.path.isdir(active):
            print(f"  !! active hub cache does not exist: {active}")
    except Exception:  # noqa: BLE001
        active = None

    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        print(f"\nscan_cache_dir: {len(list(info.repos))} repos, {info.size_on_disk / 1e9:.1f} GB")
        if args.list:
            for repo in sorted(info.repos, key=lambda r: r.repo_id):
                print(f"  {repo.repo_id} ({repo.repo_type}) {repo.size_on_disk / 1e6:.1f} MB")
    except Exception as exc:  # noqa: BLE001
        print("scan_cache_dir failed:", exc)

    print(f"\n=== {args.model} ===")
    repo_dirs = find_repo_dirs(args.model)
    if not repo_dirs:
        print("NOT FOUND in any searched cache directory.")
        print("Set HF_HOME to the volume that holds the cache, then retry:")
        print("  export HF_HOME=/path/with/cache")
        return 1

    for repo_dir in repo_dirs:
        print("found:", repo_dir)
        names = list_snapshot_files(repo_dir)
        print(f"  {len(names)} files in snapshots:")
        for name in names:
            print("   -", name)

    hub_dirs = {os.path.dirname(d) for d in repo_dirs}
    if active is not None and not any(d == active for d in hub_dirs):
        hub = sorted(hub_dirs)[0]
        hf_home = os.path.dirname(hub)
        print("\n!! The cache that holds this model is NOT the active HF cache:")
        print(f"   model hub : {hub}")
        print(f"   active hub: {active}")
        print("   Fix one of:")
        print(f"     unset HF_HOME")
        print(f"     export HF_HOME={hf_home}")
        print(f"     export HF_HUB_CACHE={hub}")

    print("\n=== loader probes (local_files_only=True) ===")
    from transformers import AutoProcessor, AutoTokenizer

    for label, fn in (
        ("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(args.model, local_files_only=True)),
        ("AutoProcessor", lambda: AutoProcessor.from_pretrained(args.model, local_files_only=True)),
    ):
        try:
            obj = fn()
            print(f"  {label}: OK ({type(obj).__name__})")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: FAIL -> {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
