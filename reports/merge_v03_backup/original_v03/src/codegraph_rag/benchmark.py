"""Prepare reproducible issue-only retrieval manifests from SWE-bench Verified.

The gold patch is only read to create a SEPARATE scoring labels file. It is
never copied into the retrieval input manifest.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Iterable


def files_in_patch(patch: str) -> list[str]:
    """Extract pre-fix Python paths from Git patch old-file headers.

    Newly added files cannot be retrieved from the pre-fix repository, so they
    are excluded from this localization label set.
    """
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            name = line[6:]
            if name != "/dev/null" and name.endswith(".py"):
                paths.add(name)
    return sorted(paths)


def choose_instances(rows: Iterable[dict], *, limit: int = 80, seed: int = 42) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["repo"]].append(row)
    rng = random.Random(seed)
    for group in buckets.values():
        group.sort(key=lambda r: r["instance_id"])
        rng.shuffle(group)
    names = sorted(buckets)
    result: list[dict] = []
    while names and len(result) < limit:
        for name in names[:]:
            if len(result) >= limit:
                break
            if buckets[name]:
                result.append(buckets[name].pop())
            else:
                names.remove(name)
    return result


def build_manifests(rows: Iterable[dict], output_dir: str | Path, *, limit: int = 80,
                    dev_count: int = 20, seed: int = 42) -> dict[str, int]:
    chosen = choose_instances(rows, limit=limit, seed=seed)
    if not 0 <= dev_count <= len(chosen):
        raise ValueError("dev_count must be between 0 and selected task count")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Shuffle the stratified selection again for dev/test allocation, fixed by seed.
    rng = random.Random(seed + 1)
    rng.shuffle(chosen)
    inputs, labels = [], []
    for index, item in enumerate(chosen):
        ident = item["instance_id"]
        split = "dev" if index < dev_count else "test"
        inputs.append({
            "instance_id": ident,
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "split": split,
        })
        labels.append({
            "instance_id": ident,
            "split": split,
            "gold_python_files": files_in_patch(item["patch"]),
        })
    for filename, content in (("issues.jsonl", inputs), ("gold_labels.jsonl", labels)):
        with (out / filename).open("w", encoding="utf-8") as handle:
            for row in content:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "manifest_metadata.json").write_text(json.dumps({
        "dataset": "SWE-bench/SWE-bench_Verified",
        "split": "test", "seed": seed,
        "requested": limit, "selected": len(chosen),
        "dev": min(dev_count, len(chosen)),
        "held_out": max(0, len(chosen) - dev_count),
        "note": "Do not give gold_labels.jsonl to the retriever. Checkout each repo at base_commit.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"selected": len(chosen), "dev": min(dev_count, len(chosen)),
            "test": max(0, len(chosen) - dev_count)}


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_verified() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies: pip install -e ".[benchmark]"') from exc
    return list(load_dataset("SWE-bench/SWE-bench_Verified", split="test"))
