"""Leak-resistant, file-level localization evaluation.

`retrieve-batch` never loads labels. `score` consumes predictions + gold labels
in a separate command. The metric is issue-level recall of changed Python
files, not SWE-bench patch success.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time

from .benchmark import read_jsonl
from .retrieval import retrieve

_SAFE_INSTANCE = re.compile(r"^[A-Za-z0-9_.-]+$")


def batch_retrieve(issues_path: str | Path, snapshots: str | Path,
                   out_path: str | Path, *, method: str = "pathpack",
                   budget: int = 2048, split: str | None = None,
                   allow_unverified: bool = False, **kwargs) -> dict:
    """Snapshot layout: snapshots/<instance_id>/<repo files>.

    Caller must check out the exact base_commit beforehand. For reproducible
    evaluations verify its actual HEAD yourself, using `git rev-parse HEAD`.
    """
    issues = read_jsonl(issues_path)
    base = Path(snapshots).resolve()
    results = []
    n_skipped = 0
    for row in issues:
        if split and row.get("split") != split:
            continue
        instance = row["instance_id"]
        if not _SAFE_INSTANCE.fullmatch(instance):
            raise ValueError(f"Unsafe instance ID: {instance}")
        repo = (base / instance).resolve()
        if not repo.is_relative_to(base) or not repo.is_dir():
            n_skipped += 1
            continue
        if not allow_unverified:
            try:
                commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
                raise ValueError(f"Snapshot {instance} cannot be verified against base_commit. "
                                 "Pass --allow-unverified-snapshots only for demos, not research.") from exc
            if commit != row["base_commit"]:
                raise ValueError(f"Snapshot {instance} is at {commit}, expected {row['base_commit']}")
        t0 = time.monotonic()
        graph, hits = retrieve(repo, row["problem_statement"],
                               method=method, budget=budget, **kwargs)
        results.append({"instance_id": instance, "method": method,
                        "budget_est": budget, "selected": [h.chunk.metadata() for h in hits],
                        "tokens_est": sum(h.chunk.tokens_est for h in hits),
                        "latency_sec": round(time.monotonic() - t0, 4),
                        "graph_stats": graph.stats()})
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"retrieved": len(results), "missing_snapshots": n_skipped,
            "output": str(dest)}


def score_retrieval(pred_path: str | Path, gold_path: str | Path) -> dict:
    preds = read_jsonl(pred_path)
    gold_rows = read_jsonl(gold_path)
    gold = {row["instance_id"]: row for row in gold_rows}
    seen: set[str] = set()
    per_task = []
    for p in preds:
        iid = p["instance_id"]
        if iid in seen:
            raise ValueError(f"Repeated prediction: {iid}")
        seen.add(iid)
        if iid not in gold:
            raise ValueError(f"Missing gold label: {iid}")
        targets = set(gold[iid]["gold_python_files"])
        if not targets:
            # No retrievable pre-fix Python patch files: exclude, but report.
            continue
        selected = {c["path"] for c in p["selected"]}
        overlap = selected & targets
        function_ids = {str(fn) for fn in gold[iid].get("gold_functions", [])}
        selected_ids = {c.get("id", c["path"] + "::" + c.get("qualname", "__module__"))
                        for c in p["selected"]}
        function_recall = len(function_ids & selected_ids)/len(function_ids) if function_ids else None
        per_task.append({"instance_id": iid, "recall": len(overlap)/len(targets),
                         "function_recall": function_recall,
                         "any_hit": bool(overlap), "gold_files": len(targets),
                         "hits": len(overlap), "selected_files": len(selected),
                         "tokens_est": p.get("tokens_est", 0)})
    n = len(per_task)
    evaluated_functions = [t["function_recall"] for t in per_task if t["function_recall"] is not None]
    return {"function_evaluable": len(evaluated_functions),
            "function_recall_macro": round(sum(evaluated_functions)/len(evaluated_functions), 5)
            if evaluated_functions else None,"evaluated": n, "not_evaluable": len(preds)-n,
            "file_recall_macro": round(sum(t["recall"] for t in per_task)/n, 5) if n else None,
            "issue_hit_rate": round(sum(t["any_hit"] for t in per_task)/n, 5) if n else None,
            "mean_tokens_est": round(sum(t["tokens_est"] for t in per_task)/n, 1) if n else None,
            "per_task": per_task}
