"""Post-retrieval gold function labels from SWE-bench patches and base snapshots.

Reads the reference patch. MUST ONLY run in the scoring/evaluation stage, never
inside retrieval or feature generation. The resulting labels are a proxy:
changed lines inside old Python function spans, NOT guaranteed bug roots.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from .benchmark import read_jsonl
from .parser import scan_repository

_HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')


def old_changed_lines(patch: str) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    current = None
    line_no = None
    for line in patch.splitlines():
        if line.startswith('--- '):
            current = line[6:] if line.startswith('--- a/') else None
            if current and not current.endswith('.py'):
                current = None
            line_no = None
        elif line.startswith('+++ '):
            continue
        elif line.startswith('@@ '):
            match = _HUNK.match(line)
            line_no = int(match.group(1)) if match else None
        elif current is not None and line_no is not None:
            if line.startswith('-'):
                result[current].add(line_no)
                line_no += 1
            elif line.startswith('+'):
                # Inserted line is not in the base snapshot. Locate it using
                # nearest old code line as a proxy, potentially ambiguous.
                result[current].add(max(1, line_no-1))
                result[current].add(max(1, line_no))
            elif line.startswith(' '):
                line_no += 1
            elif line.startswith('\\'):
                continue
    return dict(result)


def functions_touched(patch: str, snapshot: str | Path) -> list[str]:
    root = Path(snapshot)
    if not root.is_dir():
        raise ValueError(f'No snapshot at {root}')
    changes = old_changed_lines(patch)
    graph = scan_repository(root)
    by_path = defaultdict(list)
    for symbol in graph.nodes.values():
        if symbol.kind in {'function', 'method'}:
            by_path[symbol.path].append(symbol)
    touched = set()
    for path, lines in changes.items():
        functions = by_path[path]
        for line in lines:
            matches = [f for f in functions if f.line_start <= line <= f.line_end]
            if matches:
                # Select the smallest span, avoiding duplicate nested labels.
                selected = min(matches, key=lambda f:(f.line_end-f.line_start, f.id))
                touched.add(selected.id)
    return sorted(touched)


def label_from_raw(raw_dataset: str | Path, snapshots: str | Path,
                   existing_gold: str | Path, out_file: str | Path) -> dict:
    """Enrich the gold file with function labels. Do NOT write raw patch to output."""
    raw = {r['instance_id']: r for r in read_jsonl(raw_dataset)}
    labels = read_jsonl(existing_gold)
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with_functions = 0
    with out.open('w',encoding='utf8') as fh:
        for item in labels:
            ident = item['instance_id']
            base = Path(snapshots) / ident
            if ident not in raw or not base.is_dir():
                missing += 1
                continue
            ids = functions_touched(raw[ident]['patch'], base)
            enriched = dict(item)
            enriched['gold_functions'] = ids
            enriched['function_label_method'] = 'changed_base_lines_to_innermost_AST_function'
            fh.write(__import__('json').dumps(enriched,ensure_ascii=False)+'\n')
            with_functions += bool(ids)
    return {'output':str(out), 'provided':len(labels),
            'missing_raw_or_snapshot':missing,
            'with_function_labels':with_functions,
            'function_labels_caveat': 'Changes outside functions and new files excluded; insertion anchors are approximate.'}
