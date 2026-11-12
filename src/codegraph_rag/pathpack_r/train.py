"""Offline linear reranker. Inference never imports this module.

Positives are patched functions. Hard negatives are untouched functions in the
same file, high-BM25 misses, untouched graph neighbors, and other high-ranked
misses. The loss is pairwise logistic. Training refuses any instance or, when
repository-disjoint mode is on, any repository that appears in an evaluation manifest.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from ..benchmark import read_jsonl
from .retrieve import RepoIndex, build_pool
from .config import PathPackRConfig

_HUNK = __import__('re').compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')


def changed_base_lines(patch: str) -> dict[str, set[int]]:
    """Base-file lines removed or used as insertion anchors. Training only."""
    from collections import defaultdict
    result = defaultdict(set)
    current = None
    line_no = None
    for line in (patch or '').splitlines():
        if line.startswith('--- '):
            current = line[6:] if line.startswith('--- a/') else None
            if current and not current.endswith('.py'):
                current = None
            line_no = None
        elif line.startswith('@@ '):
            match = _HUNK.match(line)
            line_no = int(match.group(1)) if match else None
        elif current is not None and line_no is not None:
            if line.startswith('-'):
                result[current].add(line_no)
                line_no += 1
            elif line.startswith('+'):
                result[current].add(max(1, line_no - 1))
                result[current].add(max(1, line_no))
            elif line.startswith(' '):
                line_no += 1
    return dict(result)


FEATURES = ('lexical', 'symbol', 'graph', 'repo', 'semantic', 'same_file')


def feature_vector(region, symbol_files):
    semantic = 0.0 if region.semantic is None else (region.semantic + 1.0) / 2.0
    return [
        region.lexical,
        region.symbol,
        region.graph,
        region.repo,
        semantic,
        1.0 if region.chunk.path in symbol_files else 0.0,
    ]


def pairwise_loss(weights, positives, negatives):
    total = 0.0
    for pos in positives:
        for neg in negatives:
            margin = sum(w * (p - n) for w, p, n in zip(weights, pos, neg))
            total += math.log1p(math.exp(-max(min(margin, 20.0), -20.0)))
    count = max(1, len(positives) * len(negatives))
    return total / count


def train_linear(pairs, *, steps=300, lr=0.05, seed=42):
    """Minimize the mean pairwise logistic loss. ``pairs`` are (positive, negative) feature rows."""
    if not pairs:
        raise ValueError('No training pairs')
    width = len(pairs[0][0])
    rng = random.Random(seed)
    weights = [0.0] * width
    for _step in range(steps):
        rng.shuffle(pairs)
        grad = [0.0] * width
        for pos, neg in pairs:
            margin = sum(w * (p - n) for w, p, n in zip(weights, pos, neg))
            # d/dw log(1 + exp(-(w·(p-n)))) = -sigmoid(-margin) * (p-n)
            scale = -1.0 / (1.0 + math.exp(max(min(margin, 20.0), -20.0)))
            for index, (p, n) in enumerate(zip(pos, neg)):
                grad[index] += scale * (p - n)
        inv = 1.0 / len(pairs)
        weights = [w - lr * inv * g for w, g in zip(weights, grad)]
    return weights


def assert_disjoint(train_rows, eval_manifests, *, repository_disjoint=True):
    eval_ids = set()
    eval_repos = set()
    for path in eval_manifests:
        for row in read_jsonl(path):
            eval_ids.add(row['instance_id'])
            if row.get('repo'):
                eval_repos.add(row['repo'])
    train_ids = {row['instance_id'] for row in train_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f'Training instances overlap evaluation: {overlap[:8]}')
    train_repos = {row.get('repo') for row in train_rows if row.get('repo')}
    shared = sorted(train_repos & eval_repos)
    if repository_disjoint and shared:
        raise ValueError(f'Training repositories overlap evaluation: {shared}')
    return {'instance_overlap': overlap, 'repository_overlap': shared, 'repository_disjoint': repository_disjoint}


def build_pairs(index: RepoIndex, query: str, patch: str, config: PathPackRConfig | None = None):
    """Build one issue's pairs. The patch is read only here."""
    config = config or PathPackRConfig()
    regions, _channels, _queries, _lexical = build_pool(
        index, query, config, ('lexical', 'symbol', 'structural'),
    )
    changed = changed_base_lines(patch)
    positive_ids = set()
    positive_files = set()
    for region in regions:
        lines = changed.get(region.chunk.path, set())
        if any(region.chunk.line_start <= line <= region.chunk.line_end for line in lines):
            positive_ids.add(region.chunk.id)
            positive_files.add(region.chunk.path)
    symbol_files = {region.chunk.path for region in regions if region.symbol > 0}
    positives = [feature_vector(region, symbol_files) for region in regions if region.chunk.id in positive_ids]
    ranked = sorted(regions, key=lambda region: (-region.lexical, region.chunk.id))
    negatives = []
    for region in ranked:
        if region.chunk.id in positive_ids:
            continue
        same_file = region.chunk.path in positive_files
        hard_lexical = region.lexical >= 0.2
        graph_neighbor = region.graph > 0
        if same_file or hard_lexical or graph_neighbor:
            negatives.append(feature_vector(region, symbol_files))
    pairs = [(pos, neg) for pos in positives for neg in negatives[:8]]
    return pairs, {'positives': len(positives), 'hard_negatives': min(len(negatives), 8)}


def save_checkpoint(path, weights, meta):
    Path(path).write_text(json.dumps({'features': FEATURES, 'weights': weights, 'meta': meta}, indent=2) + '\n')
