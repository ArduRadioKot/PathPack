"""One scan per dev task, then v0.4 ranking at hops 1/2/3 and caller/callee direction.

Gold patches are read only after every context has been packed.
"""
from __future__ import annotations
import json, time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from codegraph_rag.audit import paired_bootstrap
from codegraph_rag.benchmark import read_jsonl
from codegraph_rag.config import V4Config
from codegraph_rag.context_packer import COUNTER, token_count
from codegraph_rag.evaluation_v4 import score_prediction
from codegraph_rag.experiments import serialize_hits, verify_snapshot, write_csv, write_json, write_jsonl
from codegraph_rag.hybrid_ranker import HybridIndex
from codegraph_rag.line_metrics import build_line_label, extra_metrics
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v4 import PreparedQuery
from codegraph_rag.adaptive_chunker import build_chunk_index


VARIANTS = (
    ('v4-hops-1', {'hops': 1, 'graph_direction': 'both'}),
    ('v4-hops-2', {'hops': 2, 'graph_direction': 'both'}),
    ('v4-hops-3', {'hops': 3, 'graph_direction': 'both'}),
    ('v4-callers', {'hops': 2, 'graph_direction': 'in'}),
    ('v4-callees', {'hops': 2, 'graph_direction': 'out'}),
)


def main():
    issues = [r for r in read_jsonl(ROOT / 'output/benchmark/issues.jsonl') if r['split'] == 'dev']
    snapshots = ROOT / 'data/swebench/snapshots'
    out = ROOT / 'output/v04_graph_probe'
    if (out / 'run.json').exists():
        raise SystemExit(f'Run exists: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cfg = V4Config.load(ROOT / 'configs/pathpack_v4.json')
    predictions = []
    inventory = []
    for number, row in enumerate(issues, 1):
        repo = verify_snapshot(row, snapshots)
        t = time.perf_counter()
        graph = scan_repository(repo)
        scan_sec = time.perf_counter() - t
        t = time.perf_counter()
        chunks = build_chunk_index(repo, graph, cfg)
        index = HybridIndex(chunks, graph)
        index_sec = time.perf_counter() - t
        inventory.append({'instance_id': row['instance_id'], 'scan_sec': scan_sec, 'index_sec': index_sec,
                          'nodes': len(graph.nodes), 'edges': len(graph.edges)})
        print(f'{number}/{len(issues)} {row["instance_id"]} scan {scan_sec:.1f}s index {index_sec:.1f}s', flush=True)
        for name, fields in VARIANTS:
            variant = replace(cfg, **fields)
            t = time.perf_counter()
            prepared = PreparedQuery(index, row['problem_statement'], variant)
            rank_sec = time.perf_counter() - t
            t = time.perf_counter()
            result = prepared.pack(2048, instance_id=row['instance_id'], include_candidates=False)
            pack_sec = time.perf_counter() - t
            _, selected = serialize_hits(result.hits)
            predictions.append({
                'instance_id': row['instance_id'], 'method': name, 'budget_est': 2048,
                'tokens_est': token_count(result.context), 'token_counter': COUNTER,
                'latency_sec': rank_sec + pack_sec, 'rank_sec': rank_sec, 'pack_sec': pack_sec,
                'selected': selected, 'context': result.context,
            })
    write_jsonl(out / 'predictions.jsonl', predictions)
    write_csv(out / 'graph_inventory.csv', inventory)
    write_json(out / 'run.json', {
        'status': 'retrieval_complete', 'split': 'dev', 'budget': 2048, 'variants': [n for n, _ in VARIANTS],
        'note': 'Same v0.4 ranker and token counter. Only hops and edge direction change. Gold is not in this file.',
        'index_time_excluded_from_latency': True,
    })
    print('retrieval done; scoring', flush=True)
    from codegraph_rag.experiments import load_scoring_dataset
    source = {r['instance_id']: r for r in load_scoring_dataset(None)}
    scores = []
    for row in issues:
        repo = verify_snapshot(row, snapshots)
        gold = build_line_label(source[row['instance_id']]['patch'], repo)
        for p in predictions:
            if p['instance_id'] != row['instance_id']:
                continue
            s = score_prediction(p, gold)
            s.update(extra_metrics(p['selected'], gold))
            scores.append(s)
    write_jsonl(out / 'scores.jsonl', scores)
    groups = defaultdict(list)
    for s in scores:
        groups[s['method']].append(s)
    metrics = ('file_recall', 'function_recall', 'complete_function_recall', 'changed_line_recall',
               'relevant_line_density', 'insertion_anchor_recall', 'base_touch_recall',
               'latency_sec', 'token_utilization')
    summary = []
    for method, rows in groups.items():
        item = {'method': method, 'budget': 2048, 'tasks': len(rows)}
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric] = sum(values) / len(values) if values else None
            if metric.endswith('recall') or metric == 'relevant_line_density':
                item[metric + '_evaluable'] = len(values)
        summary.append(item)
    write_csv(out / 'summary.csv', summary)
    base = {r['instance_id']: r for r in groups['v4-hops-2']}
    comparisons = []
    for method, rows in groups.items():
        if method == 'v4-hops-2':
            continue
        for metric in ('file_recall', 'function_recall', 'complete_function_recall', 'changed_line_recall'):
            a, b = {}, {}
            for r in rows:
                other = base[r['instance_id']].get(metric)
                if r[metric] is None or other is None:
                    continue
                a[r['instance_id']] = r[metric]
                b[r['instance_id']] = other
            stat = paired_bootstrap(a, b, iterations=2000, seed=42)
            comparisons.append({'candidate': method, 'baseline': 'v4-hops-2', 'metric': metric, **stat})
    write_json(out / 'comparisons.json', comparisons)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
