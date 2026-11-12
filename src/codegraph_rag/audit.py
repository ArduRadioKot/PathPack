"""Per-issue retrieval audit, held-out-gold comparison and paired bootstrap.

This module reads labels for post-hoc evaluation ONLY. Never call it from the
retrieval path or use test results to tune parameters before final reporting.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean

from .benchmark import read_jsonl


def _selected_id(chunk: dict) -> str:
    return str(chunk.get('id') or f"{chunk['path']}::{chunk.get('qualname', '__module__')}")


def _func_id(gold: str | dict) -> str:
    if isinstance(gold, str):
        return gold
    return str(gold.get('id') or f"{gold['path']}::{gold['qualname']}")


def paired_bootstrap(a: dict[str, float], b: dict[str, float], *,
                     iterations: int = 1000, seed: int = 42) -> dict:
    """Paired CI on (a - b), sampled at the issue level; no p-value claim."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {'n': 0, 'difference': None, 'ci95': None, 'wins': 0, 'ties': 0, 'losses': 0}
    ds = [a[i] - b[i] for i in ids]
    rng = Random(seed)
    boots = sorted(mean(rng.choices(ds, k=len(ds))) for _ in range(iterations)) if iterations else []
    ci = [boots[int((len(boots)-1)*0.025)], boots[int((len(boots)-1)*0.975)]] if boots else None
    return {'n': len(ds), 'difference': round(mean(ds), 6),
            'ci95': [round(v, 6) for v in ci] if ci else None,
            'wins': sum(d > 0 for d in ds), 'ties': sum(d == 0 for d in ds),
            'losses': sum(d < 0 for d in ds)}


def audit_predictions(prediction_files: list[str | Path], gold_path: str | Path, *,
                      iterations: int = 1000) -> dict:
    if iterations < 0 or iterations > 100000:
        raise ValueError('bootstrap iterations must be 0..100000')
    gold_rows = read_jsonl(gold_path)
    gold = {g['instance_id']: g for g in gold_rows}
    if len(gold) != len(gold_rows):
        raise ValueError('Duplicate gold instance_id')
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    budget_violations: list[dict] = []
    duplicates: list[dict] = []
    for source in prediction_files:
        for row in read_jsonl(source):
            instance = row['instance_id']
            if instance not in gold:
                raise ValueError(f'Missing gold for {instance}')
            method = row['method']
            if method.startswith(('pathpack-v4','pathpack-v5')):
                raise ValueError('Use audit --run for v0.4/v0.5 serialized contexts and score-run for body-line coverage')
            budget = int(row.get('budget_est', row.get('budget', 0)))
            key = f'{method}@{budget}'
            if instance in groups[key]:
                raise ValueError(f'Duplicate prediction {instance} in {key}')
            selected = row['selected']
            ids = [_selected_id(c) for c in selected]
            if len(ids) != len(set(ids)):
                duplicates.append({'group': key, 'instance_id': instance})
            tokens_sum = sum(int(c.get('tokens_est', 0)) for c in selected)
            reported = row.get('tokens_est')
            if tokens_sum > budget or (reported is not None and int(reported) != tokens_sum):
                budget_violations.append({'group': key, 'instance_id': instance,
                                          'budget': budget, 'computed': tokens_sum,
                                          'reported': reported})
            files = {c['path'] for c in selected}
            targets = set(gold[instance].get('gold_python_files', []))
            functions = {_func_id(f) for f in gold[instance].get('gold_functions', [])}
            recall = (len(files & targets)/len(targets)) if targets else None
            frecall = (len(set(ids) & functions)/len(functions)) if functions else None
            groups[key][instance] = {
                'file_recall': recall, 'function_recall': frecall,
                'selected_ids': set(ids), 'selected_files': files,
                'gold_files': targets, 'gold_functions': functions,
                'found_file_missed_function': bool((files & targets) and functions
                                                   and not (set(ids) & functions)),
            }
    # group-wise summaries including coverage. No zero-filling absent tasks.
    summaries = {}
    for key, issues in sorted(groups.items()):
        fr = [v['file_recall'] for v in issues.values() if v['file_recall'] is not None]
        fu = [v['function_recall'] for v in issues.values() if v['function_recall'] is not None]
        summaries[key] = {'tasks': len(issues), 'file_evaluable': len(fr),
                          'file_recall': round(mean(fr), 6) if fr else None,
                          'function_evaluable': len(fu),
                          'function_recall': round(mean(fu), 6) if fu else None,
                          'file_hit_function_miss': sum(v['found_file_missed_function'] for v in issues.values())}
    # Strict nestedness: all chunks selected at lower budget should still be
    # present at a larger budget for the same issue and method.
    methods = defaultdict(list)
    for key in groups:
        m, b = key.rsplit('@', 1)
        methods[m].append((int(b), key))
    nested_violations = []
    for method, levels in sorted(methods.items()):
        levels.sort()
        for (_, ka), (_, kb) in zip(levels, levels[1:]):
            for iid in sorted(groups[ka].keys() & groups[kb].keys()):
                lost = groups[ka][iid]['selected_ids'] - groups[kb][iid]['selected_ids']
                if lost:
                    nested_violations.append({'instance_id': iid, 'method': method,
                                              'small': ka, 'large': kb, 'lost': sorted(lost)[:20]})
    # Comparisons use ONLY matched issues; no pretending paired data when
    # some snapshots are missing.
    comparisons = {}
    for metric in ('file_recall', 'function_recall'):
        for budget in sorted({int(k.rsplit('@',1)[1]) for k in groups}):
            main = groups.get(f'pathpack-v3@{budget}') or groups.get(f'pathpack@{budget}')
            if not main:
                continue
            for method in ('bm25', 'graph-expand', 'pathpack'):
                other = groups.get(f'{method}@{budget}')
                if not other or main is other:
                    continue
                a = {iid:v[metric] for iid,v in main.items() if v[metric] is not None}
                b = {iid:v[metric] for iid,v in other.items() if v[metric] is not None}
                comparisons[f'{metric}:{budget}:pathpack_vs_{method}'] = paired_bootstrap(
                    a, b, iterations=iterations)
    return {'groups': summaries, 'budget_violations': budget_violations,
            'duplicate_selection': duplicates, 'nested_violations': nested_violations,
            'comparisons': comparisons,
            'diagnostic_cases': [
                {'group': key, 'instance_id': iid,
                 'correct_files': sorted(v['selected_files'] & v['gold_files']),
                 'missed_functions': sorted(v['gold_functions'] - v['selected_ids'])[:15]}
                for key, issues in sorted(groups.items())
                for iid,v in sorted(issues.items()) if v['found_file_missed_function']
            ][:250],
            'notes': ['Gold functions optional: if absent, function metrics are not calculated.',
                      'Nestedness is a desirable protocol property, not required of all algorithms.',
                      'Paired bootstrap confidence intervals describe sampling uncertainty, not proof of novelty.',
                      'Run on dev for development. Hold test out until methodology is frozen.']}
