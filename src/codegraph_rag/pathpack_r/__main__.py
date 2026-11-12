"""Run PathPack-R and the v1.0e stop fix. Scoring starts only after retrieval is written."""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from ..adaptive_chunker import build_chunk_index
from ..audit import paired_bootstrap
from ..benchmark import read_jsonl
from ..config import V4Config
from ..context_packer import COUNTER, token_count
from ..experiments import code_fingerprint, serialize_hits, serialized_legacy, verify_snapshot, write_csv, write_json, write_jsonl
from ..hybrid_ranker import HybridIndex
from ..parser import scan_repository
from ..pathpack_v10e import PreparedQueryV10E, V10EConfig
from ..pathpack_v4 import PreparedQuery
from ..retrieval import BM25, select_context
from .config import PathPackRConfig
from .pipeline import VARIANTS, execute
from .rerank import load_reranker
from .retrieve import RepoIndex
from .semantic import prepare_semantic
from .tokenizer import count_texts, probe_gemma

METHODS = (
    'bm25', 'graph-expand', 'pathpack-v4', 'pathpack-v10e',
    'pathpack-r', 'pathpack-r/semantic', 'pathpack-r/fusion',
    'pathpack-r/reranker', 'pathpack-r/reranker-graph', 'pathpack-r/reranker-active',
    'pathpack-r/active-bm25-seeds', 'pathpack-r/ranked', 'pathpack-r/fusion-epack',
    'pathpack-e', 'pathpack-e/skeleton', 'pathpack-e/nograph', 'pathpack-e/nohist', 'pathpack-e/ranked',
    'pathpack-e/nomemory', 'pathpack-eh',
    'pathpack-h', 'pathpack-h/nograph', 'pathpack-h/nohist', 'pathpack-h/nomemory', 'pathpack-h/ranked',
    'pathpack-h/skeleton', 'pathpack-u', 'pathpack-u/nobundle', 'pathpack-u/noisyor',
)
METRICS = (
    'file_recall', 'function_recall', 'complete_function_recall', 'changed_line_recall',
    'insertion_anchor_recall', 'base_touch_recall', 'relevant_line_density',
)


def _rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return rss / (1024 * 1024)
    return rss / 1024


def _is_e(method):
    return method.startswith(('pathpack-e', 'pathpack-h', 'pathpack-u'))


def _is_r(method):
    return method.startswith('pathpack-r') or _is_e(method)


def _variant(method):
    if method == 'pathpack-eh':
        return 'eh'
    for prefix, short in (('pathpack-e', 'e'), ('pathpack-h', 'h'), ('pathpack-u', 'u')):
        if method == prefix:
            return short
        if method.startswith(prefix + '/'):
            return short + '-' + method.split('/', 1)[1]
    if method == 'pathpack-r':
        return 'full'
    if method.startswith('pathpack-r/'):
        return method.split('/', 1)[1]
    return None


def run_benchmark(issues_path, snapshots, out, *, split='dev', methods=METHODS, budgets=(1024, 2048, 4096),
                  limit=None, config=None, v4_config=None, v10e_config=None, allow_download=False,
                  cache_dir=None):
    out = Path(out)
    if (out / 'run.json').exists():
        raise ValueError(f'Run exists: {out}')
    cfg = config or PathPackRConfig()
    v4 = v4_config or V4Config()
    v10e = v10e_config or V10EConfig.load(Path('configs/pathpack_v10e.yaml'))
    methods = tuple(methods)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f'Unknown method: {unknown}')
    issues = [row for row in read_jsonl(issues_path) if row['split'] == split]
    if limit:
        issues = issues[:limit]
    if not issues:
        raise ValueError('No issues selected')
    if any(key == 'patch' or key.startswith('gold_') for row in issues for key in row):
        raise ValueError('Retrieval manifest must not contain gold')
    out.mkdir(parents=True, exist_ok=True)
    reranker = load_reranker(cfg, allow_download=allow_download) if any(_is_r(method) for method in methods) else None
    edit_models = {}
    if any(_is_e(method) for method in methods):
        from . import edit_model as _edit_model
        edit_models = {'label': _edit_model.load(cfg.edit_model), 'history': _edit_model.load(cfg.history_model),
                       'both': _edit_model.load(cfg.both_model)}
    hier = None
    if any(method.startswith('pathpack-u') for method in methods):
        from . import hier_model as _hier_model
        hier = _hier_model.load(cfg.hier_model)
    meta = {
        'schema': 'pathpack-r', 'code_sha256': code_fingerprint(), 'split': split, 'seed': 42,
        'budgets': list(budgets), 'methods': list(methods), 'token_counter': COUNTER,
        'config': cfg.to_dict(), 'v4_config': v4.to_dict(), 'v10e_config': v10e.to_dict(),
        'issues': [{key: row[key] for key in ('instance_id', 'repo', 'base_commit', 'split')} for row in issues],
        'snapshots': str(Path(snapshots).resolve()), 'status': 'retrieving',
        'reranker': None if reranker is None else {k: v for k, v in reranker.info.items() if k != 'model'},
        'hardware': {'platform': platform.platform(), 'machine': platform.machine()},
        'python': sys.version,
        'hier_model': None if hier is None else {'weights': hier.weights, **{k: v for k, v in hier.meta.items() if k != 'params'}},
        'edit_models': {name: None if model is None else {k: v for k, v in model.meta.items() if k != 'task_ids'}
                        for name, model in edit_models.items()},
        'note': 'v1.0e changes only the stop rule. PathPack-R does not read the gold patch.',
    }
    write_json(out / 'run.json', meta)
    predictions = []
    not_run = []
    profiles = []
    for number, row in enumerate(issues, 1):
        iid = row['instance_id']
        repo = verify_snapshot(row, snapshots)
        t = time.perf_counter()
        graph = scan_repository(repo)
        scan_sec = time.perf_counter() - t
        t = time.perf_counter()
        chunks = build_chunk_index(repo, graph, v4)
        chunk_sec = time.perf_counter() - t
        hybrid = HybridIndex(chunks, graph)
        repo_index = RepoIndex(chunks, graph, repo)
        semantic_sec = 0.0
        if any(_is_r(method) for method in methods):
            t = time.perf_counter()
            repo_index.semantic = prepare_semantic(
                repo_index, row['problem_statement'], cfg,
                cache_dir=cache_dir, cache_key=iid, allow_download=allow_download,
            )
            repo_index.reranker = reranker
            repo_index.edit_models = edit_models
            repo_index.hier_model = hier
            if any(model is not None for model in edit_models.values()):
                from .history import load_history
                repo_index.history = load_history(
                    repo, cache_dir='output/pathpack_e/history', cache_key=iid)
            semantic_sec = time.perf_counter() - t
        legacy = None
        lexical = None
        if any(method in {'bm25', 'graph-expand'} for method in methods):
            legacy = serialized_legacy(repo, graph, chunks.functions)
            lexical = BM25(legacy).scores(row['problem_statement'])
        prepared_v4 = PreparedQuery(hybrid, row['problem_statement'], v4) if 'pathpack-v4' in methods else None
        prepared_v10e = PreparedQueryV10E(hybrid, row['problem_statement'], v10e, repo) if 'pathpack-v10e' in methods else None
        for method in methods:
            variant = _variant(method)
            if variant:
                t = time.perf_counter()
                # One execution per budget: the pack depends on the budget, the pool does not.
                # Reuse the pool by calling execute per budget. Semantic vectors stay cached.
                for budget in budgets:
                    started = time.perf_counter()
                    outcome = execute(repo_index, row['problem_statement'], cfg, variant, budget, instance_id=iid)
                    elapsed = time.perf_counter() - started
                    if outcome['status'] != 'ok':
                        not_run.append({'instance_id': iid, 'method': method, 'budget': budget, 'reason': outcome['reason']})
                        continue
                    result = outcome['result']
                    _context, selected = serialize_hits(result.hits)
                    predictions.append({
                        'instance_id': iid, 'split': split, 'method': result.diagnostics['method'],
                        'budget_est': budget, 'tokens_est': token_count(result.context), 'token_counter': COUNTER,
                        'latency_sec': elapsed, 'selected': selected, 'context': result.context,
                        'profile': {**result.diagnostics['profile'], 'scan_sec': scan_sec, 'chunk_sec': chunk_sec, 'semantic_index_sec': semantic_sec},
                        'fallback_used': False,
                    })
                    dest = out / 'diagnostics' / f'{iid}_{method.replace("/", "_")}_{budget}.json.gz'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(dest, 'wt', compresslevel=1) as handle:
                        json.dump(result.diagnostics, handle)
                profiles.append({'instance_id': iid, 'method': method, 'scan_sec': scan_sec, 'chunk_sec': chunk_sec,
                                 'semantic_index_sec': semantic_sec, 'rss_mb': round(_rss_mb(), 1),
                                 'query_sec': time.perf_counter() - t})
                continue
            for budget in budgets:
                t = time.perf_counter()
                if method == 'pathpack-v4':
                    result = prepared_v4.pack(budget, instance_id=iid, include_candidates=False)
                    context, hits = result.context, result.hits
                elif method == 'pathpack-v10e':
                    result = prepared_v10e.pack(budget, instance_id=iid, include_candidates=True)
                    context, hits = result.context, result.hits
                    dest = out / 'diagnostics' / f'{iid}_pathpack-v10e_{budget}.json.gz'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(dest, 'wt', compresslevel=1) as handle:
                        json.dump(result.diagnostics, handle)
                else:
                    hits = select_context(legacy, graph, row['problem_statement'], method=method, budget=budget, lexical_scores=lexical)
                    context, _selected = serialize_hits(hits)
                selected_rows = serialize_hits(hits)[1]
                predictions.append({
                    'instance_id': iid, 'split': split, 'method': method, 'budget_est': budget,
                    'tokens_est': token_count(context), 'token_counter': COUNTER,
                    'latency_sec': time.perf_counter() - t, 'selected': selected_rows, 'context': context,
                    'profile': {'scan_sec': scan_sec, 'chunk_sec': chunk_sec, 'rss_mb': round(_rss_mb(), 1)},
                    'fallback_used': bool(getattr(result, 'diagnostics', {}).get('fallback_used', False)) if method == 'pathpack-v10e' else False,
                })
        write_jsonl(out / f'{iid}_predictions.jsonl', [row for row in predictions if row['instance_id'] == iid])
        print(f'[{number}/{len(issues)}] {iid}: scan={scan_sec:.1f}s chunks={len(chunks.chunks)} predictions={sum(p["instance_id"]==iid for p in predictions)}', flush=True)
    write_jsonl(out / 'predictions.jsonl', predictions)
    write_jsonl(out / 'not_run.jsonl', not_run)
    write_csv(out / 'stage_profile.csv', profiles)
    meta['status'] = 'retrieval_complete'
    meta['prediction_count'] = len(predictions)
    meta['not_run'] = len(not_run)
    meta['semantic'] = None if not issues else _semantic_status(out)
    write_json(out / 'run.json', meta)
    return {'run': str(out), 'tasks': len(issues), 'predictions': len(predictions), 'not_run': len(not_run)}


def _semantic_status(out):
    return {'recorded_in': 'diagnostics', 'policy': 'not_run when the pinned code model cannot be loaded; no substitute vectors'}


def _mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return (sum(values) / len(values) if values else None), len(values)


def score_run(run, *, v10_scores=None):
    from ..evaluation_v4 import score_prediction
    from ..experiments import load_scoring_dataset
    from ..line_metrics import build_line_label, extra_metrics

    run = Path(run)
    meta = json.loads((run / 'run.json').read_text())
    if meta['status'] not in {'retrieval_complete', 'scored'}:
        raise ValueError('Retrieval must finish before scoring')
    source = {row['instance_id']: row for row in load_scoring_dataset()}
    by_issue = {row['instance_id']: row for row in meta['issues']}
    scores = []
    pool_rows = []
    predictions = read_jsonl(run / 'predictions.jsonl')
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[prediction['instance_id']].append(prediction)
    for iid, issue in by_issue.items():
        repo = verify_snapshot(issue, meta['snapshots'])
        gold = build_line_label(source[iid]['patch'], repo)
        for prediction in grouped.get(iid, []):
            scored = score_prediction(prediction, gold, function_coverage_threshold=0.20)
            scored.update(extra_metrics(prediction['selected'], gold))
            scores.append(scored)
            diag = run / 'diagnostics' / f'{iid}_{prediction["method"].replace("/", "_")}_{prediction["budget_est"]}.json.gz'
            if not diag.is_file():
                continue
            with gzip.open(diag, 'rt') as handle:
                payload = json.load(handle)
            pool = payload.get('pool_before_active') or payload.get('candidates') or []
            files = set(gold['gold_python_files'])
            functions = {fn.get('graph_id', fn['id']) for fn in gold['functions'].values()}
            pool_files = {item.get('path') for item in pool}
            pool_fns = {item.get('function_id') for item in pool if item.get('function_id')}
            packed_files = {item['path'] for item in prediction['selected']}
            packed_fns = {item.get('function_id') for item in prediction['selected'] if item.get('function_id')}
            pool_rows.append({
                'instance_id': iid, 'method': prediction['method'], 'budget': prediction['budget_est'],
                'pool_regions': len(pool),
                'file_pool_recall': None if not files else len(files & pool_files) / len(files),
                'function_pool_recall': None if not functions else len(functions & pool_fns) / len(functions),
                'file_packed_recall': None if not files else len(files & packed_files) / len(files),
                'function_packed_recall': None if not functions else len(functions & packed_fns) / len(functions),
            })
    write_csv(run / 'pool_recall.csv', pool_rows)
    if v10_scores and Path(v10_scores).is_file():
        wanted = set(meta['budgets'])
        frozen = [
            row for row in read_jsonl(v10_scores)
            if row['instance_id'] in by_issue and row['method'] == 'pathpack-v10' and row['budget'] in wanted
        ]
        for row in frozen:
            copied = dict(row)
            copied['source'] = 'frozen:' + str(v10_scores)
            scores.append(copied)
    write_jsonl(run / 'scores.jsonl', scores)
    groups = defaultdict(list)
    for row in scores:
        groups[(row['method'], row['budget'])].append(row)
    summary = []
    for (method, budget), rows in sorted(groups.items(), key=lambda item: (str(item[0][1]), item[0][0])):
        item = {'method': method, 'budget': budget, 'tasks': len({row['instance_id'] for row in rows})}
        for metric in METRICS + ('latency_sec', 'token_utilization', 'tokens_est'):
            mean, count = _mean(rows, metric)
            item[metric] = mean
            if metric.endswith('recall') or metric == 'relevant_line_density':
                item[metric + '_evaluable'] = count
        summary.append(item)
    write_csv(run / 'summary.csv', summary)
    comparisons = []
    for budget in meta['budgets']:
        base = {row['instance_id']: row for row in groups.get(('pathpack-v4', budget), [])}
        for (method, method_budget), rows in sorted(groups.items()):
            if method_budget != budget or method == 'pathpack-v4':
                continue
            for metric in ('file_recall', 'function_recall', 'complete_function_recall', 'changed_line_recall'):
                left = {iid: row[metric] for iid, row in base.items() if row.get(metric) is not None}
                right = {row['instance_id']: row[metric] for row in rows if row.get(metric) is not None}
                # Difference is candidate minus v0.4. Positive means the candidate is higher.
                stat = paired_bootstrap(right, left, iterations=2000, seed=42)
                comparisons.append({'candidate': method, 'baseline': 'pathpack-v4', 'budget': budget, 'metric': metric, **stat})
    write_json(run / 'comparisons.json', comparisons)
    write_csv(run / 'comparisons.csv', [{**row, 'ci95': json.dumps(row['ci95'])} for row in comparisons])
    gemma = probe_gemma(PathPackRConfig(**meta['config']) if meta.get('config') else None, allow_download=False)
    usage = []
    if gemma['status'] == 'available':
        texts = [row['context'] for row in predictions]
        counts, gemma = count_texts(PathPackRConfig(**meta['config']), texts, allow_download=False)
        if counts is not None:
            for row, count in zip(predictions, counts):
                usage.append({'instance_id': row['instance_id'], 'method': row['method'], 'budget_est': row['budget_est'],
                              'utf8_tokens': row['tokens_est'], 'gemma_tokens': count})
    write_json(run / 'gemma_tokenizer.json', gemma)
    write_csv(run / 'gemma_usage.csv', usage)
    audit = _audit(predictions)
    write_json(run / 'audit.json', audit)
    meta['status'] = 'scored'
    meta['gemma_tokenizer'] = gemma.get('status')
    write_json(run / 'run.json', meta)
    return {'tasks': len(by_issue), 'summary': summary, 'audit': audit['ok'], 'gemma': gemma.get('status')}


def _audit(predictions):
    violations = []
    seen = set()
    for prediction in predictions:
        key = (prediction['instance_id'], prediction['method'], prediction['budget_est'])
        if key in seen:
            violations.append({'type': 'duplicate_prediction', 'key': list(key)})
        seen.add(key)
        used = token_count(prediction['context'])
        if used != prediction['tokens_est'] or used > prediction['budget_est']:
            violations.append({'type': 'budget', 'key': list(key), 'used': used})
        ids = [item['id'] for item in prediction['selected']]
        if len(ids) != len(set(ids)):
            violations.append({'type': 'duplicate_chunk', 'key': list(key)})
        intervals = defaultdict(list)
        for chunk in prediction['selected']:
            for line, start, end in chunk.get('source_spans') or []:
                if line < 1 or start < 0 or end <= start:
                    violations.append({'type': 'bad_span', 'key': list(key), 'line': line})
                slot = (chunk['path'], line)
                if any(max(start, left) < min(end, right) for left, right in intervals[slot]):
                    violations.append({'type': 'duplicate_range', 'key': list(key), 'line': line})
                intervals[slot].append((start, end))
    return {'ok': not violations, 'predictions': len(predictions), 'violations': violations}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pathpack-r')
    parser.add_argument('cmd', choices=('benchmark', 'score'))
    parser.add_argument('--issues', default='output/benchmark/issues.jsonl')
    parser.add_argument('--snapshots', default='data/swebench/snapshots')
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument('--method', nargs='+', default=list(METHODS))
    parser.add_argument('--budgets', nargs='+', type=int, default=[1024, 2048, 4096])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--config', default='configs/pathpack_r.yaml')
    parser.add_argument('--v4-config', default='configs/pathpack_v4.json')
    parser.add_argument('--v10e-config', default='configs/pathpack_v10e.yaml')
    parser.add_argument('--cache', default='output/pathpack_r_cache')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--v10-scores', default='output/v10_diag/scores.jsonl')
    args = parser.parse_args(argv)
    if args.cmd == 'benchmark':
        print(json.dumps(run_benchmark(
            args.issues, args.snapshots, args.out, split=args.split, methods=args.method,
            budgets=args.budgets, limit=args.limit, config=PathPackRConfig.load(args.config),
            v4_config=V4Config.load(args.v4_config) if Path(args.v4_config).is_file() else V4Config(),
            v10e_config=V10EConfig.load(args.v10e_config), allow_download=args.allow_download,
            cache_dir=args.cache,
        )))
    else:
        print(json.dumps(score_run(args.out, v10_scores=args.v10_scores), default=str))


if __name__ == '__main__':
    main()
