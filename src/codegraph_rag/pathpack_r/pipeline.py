"""PathPack-R pipeline. A reference answer is not an input."""
from __future__ import annotations

from time import perf_counter

from ..pathpack_v4 import RetrievalResult
from .active import active_expand
from .edit_features import build_stage1
from .edit_pack import pack_edit
from .pack_utility import noisy_or, pack_utility
from .packing import pack_budgeted, pack_ranked
from .retrieve import build_pool

VARIANTS = {
    'full': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'rerank', 'active': True, 'seed': 'reranker', 'pack': 'budgeted',
        'needs_semantic': False, 'needs_rerank': True,
    },
    'semantic': {
        'channels': ('semantic',),
        'score': 'semantic', 'active': False, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': True, 'needs_rerank': False,
    },
    'fusion': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'fusion', 'active': False, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': True, 'needs_rerank': False,
    },
    'reranker': {
        'channels': ('lexical', 'symbol', 'semantic'),
        'score': 'rerank', 'active': False, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': False, 'needs_rerank': True,
    },
    'reranker-graph': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'rerank', 'active': False, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': False, 'needs_rerank': True,
    },
    'reranker-active': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'rerank', 'active': True, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': False, 'needs_rerank': True,
    },
    'active-bm25-seeds': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'rerank', 'active': True, 'seed': 'bm25', 'pack': 'ranked',
        'needs_semantic': False, 'needs_rerank': True,
    },
    'ranked': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'rerank', 'active': True, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': False, 'needs_rerank': True,
    },
    # PathPack-E: learned P(edit | issue, repo) with expected-coverage packing.
    'e': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'method': 'pathpack-e',
    },
    'e-skeleton': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit-skeleton',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'method': 'pathpack-e/skeleton',
    },
    'e-nograph': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': False, 'use_history': True, 'method': 'pathpack-e/nograph',
    },
    'e-nohist': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': False, 'method': 'pathpack-e/nohist',
    },
    'e-ranked': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'method': 'pathpack-e/ranked',
    },
    # Ablation: hand-weighted fusion score with the expected-coverage packer. Separates ranking from packing.
    'fusion-epack': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'fusion', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False,
    },
    # PathPack-H: the same model family trained by history replay, with zero benchmark labels.
    'h': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'use_memory': True, 'model': 'history', 'method': 'pathpack-h',
    },
    'h-nograph': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': False, 'use_history': True, 'use_memory': True, 'model': 'history', 'method': 'pathpack-h/nograph',
    },
    'h-nohist': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': False, 'use_memory': False, 'model': 'history', 'method': 'pathpack-h/nohist',
    },
    'h-nomemory': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'use_memory': False, 'model': 'history', 'method': 'pathpack-h/nomemory',
    },
    'h-ranked': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'ranked',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'use_memory': True, 'model': 'history', 'method': 'pathpack-h/ranked',
    },
    'h-skeleton': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit-skeleton',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'use_memory': True, 'model': 'history', 'method': 'pathpack-h/skeleton',
    },
    'eh': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'use_memory': True, 'model': 'both', 'method': 'pathpack-eh',
    },
    'e-nomemory': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'edit',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True,
        'use_graph': True, 'use_history': True, 'use_memory': False, 'model': 'label', 'method': 'pathpack-e/nomemory',
    },
    # PathPack-U: three-level edit probabilities (history replay) and expected-metric packing.
    'u': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'utility',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True, 'needs_hier': True,
        'use_graph': True, 'use_history': True, 'use_memory': False, 'model': 'history', 'method': 'pathpack-u',
    },
    'u-nobundle': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'utility-nobundle',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True, 'needs_hier': True,
        'use_graph': True, 'use_history': True, 'use_memory': False, 'model': 'history', 'method': 'pathpack-u/nobundle',
    },
    # Ablation: same objective, but function and file probabilities are the noisy-OR of chunk probabilities
    # instead of the two learned aggregate models.
    'u-noisyor': {
        'channels': ('lexical', 'symbol', 'semantic', 'structural'),
        'score': 'edit', 'active': False, 'seed': 'reranker', 'pack': 'utility-noisyor',
        'needs_semantic': True, 'needs_rerank': False, 'needs_edit': True, 'needs_hier': True,
        'use_graph': True, 'use_history': True, 'use_memory': False, 'model': 'history', 'method': 'pathpack-u/noisyor',
    },
    # Mechanism score for tests. Not a published reranker and not a benchmark arm.
    'heuristic': {
        'channels': ('lexical', 'symbol'),
        'score': 'heuristic', 'active': True, 'seed': 'heuristic', 'pack': 'budgeted',
        'needs_semantic': False, 'needs_rerank': False,
    },
}


def _public(info):
    if not info:
        return {'status': 'not_run'}
    return {key: value for key, value in info.items() if key not in {'tokenizer', 'model_obj', 'model'}}


def _heuristic(region):
    return 0.45 * region.lexical + 0.35 * region.symbol + 0.20 * region.graph


def _apply_scores(index, query, regions, config, mode, spec=None):
    if mode == 'edit':
        model = (getattr(index, 'edit_models', None) or {}).get(spec.get('model', 'label'))
        if model is None:
            return 'not_run'
        X, struct = build_stage1(regions, index, getattr(index, 'history', None),
                                 use_history=spec['use_history'], query=query)
        probs = model.predict(X, struct, use_graph=spec['use_graph'], use_history=spec['use_history'],
                              use_memory=spec.get('use_memory', True))
        for region, prob in zip(regions, probs):
            region.rerank = float(prob)
            region.final = float(prob)
        index.last_edit = (X, struct, probs)
        return 'edit'
    if mode == 'semantic':
        for region in regions:
            value = 0.0 if region.semantic is None else region.semantic
            region.final = (value + 1.0) / 2.0
        return 'semantic'
    if mode == 'fusion':
        for region in regions:
            semantic = 0.0 if region.semantic is None else (region.semantic + 1.0) / 2.0
            region.final = (
                config.fusion_lexical * region.lexical
                + config.fusion_symbol * region.symbol
                + config.fusion_semantic * semantic
                + config.fusion_graph * region.graph
            )
        return 'fusion'
    if mode == 'heuristic':
        for region in regions:
            region.final = max(_heuristic(region), region.hypothesis)
        return 'heuristic'
    if index.reranker is None or not index.reranker.available:
        return 'not_run'
    scores = index.reranker.score(query, regions, config)
    if scores is None:
        return 'not_run'
    for region, score in zip(regions, scores):
        region.rerank = score
        region.final = score
    return 'rerank'


def execute(index, query, config, variant, budget, *, instance_id=None):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown PathPack-R variant: {variant}')
    if budget <= 0:
        raise ValueError('budget must be positive')
    spec = VARIANTS[variant]
    profile = {}
    if spec['needs_semantic'] and (index.semantic is None or not index.semantic.available):
        reason = _public(None if index.semantic is None else index.semantic.info).get('reason', 'semantic model not_run')
        return {'status': 'not_run', 'reason': reason, 'variant': variant, 'profile': profile}
    if spec.get('needs_edit') and (getattr(index, 'edit_models', None) or {}).get(spec.get('model', 'label')) is None:
        return {'status': 'not_run', 'reason': 'edit model file not found; train it with edit_train', 'variant': variant, 'profile': profile}
    if spec.get('needs_hier') and getattr(index, 'hier_model', None) is None:
        return {'status': 'not_run', 'reason': 'hierarchy model file not found; train it with hier_model', 'variant': variant, 'profile': profile}
    if spec['needs_rerank'] and (index.reranker is None or not index.reranker.available):
        reason = _public(None if index.reranker is None else index.reranker.info).get('reason', 'reranker not_run')
        return {'status': 'not_run', 'reason': reason, 'variant': variant, 'profile': profile}
    started = perf_counter()
    regions, channels, queries, lexical = build_pool(index, query, config, spec['channels'])
    profile['candidate_sec'] = perf_counter() - started
    semantic_scores = None
    if index.semantic is not None and index.semantic.available:
        semantic_scores = {region.chunk.id: region.semantic for region in regions if region.semantic is not None}
        if len(semantic_scores) < len(index.chunks):
            from .semantic import encode_query
            vector = encode_query(index.semantic, config, query)
            semantic_scores = index.semantic.scores(vector)
    t = perf_counter()
    source = _apply_scores(index, query, regions, config, spec['score'], spec)
    profile['score_sec'] = perf_counter() - t
    if source == 'not_run':
        return {'status': 'not_run', 'reason': 'reranker returned no scores', 'variant': variant, 'profile': profile}
    before = [region.record() for region in regions]
    trace = []
    if spec['active']:
        t = perf_counter()
        regions, trace = active_expand(
            index, regions, queries, lexical, semantic_scores, config, seed=spec['seed'],
        )
        profile['active_sec'] = perf_counter() - t
        t = perf_counter()
        source = _apply_scores(index, query, regions, config, spec['score'])
        profile['rescore_sec'] = perf_counter() - t
        if source == 'not_run':
            return {'status': 'not_run', 'reason': 'reranker returned no scores', 'variant': variant, 'profile': profile}
    else:
        profile['active_sec'] = 0.0
        profile['rescore_sec'] = 0.0
    t = perf_counter()
    if spec['pack'] == 'budgeted':
        context, hits, pack_info = pack_budgeted(regions, budget, config)
    elif spec['pack'] in ('utility', 'utility-nobundle', 'utility-noisyor'):
        X, struct, probs = index.last_edit
        if spec['pack'] == 'utility-noisyor':
            p_fn, p_file = noisy_or(probs, struct['fn']), noisy_or(probs, struct['file'])
        else:
            p_fn, p_file = index.hier_model.predict(probs, X, struct)
        context, hits, pack_info = pack_utility(regions, index, budget, index.hier_model, p_fn, p_file, struct,
                                                bundles=spec['pack'] != 'utility-nobundle')
        pack_info['unit_probabilities'] = 'noisy_or' if spec['pack'] == 'utility-noisyor' else 'learned'
    elif spec['pack'] in ('edit', 'edit-skeleton'):
        context, hits, pack_info = pack_edit(regions, budget, config, skeleton=spec['pack'] == 'edit-skeleton')
    else:
        context, hits, pack_info = pack_ranked(regions, budget, config)
    profile['pack_sec'] = perf_counter() - t
    selected = {hit.chunk.id for hit in hits}
    for region in regions:
        if region.chunk.id in selected:
            region.decision = 'selected'
    method = spec.get('method') or ('pathpack-r' if variant == 'full' else f'pathpack-r/{variant}')
    diagnostics = {
        'instance_id': instance_id,
        'method': method,
        'variant': variant,
        'status': 'ok',
        'score_source': source,
        'config': config.to_dict(),
        'budget_est': budget,
        'channels': channels,
        'symbols': [{'kind': item.kind, 'span': item.span, 'source': item.source} for item in queries],
        'pool_before_active': before,
        'active_trace': trace,
        'candidates': [region.record() for region in regions],
        'pack': pack_info,
        'profile': profile,
        'semantic': _public(None if index.semantic is None else index.semantic.info),
        'reranker': _public(None if index.reranker is None else index.reranker.info),
        'named_evidence_stop': False,
        'fallback_used': False,
    }
    return {'status': 'ok', 'reason': '', 'variant': variant, 'profile': profile,
            'result': RetrievalResult(context, hits, diagnostics)}
