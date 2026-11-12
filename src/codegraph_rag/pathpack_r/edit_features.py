"""Feature extraction for PathPack-E: P(chunk is edited | issue, repository).

Stage 1 sees only the candidate itself, its file, its function and git history before
``base_commit``. Stage 2 adds one round of learned message passing over call edges:
the stage-1 probabilities of callers and callees. No feature reads a patch.
"""
from __future__ import annotations

import math

import numpy as np

CHANNELS = ('lexical', 'lexical_name', 'symbol', 'semantic', 'structural')

STAGE1 = (
    'lexical', 'semantic', 'symbol', 'graph', 'repo', 'n_channels',
    'rank_lexical', 'rank_lexical_name', 'rank_symbol', 'rank_semantic', 'rank_structural',
    'log_chunk_tokens', 'log_function_tokens', 'chunk_position', 'is_first_chunk', 'in_symbol_file',
    'file_max_lexical', 'file_max_semantic', 'log_file_regions', 'file_lexical_pct', 'file_semantic_pct',
    'lexical_rel_file', 'semantic_rel_file', 'lexical_pool_pct', 'semantic_pool_pct',
    'function_max_lexical', 'function_regions',
    'log_file_churn', 'log_file_fixes', 'cochange_top_files', 'cochange_symbol_files',
    'memory_file', 'memory_file_pct',
)
MEMORY_COLUMNS = tuple(STAGE1.index(name) for name in ('memory_file', 'memory_file_pct'))
HISTORY_COLUMNS = tuple(STAGE1.index(name) for name in (
    'log_file_churn', 'log_file_fixes', 'cochange_top_files', 'cochange_symbol_files')) + MEMORY_COLUMNS
STAGE2 = ('p0', 'callers_max_p0', 'callees_max_p0', 'log_neighbors', 'file_max_p0', 'function_max_p0')


def _pct(values):
    """Rank percentile in [0, 1]. 0 is the highest value. Deterministic on ties."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(len(values))
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks / (len(values) - 1)


def _inv_rank(region, channel):
    rank = region.ranks.get(channel)
    return 0.0 if rank is None else 1.0 / (1.0 + float(rank))


def build_stage1(regions, index, history=None, *, use_history=True, query=None):
    """Return ``(X, struct)``. ``X`` has one row per region in the given order."""
    n = len(regions)
    X = np.zeros((n, len(STAGE1)), dtype=np.float64)
    if n == 0:
        return X, {'file': np.zeros(0, dtype=int), 'fn': np.zeros(0, dtype=int), 'callers': [], 'callees': []}
    lexical = np.array([r.lexical for r in regions])
    semantic = np.array([0.0 if r.semantic is None else (r.semantic + 1.0) / 2.0 for r in regions])
    paths = [r.chunk.path for r in regions]
    file_ids = {}
    for path in paths:
        file_ids.setdefault(path, len(file_ids))
    file_idx = np.array([file_ids[p] for p in paths], dtype=int)
    fn_keys = [r.chunk.function_id or f'chunk:{r.chunk.id}' for r in regions]
    fn_ids = {}
    for key in fn_keys:
        fn_ids.setdefault(key, len(fn_ids))
    fn_idx = np.array([fn_ids[k] for k in fn_keys], dtype=int)

    n_files = len(file_ids)
    file_max_lex = np.zeros(n_files)
    file_max_sem = np.zeros(n_files)
    file_count = np.zeros(n_files)
    for i in range(n):
        f = file_idx[i]
        file_max_lex[f] = max(file_max_lex[f], lexical[i])
        file_max_sem[f] = max(file_max_sem[f], semantic[i])
        file_count[f] += 1
    n_fns = len(fn_ids)
    fn_max_lex = np.zeros(n_fns)
    fn_count = np.zeros(n_fns)
    for i in range(n):
        g = fn_idx[i]
        fn_max_lex[g] = max(fn_max_lex[g], lexical[i])
        fn_count[g] += 1
    file_lex_pct = _pct(file_max_lex)
    file_sem_pct = _pct(file_max_sem)
    lex_pct = _pct(lexical)
    sem_pct = _pct(semantic)

    symbol_files = {r.chunk.path for r in regions if r.symbol > 0}
    top_files = [p for p, _ in sorted(((p, file_max_lex[file_ids[p]]) for p in file_ids), key=lambda x: (-x[1], x[0]))[:3]]
    co_top, co_sym = {}, {}
    if use_history and history is not None:
        for path in file_ids:
            co_top[path] = history.cochange_with(path, top_files)
            co_sym[path] = history.cochange_with(path, sorted(symbol_files)) if symbol_files else 0.0

    memory = {}
    if use_history and history is not None and query:
        memory = history.memory_scores(query)
    mem_file = np.array([memory.get(p, 0.0) for p in file_ids])
    mem_pct = _pct(mem_file)

    for i, r in enumerate(regions):
        chunk = r.chunk
        owned = index.by_fn.get(chunk.function_id, []) if chunk.function_id else [chunk]
        order = sorted(owned, key=lambda c: (c.line_start or 0, c.id))
        pos = 0
        for j, c in enumerate(order):
            if c.id == chunk.id:
                pos = j
                break
        fn_tokens = sum(max(1, int(c.tokens_est or 1)) for c in owned)
        row = [
            lexical[i], semantic[i], r.symbol, r.graph, r.repo, float(len(r.channels)),
            *[_inv_rank(r, ch) for ch in CHANNELS],
            math.log1p(r.token_cost), math.log1p(fn_tokens),
            pos / max(1, len(order) - 1), 1.0 if pos == 0 else 0.0,
            1.0 if chunk.path in symbol_files else 0.0,
            file_max_lex[file_idx[i]], file_max_sem[file_idx[i]], math.log1p(file_count[file_idx[i]]),
            file_lex_pct[file_idx[i]], file_sem_pct[file_idx[i]],
            lexical[i] / max(file_max_lex[file_idx[i]], 1e-9), semantic[i] / max(file_max_sem[file_idx[i]], 1e-9),
            lex_pct[i], sem_pct[i],
            fn_max_lex[fn_idx[i]], fn_count[fn_idx[i]],
            math.log1p(history.file_churn(chunk.path)) if (use_history and history is not None) else 0.0,
            math.log1p(history.file_fixes(chunk.path)) if (use_history and history is not None) else 0.0,
            co_top.get(chunk.path, 0.0), co_sym.get(chunk.path, 0.0),
            mem_file[file_idx[i]], (1.0 - mem_pct[file_idx[i]]) if mem_file[file_idx[i]] > 0 else 0.0,
        ]
        X[i] = row

    # Call-edge neighbours as region indices. Only calls; direction is kept.
    by_function = {}
    for i, r in enumerate(regions):
        if r.chunk.function_id:
            by_function.setdefault(r.chunk.function_id, []).append(i)
    callers, callees = [], []
    cache = {}
    for i, r in enumerate(regions):
        fid = r.chunk.function_id
        if not fid:
            callers.append([])
            callees.append([])
            continue
        if fid not in cache:
            ins, outs = [], []
            for other, kind, direction in index.relations(fid):
                if kind != 'calls' or other == fid:
                    continue
                (ins if direction == 'in' else outs).extend(by_function.get(other, []))
            cache[fid] = (sorted(set(ins)), sorted(set(outs)))
        callers.append(cache[fid][0])
        callees.append(cache[fid][1])
    return X, {'file': file_idx, 'fn': fn_idx, 'callers': callers, 'callees': callees}


def stage2_features(p0, struct):
    """One message-passing round over call edges, from stage-1 probabilities ``p0``."""
    p0 = np.asarray(p0, dtype=np.float64)
    n = len(p0)
    out = np.zeros((n, len(STAGE2)))
    if n == 0:
        return out
    file_idx, fn_idx = struct['file'], struct['fn']
    file_max = np.zeros(file_idx.max() + 1)
    fn_max = np.zeros(fn_idx.max() + 1)
    for i in range(n):
        file_max[file_idx[i]] = max(file_max[file_idx[i]], p0[i])
        fn_max[fn_idx[i]] = max(fn_max[fn_idx[i]], p0[i])
    for i in range(n):
        cin, cout = struct['callers'][i], struct['callees'][i]
        out[i] = [
            p0[i],
            max((p0[j] for j in cin), default=0.0),
            max((p0[j] for j in cout), default=0.0),
            math.log1p(len(cin) + len(cout)),
            file_max[file_idx[i]],
            fn_max[fn_idx[i]],
        ]
    return out
