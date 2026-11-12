"""PathPack-U, part 2: pack to maximize the expected localization metrics under a token budget.

For a selected set S the objective is a sum of four expected recalls:

  U(S) = w_file  * sum_f P(f) [S touches file f]
       + w_fn    * sum_g P(g) [S covers at least THETA of function g]
       + w_fn    * sum_g P(g) [S covers all of function g]
       + w_chunk * sum_c P(c) [c in S]

P(.) come from the chunk model and the two aggregate models (``hier_model``). The weights are the
reciprocals of the mean number of edited units per history-replay task, so nothing is tuned.
THETA is the coverage threshold the Function Recall metric itself uses.

Selection is a density greedy over two kinds of moves: add one chunk, or add every remaining chunk
of one function (a bundle). Without bundles a greedy never pays for the completion term, because no
single chunk earns it. Named-symbol coverage is not a stop rule; the only stop is the budget.
"""
from __future__ import annotations

from ..context_packer import token_count
from .packing import _pack
from .retrieve import Region

THETA = 0.20


def noisy_or(p, groups):
    """P(unit is edited) = 1 - prod(1 - p_chunk) over the unit's chunks in the pool. Ablation baseline."""
    import numpy as np
    p = np.asarray(p, dtype=float)
    out = np.ones(int(groups.max()) + 1 if len(groups) else 0)
    np.multiply.at(out, groups, 1.0 - p)
    return 1.0 - out
MAX_BUNDLE_FUNCTIONS = 60


def pack_utility(regions, index, budget, model, p_fn, p_file, struct, *, bundles=True):
    """``p_fn`` / ``p_file`` are indexed by ``struct['fn']`` / ``struct['file']`` ids of ``regions``."""
    w = model.weights
    info = {}                      # chunk id -> dict(region, p, fn, file)
    fn_p, file_p = {}, {}
    for i, region in enumerate(regions):
        fkey = region.chunk.function_id or f'chunk:{region.chunk.id}'
        fn_p[fkey] = float(p_fn[struct['fn'][i]])
        file_p[region.chunk.path] = float(p_file[struct['file'][i]])
        info[region.chunk.id] = {'region': region, 'p': float(region.final), 'fn': fkey, 'file': region.chunk.path}
    # Functions worth completing: bring in their chunks that are outside the pool, with chunk probability 0.
    extra = 0
    if bundles:
        top = sorted((k for k in fn_p if not k.startswith('chunk:')), key=lambda k: (-fn_p[k], k))[:MAX_BUNDLE_FUNCTIONS]
        for fkey in top:
            for chunk in index.by_fn.get(fkey, ()):
                if chunk.id not in info and chunk.source_spans and (chunk.body or '').strip():
                    region = Region(chunk=chunk, token_cost=max(1, int(chunk.tokens_est or 1)), admission='completion')
                    info[chunk.id] = {'region': region, 'p': 0.0, 'fn': fkey, 'file': chunk.path}
                    extra += 1
    fn_total, fn_members = {}, {}
    for cid, item in info.items():
        fn_members.setdefault(item['fn'], []).append(cid)
    for fkey, members in fn_members.items():
        owned = index.by_fn.get(fkey) if not fkey.startswith('chunk:') else None
        fn_total[fkey] = sum(max(1, int(c.tokens_est or 1)) for c in owned) if owned else sum(info[c]['region'].token_cost for c in members)
        members.sort(key=lambda c: (info[c]['region'].chunk.line_start or 0, c))
    complete_possible = {f: (index.by_fn.get(f) is None or len(index.by_fn.get(f, ())) == len(m)) for f, m in fn_members.items()}

    selected, order = set(), []
    touched, covered = set(), {f: 0 for f in fn_members}
    used = 0

    def gain(cids):
        g, add = 0.0, {}
        files = set()
        for cid in cids:
            item = info[cid]
            g += w['chunk'] * item['p']
            add[item['fn']] = add.get(item['fn'], 0) + item['region'].token_cost
            if item['file'] not in touched:
                files.add(item['file'])
        g += sum(w['file'] * file_p.get(f, 0.0) for f in files)
        for fkey, tokens in add.items():
            before, after, total = covered[fkey], covered[fkey] + tokens, max(fn_total[fkey], 1)
            pg = fn_p.get(fkey, 0.0)
            if before / total < THETA <= after / total:
                g += w['fn'] * pg
            if complete_possible[fkey] and before < total <= after:
                g += w['fn'] * pg
        return g

    moves = {'chunk': 0, 'bundle': 0}
    while True:
        best = None
        left = budget - used
        for cid, item in info.items():
            cost = item['region'].token_cost
            if cid in selected or cost > left:
                continue
            g = gain([cid])
            key = (g / cost, g, -cost, cid)
            if best is None or key > best[0]:
                best = (key, [cid], 'chunk')
        if bundles:
            for fkey, members in fn_members.items():
                rest = [c for c in members if c not in selected]
                if len(rest) < 2 or not complete_possible[fkey]:
                    continue
                cost = sum(info[c]['region'].token_cost for c in rest)
                if cost > left:
                    continue
                g = gain(rest)
                key = (g / cost, g, -cost, rest[0])
                if best is None or key > best[0]:
                    best = (key, rest, 'bundle')
        if best is None or best[0][1] <= 0:
            break
        _key, cids, kind = best
        moves[kind] += 1
        for cid in cids:
            item = info[cid]
            selected.add(cid)
            order.append(item['region'])
            item['region'].decision = 'selected'
            touched.add(item['file'])
            covered[item['fn']] += item['region'].token_cost
            used += item['region'].token_cost
    for region in regions:
        if not region.decision:
            region.decision = 'not_selected'
    context, hits, kept = _pack(order, budget)
    for region in order:
        if region.chunk.id not in kept:
            region.decision = 'truncated_by_counter'
    return context, hits, {
        'packer': 'utility' if bundles else 'utility-nobundle', 'objective': 'expected_file_function_complete_line_recall',
        'weights': dict(w), 'theta': THETA, 'moves': moves, 'completion_chunks_outside_pool': extra,
        'tokens_planned': used, 'tokens_emitted': token_count(context),
        'stop_rule': 'budget_or_no_positive_gain', 'named_evidence_stop': False,
    }
