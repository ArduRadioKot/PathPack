"""PathPack v0.6. Whole functions locate the area; AST slices fill the budget.

No reference answer, file list, function list or line list is read here. The packed stream is the
same prefix packer and UTF-8/4 counter as v0.4, so budgets stay comparable.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from time import perf_counter

from .adaptive_chunker import build_chunk_index
from .config import V6Config
from .context_packer import COUNTER, ContextStream, token_count
from .pathpack_v4 import RetrievalResult
from .retrieval import BM25, terms


@dataclass
class V6Candidate:
    chunk: object
    lexical: float
    symbol: float
    parent: float
    graph: float
    route: list
    kinds: list
    evidence: float
    redundancy: float = 0.0
    repeat: float = 0.0
    score: float = 0.0
    utility: float = 0.0
    reason: str = ''
    order: int | None = None
    term_set: frozenset = frozenset()

    def metadata(self):
        return {'id': self.chunk.id, 'path': self.chunk.path, 'function_id': self.chunk.function_id,
                'qualname': self.chunk.qualname, 'line_start': self.chunk.line_start, 'line_end': self.chunk.line_end,
                'tokens_est': self.chunk.tokens_est, 'lexical_score': self.lexical, 'symbol_score': self.symbol,
                'parent_score': self.parent, 'graph_score': self.graph, 'graph_path': self.route, 'graph_kinds': self.kinds,
                'evidence': self.evidence, 'redundancy': self.redundancy, 'same_function_penalty': self.repeat,
                'score': self.score, 'utility': self.utility, 'order': self.order, 'decision': self.reason}


def _graph_paths(graph, seeds, cfg: V6Config):
    if not cfg.enable_graph or not seeds:
        return {}
    adjacency = defaultdict(list)
    for edge in sorted(graph.edges, key=lambda item: (item.source, item.kind, item.target)):
        if cfg.graph_direction in {'both', 'out'}:
            adjacency[edge.source].append((edge.target, edge.kind))
        if cfg.graph_direction in {'both', 'in'}:
            adjacency[edge.target].append((edge.source, edge.kind))
    paths = {}
    for seed in seeds:
        queue = deque([(seed, 1.0, [seed], [])])
        best = {(seed, 0): 1.0}
        while queue:
            node, weight, route, kinds = queue.popleft()
            previous = paths.get(node)
            if previous is None or (-weight, len(route), route) < (-previous[0], len(previous[1]), previous[1]):
                paths[node] = (weight, route, kinds)
            if len(kinds) >= cfg.hops:
                continue
            for target, kind in adjacency.get(node, []):
                gained = weight * cfg.edge_weights.get(kind, 0) * cfg.distance_decay
                state = (target, len(kinds) + 1)
                if target in route or gained <= best.get(state, 0):
                    continue
                best[state] = gained
                queue.append((target, gained, route + [target], kinds + [kind]))
    for seed in seeds:
        if seed in paths:
            weight, route, kinds = paths[seed]
            paths[seed] = (0.0, route, kinds)
    return paths


def _overlap_signature(body):
    """Bounded term sample so diversity stays deterministic on large functions."""
    words = terms(body)
    if len(words) <= 80:
        return frozenset(words)
    step = len(words) / 80
    return frozenset(words[int(i * step)] for i in range(80))


def _covered_fraction(terms, seen):
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in seen) / len(terms)


class PreparedQueryV6:
    def __init__(self, fine_index, coarse_index, graph, query, config: V6Config):
        self.config = config
        started = perf_counter()
        fine_raw = BM25(fine_index.chunks).scores(query) if fine_index.chunks else {}
        coarse_raw = BM25(coarse_index.chunks).scores(query) if coarse_index.chunks else {}
        candidate_sec = perf_counter() - started
        started = perf_counter()
        parent_raw = defaultdict(float)
        for chunk in coarse_index.chunks:
            parent_raw[chunk.function_id] = max(parent_raw[chunk.function_id], coarse_raw.get(chunk.id, 0.0))
        seeds = [item for item in sorted(parent_raw, key=lambda item: (-parent_raw[item], item))[:config.seeds] if parent_raw[item] > 0]
        paths = _graph_paths(graph, seeds, config)
        graph_sec = perf_counter() - started
        started = perf_counter()
        fine_max = max(fine_raw.values(), default=0.0) or 1.0
        parent_max = max(parent_raw.values(), default=0.0) or 1.0
        query_terms = set(terms(query))
        candidates = []
        for chunk in fine_index.chunks:
            lexical = fine_raw.get(chunk.id, 0.0) / fine_max
            names = terms(chunk.path + ' ' + chunk.qualname)
            symbol = (len(query_terms & set(names)) / max(1, len(query_terms))) if config.enable_symbol and query_terms else 0.0
            parent = (parent_raw.get(chunk.function_id, 0.0) / parent_max) if config.enable_parent else 0.0
            confidence, route, kinds = paths.get(chunk.function_id, (0.0, [], []))
            evidence = (config.lexical_weight * lexical + config.symbol_weight * symbol
                        + config.parent_weight * parent + config.graph_weight * confidence)
            item = V6Candidate(chunk, lexical, symbol, parent, confidence, route, kinds, evidence,
                               term_set=tuple(_overlap_signature(chunk.body)))
            if evidence <= config.min_relevance:
                item.reason = 'low_relevance'
            candidates.append(item)
        chunk_score_sec = perf_counter() - started
        started = perf_counter()
        valid = sorted((item for item in candidates if not item.reason), key=lambda item: (-item.evidence, item.chunk.id))
        for item in valid[config.candidate_limit:]:
            item.reason = 'candidate_limit'
        pool = valid[:config.candidate_limit]
        order = []
        taken = defaultdict(int)
        seen_terms = set()
        while pool:
            best = None
            best_key = None
            for item in pool:
                redundancy = _covered_fraction(item.term_set, seen_terms) if config.enable_diversity else 0.0
                repeat = 0.0
                if config.enable_diversity and taken[item.chunk.function_id]:
                    repeat = (taken[item.chunk.function_id] / (taken[item.chunk.function_id] + 1)) * (1 - item.lexical) * config.same_function_penalty
                score = item.evidence * (1 - config.redundancy_strength * redundancy) * (1 - repeat)
                utility = score / ((item.chunk.tokens_est + config.cost_offset) ** config.cost_exponent) if config.enable_cost else score
                key = (-utility, -score, item.chunk.id)
                if best_key is None or key < best_key:
                    best, best_key = item, key
                    item.redundancy, item.repeat, item.score, item.utility = redundancy, repeat, score, utility
            order.append(best)
            pool.remove(best)
            taken[best.chunk.function_id] += 1
            seen_terms.update(best.term_set)
        for position, item in enumerate(order):
            item.order = position
        diversity_sec = perf_counter() - started
        started = perf_counter()
        self.order = order
        self.candidates = candidates
        self.seeds = seeds
        self.stream = ContextStream(order, allow_partial=config.enable_chunking)
        pack_stream_sec = perf_counter() - started
        self.profile = {'candidate_sec': candidate_sec, 'graph_sec': graph_sec, 'file_rank_sec': 0.0,
                        'chunk_score_sec': chunk_score_sec, 'diversity_sec': diversity_sec, 'anchor_sec': 0.0,
                        'pack_stream_sec': pack_stream_sec, 'prefix_sec': 0.0,
                        'query_sec': candidate_sec + graph_sec + chunk_score_sec + diversity_sec + pack_stream_sec,
                        'pool': len(order), 'chunks_indexed': len(fine_index.chunks), 'seeds': seeds}

    def pack(self, budget, *, instance_id=None, include_candidates=True):
        started = perf_counter()
        context, hits, selected = self.stream.pack(budget)
        prefix_sec = perf_counter() - started
        logged = []
        if include_candidates:
            first_blocked = next((cid for cid, info in selected.items() if info['partial']), None)
            if first_blocked is None:
                first_blocked = next((item.chunk.id for item in self.order if item.chunk.id not in selected), None)
            for item in self.candidates:
                if item.order is None and item.reason not in {'selected', 'selected_partial'}:
                    continue
                cid = item.chunk.id
                reason = item.reason
                if cid in selected:
                    reason = 'selected_partial' if selected[cid]['partial'] else 'selected'
                elif not reason:
                    reason = 'budget_exceeded' if cid == first_blocked else 'after_budget_cutoff'
                logged.append({**item.metadata(), 'selected': cid in selected, 'decision': reason})
        profile = dict(self.profile)
        profile['prefix_sec'] = prefix_sec
        profile['query_sec'] = self.profile['query_sec'] + prefix_sec
        diagnostics = {'instance_id': instance_id, 'method': 'pathpack-v6', 'config': self.config.to_dict(),
                       'budget_est': budget, 'tokens_est': token_count(context), 'remaining_tokens': budget - token_count(context),
                       'token_utilization': token_count(context) / budget if budget else 0.0,
                       'selected_files': len({hit.chunk.path for hit in hits}),
                       'selected_functions': len({hit.chunk.function_id for hit in hits if hit.chunk.kind != 'module'}),
                       'selected_chunks': len(hits), 'token_counter': COUNTER, 'seeds': self.seeds,
                       'candidates': logged, 'profile': profile,
                       'reason_counts': _reason_counts(self.candidates, selected)}
        return RetrievalResult(context, hits, diagnostics)


def _reason_counts(candidates, selected):
    counts = defaultdict(int)
    for item in candidates:
        if item.chunk.id in selected:
            counts['selected'] += 1
        else:
            counts[item.reason or 'after_budget_cutoff'] += 1
    return dict(counts)


def make_v6_chunks(repository, graph, *, target_tokens=128, max_chunk_tokens=256, config=None):
    """Fine slices through the current AST chunker. Returns the chunk list."""
    cfg = config or V6Config(target_chunk_tokens=target_tokens, max_chunk_tokens=max_chunk_tokens)
    return build_chunk_index(repository, graph, cfg.to_v4()).chunks


def retrieve_v6(repository, graph, query, *, budget=2048, config=None):
    cfg = config or V6Config()
    fine = build_chunk_index(repository, graph, cfg.to_v4())
    coarse_cfg = cfg.to_v4()
    if cfg.enable_chunking:
        from dataclasses import replace
        coarse = build_chunk_index(repository, graph, replace(coarse_cfg, enable_chunking=False))
    else:
        coarse = fine
    return PreparedQueryV6(fine, coarse, graph, query, cfg).pack(budget)
