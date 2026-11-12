"""PathPack v0.8. v0.4 chooses the file openers and their order.

The tokens v0.4 spends after an opener, before the next new file, are the only
budget available for a better slice. A replacement cannot cost more than that
gap, so a later file stays inside the same prefix. No reference answer is read.
Packing uses the same prefix stream and UTF-8/4 counter as v0.4.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from .adaptive_chunker import build_chunk_index, context_header
from .config import V4Config, V8Config
from .context_packer import COUNTER, ContextStream, token_count
from .hybrid_ranker import HybridIndex
from .pathpack_v4 import RetrievalResult
from .retrieval import Chunk, estimated_tokens


@dataclass
class V8View:
    chunk: object
    lexical: float
    kind: str
    score: float
    symbol: float = 0.0
    graph: float = 0.0
    order: int | None = None

    def metadata(self):
        return {'id': self.chunk.id, 'path': self.chunk.path, 'function_id': self.chunk.function_id,
                'qualname': self.chunk.qualname, 'line_start': self.chunk.line_start, 'line_end': self.chunk.line_end,
                'tokens_est': self.chunk.tokens_est, 'lexical_score': self.lexical, 'symbol_score': self.symbol,
                'graph_score': self.graph, 'representation': self.kind,
                'score': self.score, 'order': self.order}


def _merged_function(chunks, limit):
    pairs = []
    seen = set()
    for chunk in sorted(chunks, key=lambda item: (item.line_start or 0, item.id)):
        offset = 0
        for line, start, end in chunk.source_spans:
            width = end - start
            piece = chunk.body[offset:offset + width]
            offset += width
            key = (line, start, end)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((line, start, end, piece))
    if not pairs or not ''.join(piece for *_, piece in pairs).strip():
        return None
    pairs.sort()
    body = ''.join(piece for *_, piece in pairs)
    spans = tuple((line, start, end) for line, start, end, _ in pairs)
    sample = chunks[0]
    header = context_header(sample.path, sample.qualname, spans[0][0], spans[-1][0], sample.signature)
    text = header + body + '\n\n'
    tokens = estimated_tokens(text)
    if tokens > limit:
        return None
    return Chunk(f'{sample.function_id}::whole', sample.path, sample.qualname, sample.kind, text, tokens,
                 sample.function_id, spans[0][0], spans[-1][0], spans, sample.signature, sample.parent_id, body)


def _view(item, kind):
    return V8View(item.chunk, item.lexical, kind, item.score, item.symbol, item.graph)


def _fill(file_chunks, opened, lexical, evidence, budget, used, cfg: V8Config):
    pool = [chunk for path in opened for chunk in file_chunks[path]]
    picked = []
    remaining = budget
    while remaining > 0:
        fresh = [chunk for chunk in pool
                 if chunk.id not in used and lexical.get(chunk.id, 0.0) > 0 and 0 < chunk.tokens_est <= remaining]
        if not fresh:
            break
        best = max(fresh, key=lambda chunk: (lexical[chunk.id], -chunk.tokens_est, chunk.id))
        siblings = [chunk for chunk in file_chunks[best.path] if chunk.function_id == best.function_id]
        emit = best
        kind = 'slice'
        if cfg.enable_adaptive:
            merged = _merged_function(siblings, min(cfg.whole_max_tokens, remaining))
            if merged is not None and merged.id not in used:
                emit = merged
                kind = 'whole'
        symbol, graph = evidence.get(best.id, (0.0, 0.0))
        if kind == 'whole':
            symbol = max(evidence.get(chunk.id, (0.0, 0.0))[0] for chunk in siblings)
            graph = max(evidence.get(chunk.id, (0.0, 0.0))[1] for chunk in siblings)
        picked.append(V8View(emit, lexical[best.id], kind, lexical[best.id], symbol, graph))
        used.add(emit.id)
        if kind == 'whole':
            used.update(chunk.id for chunk in siblings)
        remaining -= emit.tokens_est
    return picked


def _rewrite(order, chunks, lexical, cfg: V8Config, evidence=None):
    evidence = evidence or {}
    if not cfg.enable_deepen:
        return [_view(item, 'v4') for item in order]
    by_file = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.path].append(chunk)
    rewritten = []
    opened = []
    seen = set()
    used = set()
    pending = []
    items = list(order) + [None]
    for item in items:
        new_file = item is not None and item.chunk.path not in seen
        if item is None or new_file:
            if pending:
                opener = pending[0]
                tails = pending[1:]
                rewritten.append(_view(opener, 'anchor'))
                used.add(opener.chunk.id)
                budget = sum(tail.chunk.tokens_est for tail in tails)
                filled = _fill(by_file, opened, lexical, evidence, budget, used, cfg) if budget else []
                if filled:
                    rewritten.extend(filled)
                else:
                    for tail in tails:
                        if tail.chunk.id not in used:
                            rewritten.append(_view(tail, 'v4'))
                            used.add(tail.chunk.id)
                pending = []
            if item is None:
                break
        if new_file:
            seen.add(item.chunk.path)
            opened.append(item.chunk.path)
        pending.append(item)
    for position, view in enumerate(rewritten):
        view.order = position
    return rewritten


class PreparedQueryV8:
    def __init__(self, index, query, config: V8Config, v4_config: V4Config):
        self.config = config
        started = perf_counter()
        base = replace(v4_config, enable_graph=config.enable_graph, enable_diversity=config.enable_diversity)
        order, candidates, seeds = index.rank(query, base)
        lexical = {item.chunk.id: item.lexical for item in candidates}
        evidence = {item.chunk.id: (item.symbol, item.graph) for item in candidates}
        self.order = _rewrite(order, index.chunk_index.chunks, lexical, config, evidence)
        self.seeds = seeds
        self.stream = ContextStream(self.order, allow_partial=True)
        elapsed = perf_counter() - started
        self.profile = {'candidate_sec': elapsed, 'graph_sec': 0.0, 'file_rank_sec': 0.0, 'chunk_score_sec': 0.0,
                        'diversity_sec': 0.0, 'anchor_sec': 0.0, 'pack_stream_sec': 0.0, 'prefix_sec': 0.0,
                        'query_sec': elapsed, 'pool': len(self.order), 'chunks_indexed': len(index.chunk_index.chunks),
                        'seeds': seeds}

    def pack(self, budget, *, instance_id=None, include_candidates=True):
        context, hits, selected = self.stream.pack(budget)
        logged = []
        if include_candidates:
            for item in self.order:
                chosen = item.chunk.id in selected
                reason = 'selected_partial' if chosen and selected[item.chunk.id]['partial'] else ('selected' if chosen else 'after_budget_cutoff')
                logged.append({**item.metadata(), 'selected': chosen, 'decision': reason})
        used = token_count(context)
        diagnostics = {'instance_id': instance_id, 'method': 'pathpack-v8', 'config': self.config.to_dict(),
                       'budget_est': budget, 'tokens_est': used, 'remaining_tokens': budget - used,
                       'token_utilization': used / budget if budget else 0.0,
                       'selected_files': len({hit.chunk.path for hit in hits}),
                       'selected_functions': len({hit.chunk.function_id for hit in hits if hit.chunk.kind != 'module'}),
                       'selected_chunks': len(hits), 'token_counter': COUNTER, 'seeds': self.seeds,
                       'candidates': logged, 'profile': dict(self.profile)}
        return RetrievalResult(context, hits, diagnostics)


def retrieve_v8(repository, graph, query, *, budget=2048, config=None, v4_config=None):
    cfg = config or V8Config()
    base = v4_config or V4Config()
    index = HybridIndex(build_chunk_index(repository, graph, base), graph)
    return PreparedQueryV8(index, query, cfg, base).pack(budget)
