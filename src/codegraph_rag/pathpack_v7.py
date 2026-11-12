"""PathPack v0.7. v0.4 decides the file schedule. Each file then contributes
either one short whole function or its best AST slice.

No reference answer is read here. Packing uses the same prefix stream and
UTF-8/4 counter as v0.4, so 1024 is a prefix of 2048 and 4096.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, replace
from time import perf_counter
import math

from .adaptive_chunker import build_chunk_index, context_header
from .config import V4Config, V7Config
from .context_packer import COUNTER, ContextStream, token_count
from .hybrid_ranker import HybridIndex
from .pathpack_v4 import RetrievalResult
from .retrieval import Chunk, estimated_tokens


@dataclass
class V7View:
    chunk: object
    lexical: float
    kind: str
    score: float
    symbol: float = 0.0
    graph: float = 0.0
    reason: str = ''
    order: int | None = None

    def metadata(self):
        return {'id': self.chunk.id, 'path': self.chunk.path, 'function_id': self.chunk.function_id,
                'qualname': self.chunk.qualname, 'line_start': self.chunk.line_start, 'line_end': self.chunk.line_end,
                'tokens_est': self.chunk.tokens_est, 'lexical_score': self.lexical, 'symbol_score': self.symbol,
                'graph_score': self.graph, 'representation': self.kind,
                'score': self.score, 'order': self.order, 'decision': self.reason}


def _merged_function(chunks, cfg: V7Config):
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
    if tokens > cfg.whole_max_tokens:
        return None
    return Chunk(f'{sample.function_id}::whole', sample.path, sample.qualname, sample.kind, text, tokens,
                 sample.function_id, spans[0][0], spans[-1][0], spans, sample.signature, sample.parent_id, body)


def _best_slice(chunks, lexical, cfg: V7Config):
    def key(chunk):
        score = lexical.get(chunk.id, 0.0)
        utility = score / math.sqrt(chunk.tokens_est + 32) if cfg.enable_budget else score
        return (utility, score, -chunk.tokens_est, chunk.id)
    return max(chunks, key=key)


def _first_unit(file_chunks, lexical, cfg: V7Config, evidence):
    grouped = defaultdict(list)
    for chunk in file_chunks:
        grouped[chunk.function_id].append(chunk)
    best_id = max(grouped, key=lambda fid: (max(lexical.get(chunk.id, 0.0) for chunk in grouped[fid]), fid))
    slices = grouped[best_id]
    best_lexical = max(lexical.get(chunk.id, 0.0) for chunk in slices)
    if best_lexical <= 0:
        return None
    symbol = max(evidence.get(chunk.id, (0.0, 0.0))[0] for chunk in slices)
    graph = max(evidence.get(chunk.id, (0.0, 0.0))[1] for chunk in slices)
    if cfg.enable_adaptive:
        merged = _merged_function(slices, cfg)
        if merged is not None:
            return V7View(merged, best_lexical, 'whole', best_lexical, symbol, graph)
    chosen = _best_slice(slices, lexical, cfg)
    symbol, graph = evidence.get(chosen.id, (0.0, 0.0))
    return V7View(chosen, lexical.get(chosen.id, 0.0), 'slice', lexical.get(chosen.id, 0.0), symbol, graph)


def _rewrite(order, chunks, lexical, cfg: V7Config, evidence=None):
    evidence = evidence or {}
    if not cfg.enable_hierarchy:
        return [V7View(item.chunk, item.lexical, 'v4', item.score, item.symbol, item.graph) for item in order]
    by_file = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.path].append(chunk)
    rewritten = []
    seen_files = set()
    used = set()
    for item in order:
        path = item.chunk.path
        if path not in seen_files:
            seen_files.add(path)
            unit = _first_unit(by_file[path], lexical, cfg, evidence)
            if cfg.cap_to_opening and unit is not None and unit.chunk.tokens_est > item.chunk.tokens_est:
                unit = None
            if unit is not None and unit.chunk.id not in used:
                rewritten.append(unit)
                used.add(unit.chunk.id)
                if unit.kind == 'whole':
                    used.update(chunk.id for chunk in by_file[path] if chunk.function_id == unit.chunk.function_id)
                continue
        if item.chunk.id not in used:
            rewritten.append(V7View(item.chunk, item.lexical, 'v4', item.score, item.symbol, item.graph))
            used.add(item.chunk.id)
    for position, item in enumerate(rewritten):
        item.order = position
    return rewritten


class PreparedQueryV7:
    def __init__(self, index, query, config: V7Config, v4_config: V4Config):
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
        profile = dict(self.profile)
        diagnostics = {'instance_id': instance_id, 'method': 'pathpack-v7', 'config': self.config.to_dict(),
                       'budget_est': budget, 'tokens_est': used, 'remaining_tokens': budget - used,
                       'token_utilization': used / budget if budget else 0.0,
                       'selected_files': len({hit.chunk.path for hit in hits}),
                       'selected_functions': len({hit.chunk.function_id for hit in hits if hit.chunk.kind != 'module'}),
                       'selected_chunks': len(hits), 'token_counter': COUNTER, 'seeds': self.seeds,
                       'candidates': logged, 'profile': profile}
        return RetrievalResult(context, hits, diagnostics)


def retrieve_v7(repository, graph, query, *, budget=2048, config=None, v4_config=None):
    cfg = config or V7Config()
    base = v4_config or V4Config()
    index = HybridIndex(build_chunk_index(repository, graph, base), graph)
    return PreparedQueryV7(index, query, cfg, base).pack(budget)
