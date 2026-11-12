"""Union of lexical, symbol, semantic and structural candidates. Channels are not filters of each other."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..retrieval import BM25, Chunk
from .symbols import extract_symbols, matches_any


@dataclass
class Region:
    chunk: Chunk
    channels: list[str] = field(default_factory=list)
    ranks: dict = field(default_factory=dict)
    lexical: float = 0.0
    semantic: float | None = None
    symbol: float = 0.0
    graph: float = 0.0
    graph_edges: list = field(default_factory=list)
    repo: float = 0.0
    token_cost: int = 1
    rerank: float | None = None
    final: float = 0.0
    admission: str = 'retrieval'
    hypothesis: float = 0.0
    decision: str = ''

    def record(self):
        return {
            'id': self.chunk.id,
            'path': self.chunk.path,
            'function': self.chunk.qualname,
            'function_id': self.chunk.function_id,
            'line_start': self.chunk.line_start,
            'line_end': self.chunk.line_end,
            'channels': list(self.channels),
            'ranks': dict(self.ranks),
            'lexical_score': self.lexical,
            'semantic_score': self.semantic,
            'symbol_score': self.symbol,
            'graph_score': self.graph,
            'graph_edges': list(self.graph_edges),
            'repository_score': self.repo,
            'token_cost': self.token_cost,
            'rerank_score': self.rerank,
            'final_score': self.final,
            'admission': self.admission,
            'hypothesis': self.hypothesis,
            'decision': self.decision,
        }


def _normalize(raw):
    maximum = max(raw.values(), default=0.0) or 1.0
    return {key: value / maximum for key, value in raw.items()}


def _take(ordered, limit, reason):
    kept = ordered[:limit] if limit else []
    dropped = max(0, len(ordered) - len(kept))
    return kept, {'kept': len(kept), 'dropped': dropped, 'reason': reason if dropped else ''}


class RepoIndex:
    """Query-independent indexes. Semantic and reranker objects are attached by the caller."""

    def __init__(self, chunk_index, graph, repository):
        self.chunk_index = chunk_index
        self.graph = graph
        self.repository = repository
        self.chunks = list(chunk_index.chunks)
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.by_fn = defaultdict(list)
        for chunk in self.chunks:
            if chunk.function_id:
                self.by_fn[chunk.function_id].append(chunk)
        self.code_bm25 = BM25(self.chunks)
        names = [
            Chunk(chunk.id, chunk.path, chunk.qualname, chunk.kind,
                  f'{chunk.path}\n{chunk.qualname}\n{chunk.signature}', 1)
            for chunk in self.chunks
        ]
        self.name_bm25 = BM25(names)
        self.semantic = None
        self.reranker = None
        self.out = defaultdict(list)
        self.inn = defaultdict(list)
        for edge in sorted(graph.edges, key=lambda item: (item.source, item.kind, item.target)):
            self.out[edge.source].append((edge.target, edge.kind))
            self.inn[edge.target].append((edge.source, edge.kind))

    def relations(self, node_id):
        rows = [(other, kind, 'out') for other, kind in self.out.get(node_id, ())]
        rows += [(other, kind, 'in') for other, kind in self.inn.get(node_id, ())]
        rows.sort()
        return rows


def _admit(items, chunk, channel, rank, lexical, semantic, symbol, repo):
    current = items.get(chunk.id)
    if current is None:
        current = Region(
            chunk=chunk,
            lexical=lexical,
            semantic=None if semantic is None else semantic,
            symbol=1.0 if symbol else 0.0,
            repo=repo,
            token_cost=max(1, int(chunk.tokens_est or 1)),
        )
        items[chunk.id] = current
    if channel not in current.channels:
        current.channels.append(channel)
    current.ranks.setdefault(channel, rank)
    current.symbol = max(current.symbol, 1.0 if symbol else 0.0)
    current.repo = max(current.repo, repo)
    if semantic is not None:
        current.semantic = semantic if current.semantic is None else max(current.semantic, semantic)
    return current


def _chunks_for(index, function_id, lexical, limit):
    owned = list(index.by_fn.get(function_id, []))
    owned.sort(key=lambda chunk: (-lexical.get(chunk.id, 0.0), chunk.id))
    kept = owned[:limit] if limit else []
    return kept, max(0, len(owned) - len(kept))


def build_pool(index: RepoIndex, query: str, config, channels):
    """Return the union. Each channel keeps its own rank and logs its own drops."""
    queries = extract_symbols(query)
    code = _normalize(index.code_bm25.scores(query))
    name = _normalize(index.name_bm25.scores(query))
    lexical = {cid: max(code.get(cid, 0.0), name.get(cid, 0.0)) for cid in index.by_id}
    semantic_scores = None
    if 'semantic' in channels:
        if index.semantic is None or not index.semantic.available:
            semantic_scores = None
        else:
            from .semantic import encode_query
            vector = encode_query(index.semantic, config, query)
            semantic_scores = index.semantic.scores(vector)
    items = {}
    report = {}

    def symbol_hit(chunk):
        return matches_any(chunk.qualname, chunk.path, queries)

    def repo_hit(chunk):
        return 1.0 if any(query.kind == 'path' and matches_any(chunk.qualname, chunk.path, [query]) for query in queries) else 0.0

    if 'lexical' in channels:
        code_order = sorted(index.chunks, key=lambda chunk: (-code.get(chunk.id, 0.0), chunk.id))
        name_order = sorted(index.chunks, key=lambda chunk: (-name.get(chunk.id, 0.0), chunk.id))
        code_kept, code_log = _take([c for c in code_order if code.get(c.id, 0.0) > 0], config.lexical_limit, 'lexical_limit')
        name_kept, name_log = _take([c for c in name_order if name.get(c.id, 0.0) > 0], config.lexical_limit, 'name_limit')
        for rank, chunk in enumerate(code_kept):
            _admit(items, chunk, 'lexical', rank, lexical[chunk.id], None, symbol_hit(chunk), repo_hit(chunk))
        for rank, chunk in enumerate(name_kept):
            _admit(items, chunk, 'lexical_name', rank, lexical[chunk.id], None, symbol_hit(chunk), repo_hit(chunk))
        report['lexical'] = {'source': code_log, 'names': name_log}

    if 'symbol' in channels:
        hits = []
        for function in index.chunk_index.functions.values():
            if matches_any(function.qualname, function.path, queries):
                hits.append(function)
        hits.sort(key=lambda function: (-max((lexical.get(cid, 0.0) for cid in function.chunks), default=0.0), function.id))
        kept, log = _take(hits, config.symbol_limit, 'symbol_limit')
        dropped_chunks = 0
        for rank, function in enumerate(kept):
            owned, dropped = _chunks_for(index, function.id, lexical, config.max_chunks_per_function)
            dropped_chunks += dropped
            for chunk in owned:
                _admit(items, chunk, 'symbol', rank, lexical.get(chunk.id, 0.0), None, True, repo_hit(chunk))
        log['chunks_dropped_per_function_cap'] = dropped_chunks
        report['symbol'] = log

    if 'semantic' in channels:
        if semantic_scores is None:
            report['semantic'] = {
                'kept': 0, 'dropped': 0, 'status': 'not_run',
                'reason': (index.semantic.info.get('reason') if index.semantic is not None else 'semantic index was not built'),
            }
        else:
            ordered = sorted(index.chunks, key=lambda chunk: (-semantic_scores.get(chunk.id, 0.0), chunk.id))
            kept, log = _take(ordered, config.semantic_limit, 'semantic_limit')
            for rank, chunk in enumerate(kept):
                region = _admit(items, chunk, 'semantic', rank, lexical.get(chunk.id, 0.0), semantic_scores.get(chunk.id, 0.0), symbol_hit(chunk), repo_hit(chunk))
                region.semantic = semantic_scores.get(chunk.id, 0.0)
            log['status'] = 'available'
            report['semantic'] = log

    if 'structural' in channels:
        seeds = []
        for region in items.values():
            if 'symbol' in region.channels and region.chunk.function_id:
                seeds.append(region.chunk.function_id)
        lexical_seeds = sorted(
            (chunk for chunk in index.chunks if lexical.get(chunk.id, 0.0) > 0 and chunk.function_id),
            key=lambda chunk: (-lexical.get(chunk.id, 0.0), chunk.id),
        )
        for chunk in lexical_seeds:
            if chunk.function_id not in seeds:
                seeds.append(chunk.function_id)
            if len(seeds) >= config.graph_seeds + len({r.chunk.function_id for r in items.values() if 'symbol' in r.channels}):
                break
        seeds = list(dict.fromkeys(seeds))[:config.graph_seeds + config.symbol_limit]
        neighbors = []
        seen = set(seeds)
        frontier = list(seeds)
        unresolved = 0
        for hop in range(1, max(0, config.graph_hops) + 1):
            nxt = []
            for fid in frontier:
                for other, kind, direction in index.relations(fid):
                    if other in seen:
                        continue
                    if other not in index.by_fn:
                        unresolved += 1
                        continue
                    seen.add(other)
                    nxt.append((hop, other, kind, direction, fid))
            nxt.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
            neighbors.extend(nxt)
            frontier = [row[1] for row in nxt]
        kept, log = _take(neighbors, config.graph_limit, 'graph_limit')
        dropped_chunks = 0
        for rank, (hop, fid, kind, direction, source) in enumerate(kept):
            owned, dropped = _chunks_for(index, fid, lexical, config.max_chunks_per_function)
            dropped_chunks += dropped
            for chunk in owned:
                region = _admit(items, chunk, 'structural', rank, lexical.get(chunk.id, 0.0), None, symbol_hit(chunk), repo_hit(chunk))
                region.graph = max(region.graph, 1.0 / hop)
                edge = {'source': source, 'target': fid, 'kind': kind, 'direction': direction, 'hop': hop}
                if edge not in region.graph_edges:
                    region.graph_edges.append(edge)
        log['chunks_dropped_per_function_cap'] = dropped_chunks
        log['unresolved_or_non_production'] = unresolved
        log['edge_kinds_stored'] = ['calls', 'imports', 'inherits', 'defines']
        report['structural'] = log

    if semantic_scores is not None:
        for region in items.values():
            if region.semantic is None:
                region.semantic = semantic_scores.get(region.chunk.id, 0.0)
    return list(items.values()), report, queries, lexical
