"""PathPack v1.0. Evidence items, a union of retrievers, then a joint set.

No reference answer is read. A repeated evidence does not by itself spend
more tokens. A call edge is used only when the static graph resolved it.
Embeddings are omitted, not replaced, when the pinned model is absent.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .adaptive_chunker import build_chunk_index, production_path
from .config import V10Config
from .context_packer import COUNTER, ContextStream, token_count
from .embedding_retriever import probe_embedding
from .evidence import extract_evidence
from .hybrid_ranker import Candidate
from .pathpack_v4 import RetrievalResult
from .retrieval import Chunk, identifier_terms, terms


@dataclass
class PoolItem:
    candidate: Candidate
    sources: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    ranks: dict = field(default_factory=dict)

    @property
    def chunk(self):
        return self.candidate.chunk


def _best(chunks, lexical):
    return max(chunks, key=lambda chunk: (lexical.get(chunk.id, 0.0), -chunk.tokens_est, chunk.id))


def _compact(text):
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def _defines_symbol(qualname, evidence):
    """Whole class or function component. A shared CamelCase part is not a definition."""
    if evidence.type not in {'symbol', 'test'}:
        return False
    leaf = _compact(evidence.query_span.split('.')[-1])
    if len(leaf) < 4:
        return False
    return any(_compact(part) == leaf for part in qualname.split('.') if part)


def _path_match(chunk_path, span):
    left = chunk_path.lower().replace('\\', '/')
    right = span.lower().replace('\\', '/')
    return bool(left and right and (left.endswith(right) or right.endswith(left)))


def _covers(names, path, body, evidences, qualname):
    """A symbol counts as covered by its definition, not by a caller that mentions it."""
    hits = []
    for evidence in evidences:
        if evidence.type == 'path' and _path_match(path, evidence.query_span):
            hits.append(evidence.evidence_id)
        elif _defines_symbol(qualname, evidence):
            hits.append(evidence.evidence_id)
        elif evidence.type == 'exception' and evidence.terms & (names | body):
            hits.append(evidence.evidence_id)
        elif evidence.type == 'behavior' and evidence.terms & body:
            hits.append(evidence.evidence_id)
    return hits


def _channel(accepted, limit, reason):
    ordered = list(accepted)
    kept = ordered[:limit]
    return kept, max(0, len(ordered) - limit), reason if len(ordered) > limit else ''


class PreparedQueryV10:
    def __init__(self, index, query, config: V10Config, repository=None):
        self.config = config
        self.index = index
        self.repository = repository
        self.query = query
        self.fallback_used = False
        self.fallback_reason = ''
        started = perf_counter()
        self.evidences = extract_evidence(query)
        evidence_sec = perf_counter() - started
        t = perf_counter()
        raw = index.bm25.scores(query)
        maximum = max(raw.values(), default=0.0) or 1.0
        self.lexical = {cid: score / maximum for cid, score in raw.items()}
        self.items, self.channels = self._pool()
        self._link()
        candidate_sec = perf_counter() - t
        self.calls = self._call_index()
        self._nested_order = None
        if config.nested:
            self._nested_order = self._select(10**9)
        prior = self.profile if isinstance(getattr(self, 'profile', None), dict) else {}
        self.profile = {
            'candidate_sec': candidate_sec, 'graph_sec': 0.0, 'file_rank_sec': 0.0,
            'chunk_score_sec': 0.0, 'diversity_sec': 0.0, 'anchor_sec': 0.0,
            'pack_stream_sec': 0.0, 'prefix_sec': 0.0, 'query_sec': evidence_sec + candidate_sec,
            'evidence_sec': evidence_sec, 'selection_sec': prior.get('selection_sec', 0.0),
            'pool': len(self.items),
            'chunks_indexed': len(index.chunk_index.chunks), 'fallback_used': self.fallback_used,
            'embedding_status': self.channels.get('embedding', {}).get('status', 'not_run'),
        }

    def _call_index(self):
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for edge in self.index.graph.edges:
            if edge.kind != 'calls':
                continue
            outgoing[edge.source].append(edge.target)
            incoming[edge.target].append(edge.source)
        return {'out': outgoing, 'in': incoming}

    def _pool(self):
        chunks = list(self.index.chunk_index.chunks)
        by_id = {chunk.id: chunk for chunk in chunks}
        by_fn = defaultdict(list)
        for chunk in chunks:
            if chunk.function_id:
                by_fn[chunk.function_id].append(chunk)
        channels = {}
        items: dict[str, PoolItem] = {}

        def admit(chunk, source, rank):
            if chunk is None or chunk.id not in by_id and source != 'test':
                if source != 'test':
                    return
            current = items.get(chunk.id)
            if current is None:
                lexical = self.lexical.get(chunk.id, 0.0)
                items[chunk.id] = PoolItem(Candidate(chunk, lexical, lexical, 0.0, 0.0, [], []))
                current = items[chunk.id]
            if source not in current.sources:
                current.sources.append(source)
            current.ranks.setdefault(source, rank)

        ranked = sorted(chunks, key=lambda chunk: (-self.lexical.get(chunk.id, 0.0), chunk.id))
        bm25, dropped, why = _channel([c for c in ranked if self.lexical.get(c.id, 0.0) > 0], self.config.bm25_limit, 'bm25_limit')
        for rank, chunk in enumerate(bm25):
            admit(chunk, 'bm25', rank)
        channels['bm25'] = {'kept': len(bm25), 'dropped': dropped, 'reason': why}

        symbol_hits = []
        for function in self.index.chunk_index.functions.values():
            if any(_defines_symbol(function.qualname, evidence) or (
                    evidence.type == 'exception' and evidence.terms & identifier_terms(function.qualname))
                   for evidence in self.evidences):
                symbol_hits.append(function)
        symbol_hits.sort(key=lambda function: (-max((self.lexical.get(cid, 0.0) for cid in function.chunks), default=0.0), function.id))
        kept_fn, dropped, why = _channel(symbol_hits, self.config.symbol_limit, 'symbol_limit')
        for rank, function in enumerate(kept_fn):
            owned = [by_id[cid] for cid in function.chunks if cid in by_id]
            if owned:
                admit(_best(owned, self.lexical), 'symbol', rank)
        channels['symbol'] = {'kept': len(kept_fn), 'dropped': dropped, 'reason': why}

        path_hits = []
        path_terms = [evidence for evidence in self.evidences if evidence.type == 'path']
        if path_terms:
            for chunk in ranked:
                if any(_path_match(chunk.path, evidence.query_span) for evidence in path_terms):
                    path_hits.append(chunk)
        kept_paths, dropped, why = _channel(path_hits, self.config.path_limit, 'path_limit')
        for rank, chunk in enumerate(kept_paths):
            admit(chunk, 'path', rank)
        channels['path'] = {'kept': len(kept_paths), 'dropped': dropped, 'reason': why}

        if self.config.enable_semantic:
            embedding = probe_embedding()
            channels['embedding'] = {'kept': 0, 'dropped': 0, 'reason': embedding.get('reason', ''), 'status': embedding['status']}
        else:
            channels['embedding'] = {'kept': 0, 'dropped': 0, 'reason': 'semantic channel disabled', 'status': 'not_run'}

        seeds = [function.id for function in kept_fn]
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for edge in self.index.graph.edges:
            if edge.kind == 'calls':
                outgoing[edge.source].append(edge.target)
                incoming[edge.target].append(edge.source)
        neighbors = []
        seen = set(seeds)
        frontier = list(seeds)
        for _hop in range(max(0, self.config.graph_hops)):
            nxt = []
            for fid in frontier:
                for other in sorted(set(outgoing.get(fid, [])) | set(incoming.get(fid, []))):
                    if other not in seen and other in by_fn:
                        seen.add(other)
                        nxt.append(other)
            nxt.sort()
            neighbors.extend(nxt)
            frontier = nxt
        kept_nb, dropped, why = _channel(neighbors, self.config.graph_limit, 'graph_limit')
        for rank, fid in enumerate(kept_nb):
            owned = list(by_fn.get(fid, []))
            if owned:
                admit(_best(owned, self.lexical), 'graph', rank)
        channels['graph'] = {'kept': len(kept_nb), 'dropped': dropped, 'reason': why, 'unresolved_calls_omitted': True}

        test_kept = self._tests(admit)
        channels['test'] = test_kept
        return list(items.values()), channels

    def _tests(self, admit):
        wanted = {evidence.query_span for evidence in self.evidences if evidence.type == 'test'}
        if not wanted or not self.repository:
            return {'kept': 0, 'dropped': 0, 'reason': '' if wanted else 'no_test_evidence'}
        matches = []
        for node in self.index.graph.nodes.values():
            if production_path(node.path):
                continue
            leaf = node.qualname.split('.')[-1]
            if leaf in wanted or node.qualname in wanted:
                matches.append(node)
        matches.sort(key=lambda node: node.id)
        kept = matches[:self.config.test_limit]
        made = 0
        root = Path(self.repository)
        for rank, node in enumerate(kept):
            file = root / node.path
            try:
                lines = file.read_text(encoding='utf-8-sig').splitlines(keepends=True)
            except (OSError, UnicodeError):
                continue
            spans = []
            pieces = []
            for line_no in range(node.line_start, min(node.line_end, len(lines)) + 1):
                raw = lines[line_no - 1]
                if not raw.strip():
                    continue
                spans.append((line_no, 0, len(raw)))
                pieces.append(raw)
            if not spans:
                continue
            body = ''.join(pieces)
            chunk = Chunk(
                f'test::{node.id}', node.path, node.qualname, node.kind, body, max(1, token_count(body)),
                function_id=node.id, line_start=spans[0][0], line_end=spans[-1][0],
                source_spans=tuple(spans), body=body,
            )
            lexical = self.lexical.get(chunk.id, 0.0)
            self.lexical[chunk.id] = lexical
            item = PoolItem(Candidate(chunk, lexical, lexical, 0.0, 0.0, [], []), ['test'], [], {'test': rank})
            # admit() rejects unknown ids, so store the test item directly
            admit_direct = item
            # use the public dict via a side channel: caller passes admit which checks by_id.
            # Tests are outside the production index, so attach them after _pool returns is harder.
            # Store on the function attribute and merge in _pool.
            self._pending_tests = getattr(self, '_pending_tests', [])
            self._pending_tests.append(admit_direct)
            made += 1
        return {'kept': made, 'dropped': max(0, len(matches) - self.config.test_limit), 'reason': 'test_limit' if len(matches) > self.config.test_limit else ''}

    def _link(self):
        pending = getattr(self, '_pending_tests', [])
        if pending:
            known = {item.chunk.id for item in self.items}
            for item in pending:
                if item.chunk.id not in known:
                    self.items.append(item)
        for item in self.items:
            names = identifier_terms(item.chunk.qualname) | identifier_terms(Path(item.chunk.path).stem)
            body = self.index.term_sets.get(item.chunk.id, frozenset(terms(item.chunk.body or '')))
            item.evidence_ids = _covers(names, item.chunk.path, body, self.evidences, item.chunk.qualname)
            for evidence in self.evidences:
                if evidence.evidence_id in item.evidence_ids and item.chunk.function_id:
                    if item.chunk.function_id not in evidence.matched_code_nodes:
                        evidence.matched_code_nodes.append(item.chunk.function_id)

    def _gain(self, item, covered, chosen_files, chosen_ids):
        evidence_gain = 0.0
        new_evidence = False
        for evidence_id in item.evidence_ids:
            evidence = self._by_evidence[evidence_id]
            if covered[evidence_id] == 0:
                evidence_gain += evidence.confidence
                new_evidence = True
            else:
                evidence_gain += self.config.repeat_weight * evidence.confidence
        structural = 0.0
        if self.config.enable_structural and self.config.mode in {'paths', 'joint'}:
            linked = self._linked(item.chunk.function_id, chosen_ids)
            neighbor_ok = self.lexical.get(item.chunk.id, 0.0) >= self.config.min_neighbor_lexical or new_evidence
            if linked and neighbor_ok:
                structural = 1.0
        component = 1.0 if new_evidence and item.chunk.path not in chosen_files else 0.0
        redundancy = self._overlap(item, chosen_ids)
        utility = (self.config.alpha * evidence_gain + self.config.beta * structural
                   + self.config.gamma * component - self.config.delta * redundancy)
        return utility, new_evidence

    def _linked(self, function_id, chosen_ids):
        if not function_id:
            return False
        neighbors = set(self.calls['out'].get(function_id, [])) | set(self.calls['in'].get(function_id, []))
        return bool(neighbors & chosen_ids)

    def _overlap(self, item, chosen_ids):
        lines = {(item.chunk.path, line) for line, _, _ in item.chunk.source_spans}
        if not lines:
            return 1.0
        shared = 0
        for other in self.items:
            if other.chunk.id not in chosen_ids:
                continue
            shared += sum((other.chunk.path, line) in lines for line, _, _ in other.chunk.source_spans)
        return min(1.0, shared / len(lines))

    def _select(self, budget):
        started = perf_counter()
        self._by_evidence = {evidence.evidence_id: evidence for evidence in self.evidences}
        if self.config.mode == 'paths':
            chosen = self._select_paths(budget)
        elif self.config.mode == 'joint' and self.config.beam_width > 1 and self.config.enable_structural:
            chosen = self._select_beam(budget)
        else:
            chosen = self._select_greedy(budget)
        profile = getattr(self, 'profile', None)
        if profile is None:
            self.profile = profile = {}
        profile['selection_sec'] = profile.get('selection_sec', 0.0) + perf_counter() - started
        return chosen

    def _select_greedy(self, budget):
        remaining = list(self.items)
        chosen = []
        covered = Counter()
        files = set()
        ids = set()
        used = 0
        while remaining:
            best = None
            best_key = None
            for item in remaining:
                cost = item.chunk.tokens_est + 16
                if used + cost > budget:
                    continue
                utility, _new = self._gain(item, covered, files, ids)
                if utility <= 0:
                    continue
                key = (utility / cost ** 0.5, utility, -cost, item.chunk.id)
                if best_key is None or key > best_key:
                    best_key = key
                    best = item
            if best is None:
                break
            chosen.append(best)
            remaining = [item for item in remaining if item.chunk.id != best.chunk.id]
            used += best.chunk.tokens_est + 16
            files.add(best.chunk.path)
            ids.add(best.chunk.function_id)
            for evidence_id in best.evidence_ids:
                covered[evidence_id] += 1
        return chosen

    def _select_paths(self, budget):
        seeds = [item for item in self.items if 'symbol' in item.sources and item.evidence_ids]
        seeds.sort(key=lambda item: (-len(item.evidence_ids), item.chunk.id))
        groups = []
        for seed in seeds[:self.config.symbol_limit]:
            fid = seed.chunk.function_id
            neighbors = sorted(set(self.calls['out'].get(fid, [])) | set(self.calls['in'].get(fid, [])))
            eligible = []
            for other in neighbors:
                match = next((item for item in self.items if item.chunk.function_id == other), None)
                if match is None or match.chunk.id == seed.chunk.id:
                    continue
                lexical = self.lexical.get(match.chunk.id, 0.0)
                if lexical >= self.config.min_neighbor_lexical or len(neighbors) == 1:
                    eligible.append(match)
            eligible.sort(key=lambda item: (-self.lexical.get(item.chunk.id, 0.0), item.chunk.id))
            group = [seed]
            if eligible:
                group.append(eligible[0])
            groups.append(group)
        return self._take_groups(groups, budget)

    def _take_groups(self, groups, budget):
        chosen = []
        covered = Counter()
        files = set()
        ids = set()
        used = 0
        unused = list(groups)
        while unused:
            best = None
            best_key = None
            for group in unused:
                fresh = [item for item in group if item.chunk.id not in {c.chunk.id for c in chosen}]
                cost = sum(item.chunk.tokens_est + 16 for item in fresh)
                if not fresh or used + cost > budget:
                    continue
                utility = 0.0
                trial_covered = Counter(covered)
                trial_files = set(files)
                trial_ids = set(ids)
                for item in fresh:
                    gain, _new = self._gain(item, trial_covered, trial_files, trial_ids)
                    utility += gain
                    trial_files.add(item.chunk.path)
                    trial_ids.add(item.chunk.function_id)
                    for evidence_id in item.evidence_ids:
                        trial_covered[evidence_id] += 1
                if utility <= 0:
                    continue
                key = (utility / cost ** 0.5, utility, -cost, fresh[0].chunk.id)
                if best_key is None or key > best_key:
                    best_key = key
                    best = fresh
            if best is None:
                break
            for item in best:
                chosen.append(item)
                used += item.chunk.tokens_est + 16
                files.add(item.chunk.path)
                ids.add(item.chunk.function_id)
                for evidence_id in item.evidence_ids:
                    covered[evidence_id] += 1
            unused = [group for group in unused if any(item.chunk.id not in {c.chunk.id for c in chosen} for item in group)]
        return chosen

    def _select_beam(self, budget):
        # Beam over the greedy shortlist. The shortlist is the positive-gain order, so the beam cannot wander into unrelated calls.
        shortlist = self._select_greedy(budget)
        if len(shortlist) <= 2 or self.config.beam_width <= 1:
            return shortlist
        Beam = tuple  # chosen tuple, utility, used
        start = Beam(((), 0.0, 0))
        beam = [start]
        steps = min(len(shortlist), 12)
        for _step in range(steps):
            expanded = []
            for chosen, utility, used in beam:
                chosen_ids = {item.chunk.id for item in chosen}
                files = {item.chunk.path for item in chosen}
                fn_ids = {item.chunk.function_id for item in chosen}
                covered = Counter()
                for item in chosen:
                    for evidence_id in item.evidence_ids:
                        covered[evidence_id] += 1
                options = []
                for item in shortlist:
                    if item.chunk.id in chosen_ids:
                        continue
                    cost = item.chunk.tokens_est + 16
                    if used + cost > budget:
                        continue
                    gain, _new = self._gain(item, covered, files, fn_ids)
                    if gain <= 0:
                        continue
                    options.append((gain, item, cost))
                options.sort(key=lambda row: (-row[0], row[1].chunk.id))
                if not options:
                    expanded.append((chosen, utility, used))
                    continue
                for gain, item, cost in options[:self.config.beam_expand]:
                    expanded.append((chosen + (item,), utility + gain, used + cost))
            expanded.sort(key=lambda row: (-row[1], tuple(item.chunk.id for item in row[0])))
            dedup = []
            seen = set()
            for row in expanded:
                key = tuple(item.chunk.id for item in row[0])
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(row)
                if len(dedup) >= self.config.beam_width:
                    break
            if not dedup:
                break
            beam = dedup
        best = max(beam, key=lambda row: (row[1], tuple(item.chunk.id for item in row[0])))
        return list(best[0]) if best[0] else shortlist

    def pack(self, budget, *, instance_id=None, include_candidates=True):
        started = perf_counter()
        try:
            order = self._nested_order if self.config.nested and self._nested_order is not None else self._select(budget)
            for item in self.items:
                item.candidate.order = None
            for index, item in enumerate(order):
                item.candidate.order = index
            stream = ContextStream([item.candidate for item in order], allow_partial=True)
            context, hits, _selected = stream.pack(budget)
        except Exception as exc:
            self.fallback_used = True
            self.fallback_reason = type(exc).__name__
            from .config import V4Config
            from .pathpack_v4 import PreparedQuery
            prepared = PreparedQuery(self.index, self.query, V4Config())
            context, hits, _selected = prepared.stream.pack(budget)
            order = []
        selected_ids = {hit.chunk.id for hit in hits}
        elapsed = perf_counter() - started
        profile = dict(self.profile)
        profile['query_sec'] = profile['query_sec'] + elapsed
        profile['prefix_sec'] = elapsed
        profile['fallback_used'] = self.fallback_used
        logged = []
        if include_candidates:
            for item in self.items:
                logged.append({
                    **item.candidate.metadata(),
                    'sources': list(item.sources),
                    'evidence_ids': list(item.evidence_ids),
                    'ranks': dict(item.ranks),
                    'selected': item.chunk.id in selected_ids,
                })
        used = token_count(context)
        diagnostics = {
            'instance_id': instance_id, 'method': 'pathpack-v10', 'config': self.config.to_dict(),
            'budget_est': budget, 'tokens_est': used, 'remaining_tokens': budget - used,
            'token_utilization': used / budget if budget else 0.0,
            'selected_files': len({hit.chunk.path for hit in hits}),
            'selected_functions': len({hit.chunk.function_id for hit in hits if hit.chunk.kind != 'module'}),
            'selected_chunks': len(hits), 'token_counter': COUNTER,
            'candidates': logged, 'profile': profile, 'channels': self.channels,
            'evidence': [evidence.to_dict() for evidence in self.evidences],
            'fallback_used': self.fallback_used, 'fallback_reason': self.fallback_reason,
        }
        return RetrievalResult(context, hits, diagnostics)


def retrieve_v10(repository, graph, query, *, budget=2048, config=None, v4_config=None):
    from .hybrid_ranker import HybridIndex
    cfg = config or V10Config()
    base = v4_config
    index = HybridIndex(build_chunk_index(repository, graph, base), graph)
    return PreparedQueryV10(index, query, cfg, repository).pack(budget)
