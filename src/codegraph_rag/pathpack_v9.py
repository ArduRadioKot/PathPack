"""PathPack v0.9. Same evidence as v0.4, plus optional rank adjustments.

A file may contribute at most per_file_pool chunks to the ranked pool.
While the diverse order is built, a chunk that adds unseen lines of its
function, or that crosses 20% or 100% of that function's span, receives
coverage_bonus. name_prior adds evidence when a rare identifier from the
qualified name also occurs in the query. No reference answer is read.
Packing is the same prefix stream and UTF-8/4 counter as v0.4.
"""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter

from collections import defaultdict

from .adaptive_chunker import build_chunk_index
from .config import V4Config, V9Config
from .context_packer import COUNTER, ContextStream, token_count
from .hybrid_ranker import HybridIndex
from .pathpack_v4 import RetrievalResult
from .retrieval import identifier_terms


def _overlaps(left, right) -> bool:
    for line, start, end in left.chunk.source_spans:
        for other, a, b in right.chunk.source_spans:
            if left.chunk.path == right.chunk.path and line == other and max(start, a) < min(end, b):
                return True
    return False


def _name_splice(candidates, order, query, config: V9Config):
    """Keep the v0.4 head, then add chunks whose long rare name is in the query.

    The head is whatever v0.4 packs into anchor_tokens. Insertions come from
    the functions that share the single longest rare identifier, at most
    name_inserts of them. A name used by more than name_df_limit functions
    is ignored. No reference answer is read.
    """
    if config.name_inserts <= 0 or not order:
        return order
    counts = defaultdict(int)
    seen = set()
    for cand in candidates:
        qualname = cand.chunk.qualname or ''
        fid = cand.chunk.function_id
        if not qualname or qualname.startswith('__') or not fid or fid in seen:
            continue
        seen.add(fid)
        for part in identifier_terms(qualname):
            if len(part) >= config.name_min_len:
                counts[part] += 1
    rare = {part for part in identifier_terms(query)
            if len(part) >= config.name_min_len and counts[part] <= config.name_df_limit}
    best_len = {}
    for cand in candidates:
        qualname = cand.chunk.qualname or ''
        fid = cand.chunk.function_id
        if not qualname or qualname.startswith('__') or not fid:
            continue
        matched = [part for part in identifier_terms(qualname) if part in rare]
        if matched:
            best_len[fid] = max(best_len.get(fid, 0), max(len(part) for part in matched))
    if not best_len:
        return order
    longest = max(best_len.values())
    winners = {fid for fid, length in best_len.items() if length == longest}
    best = {}
    for cand in candidates:
        fid = cand.chunk.function_id
        if fid not in winners:
            continue
        current = best.get(fid)
        key = (cand.lexical, -cand.chunk.tokens_est, cand.chunk.id)
        if current is None or key > (current.lexical, -current.chunk.tokens_est, current.chunk.id):
            best[fid] = cand
    chosen = sorted(best.values(), key=lambda cand: (-cand.lexical, cand.chunk.tokens_est, cand.chunk.id))
    chosen = chosen[:config.name_inserts]
    head_hits = ContextStream(order).pack(config.anchor_tokens)[1] if config.anchor_tokens else []
    by_id = {cand.chunk.id: cand for cand in order}
    head = [by_id[hit.chunk.id] for hit in head_hits if hit.chunk.id in by_id]
    blocked = list(head)
    inserted = []
    for cand in chosen:
        if cand.chunk.id in {item.chunk.id for item in head}:
            continue
        if any(_overlaps(cand, item) for item in blocked):
            continue
        inserted.append(cand)
        blocked.append(cand)
    used = {cand.chunk.id for cand in head + inserted}
    tail = []
    for cand in order:
        if cand.chunk.id in used:
            continue
        if any(_overlaps(cand, item) for item in inserted):
            continue
        tail.append(cand)
    merged = head + inserted + tail
    for index, cand in enumerate(merged):
        cand.order = index
    return merged


class PreparedQueryV9:
    def __init__(self, index, query, config: V9Config, v4_config: V4Config):
        self.config = config
        started = perf_counter()
        base = replace(v4_config, enable_graph=config.enable_graph, enable_diversity=config.enable_diversity)
        cap = config.per_file_pool or None
        self.order, self.candidates, self.seeds = index.rank(
            query, base, per_file_pool=cap, coverage_bonus=config.coverage_bonus,
            name_prior=config.name_prior, name_df_max=config.name_df_max)
        self.order = _name_splice(self.candidates, self.order, query, config)
        self.stream = ContextStream(self.order, allow_partial=base.enable_chunking)
        elapsed = perf_counter() - started
        self.profile = {'candidate_sec': elapsed, 'graph_sec': 0.0, 'file_rank_sec': 0.0, 'chunk_score_sec': 0.0,
                        'diversity_sec': 0.0, 'anchor_sec': 0.0, 'pack_stream_sec': 0.0, 'prefix_sec': 0.0,
                        'query_sec': elapsed, 'pool': len(self.order), 'chunks_indexed': len(index.chunk_index.chunks),
                        'seeds': self.seeds}

    def pack(self, budget, *, instance_id=None, include_candidates=True):
        context, hits, selected = self.stream.pack(budget)
        logged = []
        if include_candidates:
            chosen_ids = set(selected)
            first_blocked = next((cid for cid, info in selected.items() if info['partial']), None)
            if first_blocked is None:
                first_blocked = next((entry.candidate.chunk.id for entry in self.stream.entries if entry.candidate.chunk.id not in selected), None)
            for candidate in self.order:
                cid = candidate.chunk.id
                reason = self.stream.rejections.get(cid)
                if cid in chosen_ids:
                    reason = 'selected_partial' if selected[cid]['partial'] else 'selected'
                elif not reason:
                    reason = 'budget_exceeded' if cid == first_blocked else 'after_budget_cutoff'
                logged.append({**candidate.metadata(), 'selected': cid in chosen_ids, 'decision': reason})
        used = token_count(context)
        diagnostics = {'instance_id': instance_id, 'method': 'pathpack-v9', 'config': self.config.to_dict(),
                       'budget_est': budget, 'tokens_est': used, 'remaining_tokens': budget - used,
                       'token_utilization': used / budget if budget else 0.0,
                       'selected_files': len({hit.chunk.path for hit in hits}),
                       'selected_functions': len({hit.chunk.function_id for hit in hits if hit.chunk.kind != 'module'}),
                       'selected_chunks': len(hits), 'token_counter': COUNTER, 'seeds': self.seeds,
                       'candidates': logged, 'profile': dict(self.profile)}
        return RetrievalResult(context, hits, diagnostics)


def retrieve_v9(repository, graph, query, *, budget=2048, config=None, v4_config=None):
    cfg = config or V9Config()
    base = v4_config or V4Config()
    index = HybridIndex(build_chunk_index(repository, graph, base), graph)
    return PreparedQueryV9(index, query, cfg, base).pack(budget)
