"""Two-round hypothesis check. A neighbor is not added only because the edge exists."""
from __future__ import annotations

from pathlib import Path

from ..adaptive_chunker import context_header, production_path
from ..context_packer import token_count
from ..retrieval import Chunk
from .retrieve import Region
from .symbols import matches_any


def _materialize(index, node):
    file = Path(index.repository) / node.path
    try:
        lines = file.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    except (OSError, UnicodeError):
        return None
    spans = []
    pieces = []
    for line_no in range(node.line_start, min(node.line_end, len(lines)) + 1):
        raw = lines[line_no - 1]
        if not raw.strip():
            continue
        spans.append((line_no, 0, len(raw)))
        pieces.append(raw)
    if not spans:
        return None
    body = ''.join(pieces)
    header = context_header(node.path, node.qualname, spans[0][0], spans[-1][0], '')
    return Chunk(
        f'active::{node.id}', node.path, node.qualname, node.kind, header + body + '\n\n',
        max(1, token_count(header + body + '\n\n')), node.id, spans[0][0], spans[-1][0],
        tuple(spans), '', None, body,
    )


def _independent(chunk, queries, lexical, semantic_scores, config):
    if matches_any(chunk.qualname, chunk.path, queries):
        return 'symbol'
    if lexical.get(chunk.id, 0.0) >= config.expand_lexical:
        return 'lexical'
    if semantic_scores and semantic_scores.get(chunk.id, 0.0) >= config.expand_semantic:
        return 'semantic'
    return ''


def active_expand(index, regions, queries, lexical, semantic_scores, config, *, seed='reranker'):
    """Expand a shortlist for at most ``active_rounds`` rounds.

    ``seed='reranker'`` walks from the current final scores.
    ``seed='bm25'`` walks from lexical scores. That is the ablation.
    """
    admitted = list(regions)
    known = {region.chunk.id for region in admitted}
    seen_pairs = set()
    trace = []
    for round_index in range(config.active_rounds):
        if seed == 'bm25' and round_index == 0:
            shortlist = sorted(admitted, key=lambda region: (-region.lexical, region.chunk.id))[:config.shortlist]
        else:
            shortlist = sorted(admitted, key=lambda region: (-region.final, region.hypothesis, region.chunk.id))[:config.shortlist]
        proposals = []
        for hypothesis in shortlist:
            fid = hypothesis.chunk.function_id
            if not fid:
                continue
            rels = index.relations(fid)
            callees = [other for other, kind, direction in rels if kind == 'calls' and direction == 'out']
            unique = callees[0] if len(set(callees)) == 1 else None
            for other, kind, direction in rels:
                pair = (fid, other, kind, direction)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge = {'source': fid, 'target': other, 'kind': kind, 'direction': direction, 'round': round_index}
                owned = list(index.by_fn.get(other, []))
                if not owned:
                    node = index.graph.nodes.get(other)
                    if node is not None and not production_path(node.path):
                        named = matches_any(node.qualname, node.path, queries)
                        action = 'test_named' if named else 'test_ignored'
                        trace.append({**edge, 'action': action, 'path': node.path, 'function': node.qualname})
                        if named:
                            chunk = _materialize(index, node)
                            if chunk is not None:
                                proposals.append((hypothesis, chunk, 'named_test', edge, 0.0))
                        continue
                    trace.append({**edge, 'action': 'unresolved'})
                    continue
                signal = ''
                for chunk in owned:
                    signal = _independent(chunk, queries, lexical, semantic_scores, config)
                    if signal:
                        break
                is_unique = unique is not None and other == unique and kind == 'calls' and direction == 'out'
                if not signal and not is_unique:
                    trace.append({**edge, 'action': 'rejected_neighbor', 'function': owned[0].qualname})
                    continue
                reason = 'unique_callee' if is_unique and not signal else (signal or 'linked')
                hypothesis_score = hypothesis.final * config.redirect_factor if reason == 'unique_callee' else 0.0
                owned.sort(key=lambda chunk: (-lexical.get(chunk.id, 0.0), chunk.id))
                already = [chunk for chunk in owned if chunk.id in known]
                if already and reason == 'unique_callee':
                    for region in admitted:
                        if region.chunk.id in {chunk.id for chunk in already}:
                            region.admission = 'unique_callee'
                            region.hypothesis = max(region.hypothesis, hypothesis_score)
                            if 'active' not in region.channels:
                                region.channels.append('active')
                    trace.append({**edge, 'action': 'mark_hypothesis', 'reason': reason, 'function': owned[0].qualname})
                for chunk in owned[:config.max_chunks_per_function]:
                    if chunk.id in known:
                        continue
                    proposals.append((hypothesis, chunk, reason, edge, hypothesis_score))
                if any(chunk.id not in known for chunk in owned[:config.max_chunks_per_function]):
                    trace.append({**edge, 'action': 'admit', 'reason': reason, 'function': owned[0].qualname})
        proposals.sort(key=lambda row: (-row[0].final, row[2], row[1].id))
        fresh = []
        seen_new = set()
        for hypothesis, chunk, reason, edge, hypothesis_score in proposals:
            if chunk.id in known or chunk.id in seen_new:
                continue
            seen_new.add(chunk.id)
            fresh.append((hypothesis, chunk, reason, edge, hypothesis_score))
        dropped = max(0, len(fresh) - config.max_new_per_round)
        if dropped:
            trace.append({'action': 'dropped_new', 'round': round_index, 'dropped': dropped, 'reason': 'max_new_per_round'})
        fresh = fresh[:config.max_new_per_round]
        if not fresh:
            trace.append({'action': 'stop', 'round': round_index, 'reason': 'no_new_hypothesis'})
            break
        for hypothesis, chunk, reason, edge, hypothesis_score in fresh:
            semantic = None if not semantic_scores else semantic_scores.get(chunk.id, 0.0)
            region = Region(
                chunk=chunk,
                channels=['active'],
                ranks={'active': round_index},
                lexical=lexical.get(chunk.id, 0.0),
                semantic=semantic,
                symbol=1.0 if matches_any(chunk.qualname, chunk.path, queries) else 0.0,
                graph=1.0,
                graph_edges=[edge],
                token_cost=max(1, int(chunk.tokens_est or 1)),
                admission=reason,
                hypothesis=hypothesis_score,
                final=hypothesis_score,
            )
            admitted.append(region)
            known.add(chunk.id)
        trace.append({'action': 'round_complete', 'round': round_index, 'added': len(fresh), 'seed': seed})
    return admitted, trace
