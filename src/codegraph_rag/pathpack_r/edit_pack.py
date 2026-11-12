"""Expected-edit-coverage packing for PathPack-E.

Each candidate has a probability p of containing an edited line. The objective is the expected number of
edited chunks covered per token, with a bonus for the first chunk of an unseen file (breadth). Selection is
a deterministic density greedy. A skeleton option (the def line only) is a second, cheaper resolution of
a function; it earns only a small share of p plus the file bonus. Named-symbol coverage is not a stop rule.
"""
from __future__ import annotations

from dataclasses import replace

from ..context_packer import context_header
from .packing import _lines, _overlap, _pack
from .retrieve import Region


def _skeleton(region):
    """A one-line resolution of the function that owns ``region``. ``None`` when it cannot be built."""
    chunk = region.chunk
    if not chunk.source_spans or not chunk.body:
        return None
    line, start, end = chunk.source_spans[0]
    text = chunk.body[:end - start]
    if not text.strip():
        return None
    header = context_header(chunk.path, chunk.qualname, line, line, chunk.signature)
    cost = (len((header + text).encode('utf-8')) + 3) // 4 + 1
    skel = replace(chunk, id=f'{chunk.id}#sig', source_spans=((line, start, end),), body=text,
                   line_start=line, line_end=line, tokens_est=cost)
    return Region(chunk=skel, channels=list(region.channels), lexical=region.lexical, semantic=region.semantic,
                  symbol=region.symbol, graph=region.graph, token_cost=cost, final=region.final,
                  admission='skeleton')


def pack_edit(regions, budget, config, *, skeleton=False):
    pool = list(regions)
    if skeleton:
        seen = set()
        for region in sorted(regions, key=lambda r: (-r.final, r.chunk.id)):
            fid = region.chunk.function_id
            if not fid or fid in seen:
                continue
            seen.add(fid)
            # Use the function's first chunk so the skeleton is the real def line.
            owned = sorted((r for r in regions if r.chunk.function_id == fid), key=lambda r: (r.chunk.line_start or 0, r.chunk.id))
            sk = _skeleton(owned[0])
            if sk is not None:
                sk.final = max(r.final for r in owned)
                pool.append(sk)
    file_p = {}
    for region in regions:
        file_p[region.chunk.path] = max(file_p.get(region.chunk.path, 0.0), region.final)
    chosen, covered, seen_files, used = [], set(), set(), 0
    remaining = list(pool)
    while remaining:
        best, best_key = None, None
        for region in remaining:
            if used + region.token_cost > budget:
                continue
            overlap = _overlap(region.chunk, covered)
            weight = config.edit_skeleton_weight if region.admission == 'skeleton' else 1.0
            gain = weight * region.final * (1.0 - overlap)
            if region.chunk.path not in seen_files:
                gain += config.edit_file_bonus * file_p.get(region.chunk.path, 0.0)
            if gain < config.edit_min_gain:
                continue
            key = (gain / (region.token_cost ** config.edit_cost_exponent), gain, -region.token_cost, region.chunk.id)
            if best_key is None or key > best_key:
                best, best_key = region, key
        if best is None:
            break
        best.decision = 'selected'
        chosen.append(best)
        remaining = [r for r in remaining if r.chunk.id != best.chunk.id]
        used += best.token_cost
        covered |= _lines(best.chunk)
        seen_files.add(best.chunk.path)
    for region in regions:
        if not region.decision:
            region.decision = 'not_selected'
    context, hits, kept = _pack(chosen, budget)
    for region in chosen:
        if region.chunk.id not in kept:
            region.decision = 'truncated_by_counter'
    return context, hits, {
        'packer': 'edit-skeleton' if skeleton else 'edit', 'objective': 'expected_edit_coverage_per_token',
        'file_bonus': config.edit_file_bonus, 'min_gain': config.edit_min_gain,
        'cost_exponent': config.edit_cost_exponent,
        'skeletons_selected': sum(r.admission == 'skeleton' for r in chosen),
        'stop_rule': 'marginal_gain_floor_or_budget', 'named_evidence_stop': False,
    }
