"""Budgeted context selection. Named-symbol coverage is not a stop rule."""
from __future__ import annotations

import math

from ..context_packer import ContextStream
from ..hybrid_ranker import Candidate


def _lines(chunk):
    return {(chunk.path, line) for line, _start, _end in chunk.source_spans}


def _overlap(chunk, covered):
    lines = _lines(chunk)
    if not lines:
        return 1.0
    return len(lines & covered) / len(lines)


def _as_candidate(region):
    return Candidate(
        region.chunk, region.lexical, region.lexical, region.symbol, region.graph, [], [],
        score=region.final,
    )


def _floor(regions, config):
    best = max((region.final for region in regions), default=0.0)
    return max(config.min_relevance, config.relative_floor * best), best


def _pack(chosen, budget):
    stream = ContextStream([_as_candidate(region) for region in chosen], allow_partial=True)
    context, hits, _selected = stream.pack(budget)
    kept = {hit.chunk.id for hit in hits}
    return context, hits, kept


def pack_ranked(regions, budget, config):
    floor, best = _floor(regions, config)
    order = sorted(regions, key=lambda region: (-region.final, region.chunk.id))
    chosen = []
    covered = set()
    used = 0
    for region in order:
        if region.final < floor:
            region.decision = 'below_quality'
            continue
        if _overlap(region.chunk, covered) >= config.max_overlap:
            region.decision = 'redundant'
            continue
        if used + region.token_cost > budget:
            region.decision = 'over_budget'
            continue
        region.decision = 'selected'
        chosen.append(region)
        used += region.token_cost
        covered |= _lines(region.chunk)
    context, hits, kept = _pack(chosen, budget)
    for region in regions:
        if not region.decision:
            region.decision = 'not_selected'
    for region in chosen:
        if region.chunk.id not in kept:
            region.decision = 'truncated_by_counter'
    return context, hits, {
        'packer': 'ranked', 'quality_floor': floor, 'best_score': best,
        'stop_rule': 'relative_floor_or_redundancy_or_budget',
        'named_evidence_stop': False,
    }


def _objective(order, config):
    covered = set()
    total = 0.0
    selected_ids = set()
    for region in order:
        overlap = _overlap(region.chunk, covered)
        necessity = 0.0
        if region.admission == 'unique_callee':
            low = region.rerank is not None and region.rerank < config.reject_ratio * max(region.hypothesis, region.final, 1e-9)
            if not low:
                necessity = config.necessity_bonus
        gain = region.final * (1.0 - overlap) + necessity
        total += gain
        covered |= _lines(region.chunk)
        selected_ids.add(region.chunk.id)
    return total


def pack_budgeted(regions, budget, config):
    floor, best = _floor(regions, config)
    remaining = list(regions)
    chosen = []
    covered = set()
    used = 0
    while remaining:
        best_region = None
        best_key = None
        for region in remaining:
            overlap = _overlap(region.chunk, covered)
            necessity = 0.0
            parent_signal = max(region.hypothesis, region.final)
            if region.admission == 'unique_callee':
                rejected = region.rerank is not None and region.rerank < config.reject_ratio * max(parent_signal, 1e-9)
                if not rejected:
                    necessity = config.necessity_bonus
            gain = region.final * (1.0 - overlap) + necessity
            if region.final < floor and necessity <= 0:
                continue
            if gain < config.min_marginal:
                continue
            if used + region.token_cost > budget:
                continue
            key = (gain / math.sqrt(region.token_cost), gain, -region.token_cost, region.chunk.id)
            if best_key is None or key > best_key:
                best_key = key
                best_region = region
        if best_region is None:
            break
        best_region.decision = 'selected'
        chosen.append(best_region)
        remaining = [region for region in remaining if region.chunk.id != best_region.chunk.id]
        used += best_region.token_cost
        covered |= _lines(best_region.chunk)
    rejected = sorted(
        [region for region in regions if region.decision != 'selected' and region.final >= floor],
        key=lambda region: (-region.final, region.chunk.id),
    )[:config.swap_candidates]
    swaps = 0
    for _pass in range(config.swap_passes):
        improved = False
        current = _objective(chosen, config)
        for old in list(chosen):
            for new in rejected:
                if any(region.chunk.id == new.chunk.id for region in chosen):
                    continue
                trial = [region for region in chosen if region.chunk.id != old.chunk.id] + [new]
                trial.sort(key=lambda region: (-region.final, region.chunk.id))
                if sum(region.token_cost for region in trial) > budget:
                    continue
                if _objective(trial, config) > current + 1e-9:
                    old.decision = 'swapped_out'
                    new.decision = 'selected'
                    chosen = trial
                    rejected = [region for region in rejected if region.chunk.id != new.chunk.id]
                    rejected.append(old)
                    swaps += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    for region in regions:
        if not region.decision:
            region.decision = 'not_selected'
    context, hits, kept = _pack(chosen, budget)
    for region in chosen:
        if region.chunk.id not in kept:
            region.decision = 'truncated_by_counter'
    return context, hits, {
        'packer': 'budgeted', 'quality_floor': floor, 'best_score': best, 'swaps': swaps,
        'stop_rule': 'marginal_gain_floor_or_budget', 'named_evidence_stop': False,
    }
