"""File pool, then fragment ranking. No gold, patches, or metric imports."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
import re
import time

from .chunk_ranker import ChunkCandidate, formula, order_chunks, utility_of
from .context_packer import token_count
from .file_ranker import FileEvidence, rank_files
from .adaptive_chunker import context_header
from .retrieval import BM25, terms


_DEF_LINE=re.compile(r'^\s*(?:async\s+def|def)\b')


class _Doc:
    def __init__(self,chunk):
        self.id=chunk.id
        self.text=f'{chunk.body}\n{chunk.qualname}'
        self.path=''
        self.qualname=''


@dataclass
class RankedRetrieval:
    order: list
    candidates: list
    seeds: list
    files: list
    timings: dict
    local_blend: float


def _unit(value,maximum):
    if maximum<=0:
        return 0.0
    return max(0.0,min(1.0,value/maximum))


def graph_reach(graph,seeds,cfg):
    adjacency=defaultdict(list)
    paths={}
    if not cfg.enable_graph:
        return paths,adjacency
    for edge in sorted(graph.edges,key=lambda item:(item.source,item.kind,item.target)):
        adjacency[edge.source].append((edge.target,edge.kind))
        adjacency[edge.target].append((edge.source,edge.kind))
    for seed in seeds:
        queue=deque([(seed,1.0,[seed],[])])
        best={(seed,0):1.0}
        while queue:
            node,weight,route,kinds=queue.popleft()
            previous=paths.get(node)
            if previous is None or (-weight,len(route),route)<(-previous[0],len(previous[1]),previous[1]):
                paths[node]=(weight,route,kinds)
            if len(kinds)>=cfg.hops:
                continue
            for target,kind in adjacency.get(node,[]):
                stepped=weight*cfg.edge_weights.get(kind,0)*cfg.distance_decay
                state=(target,len(kinds)+1)
                if target in route or stepped<=best.get(state,0):
                    continue
                best[state]=stepped
                queue.append((target,stepped,route+[target],kinds+[kind]))
    return paths,adjacency


def signature_lines(lines,start,end,limit):
    if limit<=0 or start<1 or start>len(lines):
        return []
    chosen=[]
    last=min(end,len(lines))
    for number in range(start,last+1):
        if len(chosen)>=limit:
            break
        chosen.append(number)
        if _DEF_LINE.match(lines[number-1]):
            break
    return chosen


def owning_definition(info,chunk):
    if info is None:
        return None
    matches=[item for item in info.definitions if item['line_start']<=(chunk.line_start or 0)<=item['line_end']]
    if matches:
        return min(matches,key=lambda item:(item['line_end']-item['line_start'],item['line_start']))
    return {'line_start':info.line_start,'line_end':info.line_end}


def apply_signature_anchor(chunk,lines,info,cfg):
    """Prepend the def/decorators when the selected slice does not already contain them.

    Neighboring body lines are not added. The anchor is source text with its own
    token cost; it is not given an extra relevance score.
    """
    if not cfg.include_signature_anchor or cfg.max_anchor_lines<=0:
        return chunk
    if chunk.kind not in {'function','method'} or lines is None:
        return chunk
    definition=owning_definition(info,chunk)
    if definition is None:
        return chunk
    occupied={span[0] for span in chunk.source_spans}
    prefix=[]
    for number in signature_lines(lines,definition['line_start'],definition['line_end'],cfg.max_anchor_lines):
        if number in occupied or number>len(lines):
            continue
        prefix.append((number,0,len(lines[number-1])))
    if not prefix:
        return chunk
    spans=tuple(sorted([*prefix,*chunk.source_spans]))
    body=''.join(lines[line-1][start:stop] for line,start,stop in spans)
    if not body.strip():
        return chunk
    header=context_header(chunk.path,chunk.qualname,spans[0][0],spans[-1][0],chunk.signature)
    text=header+body+'\n\n'
    return replace(chunk,text=text,tokens_est=token_count(text),line_start=spans[0][0],line_end=spans[-1][0],
                   source_spans=spans,body=body)


class HierarchicalIndex:
    """Query-independent indexes. ``rank`` is the per-issue cost."""

    def __init__(self,chunk_index,graph,repository,file_chunk_index=None):
        self.chunk_index=chunk_index
        self.graph=graph
        self.chunks=list(chunk_index.chunks)
        self.documents=[_Doc(chunk) for chunk in self.chunks]
        self.bm25=BM25(self.documents) if self.documents else None
        self.term_sets={chunk.id:frozenset(terms(chunk.body)) for chunk in self.chunks}
        self.symbols={chunk.id:frozenset(terms(chunk.path+' '+chunk.qualname)) for chunk in self.chunks}
        self.names={info.id:frozenset(terms(info.qualname)) for info in chunk_index.functions.values()}
        self.file_chunks=list(file_chunk_index.chunks) if file_chunk_index is not None else []
        self.file_bm25=BM25([_Doc(chunk) for chunk in self.file_chunks]) if self.file_chunks else None
        self.file_symbols={chunk.id:frozenset(terms(chunk.path+' '+chunk.qualname)) for chunk in self.file_chunks}
        self.file_terms={chunk.id:frozenset(terms(chunk.body)) for chunk in self.file_chunks}
        self.lines={}
        root=Path(repository).resolve()
        for path in sorted({chunk.path for chunk in self.chunks}):
            file=(root/path).resolve()
            if not file.is_relative_to(root) or file.is_symlink() or not file.is_file():
                continue
            try:
                self.lines[path]=file.read_text(encoding='utf-8-sig').splitlines(keepends=True)
            except (OSError,UnicodeError):
                continue

    def rank(self,query,cfg):
        started=time.perf_counter()
        raw=self.bm25.scores(query) if self.bm25 else {}
        maximum=max(raw.values(),default=0.0)
        query_terms=set(terms(query))
        by_file_max=defaultdict(float)
        function_raw=defaultdict(float)
        for chunk in self.chunks:
            by_file_max[chunk.path]=max(by_file_max[chunk.path],raw.get(chunk.id,0.0))
            function_raw[chunk.function_id]=max(function_raw[chunk.function_id],raw.get(chunk.id,0.0))
        if cfg.file_evidence=='function' and self.file_bm25:
            coarse_raw=self.file_bm25.scores(query)
            for chunk in self.file_chunks:
                function_raw[chunk.function_id]=max(function_raw[chunk.function_id],coarse_raw.get(chunk.id,0.0))
        seeds=[item for item in sorted(function_raw,key=lambda ident:(-function_raw[ident],ident))[:cfg.seeds] if function_raw[item]>0]
        candidate_sec=time.perf_counter()-started
        graph_started=time.perf_counter()
        paths,adjacency=graph_reach(self.graph,seeds,cfg)
        graph_sec=time.perf_counter()-graph_started
        file_started=time.perf_counter()
        global_lex={chunk.id:_unit(raw.get(chunk.id,0.0),maximum) for chunk in self.chunks}
        symbol={}
        for chunk in self.chunks:
            if cfg.enable_symbol and query_terms:
                symbol[chunk.id]=len(query_terms & self.symbols[chunk.id])/len(query_terms)
            else:
                symbol[chunk.id]=0.0
        function_lex=defaultdict(float)
        for chunk in self.chunks:
            function_lex[chunk.function_id]=max(function_lex[chunk.function_id],global_lex[chunk.id])
        neighbor_terms={}
        for ident in self.chunk_index.functions:
            bag=set()
            if cfg.enable_graph:
                for target,_kind in adjacency.get(ident,[]):
                    if function_lex.get(target,0.0)>0:
                        bag.update(self.names.get(target,()))
            neighbor_terms[ident]=frozenset(bag)
        graph_component={}
        route_of={}
        kinds_of={}
        path_component={}
        neighbor_component={}
        for chunk in self.chunks:
            confidence,route,kinds=paths.get(chunk.function_id,(0.0,[],[]))
            if chunk.function_id in seeds:
                confidence=0.0
            path_component[chunk.id]=confidence if cfg.enable_graph else 0.0
            names=neighbor_terms.get(chunk.function_id,frozenset())
            body=self.term_sets.get(chunk.id,frozenset())
            neighbor=(len(body & names)/len(names)) if names and cfg.enable_graph else 0.0
            neighbor_component[chunk.id]=neighbor
            graph_component[chunk.id]=min(1.0,0.75*path_component[chunk.id]+0.25*neighbor) if cfg.enable_graph else 0.0
            route_of[chunk.id]=route if cfg.enable_graph else []
            kinds_of[chunk.id]=kinds if cfg.enable_graph else []
        blend=cfg.local_chunk_blend if cfg.enable_hierarchy else 0.0
        evidence_chunks=self.chunks
        evidence_lex=global_lex
        evidence_symbol=symbol
        evidence_graph=graph_component
        if cfg.file_evidence=='function' and self.file_chunks:
            file_raw=self.file_bm25.scores(query) if self.file_bm25 else {}
            file_max=max(file_raw.values(),default=0.0)
            evidence_chunks=self.file_chunks
            evidence_lex={chunk.id:_unit(file_raw.get(chunk.id,0.0),file_max) for chunk in self.file_chunks}
            evidence_symbol={}
            evidence_graph={}
            for chunk in self.file_chunks:
                if cfg.enable_symbol and query_terms:
                    evidence_symbol[chunk.id]=len(query_terms & self.file_symbols[chunk.id])/len(query_terms)
                else:
                    evidence_symbol[chunk.id]=0.0
                confidence,_route,_kinds=paths.get(chunk.function_id,(0.0,[],[]))
                if chunk.function_id in seeds:
                    confidence=0.0
                path_score=confidence if cfg.enable_graph else 0.0
                names=neighbor_terms.get(chunk.function_id,frozenset())
                body=self.file_terms.get(chunk.id,frozenset())
                neighbor=(len(body & names)/len(names)) if names and cfg.enable_graph else 0.0
                evidence_graph[chunk.id]=min(1.0,0.75*path_score+0.25*neighbor) if cfg.enable_graph else 0.0
        file_bucket=defaultdict(list)
        for chunk in evidence_chunks:
            file_bucket[chunk.path].append(chunk)
        evidence=[]
        for path,group in file_bucket.items():
            lexical=max(evidence_lex[chunk.id] for chunk in group)
            symbols=max(evidence_symbol[chunk.id] for chunk in group)
            linked=max(evidence_graph[chunk.id] for chunk in group)
            raw_file=(cfg.file_lexical_weight*lexical+cfg.file_symbol_weight*symbols+cfg.file_graph_weight*linked)
            holder=max(group,key=lambda chunk:(evidence_lex[chunk.id],evidence_graph[chunk.id],chunk.id))
            evidence.append(FileEvidence(path,lexical,symbols,linked,raw_file,holder.id,evidence_lex[holder.id],holder.function_id,len(group)))
        seed_paths={chunk.path for chunk in self.chunks if chunk.function_id in seeds}
        decisions,chosen=rank_files(evidence,seed_paths,cfg)
        norm={item.path:item.score for item in decisions}
        file_rank_sec=time.perf_counter()-file_started
        score_started=time.perf_counter()
        candidates=[]
        for chunk in self.chunks:
            local=_unit(raw.get(chunk.id,0.0),by_file_max[chunk.path])
            text=blend*local+(1.0-blend)*global_lex[chunk.id]
            chunk_score=(1.0-cfg.chunk_symbol_weight)*text+cfg.chunk_symbol_weight*symbol[chunk.id]
            used_file=norm.get(chunk.path,0.0) if cfg.enable_file_score else 0.0
            candidate=ChunkCandidate(chunk,used_file,chunk_score,graph_component[chunk.id],global_lex[chunk.id],
                                     symbol[chunk.id],local,path_component[chunk.id],neighbor_component[chunk.id],
                                     route_of[chunk.id],kinds_of[chunk.id])
            candidate.extra['bm25']=raw.get(chunk.id,0.0)
            evidence_only=cfg.alpha_file*used_file+cfg.beta_chunk*chunk_score+cfg.gamma_graph*graph_component[chunk.id]
            if evidence_only<=cfg.min_relevance or not query_terms:
                candidate.reason='low_relevance'
            elif cfg.enable_hierarchy and cfg.enable_file_filter and chunk.path not in chosen:
                candidate.reason='outside_file_pool'
            candidates.append(candidate)
        eligible=[candidate for candidate in candidates if not candidate.reason]
        anchor_started=time.perf_counter()
        for candidate in eligible:
            candidate.chunk=apply_signature_anchor(candidate.chunk,self.lines.get(candidate.chunk.path),
                                                   self.chunk_index.functions.get(candidate.chunk.function_id),cfg)
        anchor_sec=time.perf_counter()-anchor_started
        by_file=defaultdict(list)
        for candidate in eligible:
            static=formula(candidate,1.0,cfg)
            candidate.preliminary=utility_of(static,candidate.chunk.tokens_est,cfg)
            candidate.formula_score=static
            by_file[candidate.chunk.path].append(candidate)
        pool=[]
        if cfg.enable_hierarchy and cfg.enable_file_filter:
            grouped={}
            for path,group in by_file.items():
                group.sort(key=lambda candidate:(-candidate.preliminary,-candidate.chunk_score,candidate.chunk.id))
                keep=group[:cfg.per_file_candidates]
                pool.extend(keep)
                for candidate in group[cfg.per_file_candidates:]:
                    candidate.reason='per_file_limit'
            grouped=defaultdict(list)
            for candidate in pool:
                grouped[candidate.chunk.path].append(candidate)
            reserved=[]
            rest=[]
            for path in sorted(grouped):
                items=grouped[path]
                reserved.append(items[0])
                rest.extend(items[1:])
            rest.sort(key=lambda candidate:(-candidate.preliminary,candidate.chunk.id))
            room=max(0,cfg.candidate_limit-len(reserved))
            kept=set(id(item) for item in reserved+rest[:room])
            for candidate in rest[room:]:
                candidate.reason='candidate_limit'
            pool=[candidate for candidate in reserved+rest if id(candidate) in kept]
        else:
            eligible.sort(key=lambda candidate:(-candidate.preliminary,candidate.chunk.id))
            pool=eligible[:cfg.candidate_limit]
            for candidate in eligible[cfg.candidate_limit:]:
                candidate.reason='candidate_limit'
        chunk_score_sec=time.perf_counter()-score_started
        diversity_started=time.perf_counter()
        order=order_chunks(pool,self.term_sets,cfg)
        diversity_sec=time.perf_counter()-diversity_started
        reason_counts=dict(Counter(candidate.reason or 'ranked' for candidate in candidates))
        logged=[candidate for candidate in candidates if candidate.order is not None or candidate.reason in {'per_file_limit','candidate_limit'}]
        if len(logged)>2000:
            logged=[candidate for candidate in logged if candidate.order is not None][:cfg.candidate_limit]
        timings={'candidate_sec':candidate_sec,'graph_sec':graph_sec,'file_rank_sec':file_rank_sec,
                 'chunk_score_sec':chunk_score_sec,'anchor_sec':anchor_sec,'diversity_sec':diversity_sec,
                 'pool':len(pool),'files_selected':len(chosen) if cfg.enable_hierarchy and cfg.enable_file_filter else len({c.chunk.path for c in pool}),
                 'chunks_indexed':len(self.chunks),'reason_counts':reason_counts}
        return RankedRetrieval(order,logged,seeds,[item.to_dict() for item in decisions],timings,blend)
