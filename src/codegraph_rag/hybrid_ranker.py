"""Inspectable lexical/symbol/typed graph evidence, all normalized to [0,1]."""
from collections import defaultdict, deque
from dataclasses import dataclass
import math
from .retrieval import BM25, identifier_terms, terms
from .config import V4Config


@dataclass
class Candidate:
    chunk: object
    bm25: float
    lexical: float
    symbol: float
    graph: float
    route: list
    edge_kinds: list
    diversity: float = 0.0
    name_score: float = 0.0
    score: float = 0.0
    utility: float = 0.0
    reason: str = ''
    order: int | None = None

    def metadata(self):
        return {**self.chunk.metadata(),'bm25_score':self.bm25,'lexical_score':self.lexical,
                'symbol_score':self.symbol,'graph_score':self.graph,'name_score':self.name_score,
                'diversity_contribution':self.diversity,
                'score':self.score,'utility':self.utility,'graph_path':self.route,
                'edge_kinds':self.edge_kinds,'reason':self.reason,'order':self.order}


class HybridIndex:
    def __init__(self,chunk_index,graph):
        self.chunk_index=chunk_index;self.graph=graph
        self.bm25=BM25(chunk_index.chunks)
        self.term_sets={c.id:frozenset(terms(c.body)) for c in chunk_index.chunks}
        self.symbols={c.id:frozenset(terms(c.path+' '+c.qualname)) for c in chunk_index.chunks}
        self.adj=defaultdict(list);self.adj_out=defaultdict(list);self.adj_in=defaultdict(list)
        for e in sorted(graph.edges,key=lambda e:(e.source,e.kind,e.target)):
            self.adj[e.source].append((e.target,e.kind));self.adj[e.target].append((e.source,e.kind))
            self.adj_out[e.source].append((e.target,e.kind));self.adj_in[e.target].append((e.source,e.kind))

    def rank(self,query,cfg:V4Config,per_file_pool:int|None=None,coverage_bonus:float=0.0,name_prior:float=0.0,name_df_max:float=0.01):
        raw=self.bm25.scores(query);maximum=max(raw.values(),default=0) or 1.0
        function_score=defaultdict(float)
        for c in self.chunk_index.chunks:function_score[c.function_id]=max(function_score[c.function_id],raw[c.id])
        seeds=[i for i in sorted(function_score,key=lambda i:(-function_score[i],i))[:cfg.seeds] if function_score[i]>0]
        paths={}
        neighbors={'both':self.adj,'out':self.adj_out,'in':self.adj_in}[cfg.graph_direction]
        if cfg.enable_graph:
            for seed in seeds:
                queue=deque([(seed,1.0,[seed],[])])
                best={(seed,0):1.0}
                while queue:
                    node,weight,route,kinds=queue.popleft()
                    previous=paths.get(node)
                    if previous is None or (-weight,len(route),route)<(-previous[0],len(previous[1]),previous[1]):paths[node]=(weight,route,kinds)
                    if len(kinds)>=cfg.hops:continue
                    for target,kind in neighbors.get(node,[]):
                        w=weight*cfg.edge_weights.get(kind,0)*cfg.distance_decay
                        state=(target,len(kinds)+1)
                        if target in route or w<=best.get(state,0):continue
                        best[state]=w;queue.append((target,w,route+[target],kinds+[kind]))
        query_terms=set(terms(query));candidates=[]
        rare_names=frozenset()
        if name_prior:
            # A name used by a handful of functions is informative. A name used
            # across the repository (check, get, value) is not, even when the
            # issue text contains that word.
            df=defaultdict(int); seen=set()
            for fn in self.chunk_index.functions.values():
                if fn.id in seen or not fn.qualname or fn.qualname.startswith('__'):
                    continue
                seen.add(fn.id)
                for part in identifier_terms(fn.qualname):
                    if len(part)>=5: df[part]+=1
            limit=max(3, math.ceil(name_df_max*max(1,len(seen))))
            rare_names=frozenset(part for part in identifier_terms(query) if len(part)>=5 and df[part]<=limit)
        for c in self.chunk_index.chunks:
            norm=raw[c.id]/maximum
            names=len(query_terms & self.symbols[c.id])/max(1,len(query_terms)) if cfg.enable_symbol else 0.0
            confidence,route,kinds=paths.get(c.function_id,(0.0,[],[]))
            # A lexical anchor does not receive its own graph evidence bonus.
            confidence=confidence if c.function_id not in seeds else 0.0
            name_score=0.0
            if rare_names and c.qualname and not c.qualname.startswith('__'):
                name_score=1.0 if identifier_terms(c.qualname) & rare_names else 0.0
            cand=Candidate(c,raw[c.id],norm,names,confidence,route,kinds,name_score=name_score)
            evidence=cfg.lexical_weight*norm+cfg.symbol_weight*names+cfg.graph_weight*confidence
            if name_prior: evidence+=name_prior*name_score
            cand.score=evidence
            cand.utility=evidence/((c.tokens_est+cfg.cost_offset)**cfg.cost_exponent) if cfg.enable_cost else evidence
            if evidence<=cfg.min_relevance:cand.reason='low_relevance'
            candidates.append(cand)
        valid=sorted((c for c in candidates if not c.reason),key=lambda c:(-c.utility,-c.score,c.chunk.id))
        if per_file_pool is None:
            for c in valid[cfg.candidate_limit:]:c.reason='candidate_limit'
            pool=valid[:cfg.candidate_limit]
        else:
            pool=[];used_ids=set();per_file=defaultdict(int)
            for c in valid:
                if len(pool)>=cfg.candidate_limit:break
                if per_file[c.chunk.path]>=per_file_pool:
                    c.reason='file_pool';continue
                per_file[c.chunk.path]+=1;pool.append(c);used_ids.add(id(c))
            for c in valid:
                if id(c) not in used_ids and not c.reason:c.reason='candidate_limit'
        covered=defaultdict(set);span_lines={};func_size={}
        if coverage_bonus:
            for cand in candidates:
                span_lines[cand.chunk.id]={ln for ln,_,_ in cand.chunk.source_spans}
            for fid,fn in self.chunk_index.functions.items():
                func_size[fid]=max(1,(fn.line_end or 1)-(fn.line_start or 1)+1)
        order=[];file_counts=defaultdict(int);similarities=defaultdict(float)
        while pool:
            for c in pool:
                if cfg.enable_diversity:
                    novelty=(1-similarities[c.chunk.id])/((1+file_counts[c.chunk.path])**cfg.file_diversity_exponent)
                    c.diversity=cfg.diversity_weight*novelty
                    # Redundancy attenuates evidence, rather than imposing a file quota.
                    attenuation=1-cfg.redundancy_strength*similarities[c.chunk.id]
                else:c.diversity=0.0;attenuation=1.0
                base=cfg.lexical_weight*c.lexical+cfg.symbol_weight*c.symbol+cfg.graph_weight*c.graph
                if name_prior: base+=name_prior*c.name_score
                c.score=base*attenuation+c.diversity
                if coverage_bonus:
                    fid=c.chunk.function_id;lines=span_lines.get(c.chunk.id,())
                    new=len(lines-covered[fid]) if lines else 0
                    size=func_size.get(fid,max(1,new))
                    before=len(covered[fid])/size;after=(len(covered[fid])+new)/size
                    cross=(1.0 if before<0.2<=after else 0.0)+(1.0 if before<1.0<=after else 0.0)
                    c.score+=coverage_bonus*(new/size+cross)
                c.utility=c.score/((c.chunk.tokens_est+cfg.cost_offset)**cfg.cost_exponent) if cfg.enable_cost else c.score
            chosen=min(pool,key=lambda c:(-c.utility,-c.score,c.chunk.id));pool.remove(chosen)
            chosen.order=len(order);order.append(chosen);file_counts[chosen.chunk.path]+=1
            if coverage_bonus:
                covered[chosen.chunk.function_id].update(span_lines.get(chosen.chunk.id,()))
            if cfg.enable_diversity:
                a=self.term_sets[chosen.chunk.id]
                for c in pool:
                    b=self.term_sets[c.chunk.id]
                    similarities[c.chunk.id]=max(similarities[c.chunk.id],len(a & b)/max(1,len(a | b)))
        return order,candidates,seeds
