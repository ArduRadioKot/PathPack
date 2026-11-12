"""One budget-independent stream. Valid prefixes end after source atoms, never a header."""
from dataclasses import dataclass, replace
from bisect import bisect_right
from collections import defaultdict
from .adaptive_chunker import context_header
from .retrieval import estimated_tokens, Hit

COUNTER='utf8_bytes_div4_ceil_serialized_context_v1'


def token_count(text):return estimated_tokens(text) if text else 0


@dataclass
class StreamEntry:
    candidate: object
    header: str
    pieces: list  # (source span, emitted code, stream character end)
    start: int
    end: int


class ContextStream:
    def __init__(self,order,*,allow_partial=True):
        self.entries=[];self.boundaries=[0];self.text='';self.rejections={}
        covered=defaultdict(list);content_seen=set();parts=[];length=0
        for candidate in order:
            chunk=candidate.chunk
            if not chunk.source_spans or not chunk.body.strip():
                self.rejections[chunk.id]='invalid_chunk';continue
            fingerprint='\n'.join(x.rstrip() for x in chunk.body.strip().splitlines())
            if fingerprint in content_seen:
                self.rejections[chunk.id]='duplicate_content';continue
            pieces=[];offset=0
            for ln,a,b in chunk.source_spans:
                original=chunk.body[offset:offset+(b-a)];offset+=b-a
                remaining=[(a,b)]
                for x,y in covered[(chunk.path,ln)]:
                    remaining=[piece for lo,hi in remaining for piece in ([(lo,min(hi,x))] if lo<x else [])+([(max(lo,y),hi)] if hi>y else []) if piece[1]>piece[0]]
                for lo,hi in remaining:
                    pieces.append(((ln,lo,hi),original[lo-a:hi-a]))
            if not pieces or not ''.join(s for _,s in pieces).strip():
                self.rejections[chunk.id]='duplicate_content';continue
            header=context_header(chunk.path,chunk.qualname,pieces[0][0][0],pieces[-1][0][0],chunk.signature)
            entry_start=length;parts.append(header);length+=len(header);record=[]
            for span,payload in pieces:
                parts.append(payload);length+=len(payload);record.append((span,payload,length))
                if allow_partial:self.boundaries.append(length)
                covered[(chunk.path,span[0])].append((span[1],span[2]))
            parts.append('\n\n');length+=2;self.boundaries.append(length)
            self.entries.append(StreamEntry(candidate,header,record,entry_start,length))
            content_seen.add(fingerprint)
        self.text=''.join(parts)
        # Prefix costs are monotonic for the documented UTF-8 counter.
        self.costs=[];last=0;bytes_so_far=0
        for boundary in self.boundaries:
            bytes_so_far+=len(self.text[last:boundary].encode('utf-8'));last=boundary
            self.costs.append((bytes_so_far+3)//4)

    def pack(self,budget):
        if budget<=0:raise ValueError('budget must be positive')
        end=self.boundaries[bisect_right(self.costs,budget)-1]
        text=self.text[:end];hits=[];selected={};previous_cost=0
        for entry in self.entries:
            included=[(span,payload) for span,payload,stop in entry.pieces if stop<=end]
            if not included:break
            chunk=entry.candidate.chunk
            entry_end=min(end,entry.end)
            prefix_cost=token_count(self.text[:entry_end])
            cost=prefix_cost-previous_cost;previous_cost=prefix_cost
            spans=tuple(s for s,_ in included);body=''.join(p for _,p in included)
            selected_chunk=replace(chunk,text=self.text[entry.start:entry_end],tokens_est=cost,
                                   source_spans=spans,line_start=spans[0][0],line_end=spans[-1][0],body=body)
            hits.append(Hit(selected_chunk,entry.candidate.score,'hybrid evidence; fixed stream prefix'))
            selected[chunk.id]={'partial':len(included)<len(entry.pieces),'emitted_spans':spans}
        return text,hits,selected
