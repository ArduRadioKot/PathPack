"""AST boundaries and function ownership. Function Graph stays unchanged."""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from .config import V4Config
from .graph import CodeGraph
from .retrieval import Chunk, estimated_tokens


def production_path(path):
    p=Path(path)
    return not (any(x in {'test','tests'} for x in p.parts) or p.name.startswith('test_') or p.name.endswith('_test.py'))


def context_header(path,qualname,start,end,signature=''):
    # Signature is reading context, never evidence of source-line coverage.
    sig=' '.join(signature.split())
    if len(sig)>160: sig=sig[:157]+'...'
    return f'# File: {path}\n# Function: {qualname}\n# Lines: {start}-{end}\n'+(f'# Signature: {sig}\n' if sig else '')


@dataclass
class FunctionInfo:
    id: str
    path: str
    qualname: str
    kind: str
    line_start: int
    line_end: int
    signature: str
    parent_id: str | None
    chunks: list[str] = field(default_factory=list)
    dependency_ids: list[str] = field(default_factory=list)
    definitions: list[dict] = field(default_factory=list)


@dataclass
class ChunkIndex:
    chunks: list[Chunk]
    functions: dict[str,FunctionInfo]
    errors: dict[str,str]


def build_chunk_index(repository, graph: CodeGraph, config: V4Config | None=None):
    cfg=config or V4Config();root=Path(repository).resolve()
    chunks=[];functions={};errors=dict(graph.errors);chunk_numbers=defaultdict(int)
    by_file=defaultdict(dict)
    for node in graph.nodes.values():
        if production_path(node.path): by_file[node.path][node.qualname]=node
    links=defaultdict(set)
    for e in graph.edges:
        links[e.source].add(e.target);links[e.target].add(e.source)
    for rel,symbols in sorted(by_file.items()):
        file=(root/rel).resolve()
        if not file.is_relative_to(root) or file.is_symlink():
            errors[rel]='Unsafe source path';continue
        try:
            text=file.read_text(encoding='utf-8-sig');lines=text.splitlines(keepends=True);tree=ast.parse(text)
        except (OSError,UnicodeError,SyntaxError,ValueError) as exc:
            errors[rel]=str(exc);continue
        found=[]
        def visit(node,scope=()):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                q='.'.join((*scope,node.name))
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and q in symbols:
                    found.append((node,q,'.'.join(scope) if scope else None))
                for child in node.body: visit(child,(*scope,node.name))
            else:
                for child in ast.iter_child_nodes(node):visit(child,scope)
        visit(tree)
        if not found:
            module=symbols.get('__module__')
            if module:
                found=[(tree,'__module__',None)]
        for node,q,parent in found:
            sym=symbols.get(q)
            if not sym:continue
            is_func=isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
            start=min([getattr(node,'lineno',1),*[d.lineno for d in getattr(node,'decorator_list',[])]])
            end=getattr(node,'end_lineno',len(lines)) or len(lines)
            signature=(('async ' if isinstance(node,ast.AsyncFunctionDef) else '')+f'def {node.name}({ast.unparse(node.args)})'+(' -> '+ast.unparse(node.returns) if node.returns else '')+':') if is_func else ''
            info=FunctionInfo(sym.id,rel,q,sym.kind,start,end,signature,f'{rel}::{parent}' if parent else None,dependency_ids=sorted(links[sym.id]))
            previous=functions.get(sym.id)
            if previous:
                info.chunks=list(previous.chunks)
                info.definitions=list(previous.definitions)
                errors[sym.id]='Multiple AST definitions share the legacy function-graph symbol; chunk ids remain unique and source ranges distinguish definitions'
            info.definitions.append({'line_start':start,'line_end':end,'signature':signature,'parent_id':info.parent_id})
            functions[sym.id]=info
            excluded=set()
            # Every source line has one function owner, including decorators.
            for child,cq,_ in found:
                if is_func and cq.startswith(q+'.'):
                    cs=min([child.lineno,*[d.lineno for d in getattr(child,'decorator_list',[])]])
                    excluded.update(range(cs,child.end_lineno+1))
            owned=set(range(start,end+1))-excluded
            def spans_for(lo,hi):return [(ln,0,len(lines[ln-1])) for ln in range(lo,hi+1) if ln in owned and ln<=len(lines)]
            def body(spans):return ''.join(lines[ln-1][a:b] for ln,a,b in spans)
            def cost(spans):
                if not spans:return 0
                return estimated_tokens(context_header(rel,q,spans[0][0],spans[-1][0],signature)+body(spans)+'\n\n')
            whole=spans_for(start,end)
            if not whole or not body(whole).strip():continue
            groups=[]
            def split_ast(stmt):
                lo=getattr(stmt,'lineno',start);hi=getattr(stmt,'end_lineno',end)
                current=spans_for(lo,hi)
                if not current:return []
                if cost(current)<=cfg.max_chunk_tokens:return [current]
                children=[]
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child,ast.stmt):children.append(child)
                    elif isinstance(child,ast.ExceptHandler):children.extend(child.body)
                children=sorted(children,key=lambda x:x.lineno)
                result=[];cursor=lo
                for child in children:
                    if child.lineno<cursor:continue
                    if child.lineno>cursor:result.append(spans_for(cursor,child.lineno-1))
                    result.extend(split_ast(child));cursor=child.end_lineno+1
                if cursor<=hi:result.append(spans_for(cursor,hi))
                return [x for x in result if x] if children else [current]
            if not cfg.enable_chunking or cost(whole)<=cfg.max_chunk_tokens:
                groups=[whole]
            else:
                first_body=node.body[0].lineno if node.body else end+1
                units=[spans_for(start,first_body-1)]
                for stmt in node.body:units.extend(split_ast(stmt))
                atoms=[]
                for unit in units:
                    if not unit:continue
                    if cost(unit)<=cfg.max_chunk_tokens:atoms.append(unit);continue
                    for span in unit:
                        if cost([span])<=cfg.max_chunk_tokens:atoms.append([span]);continue
                        ln,a,b=span
                        while a<b:
                            # Fall back to source columns for a single huge line.
                            lo,hi=a+1,b;fit=a
                            while lo<=hi:
                                mid=(lo+hi)//2
                                if cost([(ln,a,mid)])<=cfg.max_chunk_tokens:fit=mid;lo=mid+1
                                else:hi=mid-1
                            if fit==a:
                                errors[f'{rel}::{q}']='Header leaves no space for source under max_chunk_tokens';break
                            atoms.append([(ln,a,fit)]);a=fit
                pending=[]
                for atom in atoms:
                    if pending and cost(pending+atom)>cfg.target_chunk_tokens:
                        groups.append(pending);pending=[]
                    pending+=atom
                if pending:
                    if groups and cost(pending)<cfg.min_chunk_tokens and cost(groups[-1]+pending)<=cfg.max_chunk_tokens:groups[-1]+=pending
                    else:groups.append(pending)
                # Optional index overlap; the packer removes repeated source spans.
                if cfg.chunk_overlap_lines:
                    original=list(groups)
                    for i in range(1,len(groups)):
                        overlap=original[i-1][-cfg.chunk_overlap_lines:]
                        if cost(overlap+groups[i])<=cfg.max_chunk_tokens:groups[i]=overlap+groups[i]
            for number,spans in enumerate(groups):
                if not spans or not body(spans).strip():continue
                payload=body(spans);header=context_header(rel,q,spans[0][0],spans[-1][0],signature)
                serialized=header+payload+'\n\n'
                if cfg.enable_chunking and estimated_tokens(serialized)>cfg.max_chunk_tokens:
                    errors[f'{sym.id}::chunk_{number}']='Chunk exceeds configured max';continue
                cid=f'{sym.id}::chunk_{chunk_numbers[sym.id]}'
                chunk_numbers[sym.id]+=1
                chunks.append(Chunk(cid,rel,q,sym.kind,serialized,estimated_tokens(serialized),sym.id,spans[0][0],spans[-1][0],tuple(spans),signature,info.parent_id,payload))
                info.chunks.append(cid)
    return ChunkIndex(chunks,functions,errors)
