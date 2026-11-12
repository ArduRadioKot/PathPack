"""Gold-only source coverage metrics. Never imported by chunking/ranking/packing."""
from __future__ import annotations
import ast
from collections import defaultdict
from pathlib import Path
import math,re

_HUNK=re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')


def patch_events(patch):
    files=[];current=None;line=None;removed=set();anchors=set();added=[]
    def flush():
        if current is not None and added and not removed:
            current['insertions'].append({'anchors':sorted(anchors),'added':list(added)})
        removed.clear();anchors.clear();added.clear()
    for text in patch.splitlines():
        if text.startswith('diff --git '):flush();current=None;line=None
        elif text.startswith('--- '):
            flush()
            path=text[6:] if text.startswith('--- a/') else None
            current={'path':path,'new_path':None,'removed':set(),'insertions':[]}
            files.append(current);line=None
        elif text.startswith('+++ ') and current is not None:
            current['new_path']=text[6:] if text.startswith('+++ b/') else None
        elif text.startswith('@@ '):
            flush();m=_HUNK.match(text);line=int(m.group(1)) if m else None
        elif current is not None and line is not None:
            if text.startswith('-'):
                current['removed'].add(line);removed.add(line);line+=1
            elif text.startswith('+'):
                added.append(text[1:]);anchors.update((max(0,line-1),max(1,line)))
            elif text.startswith(' '):flush();line+=1
    flush();return files


def source_functions(text,path):
    tree=ast.parse(text);lines=text.splitlines();records=[]
    def visit(node,scope=()):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            q='.'.join((*scope,node.name))
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                start=min([node.lineno,*[d.lineno for d in node.decorator_list]])
                owned=set(range(node.body[0].lineno,node.end_lineno+1))
                nested=[c for c in ast.walk(node) if c is not node and isinstance(c,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
                for child in nested:
                    first=min([child.lineno,*[d.lineno for d in getattr(child,'decorator_list',[])]])
                    owned.difference_update(range(first,child.end_lineno+1))
                if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                    owned.difference_update(range(node.body[0].lineno,node.body[0].end_lineno+1))
                owned={ln for ln in owned if lines[ln-1].strip() and not lines[ln-1].lstrip().startswith('#')}
                records.append({'id':f'{path}::{q}','path':path,'qualname':q,'start':start,'end':node.end_lineno,'body_lines':sorted(owned)})
            for child in node.body:visit(child,(*scope,node.name))
        else:
            for child in ast.iter_child_nodes(node):visit(child,scope)
    visit(tree)
    counts=defaultdict(int)
    for record in records:counts[record['id']]+=1
    for record in records:
        record['graph_id']=record['id']
        if counts[record['id']]>1:record['id']+=f'::definition_line_{record["start"]}'
    return records


def label_patch(patch,repository):
    root=Path(repository).resolve();targets={};files=set();changed=defaultdict(set);excluded=[];source={}
    for event in patch_events(patch):
        rel=event['path'];new=event['new_path']
        if rel is None:
            if new and new.endswith('.py'):excluded.append({'path':new,'reason':'new_file_no_base_source'})
            continue
        if not rel.endswith('.py'):continue
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            excluded.append({'path':rel,'reason':'missing_base_file'});continue
        files.add(rel)
        try:
            text=path.read_text(encoding='utf-8-sig');records=source_functions(text,rel);source[rel]=text.splitlines(keepends=True)
        except (OSError,UnicodeError,SyntaxError,ValueError) as exc:
            excluded.append({'path':rel,'reason':'parse_error','detail':str(exc)[:200]});continue
        removed=event['removed'];changed[rel].update(removed)
        def owner_at(ln):
            matches=[r for r in records if r['start']<=ln<=r['end']]
            return min(matches,key=lambda r:(r['end']-r['start'],r['id'])) if matches else None
        def add_owner(owner,ln):
            if owner and owner['body_lines']:targets[owner['id']]=owner
            elif owner:excluded.append({'path':rel,'line':ln,'reason':'no_executable_body'})
            else:excluded.append({'path':rel,'line':ln,'reason':'module_level_change'})
        for ln in sorted(removed):add_owner(owner_at(ln),ln)
        for insertion in event['insertions']:
            if any(re.match(r'^\s*(?:async\s+)?(?:def|class)\s+',line) for line in insertion['added']):
                excluded.append({'path':rel,'reason':'new_definition_no_base_function'});continue
            owners=[owner_at(ln) for ln in insertion['anchors']]
            if owners and all(o is not None and o['id']==owners[0]['id'] for o in owners):
                add_owner(owners[0],insertion['anchors'][0])
                excluded.append({'path':rel,'reason':'insertion_only_no_exact_old_changed_line'})
            else:excluded.append({'path':rel,'reason':'insertion_boundary_unmapped'})
    return {'gold_python_files':sorted(files),'functions':targets,'changed_lines':{p:sorted(v) for p,v in changed.items()},
            'exclusions':excluded,'source_lines':source,'label_completeness':'partial' if excluded else 'complete'}


def covered_lines(selected,source):
    spans=defaultdict(list)
    for c in selected:
        # Headers/names are never used to infer coverage.
        for ln,a,b in c.get('source_spans',[]):spans[(c['path'],int(ln))].append((int(a),int(b)))
    covered=set()
    for (path,ln),ranges in spans.items():
        lines=source.get(path,[])
        if not 1<=ln<=len(lines):continue
        raw=lines[ln-1];lo=len(raw)-len(raw.lstrip());hi=len(raw.rstrip())
        if lo>=hi:continue
        cursor=lo
        for a,b in sorted(ranges):
            if a>cursor:break
            cursor=max(cursor,b)
        if cursor>=hi:covered.add((path,ln))
    return covered


def score_prediction(prediction,gold,*,function_coverage_threshold=0.20):
    selected=prediction['selected'];paths={c['path'] for c in selected};files=set(gold['gold_python_files'])
    covered=covered_lines(selected,gold['source_lines']);found=set();complete=set();fractions={}
    for ident,fn in gold['functions'].items():
        body=set(fn['body_lines']);n=sum((fn['path'],ln) in covered for ln in body)
        fractions[ident]=n/len(body)
        if n>=max(1,math.ceil(function_coverage_threshold*len(body))):found.add(ident)
        if n==len(body):complete.add(ident)
    changed={(p,ln) for p,ls in gold['changed_lines'].items() for ln in ls
             if ln<=len(gold['source_lines'].get(p,[])) and gold['source_lines'][p][ln-1].strip()}
    used=prediction['tokens_est'];budget=prediction.get('budget_est',prediction.get('budget'))
    return {'instance_id':prediction['instance_id'],'method':prediction['method'],'budget':budget,
            'file_recall':len(paths & files)/len(files) if files else None,
            'function_recall':len(found)/len(gold['functions']) if gold['functions'] else None,
            'complete_function_recall':len(complete)/len(gold['functions']) if gold['functions'] else None,
            'changed_line_recall':len(changed & covered)/len(changed) if changed else None,
            'latency_sec':prediction['latency_sec'],'tokens_est':used,'token_utilization':used/budget,
            'useful_functions_per_1k':(len(found)*1000/used if used else 0.0) if gold['functions'] else None,
            'gold_files':len(files),'gold_functions':len(gold['functions']),'found_functions':sorted(found),
            'function_coverage':fractions,'changed_lines_evaluable':len(changed),
            'label_completeness':gold['label_completeness']}


def diagnose_failures(prediction,gold,score,candidates,graph=None):
    found=set(score['found_functions']);missed=set(gold['functions'])-found
    missed_graph_ids={gold['functions'][fn].get('graph_id',fn) for fn in missed}
    selected_paths={c['path'] for c in prediction['selected']};flags=[]
    if selected_paths & set(gold['gold_python_files']) and missed:flags.append('correct_file_missed_function')
    if any(c.get('function_id') in missed_graph_ids and c.get('order') is not None and c.get('decision') in {'budget_exceeded','after_budget_cutoff'} for c in candidates):flags.append('ranked_target_did_not_fit')
    chosen=[c for c in candidates if c.get('selected')]
    if chosen and sum(c.get('lexical_score', 0)<.1 and c.get('symbol_score', 0)==0 and c.get('graph_score', 0)==0 for c in chosen)/len(chosen)>.5:flags.append('mostly_low_evidence_context')
    selected_fn={c.get('function_id') for c in prediction['selected']}
    if graph:
        neighbors={e.target for e in graph.edges if e.source in selected_fn}|{e.source for e in graph.edges if e.target in selected_fn}
        if any(gold['functions'][fn].get('graph_id',fn) in neighbors and gold['functions'][fn]['path'] not in selected_paths for fn in missed):flags.append('target_file_missed_selected_graph_neighbor')
    covered=covered_lines(prediction['selected'],gold['source_lines'])
    for fn in gold['functions'].values():
        if any(c.get('function_id')==fn.get('graph_id',fn['id']) for c in prediction['selected']):
            changed=set(gold['changed_lines'].get(fn['path'],[])) & set(fn['body_lines'])
            if changed and not any((fn['path'],ln) in covered for ln in changed):
                flags.append('partial_function_missed_changed_lines');break
    return sorted(set(flags))
