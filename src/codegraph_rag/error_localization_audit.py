"""Post-hoc localization audit. Gold is read only after retrieval has finished."""
from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .benchmark import read_jsonl
from .experiments import write_json
from .line_metrics import build_line_label, evaluable_changed, extra_metrics


def _spans(chunk):
    return [(chunk.get('path'),int(line)) for line,_start,_stop in chunk.get('source_spans') or []]


def classify_observation(obs):
    """Return case letters. Thresholds are diagnostic rules, not trained cutoffs.

    A correct file, function below the 20% body rule.
    B function body partially emitted, changed lines inside it missed.
    C a chunk overlapping the gold body or changed lines was ranked and then
      cut by the budget or the candidate cap, and its score is at least the
      median score of selected chunks (or it never entered the ordered pool).
    D a selected chunk from a gold file does not touch the gold body or the
      evaluable changed lines.
    E the miss is explained by the label: no evaluable changed lines, an
      unmapped insertion, a new file, or changed lines that lie outside every
      function body while no function miss remains.
    F the gold function has graph score 0. Subtype ``absent`` means the saved
      graph has no incident edge; ``unreached`` means edges exist but this
      query's walk did not score them.
    """
    cases=[]
    selected=set(obs['selected_paths']);gold_files=set(obs['gold_files']);found=set(obs['found_functions'])
    covered=set(map(tuple,obs['covered']));changed=set(map(tuple,obs['changed']))
    needed_body=set()
    for fn in obs['functions'].values():
        for line in fn['body_lines']:
            needed_body.add((fn['path'],line))
    gold_touch=needed_body|changed
    if selected & gold_files and any(fn_id not in found for fn_id in obs['functions']):
        cases.append('A')
    partial=False
    for fn in obs['functions'].values():
        body={(fn['path'],line) for line in fn['body_lines']}
        own_changed=changed & {(fn['path'],line) for line in fn['body_lines']}
        if not body:
            continue
        hit=len(body & covered)
        if 0<hit<len(body) and own_changed and not own_changed<=covered:
            partial=True
        elif fn['id'] in found and own_changed and not own_changed<=covered:
            partial=True
    if partial:
        cases.append('B')
    selected_rows=[c for c in obs['candidates'] if c.get('selected')]
    score_floor=median([c['score'] for c in selected_rows if c.get('score') is not None]) if selected_rows else 0.0
    lexical_values=[c['lexical_score'] for c in selected_rows if c.get('lexical_score') is not None]
    lexical_floor=median(lexical_values) if lexical_values else 0.0
    high_cut=False
    for cand in obs['candidates']:
        ident=cand.get('function_id')
        owns=ident in obs['functions'] or any(fn.get('graph_id')==ident for fn in obs['functions'].values())
        overlap=bool(set(_spans(cand)) & gold_touch) or owns
        if not overlap:
            continue
        decision=cand.get('decision') or cand.get('reason')
        if decision not in {'budget_exceeded','after_budget_cutoff','candidate_limit'}:
            continue
        lexical=cand.get('lexical_score') or 0.0
        score=cand.get('score') or 0.0
        if (cand.get('order') is not None and score>=score_floor) or lexical>=lexical_floor:
            high_cut=True
    if high_cut:
        cases.append('C')
    if any(c.get('selected') and c.get('path') in gold_files and not (set(_spans(c)) & gold_touch) for c in obs['candidates']):
        cases.append('D')
    label_gap=obs.get('changed_line_recall') is None or bool(obs.get('changed_outside_functions'))
    if obs.get('label_completeness')!='complete' and (obs.get('changed_line_recall') is None or obs.get('changed_outside_functions')):
        label_gap=True
    if any(item.get('reason') in {'new_file_no_base_source','insertion_anchor_unmapped','insertion_boundary_unmapped'} for item in obs.get('exclusions',[])):
        if obs.get('changed_line_recall') is None:
            label_gap=True
    if label_gap:
        cases.append('E')
    for fn in obs['functions'].values():
        gid=fn.get('graph_id',fn['id'])
        related=[c for c in obs['candidates'] if c.get('function_id')==gid]
        graph_score=max((c.get('graph_score') or 0) for c in related) if related else 0
        if graph_score==0:
            degree=obs.get('graph_degree',{}).get(gid,0)
            cases.append('F')
            obs['graph_subtype']='absent' if degree==0 else 'unreached'
            break
    return cases


def _load_prediction(run,instance_id,method,budget):
    rows=read_jsonl(run/f'{instance_id}_predictions.jsonl')
    return next(row for row in rows if row['method']==method and int(row['budget_est'])==int(budget))


def _load_diag(run,instance_id,method,budget):
    tag='full' if method=='pathpack-v4' else 'v5full' if method=='pathpack-v5' else method.split('/',1)[-1]
    if method.startswith('pathpack-v5/') :
        tag='v5_'+method.split('/',1)[1]
    path=run/'diagnostics'/f'{instance_id}_{tag}_{budget}.json.gz'
    if not path.exists():
        return {'candidates':[],'files':[]}
    import gzip
    with gzip.open(path,'rt') as handle:
        return json.load(handle)


def _covered_from_prediction(prediction,source):
    from .evaluation_v4 import covered_lines
    return covered_lines(prediction['selected'],source)


def audit_run(run,out,*,method='pathpack-v4',budget=2048,limit=10,raw=None):
    from .evaluation_v4 import score_prediction
    from .experiments import load_scoring_dataset
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((run/'run.json').read_text())
    issues={row['instance_id']:row for row in meta['issues']}
    labels={row['instance_id']:row for row in read_jsonl(run/'scoring_labels.jsonl')} if (run/'scoring_labels.jsonl').exists() else {}
    scores={row['instance_id']:row for row in read_jsonl(run/'scores.jsonl') if row['method']==method and int(row['budget'])==int(budget)} if (run/'scores.jsonl').exists() else {}
    source_rows={row['instance_id']:row for row in load_scoring_dataset(raw)}
    observations=[]
    for issue in meta['issues']:
        iid=issue['instance_id']
        repo=Path(meta['snapshots'])/iid
        patch=source_rows[iid]['patch']
        gold=build_line_label(patch,repo)
        prediction=_load_prediction(run,iid,method,budget)
        score=scores.get(iid) or score_prediction(prediction,gold)
        diag=_load_diag(run,iid,method,budget)
        covered=_covered_from_prediction(prediction,gold['source_lines'])
        degree=defaultdict(int)
        graph_path=run/f'{iid}_graph.json.gz'
        if graph_path.exists():
            import gzip
            with gzip.open(graph_path,'rt') as handle:
                graph=json.load(handle)
            for edge in graph['edges']:
                degree[edge['source']]+=1;degree[edge['target']]+=1
        changed=evaluable_changed(gold)
        obs={'instance_id':iid,'selected_paths':sorted({c['path'] for c in prediction['selected']}),
             'gold_files':gold['gold_python_files'],'found_functions':score.get('found_functions',[]),
             'functions':gold['functions'],'covered':sorted(covered),'changed':sorted(changed),
             'candidates':diag.get('candidates',[]),'graph_degree':dict(degree),
             'label_completeness':gold['label_completeness'],'exclusions':gold['exclusions']+gold['anchor_exclusions'],
             'changed_line_recall':score.get('changed_line_recall'),
             'changed_outside_functions':gold['line_audit']['changed_lines_outside_function_bodies'],
             'file_recall':score.get('file_recall'),'function_recall':score.get('function_recall'),
             'prediction':prediction,'gold':gold,'score':score,'problem':issue.get('problem_statement','')}
        obs['cases']=classify_observation(obs)
        obs['extra']=extra_metrics(prediction['selected'],gold)
        observations.append(obs)
    picked=_pick(observations,limit)
    reports=[]
    for obs in picked:
        report=_report(obs,method,budget)
        write_json(out/'cases'/f"{obs['instance_id']}.json",report)
        reports.append(report)
        if report['illustrative']:
            _render(obs,out/'figures',method,budget)
    summary={'run':str(run),'method':method,'budget':budget,'tasks':len(observations),
             'case_counts':dict(Counter(case for obs in observations for case in obs['cases'])),
             'reports':[item['instance_id'] for item in reports],
             'label_audit':_label_summary(observations),
             'note':'Gold patches were read only inside this audit, after the stored retrieval run.'}
    write_json(out/'summary.json',summary)
    write_json(out/'label_audit.json',summary['label_audit'])
    return summary


def _pick(observations,limit):
    remaining=list(observations)
    chosen=[]
    for case in ('B','C','A','D','F','E'):
        for obs in remaining:
            if case in obs['cases'] and obs not in chosen:
                chosen.append(obs);break
        if len(chosen)>=limit:
            return chosen
    for obs in sorted(remaining,key=lambda item:(item.get('function_recall') is None,item.get('function_recall') if item.get('function_recall') is not None else 1,item['instance_id'])):
        if obs in chosen:
            continue
        if obs['cases']:
            chosen.append(obs)
        if len(chosen)>=limit:
            break
    if len(chosen)<limit:
        for obs in remaining:
            if obs not in chosen:
                chosen.append(obs)
            if len(chosen)>=limit:
                break
    return chosen[:limit]


def _report(obs,method,budget):
    gold=obs['gold'];prediction=obs['prediction']
    functions=[]
    for ident,fn in gold['functions'].items():
        gid=fn.get('graph_id',ident)
        related=[c for c in obs['candidates'] if c.get('function_id')==gid]
        related.sort(key=lambda c:(c.get('order') is None,c.get('order') if c.get('order') is not None else 10**9))
        best=related[0] if related else None
        functions.append({'id':ident,'qualname':fn['qualname'],'path':fn['path'],'coverage':obs['score'].get('function_coverage',{}).get(ident),
                          'best_candidate':None if best is None else {key:best.get(key) for key in ('id','order','decision','score','utility','lexical_score','symbol_score','graph_score','line_start','line_end','tokens_est','reason')}})
    text=_narrative(obs,functions)
    return {'instance_id':obs['instance_id'],'method':method,'budget':budget,'cases':obs['cases'],
            'graph_subtype':obs.get('graph_subtype'),'illustrative':True,
            'problem_statement':obs['problem'][:1500],'gold_files':gold['gold_python_files'],
            'gold_functions':functions,'changed_lines':gold['changed_lines'],
            'insertion_anchors':gold['insertion_anchors'],'exclusions':gold['exclusions'],
            'anchor_exclusions':gold['anchor_exclusions'],'line_audit':gold['line_audit'],
            'metrics':{key:obs['score'].get(key) for key in ('file_recall','function_recall','complete_function_recall','changed_line_recall','label_completeness')},
            'extra_metrics':obs['extra'],
            'selected_files':obs['selected_paths'],
            'selected_chunks':[{key:c.get(key) for key in ('id','path','qualname','function_id','line_start','line_end','tokens_est','source_spans')} for c in prediction['selected']],
            'context':prediction.get('context',''),'summary_ru':text}


def _narrative(obs,functions):
    bits=[f"Случаи {', '.join(obs['cases']) or 'без флага'}."]
    metrics=obs['score']
    bits.append(f"File Recall={metrics.get('file_recall')}, Function Recall={metrics.get('function_recall')}, Changed Line Recall={metrics.get('changed_line_recall')}.")
    for fn in functions:
        best=fn['best_candidate']
        if best is None:
            bits.append(f"Для {fn['id']} в диагностике нет чанка (файл мог не войти в пул).")
        else:
            bits.append(f"Лучший чанк {fn['id']}: order={best.get('order')}, decision={best.get('decision') or best.get('reason')}, score={best.get('score')}, lexical={best.get('lexical_score')}, graph={best.get('graph_score')}, строки {best.get('line_start')}-{best.get('line_end')}.")
    if obs.get('graph_subtype'):
        bits.append(f"Граф: {obs['graph_subtype']}.")
    if obs['extra'].get('insertion_anchor_reason'):
        bits.append(f"Якоря вставок: {obs['extra']['insertion_anchor_reason']}, recall={obs['extra'].get('insertion_anchor_recall')}. Это не Changed Line Recall.")
    return ' '.join(bits)


def _label_summary(observations):
    reasons=Counter()
    for obs in observations:
        for item in obs['gold']['exclusions']:
            reasons[item.get('reason','unknown')]+=1
        for item in obs['gold']['anchor_exclusions']:
            reasons['anchor:'+item.get('reason','unknown')]+=1
    return {'tasks':len(observations),
            'complete_labels':sum(obs['gold']['label_completeness']=='complete' for obs in observations),
            'changed_line_evaluable':sum(obs['score'].get('changed_line_recall') is not None for obs in observations),
            'function_evaluable':sum(obs['score'].get('function_recall') is not None for obs in observations),
            'insertion_only_tasks':sum(obs['score'].get('changed_line_recall') is None and obs['gold']['line_audit']['insertion_hunks']>0 for obs in observations),
            'changed_lines_outside_functions':sum(len(obs['gold']['line_audit']['changed_lines_outside_function_bodies']) for obs in observations),
            'removed_lines_out_of_range':sum(len(obs['gold']['line_audit']['removed_lines_out_of_range']) for obs in observations),
            'nested_functions':sum(len(obs['gold']['line_audit']['nested_functions']) for obs in observations),
            'class_methods':sum(len(obs['gold']['line_audit']['class_methods']) for obs in observations),
            'exclusion_reasons':dict(reasons),
            'rules':{'changed_line_recall':'non-empty removed/replaced base lines; pure insertions excluded',
                     'insertion_anchor_recall':'separate; blank skip of one line; not mixed into changed_line_recall',
                     'name_match':'function recall uses owned body lines, not the function id string'}}


def _render(obs,out,method,budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    gold=obs['gold']
    if not gold['gold_python_files']:
        return
    path=gold['gold_python_files'][0]
    lines=gold['source_lines'].get(path,[])
    if not lines:
        return
    changed={line for p,line in evaluable_changed(gold) if p==path}
    bodies={line for fn in gold['functions'].values() if fn['path']==path for line in fn['body_lines']}
    emitted={line for p,line in obs['covered'] if p==path}
    selected={line for chunk in obs['prediction']['selected'] if chunk['path']==path for _p,line in _spans(chunk)}
    interest=set(changed)|set(bodies)|set(selected)
    if not interest:
        interest={1}
    lo=max(1,min(interest)-3);hi=min(len(lines),max(interest)+3)
    rows=[]
    for number in range(lo,hi+1):
        needed=number in changed or number in bodies
        got=number in emitted
        if needed and got:
            kind='found'
        elif needed:
            kind='missed'
        elif number in selected:
            kind='extra'
        else:
            kind='plain'
        rows.append((number,kind,lines[number-1].rstrip('\n')))
    colors={'found':'#1f7a4d','missed':'#b42318','extra':'#b58105','plain':'#444444'}
    fig,ax=plt.subplots(figsize=(12,max(3,0.28*len(rows))))
    ax.set_xlim(0,1);ax.axis('off');ax.set_title(f"{obs['instance_id']} {path} ({method} @{budget})")
    for index,(number,kind,text) in enumerate(rows):
        y=1-(index+1)/(len(rows)+1)
        ax.text(0.01,y,f'{number:>5}  {text[:140]}',color=colors[kind],fontsize=7,family='monospace',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(out/f"{obs['instance_id']}_lines.png",dpi=140);plt.close(fig)
    parts=['<html><head><meta charset="utf-8"><style>',
           'body{font-family:ui-monospace,monospace;background:#f7f6f3;color:#222}',
           '.found{background:#cfe8d8}.missed{background:#f6c9c4}.extra{background:#f6e7b8}',
           '</style></head><body>',
           f'<h1>{html.escape(obs["instance_id"])}</h1>',
           '<p>Зелёный: нужный код найден. Красный: нужный код пропущен. Жёлтый: выбранный фрагмент без пересечения с эталоном.</p>',
           f'<p>Файлы контекста: {html.escape(", ".join(obs["selected_paths"][:24]))}</p>',
           '<pre>']
    for number,kind,text in rows:
        cls='' if kind=='plain' else f' class="{kind}"'
        parts.append(f'<span{cls}>{number:>5}  {html.escape(text)}</span>\n')
    parts.append('</pre></body></html>')
    (out/f"{obs['instance_id']}_lines.html").write_text(''.join(parts),encoding='utf-8')
