"""Figures from persisted predictions, graph paths and measured metrics only."""
import csv,gzip,json
from pathlib import Path
from .experiments import write_json,write_csv


def visualize_run(run,out=None,*,instance_id=None,budget=2048):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    run=Path(run);out=Path(out) if out else run/'figures';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((run/'run.json').read_text());iid=instance_id or meta['issues'][0]['instance_id']
    diag_path=run/'diagnostics'/f'{iid}_full_{budget}.json.gz'
    shown='pathpack-v4'
    if not diag_path.exists():
        diag_path=run/'diagnostics'/f'{iid}_v5full_{budget}.json.gz';shown='pathpack-v5'
    with gzip.open(diag_path,'rt') as f:diag=json.load(f)
    with gzip.open(run/f'{iid}_graph.json.gz','rt') as f:graph=json.load(f)
    lookup={n['id']:n for n in graph['nodes']};seeds=set(diag['seeds']);chosen=[c for c in diag['candidates'] if c['selected']]
    selected={c['function_id'] for c in chosen};extra=[c for c in sorted(diag['candidates'],key=lambda c:(c['order'] if c['order'] is not None else 10**9,-c['score'],c['id'])) if c['function_id'] not in selected][:6]
    nodes=set(seeds)|set(sorted(selected)[:12])|{c['function_id'] for c in extra}
    routes=[]
    for c in chosen:
        if c['graph_path'] and len(nodes|set(c['graph_path']))<=32:nodes.update(c['graph_path']);routes.append(c['graph_path'])
    g=nx.DiGraph();g.add_nodes_from(sorted(nodes))
    for edge in graph['edges']:
        if edge['source'] in nodes and edge['target'] in nodes:g.add_edge(edge['source'],edge['target'],kind=edge['kind'])
    pos=nx.spring_layout(g,seed=42,k=1.4)
    fig,ax=plt.subplots(figsize=(15,10))
    rejected={c['function_id'] for c in extra}
    colors=['#2878b5' if n in seeds else '#36a269' if n in selected else '#c8c8c8' if n in rejected else '#efd28a' for n in g]
    labels={n:lookup.get(n,{}).get('qualname',n.split('::')[-1])[-48:] for n in g}
    nx.draw_networkx(g,pos,ax=ax,labels=labels,node_color=colors,node_size=1000,font_size=7,arrowsize=13,edge_color='#bbbbbb')
    path_edges=[(a,b) if g.has_edge(a,b) else (b,a) for route in routes for a,b in zip(route,route[1:]) if g.has_edge(a,b) or g.has_edge(b,a)]
    if path_edges:nx.draw_networkx_edges(g,pos,edgelist=path_edges,ax=ax,edge_color='#e28d28',width=2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2878b5',label='BM25 seeds'),Patch(color='#36a269',label='Selected functions'),Patch(color='#c8c8c8',label='Rejected candidates'),Patch(color='#efd28a',label='Graph bridge nodes'),Patch(color='#e28d28',label='Evidence paths')],loc='upper left')
    ax.set_title(f'{iid} | {shown} | budget={budget} (UTF-8 proxy)');ax.axis('off');fig.tight_layout();fig.savefig(out/'function_graph.png',dpi=160);fig.savefig(out/'function_graph.svg');plt.close(fig)
    predictions=[json.loads(s) for s in (run/f'{iid}_predictions.jsonl').read_text().splitlines()]
    prediction=next(p for p in predictions if p['method']==shown and p['budget_est']==budget)
    fig,ax=plt.subplots(figsize=(12,3.5));left=0
    for i,c in enumerate(prediction['selected']):
        ax.barh(['context'],[c['tokens_est']],left=left,color=plt.get_cmap('tab20')(i%20),label=c['qualname'][-35:] if i<8 else None);left+=c['tokens_est']
    ax.barh(['context'],[budget-left],left=left,color='#eeeeee',label='unused budget');ax.set_xlim(0,budget);ax.set_xlabel('Estimated serialized-context tokens');ax.set_title(f'{iid}: {left}/{budget} tokens, {len(prediction["selected"])} chunks');ax.legend(fontsize=6,bbox_to_anchor=(1.01,1));fig.tight_layout();fig.savefig(out/'budget_allocation.png',dpi=160);plt.close(fig)
    write_csv(out/'example_selection.csv',prediction['selected'])
    write_json(out/'example_paths.json',{'instance_id':iid,'seeds':sorted(seeds),'selected':sorted(selected),'paths':routes})
    if (run/'summary.csv').exists():
        with (run/'summary.csv').open() as f:summary=list(csv.DictReader(f))
        methods=('bm25','graph-expand','pathpack','pathpack-v3','pathpack-v4','pathpack-v5')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,metric in zip(axes,('file_recall','function_recall')):
            for method in methods:
                values=sorted((int(r['budget']),float(r[metric])) for r in summary if r['method']==method and r[metric])
                if values:ax.plot([x for x,y in values],[y for x,y in values],marker='o',label=method)
            ax.set_title(metric.replace('_',' ').title());ax.set_xlabel('Estimated token budget');ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'Measured {meta["split"]}, n={len(meta["issues"])}; function recall requires 20% owned body-line coverage')
        fig.tight_layout();fig.savefig(out/'recall_vs_budget.png',dpi=170);fig.savefig(out/'recall_vs_budget.svg');plt.close(fig)
        ablations=[r for r in summary if r['method'].startswith('pathpack-v4') and int(r['budget'])==2048]
        fig,ax=plt.subplots(figsize=(10,4.5));x=list(range(len(ablations)))
        ax.bar([i-.18 for i in x],[float(r['file_recall'] or 0) for r in ablations],width=.36,label='File recall')
        ax.bar([i+.18 for i in x],[float(r['function_recall'] or 0) for r in ablations],width=.36,label='Function recall')
        ax.set_xticks(x,[r['method'].replace('pathpack-v4/','').replace('pathpack-v4','full') for r in ablations],rotation=15);ax.set_ylim(0,1);ax.legend();ax.set_title('Measured ablations @2048');fig.tight_layout();fig.savefig(out/'ablations_2048.png',dpi=160);plt.close(fig)
        v5=[r for r in summary if r['method'].startswith('pathpack-v5') and int(r['budget'])==2048]
        if v5:
            fig,ax=plt.subplots(figsize=(10,4.5));x=list(range(len(v5)))
            ax.bar([i-.18 for i in x],[float(r['file_recall'] or 0) for r in v5],width=.36,label='File recall')
            ax.bar([i+.18 for i in x],[float(r['function_recall'] or 0) for r in v5],width=.36,label='Function recall')
            ax.set_xticks(x,['full' if r['method']=='pathpack-v5' else r['method'].replace('pathpack-v5/','') for r in v5],rotation=15);ax.set_ylim(0,1);ax.legend();ax.set_title('Measured v0.5 ablations @2048');fig.tight_layout();fig.savefig(out/'v05_ablations_2048.png',dpi=160);plt.close(fig)
    return {'figures':str(out),'instance_id':iid,'seed_nodes':len(seeds),'selected_functions':len(selected),'chunks':len(chosen)}
