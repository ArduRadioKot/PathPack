"""Dev comparisons for PathPack v0.5. Scoring stays outside retrieval."""
from __future__ import annotations

from collections import defaultdict

from .config import V5_ABLATIONS

COMPARISON_METRICS=('file_recall','function_recall','complete_function_recall','changed_line_recall','relevant_line_density')
PROFILE_KEYS=('candidate_sec','graph_sec','file_rank_sec','chunk_score_sec','diversity_sec','anchor_sec','pack_stream_sec','prefix_sec','query_sec')


def comparisons(groups,budgets,bootstrap):
    """Paired task-level bootstrap of v0.5 and v0.5.1 against every other method."""
    rows=[]
    for candidate in ('pathpack-v5','pathpack-v51','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10','pathpack-v10/coverage','pathpack-v10/paths','pathpack-v10/no-structural','pathpack-v10/nested','pathpack-v10/no-semantic'):
        for budget in budgets:
            target=groups.get((candidate,budget))
            if not target:
                continue
            by_id={row['instance_id']:row for row in target}
            for (method,method_budget),other in sorted(groups.items()):
                if method_budget!=budget or method==candidate:
                    continue
                for metric in COMPARISON_METRICS:
                    left={iid:row[metric] for iid,row in by_id.items() if row.get(metric) is not None}
                    right={row['instance_id']:row[metric] for row in other if row.get(metric) is not None}
                    rows.append({'candidate':candidate,'baseline':method,'budget':budget,'metric':metric,
                                 **bootstrap(left,right,iterations=2000,seed=42)})
    return rows


def profile_table(predictions):
    """One query profile per instance and method, taken at the largest budget."""
    grouped=defaultdict(list)
    for row in predictions:
        if not str(row['method']).startswith(('pathpack-v5','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10')) or 'profile' not in row:
            continue
        grouped[(row['instance_id'],row['method'])].append(row)
    table=[]
    for (iid,method),rows in sorted(grouped.items()):
        row=max(rows,key=lambda item:item['budget_est'])
        profile=row['profile']
        table.append({'instance_id':iid,'method':method,'budget':row['budget_est'],
                      **{key:profile.get(key) for key in PROFILE_KEYS},
                      'files_selected':profile.get('files_selected'),'pool':profile.get('pool'),
                      'chunks_indexed':profile.get('chunks_indexed')})
    return table


def mean_profile(table):
    if not table:
        return []
    grouped=defaultdict(list)
    for row in table:
        grouped[row['method']].append(row)
    means=[]
    for method,rows in sorted(grouped.items()):
        item={'method':method,'queries':len(rows)}
        for key in PROFILE_KEYS:
            values=[row[key] for row in rows if row.get(key) is not None]
            item[key]=sum(values)/len(values) if values else None
        means.append(item)
    return means


ABLATION_METHODS=tuple('pathpack-v5' if name=='full' else f'pathpack-v5/{name}' for name in V5_ABLATIONS)
