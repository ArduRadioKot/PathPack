from pathlib import Path
import sys, json
root=Path(__file__).parent
sys.path.insert(0,str(root/'src'))
from codegraph_rag.parser import scan_repository
from codegraph_rag.retrieval import make_chunks, select_context
from codegraph_rag.audit import audit_predictions

snapshot=root/'examples/issue_demo/snapshot'
issues=[('demo-login','Login cookie causes failure','store.py'),
        ('demo-invoice','Checkout invoice amount is incorrect','taxes.py'),
        ('demo-upload','Upload image causes crash','filesys.py'),
        ('demo-notify','Send notification fails','packets.py')]
output=root/'reports/v03_demo'; output.mkdir(exist_ok=True)
graph=scan_repository(snapshot)
chunks=make_chunks(snapshot,graph)
gold=output/'gold_demo.jsonl'
with gold.open('w',encoding='utf8') as f:
 for iid,q,p in issues:
  f.write(json.dumps({'instance_id':iid,'gold_python_files':[p]},ensure_ascii=False)+'\n')
paths=[]
for method in ('bm25','graph-expand','pathpack','pathpack-v3'):
 for budget in (45,70,110,160):
  dst=output/f'{method}_{budget}.jsonl'
  with dst.open('w',encoding='utf8') as f:
   for iid,q,p in issues:
    hits=select_context(chunks,graph,q,method=method,budget=budget,seeds=1,hops=2)
    f.write(json.dumps({'instance_id':iid,'method':method,'budget_est':budget,
                       'selected':[hit.chunk.metadata() for hit in hits],
                       'tokens_est':sum(h.chunk.tokens_est for h in hits)},ensure_ascii=False)+'\n')
  paths.append(dst)
report=audit_predictions(paths,gold,iterations=200)
(output/'audit_demo.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print('TOY ONLY, not SWE-bench:',json.dumps(report['groups'],indent=2))
