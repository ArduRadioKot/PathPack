"""Rebuild notebooks/07_pathpack_v06.ipynb from the integrated v0.6 code and dev CSVs."""
from __future__ import annotations
import base64, io, os, sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/codegraph-mpl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

CELLS = [
("md", """# PathPack v0.6

Первая часть проверяет интеграцию на искусственном репозитории `examples/v06_demo`: длинная функция, нужные слова стоят в конце. Это не результат SWE-bench.

Вторая часть читает прогон `output/v06_dev` на 20 dev-задачах. v0.4 остаётся основной версией: File Recall v0.6 ниже, а прирост Changed Line Recall не выходит за парный интервал, включающий ноль.
"""),
("code", r"""from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if not (ROOT/'src').exists():
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from codegraph_rag.parser import scan_repository
from codegraph_rag.retrieval import retrieve
from codegraph_rag.pathpack_v6 import make_v6_chunks
SRC=ROOT/'examples/v06_demo/snapshot'
graph=scan_repository(SRC)
fine=make_v6_chunks(SRC,graph,target_tokens=80,max_chunk_tokens=140)
print('nodes',len(graph.nodes),'edges',len(graph.edges),'fine',len(fine))
slices=[c for c in fine if c.qualname=='validate_cookie']
print('validate_cookie slices',len(slices))
print([(c.line_start,c.line_end,'expired_session_signature' in c.body) for c in slices])
"""),
("md", """## Граф учебного примера

Рёбра `calls` построены из AST этого примера. Это не граф задачи SWE-bench.
"""),
("code", r"""import matplotlib.pyplot as plt
import networkx as nx
G=nx.DiGraph()
funcs={k:v for k,v in graph.nodes.items() if v.kind in ('function','method')}
for key,value in funcs.items():
    G.add_node(key,label=value.qualname)
for edge in graph.edges:
    if edge.kind=='calls' and edge.source in funcs and edge.target in funcs:
        G.add_edge(edge.source,edge.target)
fig,ax=plt.subplots(figsize=(8,3.2))
pos=nx.spring_layout(G,seed=42)
nx.draw_networkx(G,pos,ax=ax,labels={n:funcs[n].qualname for n in G},node_color='#d9e4f5',font_size=9,arrows=True)
ax.set_axis_off()
fig.tight_layout()
plt.show()
"""),
("md", """## Срез, а не начало функции

Запрос говорит про просроченную подпись. Целая функция длиннее 180 токенов, поэтому BM25 и v0.3 при этом бюджете её не вмещают. v0.6 кладёт в контекст срез со строками 71–84, где стоит `expired_session_signature`.
"""),
("code", r"""query='expired session signature invalid cookie'
for method in ['bm25','pathpack-v3','pathpack-v6']:
    _,hits=retrieve(SRC,query,method=method,budget=180)
    text=''.join(h.chunk.text for h in hits)
    print(method,'chunks',len(hits),'has marker', 'expired_session_signature' in text,
          'lines',[(h.chunk.line_start,h.chunk.line_end) for h in hits])
"""),
("md", """## 20 dev-задач, бюджет 2048

Числа читаются из `output/v06_dev/summary.csv`. Число оцениваемых задач: File 20, Function и Complete 18, Changed Line 14.
"""),
("code", r"""import csv
def load(path):
    with open(path,newline='') as f:
        return list(csv.DictReader(f))
rows=load(ROOT/'output/v06_dev/summary.csv')
want=['bm25','graph-expand','pathpack-v4','pathpack-v51','pathpack-v6','pathpack-v6/hops-1','pathpack-v6/hops-2','pathpack-v6/callers','pathpack-v6/callees','pathpack-v6/no-graph','pathpack-v6/no-parent','pathpack-v6/no-chunking','pathpack-v6/no-diversity']
print(f"{'method':<28} {'file':>7} {'fn':>7} {'complete':>8} {'line':>7} {'latency':>8}")
for method in want:
    r=next(item for item in rows if item['method']==method and item['budget']=='2048')
    print(f"{method:<28} {float(r['file_recall']):7.3f} {float(r['function_recall']):7.3f} {float(r['complete_function_recall']):8.3f} {float(r['changed_line_recall']):7.3f} {float(r['latency_sec']):8.3f}")
fig,axes=plt.subplots(1,3,figsize=(11,3.3),sharex=True)
for ax,(key,title) in zip(axes,[('file_recall','File'),('function_recall','Function'),('changed_line_recall','Changed line')]):
    for method,label in [('pathpack-v4','v0.4'),('pathpack-v51','v0.5.1'),('pathpack-v6','v0.6')]:
        series=sorted((item for item in rows if item['method']==method), key=lambda item:int(item['budget']))
        ax.plot([int(item['budget']) for item in series],[float(item[key]) for item in series],marker='o',label=label)
    ax.set_title(title); ax.set_xticks([1024,2048,4096]); ax.set_ylim(0,1)
axes[0].legend(fontsize=8)
fig.tight_layout(); plt.show()
"""),
("md", """## Парное сравнение

Разница — v0.6 минус v0.4, по одним и тем же задачам. Интервал, который включает 0, не считается подтверждением.
"""),
("code", r"""import json
comp=json.loads((ROOT/'output/v06_dev/comparisons.json').read_text())
print(f"{'metric':<28} {'diff':>8} {'ci':>24} {'+/=/-':>10}")
for item in comp:
    if item.get('candidate')=='pathpack-v6' and item.get('baseline')=='pathpack-v4' and item['budget']==2048:
        lo,hi=item['ci95']
        print(f"{item['metric']:<28} {item['difference']:8.4f} [{lo:7.3f}, {hi:7.3f}] {item['wins']:>3}/{item['ties']:>2}/{item['losses']}")
"""),
("md", """## Вывод

На учебном примере v0.6 доходит до строки внутри длинной функции. На 20 dev-задачах при 2048 токенах он проигрывает v0.4 по File Recall (0.600 против 0.725) и не даёт прироста Changed Line Recall, чей интервал включает ноль. Holdout не запускался. Конфигурация после прогона не менялась.
"""),
]

def execute(nb):
    ns={'plt':plt}
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':
            continue
        count+=1
        buf=io.StringIO(); old=sys.stdout; sys.stdout=buf; plt.close('all'); error=None
        try:
            exec(compile(cell.source,'<cell>','exec'),ns)
        except Exception:
            import traceback; error=traceback.format_exc()
        finally:
            sys.stdout=old
        outputs=[]
        text=buf.getvalue()
        if text:
            outputs.append(new_output(output_type='stream',name='stdout',text=text))
        for number in plt.get_fignums():
            fig=plt.figure(number); raw=io.BytesIO(); fig.savefig(raw,format='png',dpi=120,bbox_inches='tight')
            outputs.append(new_output(output_type='display_data',data={'image/png':base64.b64encode(raw.getvalue()).decode('ascii'),'text/plain':'<Figure>'},metadata={}))
        plt.close('all')
        if error:
            outputs.append(new_output(output_type='stream',name='stderr',text=error))
            cell.outputs=outputs
            raise RuntimeError(error)
        cell.outputs=outputs; cell.execution_count=count

def main():
    nb=new_notebook(cells=[new_markdown_cell(src) if kind=='md' else new_code_cell(src) for kind,src in CELLS])
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    os.chdir(ROOT)
    execute(nb)
    dest=ROOT/'notebooks/07_pathpack_v06.ipynb'
    nbformat.write(nb,dest)
    print(dest)

if __name__=='__main__':
    main()
