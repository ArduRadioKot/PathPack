"""Build notebooks/05_pathpack_v05.ipynb from the measured v0.5 dev run."""
import ast, csv, gzip, json, traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output' / 'v05_dev'
AUDIT = ROOT / 'output' / 'v05_audit'
OUT = ROOT / 'notebooks' / '05_pathpack_v05.ipynb'


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    meta = json.loads((RUN / 'run.json').read_text())
    audit = json.loads((RUN / 'audit.json').read_text())
    if meta['status'] != 'scored' or not audit['ok']:
        raise SystemExit('Notebook requires a scored run whose audit passed')
    summary = rows(RUN / 'summary.csv')
    v5 = next(r for r in summary if r['method'] == 'pathpack-v5' and r['budget'] == '2048')
    v4 = next(r for r in summary if r['method'] == 'pathpack-v4' and r['budget'] == '2048')
    conclusion = (
        f"Двухэтапный поиск на этих 20 задачах не улучшил локализацию. При 2048 токенах File Recall {float(v5['file_recall']):.4f} "
        f"против {float(v4['file_recall']):.4f} у v0.4, Function Recall {float(v5['function_recall']):.4f} против {float(v4['function_recall']):.4f}, "
        f"Complete Function Recall {float(v5['complete_function_recall']):.4f} против {float(v4['complete_function_recall']):.4f}, "
        f"Changed Line Recall {float(v5['changed_line_recall']):.4f} против {float(v4['changed_line_recall']):.4f}. "
        "По File Recall и Function Recall у v0.5 нет побед над v0.4; Changed Line Recall совпал на всех 14 оцениваемых задачах. "
        "Метрики v0.5 при 1024 и 2048 совпали: лишняя тысяча токенов ушла в первые чанки новых, неэталонных файлов. "
        "Абляция no-hierarchy возвращает File Recall к 0.7000 и не меняет Function и Changed Line Recall. "
        "Отключение графа не ухудшает локализацию строк. "
        "Это одна зависимая выборка. Интервалы bootstrap в отчёте важнее разницы средних. "
        "Holdout из 60 задач не оценивался. Embedding retrieval не запускался."
    )
    cells = [
        md(f'''# PathPack v0.5 — иерархический поиск

Блокнот читает измеренный прогон `output/v05_dev` и аудит ошибок v0.4 в `output/v05_audit`. Числа не заданы вручную.

Двухэтапный отбор:

1. **Файлы.** BM25 тела, имена символов и типизированные пути графа дают FileScore. В пул входят файлы зёрен, файлы с raw ≥ 12% лучшего и добор до 8 файлов. Потолок 24 не выбрасывает файлы зёрен.
2. **Фрагменты.** Внутри файла отдельно оцениваются функции, методы и AST-чанки. Формула `0.20·File + 0.50·Chunk + 0.18·Graph + 0.12·Context`. Повтор текста и повтор файла штрафуются, но высокий ChunkScore ослабляет штраф файла. Первый чанк файла получает небольшой бонус, чтобы один файл не занял весь бюджет только за счёт FileScore и чтобы File Recall не схлопнулся.
3. **Бюджет.** Порядок фрагментов один. Контексты 1024, 2048 и 4096 — префиксы этого порядка. Заголовки, пути и сигнатурный якорь входят в стоимость. Соседние строки тела сами по себе не добавляются.

Счётчик токенов: `{meta['token_counter']}`. Это не токенизатор Gemma. Gold используется только в аудите и в оценке после retrieval.
'''),
        code('''from pathlib import Path
import csv, gzip, json, sys
import matplotlib.pyplot as plt
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/codegraph_rag').is_dir())
sys.path.insert(0, str(ROOT/'src'))
RUN, AUDIT = ROOT/'output/v05_dev', ROOT/'output/v05_audit'
meta = json.loads((RUN/'run.json').read_text())
summary = list(csv.DictReader((RUN/'summary.csv').open()))
print('tasks', len(meta['issues']), 'counter', meta['token_counter'])
print('embedding', meta['embedding_baseline']['status'], meta['embedding_baseline'].get('reason',''))
print('audit_ok', json.loads((RUN/'audit.json').read_text())['ok'])
'''),
        md('''## Пример двухэтапного поиска

Ниже — реальная диагностика `pathpack-v5` @2048 для первой dev-задачи прогона. Таблица файлов — этап 1. Таблица чанков — этап 2. Эталонные строки в этот пример не подставляются.
'''),
        code('''diag = next((RUN/'diagnostics').glob('*_v5full_2048.json.gz'))
doc = json.loads(gzip.open(diag).read())
print(doc['instance_id'], 'seeds', doc['seeds'])
print('selected files', doc['selected_files'], 'selected chunks', doc['selected_chunks'])
print('profile', {k: round(doc['profile'][k], 4) for k in ('candidate_sec','graph_sec','file_rank_sec','chunk_score_sec','diversity_sec','query_sec')})
chosen = [row for row in doc['files'] if row['selected']]
chosen.sort(key=lambda row: -row['file_score_raw'])
print('top selected files:')
for row in chosen[:8]:
    print(f"  {row['file_score_raw']:.3f}  {row['path']}  best={row['best_function_id']}")
'''),
        md('''## Граф зависимостей

Рёбра берутся из сохранённого графа той же задачи. На рисунке — функции, которые v0.5 передал в контекст, и рёбра между ними, если статический граф их содержит. Отсутствие ребра на рисунке не означает, что зависимости нет: показаны только выбранные функции.
'''),
        code('''graph_path = RUN/f"{doc['instance_id']}_graph.json.gz"
graph = json.loads(gzip.open(graph_path).read())
selected = {c['function_id'] for c in doc['candidates'] if c.get('selected') and c.get('function_id')}
if len(selected) > 24:
    selected = set(sorted(selected)[:24])
edges = [e for e in graph['edges'] if e['source'] in selected and e['target'] in selected]
print('selected functions shown', len(selected), 'edges among them', len(edges))
fig, ax = plt.subplots(figsize=(8, max(3, 0.28*len(selected))))
names = sorted(selected)
pos = {name: i for i, name in enumerate(names)}
for edge in edges[:80]:
    ax.plot([0, 1], [pos[edge['source']], pos[edge['target']]], color='#9aa0a6', lw=0.6)
ax.scatter([0]*len(names), list(pos.values()), s=12, c='#1f4e79')
ax.set_yticks(list(pos.values()), [name.split('::')[-1][:48] for name in names], fontsize=7)
ax.set_xticks([])
ax.set_title(doc['instance_id'] + ': edges among selected functions')
fig.tight_layout()
'''),
        md('''## Ранжирование файлов

Полный список выбранных файлов той же задачи, по FileScore. Причина `selected_seed` оставляет файл даже ниже порога.
'''),
        code('''print(f"{'score':>7} {'reason':<22} path")
for row in chosen:
    print(f"{row['file_score_raw']:7.3f} {row['reason']:<22} {row['path']}")
'''),
        md('''## Ранжирование функций внутри файла

Для файла с наибольшим FileScore показаны его чанки в порядке v0.5. `chunk_score` — соответствие фрагмента запросу внутри файла, `graph_score` — связь с релевантными функциями, `context_score` — новизна относительно уже выбранного текста.
'''),
        code('''focus = chosen[0]['path']
local = [c for c in doc['candidates'] if c['path']==focus and c.get('order') is not None]
local.sort(key=lambda c: c['order'])
print(focus)
print(f"{'ord':>4} {'score':>6} {'chunk':>6} {'graph':>6} {'ctx':>6} {'dec':<22} function")
for c in local[:12]:
    print(f"{c['order']:4} {c['score']:6.3f} {c['chunk_score']:6.3f} {c['graph_score']:6.3f} {c['context_score']:6.3f} {c['decision']:<22} {c['qualname']}")
'''),
        md('''## Найденные и эталонные строки

Цвета построены аудитом **после** retrieval v0.4 @2048, потому что вопрос этого раздела — почему правильный файл не передаёт нужные строки. Зелёный: нужная строка попала в контекст. Красный: нужная строка пропущена. Жёлтый: выбранный фрагмент не пересекается с эталонным исправлением. Эталон не участвовал в ранжировании.
'''),
        code('''case = json.loads((AUDIT/'summary.json').read_text())
print('case counts', case['case_counts'])
print('reports', case['reports'])
image = AUDIT/'figures'/'sympy__sympy-24213_lines.png'
print('line figure', image.exists())
pixels = plt.imread(image)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(pixels); ax.axis('off')
ax.set_title('sympy-24213: green found, red missed, yellow extra')
fig.tight_layout()
'''),
        md('''## Recall по бюджетам

Macro mean только по задачам, где метрика определена. Changed Line Recall не включает вставки, которых нет в `base_commit`.
'''),
        code('''methods = ('bm25','graph-expand','pathpack-v4','pathpack-v5')
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, metric, title in zip(axes, ('file_recall','function_recall','changed_line_recall'), ('File Recall','Function Recall','Changed Line Recall')):
    for method in methods:
        part = sorted((r for r in summary if r['method']==method), key=lambda r: int(r['budget']))
        ax.plot([int(r['budget']) for r in part], [float(r[metric]) for r in part], marker='o', label=method)
    ax.set_title(title); ax.set_xlabel('budget'); ax.set_xticks([1024,2048,4096])
axes[0].legend(fontsize=7)
fig.tight_layout()
'''),
        md('''## Абляции @2048

Каждая строка выключает один механизм v0.5. `no-hierarchy` убирает отбор файлов и внутрифайловую перенормировку. `no-graph` обнуляет граф и в FileScore, и в GraphScore. `no-diversity` убирает штраф повтора файла и бонус первого чанка. `no-chunking` берёт функции целиком. `no-file-score` обнуляет α, но не меняет состав пула файлов.
'''),
        code('''cols = ['method','file_recall','function_recall','complete_function_recall','changed_line_recall','relevant_line_density','latency_sec','useful_functions_per_1k']
print(' | '.join(cols))
for row in summary:
    if row['budget']=='2048' and row['method'].startswith('pathpack-v5'):
        print(' | '.join(row[c] for c in cols))
'''),
        md('''## Latency

Время запроса не включает сканирование репозитория и построение индекса. Профиль усреднён по максимальному бюджету, чтобы три бюджета одной задачи не посчитались трижды.
'''),
        code('''methods = ('bm25','graph-expand','pathpack-v4','pathpack-v5')
main = [next(r for r in summary if r['method']==m and r['budget']=='2048') for m in methods]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar([r['method'] for r in main], [float(r['latency_sec']) for r in main], color='#1f4e79')
ax.set_ylabel('seconds per query')
ax.set_title('Measured query latency @2048')
fig.tight_layout()
profile = list(csv.DictReader((RUN/'stage_profile_mean.csv').open()))
print('stage means, seconds')
for row in profile:
    if row['method'] in {'pathpack-v5','pathpack-v5/no-graph','pathpack-v5/no-diversity'}:
        print(row['method'], {k: round(float(row[k]), 4) for k in ('graph_sec','diversity_sec','chunk_score_sec','query_sec')})
'''),
        md('## Выводы и ограничения\n\n' + conclusion + '''

Ограничения: 20 задач, неполная разметка на большинстве из них, один проход latency, статический граф, невыполненный embedding baseline и не оценённый holdout `output/v05_holdout` (60 instance id, seed 44, пересечение с dev и прежними списками равно нулю). Подробности и парные доверительные интервалы — в `docs/PATHPACK_V05_REPORT_RU.md`.
'''),
    ]
    nb = execute(cells)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(json.dumps({'notebook': str(OUT), 'cells': len(nb['cells'])}))


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)}


def execute(cells):
    ns = {'plt': plt}
    count = 0
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        cell['execution_count'] = count
        plt.close('all')
        import io, contextlib
        buffer = io.StringIO()
        source = ''.join(cell['source'])
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, f'cell{count}', 'exec'), ns, ns)
            outputs = []
            if buffer.getvalue():
                outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': buffer.getvalue().splitlines(keepends=True)})
            for number in plt.get_fignums():
                fig = plt.figure(number)
                raw = io.BytesIO()
                fig.savefig(raw, format='png', dpi=140)
                import base64
                outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': base64.b64encode(raw.getvalue()).decode('ascii'), 'text/plain': ['<Figure>']}})
            cell['outputs'] = outputs
        except Exception:
            cell['outputs'] = [{'output_type': 'error', 'ename': 'ExecutionError', 'evalue': 'cell failed', 'traceback': traceback.format_exc().splitlines()}]
            raise
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'}}, 'cells': cells}


if __name__ == '__main__':
    main()
