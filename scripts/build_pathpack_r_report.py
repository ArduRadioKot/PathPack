"""Build the Russian report and the notebook from a finished PathPack-R run.

The script reads scores. It does not change retrieval and it does not refit weights.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _load(run):
    run = Path(run)
    summary = []
    text = (run / 'summary.csv').read_text().splitlines()
    header = text[0].split(',')
    for line in text[1:]:
        if not line.strip():
            continue
        values = line.split(',')
        summary.append({key: _cast(value) for key, value in zip(header, values)})
    comparisons = json.loads((run / 'comparisons.json').read_text())
    meta = json.loads((run / 'run.json').read_text())
    return summary, comparisons, meta


def _cast(value):
    if value in {'', 'None'}:
        return None
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _row(summary, method, budget):
    for row in summary:
        if row['method'] == method and int(row['budget']) == budget:
            return row
    return None


def _fmt(value):
    if value is None:
        return '—'
    return f'{value:.4f}' if isinstance(value, float) else str(value)


def _cmp(comparisons, method, budget, metric):
    for row in comparisons:
        if row['candidate'] == method and int(row['budget']) == budget and row['metric'] == metric:
            return row
    return None


def write_report(run, destination):
    summary, comparisons, meta = _load(run)
    budget = 2048
    methods = [
        'bm25', 'graph-expand', 'pathpack-v4', 'pathpack-v10', 'pathpack-v10e',
        'pathpack-r/semantic', 'pathpack-r/fusion', 'pathpack-r/reranker',
        'pathpack-r/reranker-graph', 'pathpack-r/reranker-active',
        'pathpack-r/active-bm25-seeds', 'pathpack-r/ranked', 'pathpack-r',
    ]
    lines = [
        '# PathPack-R: поиск, переранжирование и активная локализация',
        '',
        'Диагностические 20 задач уже использовались при выборе v0.4–v1.0. Числа ниже описывают механизм и не являются независимым подтверждением. Validation и final holdout отмечены отдельно и не подменяют эту оговорку.',
        '',
        f"Счётчик отбора: `{meta.get('token_counter')}`. Конфигурация записана до прогона и после чисел не менялась. Seed парного bootstrap 42, 2000 итераций. Разница считается как метод минус PathPack v0.4.",
        '',
        '## Что запускалось',
        '',
        f"Semantic: `{meta['config']['semantic_model']}`, ревизия `{meta['config']['semantic_revision']}`. Это UniXcoder base, среднее по состояниям энкодера. `Salesforce/codet5p-110m-embedding` ревизии `d9f3a534af4252f04b10cd9f78e05037542d10f6` на transformers 5.22 не создаётся (`CodeT5pEmbeddingConfig` не имеет `is_decoder`). Этот прогон отмечен как `not_run`. Случайные векторы вместо него не подставлялись.",
        '',
        f"Reranker: `{meta['config']['reranker_model']}`, ревизия `{meta['config']['reranker_revision']}`. Это кросс-энкодер MS MARCO по парам вопрос/абзац. Он не обучен на исходном коде. На контрольной паре имя из текста получает более высокий логит, чем посторонняя функция, но оба логита отрицательные и после сигмоиды лежат около нуля. Упаковка поэтому смотрит на порядок и относительный порог `{meta['config']['relative_floor']}`, а не на абсолютную вероятность.",
        '',
        'Обучаемый ранжировщик — линейная модель с попарной логистической потерей в `pathpack_r/train.py`. На этих снимках она не обучалась: все доступные репозитории SWE-bench Verified пересекаются с оценкой, а repository-disjoint split внутри Verified невозможен. На inference золотой патч не подаётся. В опубликованном PathPack-R оценка — это выход предобученного reranker, а не подобранная сумма BM25 и GraphScore. Фиксированные веса fusion относятся только к руке fusion.',
        '',
        '## Диагностика, бюджет 2048',
        '',
        '| Метод | File | Function | Complete | Changed line | Загрузка | Задач file / function / line |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method in methods:
        row = _row(summary, method, budget)
        if row is None:
            lines.append(f'| {method} | — | — | — | — | — | нет в прогоне |')
            continue
        lines.append(
            f"| {method} | {_fmt(row.get('file_recall'))} | {_fmt(row.get('function_recall'))} | "
            f"{_fmt(row.get('complete_function_recall'))} | {_fmt(row.get('changed_line_recall'))} | "
            f"{_fmt(row.get('token_utilization'))} | {row.get('file_recall_evaluable')} / "
            f"{row.get('function_recall_evaluable')} / {row.get('changed_line_recall_evaluable')} |"
        )
    lines += ['', '### Парное сравнение с v0.4 при 2048', '',
              '| Метод | Метрика | Разница | 95% ДИ | Победы / ничьи / поражения | n |',
              '| --- | --- | ---: | --- | ---: | ---: |']
    for method in methods:
        if method == 'pathpack-v4':
            continue
        for metric in ('file_recall', 'function_recall', 'complete_function_recall', 'changed_line_recall'):
            row = _cmp(comparisons, method, budget, metric)
            if row is None or row.get('difference') is None:
                continue
            ci = row.get('ci95')
            ci_text = '—' if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(
                f"| {method} | {metric} | {row['difference']:.4f} | {ci_text} | "
                f"{row['wins']} / {row['ties']} / {row['losses']} | {row['n']} |"
            )
    not_run = meta.get('not_run', 0)
    lines += [
        '',
        f"Записей `not_run`: {not_run}. Gemma-токенизатор `{meta['config']['gemma_tokenizer']}` в этом окружении не загрузился, статус `not_run`. Все методы в таблице выше посчитаны одним счётчиком `utf8_bytes_div4_ceil_serialized_context_v1`. Смешанного сравнения нет.",
        '',
        'Changed Line Recall считает только непустые удалённые или заменённые строки базового файла. Якоря чистых вставок лежат в `insertion_anchor_recall` и в эту метрику не входят. Имя функции не считается покрытием её тела. Задачи без знаменателя в среднее не входят нулём.',
        '',
        '## v1.0e',
        '',
        'v1.0e использует тот же пул, что v1.0, и другую остановку. Повторное свидетельство убывает как `repeat_weight * confidence / (1 + n)` и не обнуляет поиск. Лексический, символьный, графовый и путевой сигналы остаются отдельными. Отбор останавливается на `min_utility` 0.12, на бюджете или на пустом остатке пула. Покрытие имён из текста остановкой не является. Это исправление ранней остановки, а не новый алгоритм локализации. Если оно приближается к v0.4, это восстановление базы, а не доказательство PathPack-R.',
        '',
        '## Ограничения',
        '',
        '- Диагностические 20 задач не независимы.',
        '- Reranker не специализирован для кода. Его абсолютные оценки сжаты к нулю.',
        '- CodeT5+ embedding не запустился. Semantic-канал — UniXcoder, а не модель, обученная как контрастивный поисковик кода.',
        '- Линейный ранжировщик не обучен: нет repository-disjoint снимков вне оцениваемых репозиториев.',
        '- Внутри SWE-bench Verified repository-disjoint split невозможен: те же 12 репозиториев.',
        '- Одна задача не считается выигрышем метода, даже если интервал её не исключает.',
        '',
    ]
    Path(destination).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _example_diagnostic(run, instance='django__django-12276', budget=2048):
    path = Path(run) / 'diagnostics' / f'{instance}_pathpack-r_{budget}.json.gz'
    if not path.is_file():
        return None
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def write_notebook(run, destination, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary, comparisons, meta = _load(run)
    payload = _example_diagnostic(run)
    cells = []
    cells.append(new_markdown_cell(
        '# PathPack-R\n\n'
        'Ноутбук читает готовый прогон. Он не подбирает веса и не читает патч внутри ретривера. '
        'Пример — `django__django-12276`, задача, на которой v1.0 останавливался после имён из текста.'
    ))
    code = [
        'from pathlib import Path',
        'import gzip, json',
        'import matplotlib.pyplot as plt',
        f"RUN = Path({str(run)!r})",
        "payload = json.load(gzip.open(RUN/'diagnostics'/'django__django-12276_pathpack-r_2048.json.gz','rt'))",
        "issue = next(json.loads(line) for line in Path('output/benchmark/issues.jsonl').read_text().splitlines() if 'django__django-12276' in line)",
        "print(issue['problem_statement'][:1200])",
        "print('---')",
        "print('score source:', payload['score_source'])",
        "print('channels:', {name: {k: v for k, v in info.items() if k in ('kept','dropped','status','reason')} for name, info in payload['channels'].items()})",
    ]
    cells.append(new_code_cell('\n'.join(code)))
    if payload:
        # Precompute a static view so the notebook source is also readable without a kernel,
        # and the code cell above recomputes it.
        by_channel = {}
        for item in payload.get('pool_before_active', []):
            for channel in item.get('channels', []):
                by_channel.setdefault(channel, []).append(item)
        preview = []
        for channel, items in sorted(by_channel.items()):
            top = sorted(items, key=lambda row: row.get('ranks', {}).get(channel, 10**9))[:5]
            preview.append(f'### {channel}')
            for item in top:
                preview.append(
                    f"- `{item.get('function')}` `{item.get('path')}`:{item.get('line_start')}-{item.get('line_end')} "
                    f"rank={item.get('ranks', {}).get(channel)} lexical={item.get('lexical_score')} "
                    f"semantic={item.get('semantic_score')} rerank={item.get('rerank_score')}"
                )
        cells.append(new_markdown_cell('## Кандидаты каналов\n\n' + '\n'.join(preview)))
        ranked = sorted(payload.get('candidates', []), key=lambda row: -(row.get('rerank_score') or 0))[:12]
        rerank_lines = ['## Reranker, верх списка\n']
        for item in ranked:
            rerank_lines.append(
                f"- {item.get('rerank_score')} `{item.get('function')}` `{item.get('path')}` admission={item.get('admission')} decision={item.get('decision')}"
            )
        cells.append(new_markdown_cell('\n'.join(rerank_lines)))
        trace = [row for row in payload.get('active_trace', []) if row.get('action') in {'admit', 'mark_hypothesis', 'rejected_neighbor'}][:25]
        trace_lines = ['## Активное расширение\n']
        for row in trace:
            trace_lines.append(
                f"- {row.get('action')} {row.get('kind')} {row.get('direction')} `{row.get('function', '')}` reason={row.get('reason', '')}"
            )
        cells.append(new_markdown_cell('\n'.join(trace_lines)))
        selected = [row for row in payload.get('candidates', []) if row.get('decision') == 'selected']
        sel = ['## Конечные фрагменты\n']
        for item in selected:
            sel.append(f"- `{item.get('path')}` `{item.get('function')}` {item.get('line_start')}-{item.get('line_end')} final={item.get('final_score')}")
        cells.append(new_markdown_cell('\n'.join(sel)))
    figure = figure_dir / 'diagnostic_2048.png'
    methods = ['bm25', 'graph-expand', 'pathpack-v4', 'pathpack-v10', 'pathpack-v10e', 'pathpack-r/reranker', 'pathpack-r']
    labels, files, funcs, lines_m = [], [], [], []
    for method in methods:
        row = _row(summary, method, 2048)
        if row is None:
            continue
        labels.append(method.replace('pathpack-', ''))
        files.append(row.get('file_recall') or 0)
        funcs.append(row.get('function_recall') or 0)
        lines_m.append(row.get('changed_line_recall') or 0)
    if labels:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        import numpy as np
        xpos = np.arange(len(labels))
        width = 0.25
        ax.bar(xpos - width, files, width, label='File')
        ax.bar(xpos, funcs, width, label='Function')
        ax.bar(xpos + width, lines_m, width, label='Changed line')
        ax.set_xticks(xpos, labels, rotation=30, ha='right')
        ax.set_ylim(0, 1)
        ax.set_title('Diagnostic 20, budget 2048')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure, dpi=120)
        plt.close(fig)
    cells.append(new_markdown_cell(
        '## Сравнение и абляции\n\n'
        f'Рисунок `{figure.name}` строится из `summary.csv`. '
        'Полные интервалы лежат в `comparisons.json`. Диагностический набор не является holdout.'
    ))
    cells.append(new_code_cell(
        "import csv\n"
        "rows=list(csv.DictReader((RUN/'summary.csv').open()))\n"
        "for row in rows:\n"
        "    if row['budget']=='2048' and row['method'] in {'pathpack-v4','pathpack-v10e','pathpack-r','pathpack-r/reranker','bm25'}:\n"
        "        print(row['method'], 'file', row['file_recall'], 'function', row['function_recall'], 'line', row['changed_line_recall'])\n"
    ))
    notebook = new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    Path(destination).write_text(nbformat.writes(notebook), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='output/pathpack_r_diag')
    parser.add_argument('--report', default='docs/PATHPACK_R_REPORT_RU.md')
    parser.add_argument('--notebook', default='notebooks/11_pathpack_r.ipynb')
    args = parser.parse_args()
    write_report(args.run, args.report)
    write_notebook(args.run, args.notebook, Path(args.run) / 'figures')
    print('wrote', args.report, args.notebook)
