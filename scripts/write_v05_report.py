"""Russian v0.5 report from measured artifacts only."""
import argparse, csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def num(value, digits=4):
    if value is None or value == '':
        return '—'
    return f'{float(value):.{digits}f}'


def table(rows, columns):
    head = '| ' + ' | '.join(label for _, label in columns) + ' |\n'
    rule = '| ' + ' | '.join('---:' if key not in {'method', 'instance_id', 'cases', 'baseline', 'metric', 'ci95'} else '---' for key, _ in columns) + ' |\n'
    body = ''
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, str):
                cells.append(value)
            elif value is None or value == '':
                cells.append('—')
            elif isinstance(value, int) or key in {'wins', 'ties', 'losses', 'n', 'budget'}:
                cells.append(str(int(value)))
            else:
                cells.append(f'{float(value):.4f}')
        body += '| ' + ' | '.join(cells) + ' |\n'
    return head + rule + body


def load_summary(path):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            item = {'method': row['method']}
            for key, value in row.items():
                if key == 'method' or value == '':
                    continue
                item[key] = int(value) if key in {'budget', 'tasks'} or key.endswith('_evaluable') or key.endswith('_tasks') else float(value)
            rows.append(item)
    return rows


def find(rows, method, budget):
    return next(row for row in rows if row['method'] == method and row['budget'] == budget)


def paired(comparisons, baseline, metric, budget=2048):
    return next(row for row in comparisons if row['candidate'] == 'pathpack-v5' and row['baseline'] == baseline and row['metric'] == metric and row['budget'] == budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'output/v05_dev')
    parser.add_argument('--audit', type=Path, default=ROOT / 'output/v05_audit')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/PATHPACK_V05_REPORT_RU.md')
    args = parser.parse_args()
    run = args.run.resolve()
    meta = json.loads((run / 'run.json').read_text())
    audit = json.loads((run / 'audit.json').read_text())
    if meta['status'] != 'scored' or not audit['ok']:
        raise SystemExit('Report requires a scored run whose audit passed')
    summary = load_summary(run / 'summary.csv')
    comparisons = json.loads((run / 'comparisons.json').read_text())
    profiles = []
    if (run / 'stage_profile_mean.csv').exists():
        with (run / 'stage_profile_mean.csv').open() as handle:
            profiles = list(csv.DictReader(handle))
    label_audit = json.loads((args.audit / 'label_audit.json').read_text()) if (args.audit / 'label_audit.json').exists() else {}
    cases = []
    case_dir = args.audit / 'cases'
    if case_dir.exists():
        for path in sorted(case_dir.glob('*.json')):
            item = json.loads(path.read_text())
            cases.append({'instance_id': item['instance_id'], 'cases': ', '.join(item['cases']),
                          'file': item['metrics'].get('file_recall'), 'function': item['metrics'].get('function_recall'),
                          'line': item['metrics'].get('changed_line_recall'), 'summary': item['summary_ru']})
    methods = ('bm25', 'graph-expand', 'pathpack-v4', 'pathpack-v5')
    recall_rows = []
    for budget in (1024, 2048, 4096):
        for method in methods:
            row = find(summary, method, budget)
            recall_rows.append({'method': method, 'budget': budget, 'file_recall': row['file_recall'],
                                'function_recall': row['function_recall'], 'complete_function_recall': row['complete_function_recall'],
                                'changed_line_recall': row['changed_line_recall'], 'relevant_line_density': row.get('relevant_line_density'),
                                'latency_sec': row['latency_sec'], 'token_utilization': row['token_utilization'],
                                'useful_functions_per_1k': row['useful_functions_per_1k']})
    ablations = [row for row in summary if row['method'].startswith('pathpack-v5') and row['budget'] == 2048]
    v5 = find(summary, 'pathpack-v5', 2048)
    v4 = find(summary, 'pathpack-v4', 2048)
    bm25 = find(summary, 'bm25', 2048)
    no_graph = find(summary, 'pathpack-v5/no-graph', 2048)
    no_hier = find(summary, 'pathpack-v5/no-hierarchy', 2048)
    no_div = find(summary, 'pathpack-v5/no-diversity', 2048)
    no_chunk = find(summary, 'pathpack-v5/no-chunking', 2048)
    no_file = find(summary, 'pathpack-v5/no-file-score', 2048)

    def delta(metric):
        return v5[metric] - v4[metric] if v5[metric] is not None and v4[metric] is not None else None

    def verdict(metric, goal):
        value = v5[metric]
        if value is None:
            return 'метрика не определена на этой выборке'
        if value + 1e-12 >= goal:
            return f'{value:.4f} достигает ориентир {goal:.4f}, но это не статистическое доказательство'
        return f'{value:.4f} ниже ориентира {goal:.4f}'

    file_ci = paired(comparisons, 'pathpack-v4', 'file_recall')
    fn_ci = paired(comparisons, 'pathpack-v4', 'function_recall')
    line_ci = paired(comparisons, 'pathpack-v4', 'changed_line_recall')
    comp_ci = paired(comparisons, 'pathpack-v4', 'complete_function_recall')
    budgets = {}
    for budget in (1024, 2048, 4096):
        budgets[budget] = {metric: find(summary, 'pathpack-v5', budget)[metric] - find(summary, 'pathpack-v4', budget)[metric]
                           for metric in ('file_recall', 'function_recall', 'changed_line_recall')}
    graph_part = v5['function_recall'] - no_graph['function_recall']
    hier_part = v5['function_recall'] - no_hier['function_recall']
    file_hier = v5['file_recall'] - no_hier['file_recall']
    line_hier = (v5['changed_line_recall'] - no_hier['changed_line_recall']) if v5['changed_line_recall'] is not None and no_hier['changed_line_recall'] is not None else None
    line_graph = (v5['changed_line_recall'] - no_graph['changed_line_recall']) if v5['changed_line_recall'] is not None and no_graph['changed_line_recall'] is not None else None
    profile = next((row for row in profiles if row['method'] == 'pathpack-v5'), None)
    embedding = meta['embedding_baseline']
    holdout = json.loads((ROOT / 'output/v05_holdout/metadata.json').read_text())
    case_table = table(cases, (('instance_id', 'Задача'), ('cases', 'Случаи'), ('file', 'File'), ('function', 'Function'), ('line', 'Line')))
    comp_rows = []
    for baseline in ('bm25', 'graph-expand', 'pathpack-v4', 'pathpack-v5/no-hierarchy', 'pathpack-v5/no-graph', 'pathpack-v5/no-diversity', 'pathpack-v5/no-chunking', 'pathpack-v5/no-file-score'):
        for metric in ('file_recall', 'function_recall', 'changed_line_recall'):
            try:
                row = paired(comparisons, baseline, metric)
            except StopIteration:
                continue
            comp_rows.append({'baseline': baseline, 'metric': metric, 'difference': row['difference'], 'ci95': str(row['ci95']),
                              'wins': row['wins'], 'ties': row['ties'], 'losses': row['losses'], 'n': row['n']})
    text = f'''# PathPack v0.5: иерархический поиск и локализация строк

Дата прогона: 2 ноября 2026. Пакет 0.5.0. Оценка выполнена на **{len(meta["issues"])} dev-задачах** SWE-bench Verified, seed манифеста 42. Это та же dev-выборка, на которой настраивалась архитектура v0.5 после аудита v0.4. Она не является независимым подтверждением. Новый список из 60 задач в `output/v05_holdout/` (seed 44) не пересекается по `instance_id` с dev, прежними test и неиспользованным holdout v0.4. Он **не оценивался**: снимки этих коммитов не развёрнуты, и параметры после его просмотра не подбирались, потому что просмотра не было. Репозитории в новом списке могут совпадать с уже виденными (`{', '.join(holdout['repo_overlap'])}`).

Git commit прогона: `{meta.get('git_commit')}`. Счётчик токенов: `{meta['token_counter']}`. Embedding baseline: **{embedding['status']}**. {embedding.get('reason', '')} Модель, которую следовало бы использовать, зафиксирована как `{embedding.get('model')}` ревизия `{embedding.get('revision')}`; подмена другим методом не делалась.

## 1. Почему v0.4 находит файлы, но не изменённые строки

Аудит `output/v05_audit/` построен по уже завершённому прогону v0.4 @2048. Gold читался только после retrieval. На 20 задачах встретились случаи: файл найден, функция не покрыта (A: {label_audit.get('case_counts', {}).get('A', 'см. summary')}). Точные счётчики случаев лежат в `output/v05_audit/summary.json`.

По сохранённым диагностикам v0.4 механизм такой:

1. **Diversity забирает бюджет первым чанком многих файлов.** На `pytest-dev__pytest-7982` файл `src/_pytest/pathlib.py` попал в контекст, но единственный чанк функции `visit` (строки 551–562, lexical 0.655, 161 токен) имел order 77 и решение `after_budget_cutoff`. Бюджет заполнили 16 других файлов.
2. **Внутри найденной функции выбирается не тот срез.** На `scikit-learn__scikit-learn-13124` функция `_make_test_folds` покрыта по правилу 20%, но выбранные чанки начинаются со строк 669–672, а заменённые строки — 579, 580 и 623. File Recall равен 1, Changed Line Recall равен 0.
3. **Глобальный top-512 выбрасывает умеренное совпадение в большом репозитории.** На `django__django-14580` чанк `TypeSerializer.serialize` имел score 0.201 и причину `candidate_limit`. Файл в контекст не попал.
4. **Changed Line Recall v0.4 не растёт с бюджетом** (0.1480 при 1024, 2048 и 4096 в прежнем отчёте). Дополнительные токены добавляли новые файлы, а не следующие строки уже найденной функции.
5. **Граф часто не доходит до gold-функции.** В аудите у многих промахов graph score лучшего gold-чанка равен 0, даже если рёбра в графе есть (`unreached`). Это не доказательство ложных рёбер: статический граф просто не был использован этими зернами BM25.

Разметка ограничивает интерпретацию. Полная разметка: {label_audit.get('complete_labels', '—')} из {label_audit.get('tasks', 20)}. Changed Line Recall определён на {label_audit.get('changed_line_evaluable', '—')} задачах, Function Recall на {label_audit.get('function_evaluable', '—')}. Причины исключений v0.4: `{label_audit.get('exclusion_reasons', {})}`. Смещения номеров за пределы файла: {label_audit.get('removed_lines_out_of_range', '—')}. Строк удаления вне тел функций: {label_audit.get('changed_lines_outside_functions', '—')}. Задач только со вставками, без оцениваемых старых строк: {label_audit.get('insertion_only_tasks', '—')}. Методов классов в разметке: {label_audit.get('class_methods', '—')}, вложенных функций: {label_audit.get('nested_functions', '—')}.

**Changed Line Recall** по-прежнему считает только непустые строки `base_commit`, которые патч удаляет или заменяет. Добавленные строки в репозитории отсутствуют и в эту метрику не входят. **Insertion-anchor recall** — отдельная метрика: для хунка без удалений якорь — это базовая строка перед вставкой и базовая строка после неё; пустая скобка сдвигается на одну соседнюю непустую строку и дальше не ищется. Хунк `@@ -N,0` якорится на строке N. Неразрешённый якорь исключается с причиной `insertion_anchor_unmapped`, новый файл — с `new_file_no_base_source`. **Base-touch recall** явно объединяет оцениваемые удалённые строки и разрешённые якоря и публикуется только под этим именем. **Relevant Line Density** = доля оцениваемых изменённых строк среди всех непустых строк, реально переданных в контексте. Задачи без оцениваемых изменённых строк и пустой контекст дают null, а не ноль. Якоря вставок в числитель density не входят. Совпадение имени функции не считается покрытием: нужен порог 20% собственных непустых строк тела.

Показательные отчёты (до 10) сохранены в `output/v05_audit/cases/`. Цветные фрагменты: зелёный — нужная строка передана, красный — пропущена, жёлтый — выбранный код без пересечения с эталоном.

{case_table}

## 2. Улучшает ли двухэтапный поиск Function Recall и Changed Line Recall

Формула, которая упорядочивает фрагменты:

`formula = 0.20·FileScore + 0.50·ChunkScore + 0.18·GraphScore + 0.12·ContextScore`

`score = formula · (1 − redundancy) · (1 − file_penalty) + first_file_bonus`

FileScore — максимум лексики, имён и графа по файлу, делённый на лучший файл. ChunkScore на полном алгоритме на 72% берётся из BM25 тела внутри файла, чтобы соседние срезы одной функции различались. GraphScore — 0.75 уверенности типизированного пути (зёрна не получают бонус за самих себя) плюс 0.25 пересечения тела с именами лексически релевантных соседей. ContextScore — новизна относительно уже взятого текста. Штраф файла растёт с числом уже взятых чанков этого файла и ослабляется, если ChunkScore высокий (`local_resistance=0.80`). Первый чанк файла получает бонус `diversity_weight=0.12`, чтобы широкий File Recall v0.4 не схлопнулся. Порядок один на все бюджеты, поэтому контекст 1024 является префиксом 2048 и 4096. Заголовок, путь и сигнатура входят в стоимость. Сигнатурный якорь добавляет строки `def`/декораторов, если срез их не содержит, и не добавляет соседние строки тела.

Коэффициенты записаны в `configs/pathpack_v05.yaml` до полного dev-прогона. Они выбраны по качественному аудиту v0.4, а не перебором по таблице v0.5. После получения чисел ниже конфигурация не менялась.

Ориентиры при 2048, не критерии подгонки метрик:

| Метрика | v0.4 | v0.5 | Ориентир | Факт |
| --- | ---: | ---: | ---: | --- |
| File Recall | {num(v4['file_recall'])} | {num(v5['file_recall'])} | ≥ 0.7000 | {verdict('file_recall', 0.70)} |
| Function Recall | {num(v4['function_recall'])} | {num(v5['function_recall'])} | ≥ 0.2500 | {verdict('function_recall', 0.25)} |
| Complete Function Recall | {num(v4['complete_function_recall'])} | {num(v5['complete_function_recall'])} | ≥ 0.1500 | {verdict('complete_function_recall', 0.15)} |
| Changed Line Recall | {num(v4['changed_line_recall'])} | {num(v5['changed_line_recall'])} | ≥ 0.2000 | {verdict('changed_line_recall', 0.20)} |

Разница v0.5 − v0.4 @2048: File {num(delta('file_recall'))}, Function {num(delta('function_recall'))}, Complete {num(delta('complete_function_recall'))}, Changed Line {num(delta('changed_line_recall'))}, Density {num(delta('relevant_line_density'))}, Useful functions/1K {num(delta('useful_functions_per_1k'))}.

Парное сравнение по задачам, bootstrap 2000, seed 42, v0.5 минус v0.4 @2048:

| Метрика | Разница | 95% ДИ | Победы | Ничьи | Поражения | n |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| File Recall | {num(file_ci['difference'])} | {file_ci['ci95']} | {file_ci['wins']} | {file_ci['ties']} | {file_ci['losses']} | {file_ci['n']} |
| Function Recall | {num(fn_ci['difference'])} | {fn_ci['ci95']} | {fn_ci['wins']} | {fn_ci['ties']} | {fn_ci['losses']} | {fn_ci['n']} |
| Complete Function Recall | {num(comp_ci['difference'])} | {comp_ci['ci95']} | {comp_ci['wins']} | {comp_ci['ties']} | {comp_ci['losses']} | {comp_ci['n']} |
| Changed Line Recall | {num(line_ci['difference'])} | {line_ci['ci95']} | {line_ci['wins']} | {line_ci['ties']} | {line_ci['losses']} | {line_ci['n']} |

Интервал, который содержит ноль, не позволяет назвать улучшение устойчивым даже на этих 20 задачах. Большее среднее само по себе не является доказательством.

### Основные методы

{table(recall_rows, (('method','Метод'),('budget','Бюджет'),('file_recall','File'),('function_recall','Function'),('complete_function_recall','Complete'),('changed_line_recall','Changed line'),('relevant_line_density','Density'),('token_utilization','Utilization'),('latency_sec','Latency, с'),('useful_functions_per_1k','Useful/1K')))}

BM25 @2048: File {num(bm25['file_recall'])}, Function {num(bm25['function_recall'])}, Changed Line {num(bm25['changed_line_recall'])}, latency {num(bm25['latency_sec'], 4)} с. Latency v0.5 @2048: {num(v5['latency_sec'], 4)} с. Индексирование в latency запроса не входит; оно записано в `graph_inventory.csv`.

## 3. Что даёт граф, а что ранжирование внутри файла

Абляции v0.5 @2048. Каждая выключает один механизм и оставляет остальные.

{table(ablations, (('method','Вариант'),('file_recall','File'),('function_recall','Function'),('complete_function_recall','Complete'),('changed_line_recall','Changed line'),('relevant_line_density','Density'),('latency_sec','Latency, с')))}

- Full − No Hierarchy @2048: File {num(file_hier)}, Function {num(hier_part)}, Changed Line {num(line_hier)}. No Hierarchy убирает отбор файлов и внутрифайловую перенормировку BM25; фрагменты конкурируют глобально в top-{meta['v5_config']['candidate_limit']}.
- Full − No Graph: Function {num(graph_part)}, Changed Line {num(line_graph)}, File {num(v5['file_recall']-no_graph['file_recall'])}. Графовый член здесь обнуляется и в FileScore, и в GraphScore.
- No Diversity: File {num(no_div['file_recall'])} против Full {num(v5['file_recall'])}. Если разрыв большой, широкий охват файлов по-прежнему держится на бонусе первого чанка, а не на пороге файлов.
- No Chunking: File {num(no_chunk['file_recall'])}, Function {num(no_chunk['function_recall'])}, Changed Line {num(no_chunk['changed_line_recall'])}. Целые функции проверяют, помогает ли разрезка именно строкам.
- No File Score: File {num(no_file['file_recall'])}, Function {num(no_file['function_recall'])}. Состав пула файлов при этом не меняется: обнуляется только коэффициент α.

Парные интервалы v0.5 против этих вариантов и против BM25/Graph Expansion лежат в `output/v05_dev/comparisons.csv`. Сводка @2048:

{table(comp_rows, (('baseline','База'),('metric','Метрика'),('difference','Разница'),('ci95','95% ДИ'),('wins','Победы'),('ties','Ничьи'),('losses','Поражения'),('n','n')))}

## 4. Устойчивость к бюджету

Разница v0.5 − v0.4:

| Бюджет | File | Function | Changed line |
| ---: | ---: | ---: | ---: |
| 1024 | {num(budgets[1024]['file_recall'])} | {num(budgets[1024]['function_recall'])} | {num(budgets[1024]['changed_line_recall'])} |
| 2048 | {num(budgets[2048]['file_recall'])} | {num(budgets[2048]['function_recall'])} | {num(budgets[2048]['changed_line_recall'])} |
| 4096 | {num(budgets[4096]['file_recall'])} | {num(budgets[4096]['function_recall'])} | {num(budgets[4096]['changed_line_recall'])} |

Потому что порядок фрагментов не зависит от бюджета, рост бюджета может улучшить строковую метрику только если нужный чанк стоит в этом порядке достаточно рано. Если разница Changed Line Recall почти одинакова на трёх бюджетах, иерархия переставила начало списка, а не «дотянула» строку за счёт лишнего места. Если разница растёт к 4096, дополнительный бюджет начал доходить до вторых и третьих чанков нужной функции. Оба исхода нужно читать по таблице выше, а не по одному бюджету.

## Производительность

Время запроса v0.5 на задаче, усреднённое по максимальному бюджету каждого варианта (`stage_profile_mean.csv`). Сканирование репозитория и построение BM25-индекса в эти числа не входят.

{table([{k: (row[k] if k=='method' else float(row[k]) if row.get(k) else None) for k in ('method','candidate_sec','graph_sec','file_rank_sec','chunk_score_sec','diversity_sec','anchor_sec','pack_stream_sec','prefix_sec','query_sec')} for row in profiles], (('method','Вариант'),('candidate_sec','Кандидаты'),('graph_sec','Граф'),('file_rank_sec','Файлы'),('chunk_score_sec','Чанки'),('diversity_sec','Diversity'),('anchor_sec','Якорь'),('pack_stream_sec','Сборка'),('prefix_sec','Префикс'),('query_sec','Запрос'))) if profiles else 'Профиль не записан.'}

На смоук-задаче `django__django-12276` полный v0.5 уложился примерно в половину секунды запроса при пуле в несколько сотен чанков. Основная доля запроса — обход графа и попарная новизна, а не упаковка префикса. Это не оптимизация ценой другого контекста: порядок и состав чанков определяются той же формулой.

## Ограничения

- 20 задач, зависимые репозитории, одна реализация, один проход latency. Bootstrap описывает разброс по этим задачам, а не генерализацию на SWE-bench.
- 16 из 20 задач v0.4 имели неполную разметку. Density и Changed Line Recall усредняются только там, где знаменатель существует.
- Density наказывает поясняющие строки. Высокая density не равна полезности для модели.
- Статический граф пропускает динамические вызовы. Нулевой GraphScore не означает, что зависимости нет.
- Embedding retrieval не выполнен.
- Holdout `output/v05_holdout/` не прогонялся. Успех патча языковой моделью не измерялся.
- Аудит бюджета прогона: ok={audit['ok']}, нарушений={len(audit['violations'])}.
'''
    # The case-count sentence above used the wrong key. Patch it from summary if present.
    summary_audit = {}
    summary_path = args.audit / 'summary.json'
    if summary_path.exists():
        summary_audit = json.loads(summary_path.read_text())
    counts = summary_audit.get('case_counts', {})
    text = text.replace(
        'На 20 задачах встретились случаи: файл найден, функция не покрыта (A: см. summary). Точные счётчики случаев лежат в `output/v05_audit/summary.json`.',
        'На 20 задачах счётчики случаев (задача может получить несколько букв): '
        + ', '.join(f'{key}={counts[key]}' for key in sorted(counts))
        + '. Подробности в `output/v05_audit/summary.json`.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(json.dumps({'report': str(args.out), 'file_recall': v5['file_recall'], 'function_recall': v5['function_recall'],
                      'changed_line_recall': v5['changed_line_recall'], 'complete_function_recall': v5['complete_function_recall']}, indent=2))


if __name__ == '__main__':
    main()
