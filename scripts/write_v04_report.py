"""Build the Russian research report exclusively from measured artifacts."""
import argparse,csv,json,hashlib,re
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]

def markdown_table(rows,columns):
    def val(v):
        if v is None or v=='':return '—'
        if isinstance(v,float):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(label for key,label in columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+''.join('| '+' | '.join(val(row.get(key)) for key,label in columns)+' |\n' for row in rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=ROOT/'output/v04_dev');parser.add_argument('--out',type=Path,default=ROOT/'docs/PATHPACK_V04_RU.md');args=parser.parse_args()
    run=args.run.resolve();meta=json.loads((run/'run.json').read_text());audit=json.loads((run/'audit.json').read_text())
    assert meta['status']=='scored' and audit['ok'],'Only completed runs passing the audit can enter the report'
    with (run/'summary.csv').open() as f:
        summary=[{k:float(v) if k not in {'method'} and v else v for k,v in row.items()} for row in csv.DictReader(f)]
    for row in summary:row['budget']=int(row['budget']);row['tasks']=int(row['tasks'])
    main_rows=sorted((r for r in summary if '/' not in r['method']),key=lambda r:(r['budget'],r['method']))
    ablation=[r for r in summary if r['method'].startswith('pathpack-v4') and r['budget']==2048]
    full=next(r for r in summary if r['method']=='pathpack-v4' and r['budget']==2048)
    v4_by_budget={r['budget']:r for r in summary if r['method']=='pathpack-v4'}
    diversity_off=next(r for r in summary if r['method']=='pathpack-v4/no-diversity' and r['budget']==2048)
    chunking_off=next(r for r in summary if r['method']=='pathpack-v4/no-chunking' and r['budget']==2048)
    graph_large=next(r for r in summary if r['method']=='graph-expand' and r['budget']==4096)
    bm25=next(r for r in summary if r['method']=='bm25' and r['budget']==2048)
    no_graph=next(r for r in summary if r['method']=='pathpack-v4/no-graph' and r['budget']==2048)
    comparisons=json.loads((run/'comparisons.json').read_text())
    chosen=[]
    for r in comparisons:
        if r['budget']==2048 and r['metric'] in {'file_recall','function_recall'} and r['baseline'] in {'bm25','graph-expand','pathpack','pathpack-v3','pathpack-v4/no-graph','pathpack-v4/no-chunking','pathpack-v4/no-diversity','pathpack-v4/no-cost','pathpack-v4/no-symbol'}:
            chosen.append({**r,'ci95':str(r['ci95'])})
    labels=[json.loads(s) for s in (run/'scoring_labels.jsonl').read_text().splitlines()]
    completeness=Counter(r['label_completeness'] for r in labels)
    exclusions=Counter(e['reason'] for r in labels for e in r['exclusions'])
    losing=json.loads((run/'losing_cases.json').read_text())
    failures=[json.loads(s) for s in (run/'retrieval_failures.jsonl').read_text().splitlines()]
    flags=Counter(flag for r in failures if r['method']=='pathpack-v4' for flag in r['flags'])
    with (run/'budget_summary.csv').open() as f:budget_rows=list(csv.DictReader(f))
    utilization=[]
    for budget in meta['budgets']:
        rows=[r for r in budget_rows if r['method']=='pathpack-v4' and int(r['budget_est'])==budget]
        utilization.append({'budget':budget,'contexts':len(rows),'at_least_95_percent':sum(float(r['utilization'])>=.95 for r in rows),'mean':sum(float(r['utilization']) for r in rows)/len(rows)})
    test_log=(ROOT/'output/v04_validation/tests.log').read_text();tests=re.search(r'Ran (\d+) tests',test_log).group(1)
    assert '\nOK' in test_log,'Tests must pass'
    baseline=json.loads((ROOT/'reports/v04_baseline.json').read_text())
    for path,expected in baseline['artifacts_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,f'Historical artifact changed: {path}'
    graph_effect=full['file_recall']-no_graph['file_recall']
    graph_interpretation=('На dev включение графа повысило File Recall@2048 на '+f'{graph_effect:.4f} относительно No Graph; парный CI указан ниже. Это ещё не подтверждение обобщения.') if graph_effect>0 else ('На dev граф не дал преимущества по File Recall@2048: Full − No Graph = '+f'{graph_effect:.4f}. Инженерная работоспособность не доказывает полезность графового сигнала.')
    report=f'''# CodeGraph-RAG / PathPack v0.4: реализация и dev-исследование

Дата: 1 ноября 2026. Пакет 0.4.0, существующий проект v0.3. Оценка только на **{len(meta['issues'])} dev-задачах**, по одному последовательному измерению latency; старые test и новый holdout не оценивались. Научная новизна и patch success не установлены.

## Реализовано

1. Adaptive AST Chunking: обычные/async/вложенные функции, методы, декораторы, AST-блоки и fallback по строкам/колонкам. Размеры 128/256/32 proxy-токена, overlap 2 строки. Чанки сохраняют точные исходные интервалы, сигнатуру, родителя и function_id. Function Graph не размножается до графа чанков.
2. Hybrid Ranking: нормализованный BM25, имена/пути, двунаправленные типизированные графовые пути до двух hops, мягкое разнообразие и стоимость. Индексирование отдельно от запроса. Каждый компонент выключается конфигурацией.
3. Packing: единый порядок без зависимости от бюджета, минимальные заголовки, удаление исходных перекрытий, префиксы по исходным атомам, частичный последний чанк. Префикс не обрывает самостоятельный заголовок без кода.
4. Диагностика: компоненты score, graph paths, размеры, реальные причины отказа, budget summary; JSON и CSV (сжатые gzip для полных кандидатов). Scoring отдельно формирует ошибки и проигрыши baselines.
5. Benchmark: пять методов и шесть вариантов v0.4, исходные ids/split/base_commit, проверка clean snapshots, bootstrap 2000/seed42, версии зависимостей/платформа и снимки исходников. CLI и прежние методы совместимы.

Точная эвристика: `E=.45L+.20N+.25G`; `J` — максимальное Jaccard-сходство с уже упорядоченным контекстом; `D=(1-J)/(1+n_file)^1`; `S=E*(1-.5J)+.10D`; `U=S/(T+32)^.5`. Все коэффициенты вынесены в JSON. BM25 нормализуется делением на максимум: нулевые оценки остаются нулевыми, одинаковые положительные становятся единицами. Graph score — произведение весов рёбер и decay .8 на переход. Префильтр top-512 evidence-кандидатов ограничивает стоимость динамического порядка и явно диагностируется. Параметры не подбирались по результатам этого прогона.

Условные/повторные определения с одним qualname имеют различные стабильные номера чанков и точные интервалы. Старый Function Graph объединяет такой символ; неоднозначность отмечена в диагностике. Gold-разметка различает определения по строкам.

## Протокол и определения метрик

Существующий манифест 80 задач сохранён: 20 dev / 60 исследовательских test, seed42. Все методы пересчитаны на одинаковых задачах при 1024/2048/4096. Baselines используют прежние алгоритмы с общим сериализатором заголовков, строк и разделителей; результаты исторических таблиц с другим счётчиком/разметкой не смешиваются с этими числами.

- **File Recall**: доля существовавших Python-файлов gold patch, переданных контекстом; macro mean только по задачам с такими файлами.
- **Function Recall**: функция найдена, когда полностью передано ≥20% её собственных непустых строк тела (округление вверх, минимум одна). Docstring, комментарии и вложенные определения исключены. Заголовки, имя, дополнительная сигнатура и неполная строка не дают попадание. Macro mean по задачам с сопоставимыми функциями. Полное 100% покрытие публикуется отдельно.
- **Changed Line Recall**: доля покрытых точных старых удалённых/заменённых непустых строк; pure insertions не считаются точными старыми строками. Macro mean по задачам с таким знаменателем.
- **Latency**: BM25 query scoring + порядок + packing; AST и построение индексов отдельно. Выгрузка полной диагностики не включена. Это одиночный прогон, без доверительного интервала времени.
- **Utilization**: стоимость полного сериализованного контекста / бюджет, затем macro mean по всем задачам.
- **Useful Functions/1K**: число gold-функций, удовлетворяющих правилу покрытия, ×1000 / использованные токены; при пустом контексте — 0, затем macro mean только по function-evaluable задачам.

Единый счётчик: `{meta['token_counter']}` — **приближённые UTF-8 bytes/4**, не Gemma tokenizer. Baseline-упаковщики используют консервативную сумму округлённых стоимостей чанков, итоговый контекст всех методов проверяется точным для этого proxy глобальным счётчиком.

Из {len(labels)} задач полная разметка: **{completeness.get('complete',0)}** ({100*completeness.get('complete',0)/len(labels):.1f}%); частичная: **{completeness.get('partial',0)}** ({100*completeness.get('partial',0)/len(labels):.1f}%). Сопоставимые функции при @2048: {int(full['function_recall_evaluable'])}; точные старые изменённые строки: {int(full['changed_line_recall_evaluable'])}. Неизвестные значения исключаются из соответствующего знаменателя, а не превращаются в отрицательные примеры. Причины исключений: `{dict(exclusions)}`.

Замены и вставки разобраны отдельными diff-блоками: replacement не расширяет gold на соседнюю неизменённую функцию. Чистая вставка привязывается к старой функции только при согласованных соседних якорях; новые определения, файлы и неопределённые границы отмечаются отдельно. Переименование использует старый путь на base_commit.

## Измеренные основные результаты

{markdown_table(main_rows,[('method','Метод'),('budget','Budget'),('file_recall','File Recall'),('function_recall','Function Recall'),('changed_line_recall','Changed Lines'),('token_utilization','Utilization'),('latency_sec','Latency, s')])}

При 2048 v0.4 − BM25: File Recall **{full['file_recall']-bm25['file_recall']:+.4f}**, Function Recall **{full['function_recall']-bm25['function_recall']:+.4f}**. Цели ТЗ (0.65 / 0.20) являются исследовательскими ориентирами, а не основанием менять метрики. Фактически v0.4 @2048: **{full['file_recall']:.4f} / {full['function_recall']:.4f}**.

## Абляции @2048

{markdown_table(ablation,[('method','Режим'),('file_recall','File Recall'),('function_recall','Function Recall'),('changed_line_recall','Changed Lines'),('token_utilization','Utilization'),('latency_sec','Latency, s')])}

{graph_interpretation}

Важная отрицательная сторона: Changed Line Recall Full = **{full['changed_line_recall']:.4f}**, No Graph = **{no_graph['changed_line_recall']:.4f}**. Рост File Recall не означает лучшего покрытия исправляемых строк. У Full Changed Line Recall при 1024/2048/4096 равен **{v4_by_budget[1024]['changed_line_recall']:.4f} / {v4_by_budget[2048]['changed_line_recall']:.4f} / {v4_by_budget[4096]['changed_line_recall']:.4f}** — дополнительные токены не улучшили эту метрику.

На @2048 отключение Diversity снижает File Recall с {full['file_recall']:.4f} до {diversity_off['file_recall']:.4f}; отключение Chunking даёт {chunking_off['file_recall']:.4f}. Поэтому основной прирост нельзя автоматически приписать графу или AST-разбиению. Main-сравнение объединяет изменения корпуса, ранжирования и packing; только контролируемые абляции отделяют их вклад. Latency Full **{full['latency_sec']:.4f}s**, BM25 **{bm25['latency_sec']:.4f}s**, примерно **{full['latency_sec']/bm25['latency_sec']:.1f}×**; более высокая полнота оплачивается временем поиска.

При 4096 Function Recall Graph Expansion = **{graph_large['function_recall']:.4f}**, у v0.4 — **{v4_by_budget[4096]['function_recall']:.4f}**; Changed Line Recall соответственно **{graph_large['changed_line_recall']:.4f} / {v4_by_budget[4096]['changed_line_recall']:.4f}**. v0.4 не доминирует baselines по всем метрикам и бюджетам.

Сравнение Full/No Chunking отдельно показывает эффект размера/packing; No Diversity, No Symbol и No Cost изолируют соответствующие части эвристики.

Заполнение ≥95% по задачам:

{markdown_table(utilization,[('budget','Budget'),('at_least_95_percent','≥95%'),('contexts','Всего'),('mean','Средняя доля')])}

Высокое заполнение само по себе не означает релевантность: полезность оценивается отдельными recall и changed-line метриками.

## Парный bootstrap @2048

Разница = v0.4 − baseline; единица ресэмплинга — задача, seed42, 2000 повторений. Bootstrap не учитывает корреляцию задач одного репозитория, поэтому интервалы не заменяют более крупную независимую проверку.

{markdown_table(chosen,[('baseline','Baseline'),('metric','Метрика'),('difference','Разница'),('ci95','95% CI'),('n','n'),('wins','Лучше'),('ties','Равно'),('losses','Хуже')])}

## Где алгоритм проигрывает

В `losing_cases.json/csv` сохранено {len(losing)} сравнений задача×бюджет×baseline×метрика, где v0.4 хуже BM25 / legacy PathPack / v0.3. Это не число уникальных задач. Диагностические флаги Full (60 контекстов): `{dict(flags)}`. Примеры:

{markdown_table(losing[:12],[('instance_id','Задача'),('budget','Budget'),('baseline','Baseline'),('metric','Метрика'),('v4','v0.4'),('baseline_value','Baseline value'),('diagnostics','Диагностика')])}

Причины — диагностические признаки, а не доказанные причинные объяснения: верный файл с пропущенной функцией, релевантный кандидат за границей бюджета, недостаточное покрытие изменённых строк. Для окончательного анализа нужно читать соответствующие `source_spans`, score и graph paths, а не только агрегаты.

## Проверки и воспроизводимость

**{tests} тестов прошли**: AST-случаи, стабильные ids, условные повторные определения, unicode/длинная строка, max size, ссылки на граф, отсутствие повторения source spans/тел, exact budget prefixes, детерминизм, отключения компонентов, корректность известных метрик, no gold leakage и совместимость старых методов. Малый fixture benchmark проходит все этапы, затем выполнены реальный smoke и полный dev.

Аудит итогового запуска: **{audit['predictions']} предсказаний, {len(audit['violations'])} нарушений**. Старые greedy baselines могут менять состав при росте бюджета; это фиксируется отдельно и не является нарушением инварианта v0.4.

Первый полный запуск выявил повторный id условных реализаций `six.b` в Requests. Он сохранён в `output/v04_dev_initial_failed/` со статусом invalidated и не используется для итоговых таблиц. После исправления и регрессионного теста весь dev повторён с теми же весами и задачами. Это исправление корректности, а не подбор по gold.

Код retrieval: `{meta['code_sha256']}`; Git commit: `{meta['git_commit']}`. Для рабочей копии без `.git` сохранены SHA256 и `code_snapshot/`. Код scoring отдельно сохранён в `scoring_code_snapshot/` и имеет SHA256 `{meta.get('scoring_code_sha256')}`. Конфигурация, issues/base_commit, зависимости, Python и платформа — в `run.json`. Исходные v0.2/v0.3 результаты проверены по сохранённым SHA256 и не изменены.

Повтор: `RUN_OUT=output/v04_dev_repeat bash scripts/run_v04_dev.sh`. Блокнот: [04_pathpack_v04.ipynb](../notebooks/04_pathpack_v04.ipynb). Результаты: [summary.csv](../output/v04_dev/summary.csv), [comparisons.json](../output/v04_dev/comparisons.json), [audit.json](../output/v04_dev/audit.json). Графики PNG/SVG — `output/v04_dev/figures/`.

## Ограничения и следующие эксперименты

- Нужен настоящий tokenizer целевой модели и повтор всех методов при его лимите.
- Function Graph статически приближённый: динамические вызовы, полиморфизм и общие qualname условных определений не полностью разрешены. Top-512 — дополнительное ограничение recall.
- AST-чанки могут потерять imports/модульные переменные и часть тела; body coverage и changed-line recall отражают это лишь частично.
- Порог 20% — заранее заданное инженерное определение, не гарантия достаточного контекста для исправления. Полное покрытие опубликовано отдельно в CSV.
- В текущем окружении нет совместимого локального sentence/code embedding encoder с фиксированными весами. Dense baseline честно пропущен; облачных API и загрузки весов не было.
- Новый holdout содержит 60 instance-disjoint задач, исключены прежние 80. Репозитории пересекаются; это не repository-disjoint validation. Holdout не запускался и не участвовал в настройке.
- Нужны предрегистрация последующего протокола, больше задач, bootstrap с учётом репозиториев, несколько временных повторов, локальный embedding baseline и отдельная оценка patch success.

Инженерные компоненты реализованы и проверены. Научную ценность графового сигнала следует оценивать по абляциям и новой независимой проверке; одна dev-таблица не устанавливает новизну.
'''
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(report)
    print(args.out)

if __name__=='__main__':main()
