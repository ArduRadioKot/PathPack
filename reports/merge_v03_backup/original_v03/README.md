# CodeGraph-RAG / PathPack v0.3

**Экспериментальное обновление от 1 ноября 2026.** Включает исходный код v0.2, новый метод `pathpack-v3`, инструменты оценки, диагностику ошибок, исследовательский ноутбук и визуализации.

**[Главный отчёт на русском](docs/STATUS_V03_RU.md)** · [Ноутбук v0.3](notebooks/03_pathpack_v03_audit.ipynb) · [Результаты пользователя в CSV](reports/user_aggregate_2026-11-01.csv)

## Запуск на Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[viz]'
python -m unittest discover -s tests -v
python scripts_make_demo.py
codegraph retrieve examples/issue_demo/snapshot --query 'login cookie causes failure' --method pathpack-v3 --budget 2048
```

Если уже используется v0.2, создайте отдельное окружение для v0.3. Метод `pathpack` оставлен неизменённым, новый метод называется `pathpack-v3`.

## Реальный эксперимент (после подготовки датасета и проверенных Git snapshots)

```bash
pip install -e '.[benchmark,viz]'
codegraph prepare --out output/benchmark --limit 80 --dev-count 20 --seed 42
# Необходимо скачать snapshots/<instance_id> на точные base_commit.
bash scripts/run_dev.sh
```

Если нужны gold-метки функций, создайте **отдельный** `raw_verified.jsonl` с оригинальными SWE-bench строками, включая patch. Никогда не передавайте этот файл коду retrieval или модели.

```bash
codegraph gold-functions --raw raw_verified.jsonl --snapshots snapshots \
  --gold output/benchmark/gold_labels.jsonl \
  --out output/benchmark/gold_functions.jsonl
codegraph audit --predictions output/v03_dev/*.jsonl \
  --gold output/benchmark/gold_functions.jsonl --out reports/audit_functions_dev.json
```

## Что не доказано

- PathPack v0.3 пока не улучшает File/Function Recall на реальном SWE-bench: доступа к сырым выводам пользователя в данной рабочей среде не было.
- На **4 синтетических задачах** при бюджете 110 PathPack v0.3 даёт **3/4**, а старый PathPack и Graph Expansion **4/4**. Это честный отрицательный результат.
- Метрики со скриншота пользователя использованы только для анализа исходных агрегатов; новые значения не приписываются v0.3.
- Бюджеты считаются приблизительным UTF-8 proxy, а не токенизатором Gemma.

`gold-functions` и `audit` работают **только для оценки после retrieval**, во избежание утечки ответов в поисковый алгоритм.

## Структура

- `src/codegraph_rag/retrieval.py` - BM25 / graph-expand / PathPack / PathPack v0.3.
- `src/codegraph_rag/audit.py` - проверки и парный bootstrap.
- `src/codegraph_rag/function_gold.py` - приближённая функция-level разметка gold.
- `scripts/run_dev.sh` - воспроизводимые прогоны на dev.
- `scripts_make_demo.py` - искусственные результаты со скриптом аудита.
- `notebooks/03_pathpack_v03_audit.ipynb` - визуализации и воспроизводимые примеры.
- `docs/STATUS_V03_RU.md` - выводы, риски, план.

В архиве сохранены предыдущие документы и графы v0.2.
