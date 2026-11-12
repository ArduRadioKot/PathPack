"""Write notebooks/11_pathpack_h.ipynb. Every number is read from saved run files; nothing is typed in by hand."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text.strip()))
code = lambda text: cells.append(nbf.v4.new_code_cell(text.strip()))

md('''
# PathPack-H: локализация правок, обученная на истории репозитория

Все числа читаются из сохранённых прогонов:

- `output/pathpack_val_all`: validation, 99 непросмотренных задач (использован для выбора варианта);
- `output/pathpack_holdout`: final holdout, 100 задач, один прогон после заморозки `output/pathpack_e/FREEZE_before_holdout.json`.

Токены условные: `ceil(UTF-8 bytes / 4)`. Это не токенизатор Gemma. Долю исправленных ошибок ноутбук не измеряет.
''')

code('''
import csv, gzip, json, collections
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / 'output').is_dir() else Path.cwd().parent
VAL, HOLD = ROOT / 'output/pathpack_val_all', ROOT / 'output/pathpack_holdout'
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#6b6a64', '#e4e3de'
BLUE, ORANGE, AQUA, YELLOW, NEUTRAL = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#b9b8b1'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                     'text.color': INK, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.grid': False, 'figure.dpi': 110})
METRICS = [('file_recall', 'File Recall'), ('function_recall', 'Function Recall'),
           ('complete_function_recall', 'Complete Function Recall'), ('changed_line_recall', 'Changed Line Recall')]
NAMES = {'bm25': 'BM25', 'pathpack-v4': 'PathPack v0.4', 'pathpack-v10e': 'v1.0e (исправленная остановка)',
         'pathpack-r/fusion': 'Fusion (ручная формула)', 'pathpack-e': 'PathPack-E (метки, 34 задачи)',
         'pathpack-h': 'PathPack-H с памятью коммитов', 'pathpack-h/nomemory': 'PathPack-H (основной)',
         'pathpack-h/nohist': 'H без признаков истории', 'pathpack-h/nograph': 'H без графового раунда',
         'pathpack-h/ranked': 'H, упаковка ранжированием', 'pathpack-h/skeleton': 'H со скелетами',
         'pathpack-eh': 'PathPack-EH (метки + история)', 'pathpack-r/fusion-epack': 'Fusion + упаковка по покрытию',
         'pathpack-r': 'PathPack-R (reranker + active)', 'pathpack-r/reranker-graph': 'Reranker MS-MARCO + граф'}
PRIMARY = 'pathpack-h/nomemory'

def summary(run):
    df = pd.read_csv(run / 'summary.csv')
    return df

def scores(run):
    return pd.DataFrame([json.loads(l) for l in open(run / 'scores.jsonl')])

freeze = json.load(open(ROOT / 'output/pathpack_e/FREEZE_before_holdout.json'))
print('Основной вариант (выбран по validation до holdout):', freeze['primary'])
print('Правило выбора:', freeze['selection_rule'])
''')

md('''
## 1. Итог на holdout при бюджете 2048

Одна панель на метрику, общий порядок методов. Синим выделен основной вариант, выбранный до прогона. Остальные методы серые: идентичность несёт подпись строки.
''')

code('''
hold = summary(HOLD)
order = [PRIMARY, 'pathpack-e', 'pathpack-r/fusion', 'pathpack-v10e', 'pathpack-v4', 'bm25']
at = hold[hold.budget == 2048].set_index('method').loc[order]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
for ax, (key, title) in zip(axes, METRICS):
    vals = at[key].values
    bars = ax.barh(range(len(order)), vals, height=0.62, color=[BLUE if m == PRIMARY else NEUTRAL for m in order])
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_title(title, fontsize=10, loc='left', color=INK)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[m] for m in order])
    ax.set_xticks([0, 0.5, 1]); ax.tick_params(length=0)
    n = int(at[key + '_evaluable'].iloc[0]); ax.set_xlabel(f'n = {n} задач')
axes[0].invert_yaxis()
fig.suptitle('Final holdout, 100 задач, бюджет 2048 условных токенов', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
at[[k for k, _ in METRICS]].rename(index=NAMES).round(3)
''')

md('''
## 2. Парные разницы с доверительными интервалами

Точка: средняя разница по задачам, отрезок: 95% интервал парного bootstrap по задачам (2000 итераций, seed 42). Для основных сравнений ниже отдельной таблицей дан интервал с кластеризацией по репозиториям.
''')

code('''
import random
def paired(df, cand, base, metric, budget=2048, iters=2000, seed=42):
    a = df[(df.method == cand) & (df.budget == budget)].set_index('instance_id')[metric].dropna()
    b = df[(df.method == base) & (df.budget == budget)].set_index('instance_id')[metric].dropna()
    ids = sorted(set(a.index) & set(b.index)); d = [a[i] - b[i] for i in ids]
    rng = random.Random(seed); bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return dict(diff=sum(d) / len(d), lo=bs[int(0.025 * iters)], hi=bs[int(0.975 * iters) - 1], n=len(d),
                wins=sum(x > 0 for x in d), ties=sum(x == 0 for x in d), losses=sum(x < 0 for x in d))

hs = scores(HOLD)
rows = []
for base in ('pathpack-v4', 'pathpack-r/fusion', 'pathpack-e'):
    for key, title in METRICS:
        rows.append({'сравнение': f'H − {NAMES[base]}', 'метрика': title, **paired(hs, PRIMARY, base, key)})
cmp = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), sharex=True, sharey=True)
for ax, (name, part) in zip(axes, cmp.groupby('сравнение', sort=False)):
    y = range(len(part))
    ax.axvline(0, color=MUTED, lw=1)
    ax.hlines(y, part.lo, part.hi, color=BLUE, lw=2)
    ax.plot(part['diff'], y, 'o', color=BLUE, ms=7, mec=SURFACE, mew=2)
    for i, r in enumerate(part.itertuples()):
        ax.text(r.hi + 0.015, i, f'{r.diff:+.3f}', va='center', fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(part['метрика'])
    ax.set_title(name, fontsize=10, loc='left'); ax.tick_params(length=0); ax.set_xlim(-0.2, 0.55)
    ax.set_xlabel('разница, 95% ДИ по задачам')
axes[0].invert_yaxis()
fig.suptitle('Holdout, бюджет 2048: основной вариант минус baseline', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
cmp.round(3)
''')

code('''
cluster = json.load(open(HOLD / 'cluster_bootstrap.json'))
pd.DataFrame([{'сравнение': k.rsplit(' ', 1)[0], 'метрика': k.rsplit(' ', 1)[1], 'разница': round(v['difference'], 3),
               'ДИ по репозиториям': [round(x, 3) for x in v['ci95_repo_cluster']], 'кластеров': v['repos']}
              for k, v in cluster.items()])
''')

md('''
## 3. Зависимость от бюджета

Проверка, что вывод не привязан к одному бюджету. У v0.4 покрытие функций и строк почти не растёт с бюджетом: лишние токены уходят на новые файлы.
''')

code('''
lines = [('pathpack-v4', ORANGE), ('bm25', AQUA), ('pathpack-r/fusion', YELLOW), (PRIMARY, BLUE)]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
pos = {1024: 0, 2048: 1, 4096: 2}
for ax, (key, title) in zip(axes, METRICS):
    for method, color in lines:
        part = hold[hold.method == method].sort_values('budget')
        ax.plot([pos[b] for b in part.budget], part[key], color=color, lw=2, marker='o', ms=7, mec=SURFACE, mew=1.5,
                label=NAMES[method])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['1024', '2048', '4096']); ax.set_xlim(-0.25, 2.25); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, loc='left'); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0); ax.set_xlabel('бюджет, усл. токены')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc='lower left', bbox_to_anchor=(0.03, -0.02))
fig.suptitle('Holdout: метрики при трёх бюджетах', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0.07, 1, 1)); plt.show()
hold[hold.method.isin([m for m, _ in lines])].pivot(index='method', columns='budget', values=[k for k, _ in METRICS]).rename(index=NAMES).round(3)
''')

md('''
## 4. Абляции: что в PathPack-H даёт вклад

Слева validation (по нему выбирался вариант), справа holdout. Различия между вариантами H малы относительно интервалов; это вторичный анализ без поправки на множественные сравнения.
''')

code('''
val = summary(VAL)
abl_val = ['pathpack-h/nomemory', 'pathpack-h', 'pathpack-h/nohist', 'pathpack-h/nograph', 'pathpack-h/skeleton',
           'pathpack-h/ranked', 'pathpack-eh', 'pathpack-e', 'pathpack-r/fusion', 'pathpack-r/fusion-epack', 'pathpack-v4']
abl_hold = [m for m in abl_val if m in set(hold.method)]
def table(df, methods):
    return df[df.budget == 2048].set_index('method').loc[methods][[k for k, _ in METRICS]].rename(index=NAMES).round(3)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
marks = dict(zip([k for k, _ in METRICS], [(BLUE, 'o'), (ORANGE, 's'), (AQUA, '^'), (YELLOW, 'D')]))
for ax, (df, methods, title) in zip(axes, [(val, abl_val, 'Validation, 99 задач'), (hold, abl_hold, 'Holdout, 100 задач')]):
    t = df[df.budget == 2048].set_index('method').loc[methods]
    for i in range(len(methods)):
        ax.axhline(i, color=GRID, lw=0.8, zorder=0)
    for key, label in METRICS:
        color, marker = marks[key]
        ax.plot(t[key], range(len(methods)), marker, color=color, ms=7, mec=SURFACE, mew=1.5, label=label)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([NAMES[m] for m in methods]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_title(title, fontsize=10, loc='left'); ax.tick_params(length=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc='lower left', bbox_to_anchor=(0.18, -0.02))
fig.suptitle('Абляции при бюджете 2048', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()
display(table(val, abl_val)); table(hold, abl_hold)
''')

md('''
## 5. Где теряется код: пул против упакованного контекста

Доля золотых файлов и функций в пуле кандидатов и в итоговом контексте (holdout, 2048). Пул у всех вариантов PathPack-R/E/H общий, различается выбор из него.
''')

code('''
pool = pd.read_csv(HOLD / 'pool_recall.csv'); pool = pool[pool.budget == 2048]
agg = pool.groupby('method')[['file_pool_recall', 'file_packed_recall', 'function_pool_recall', 'function_packed_recall']].mean()
show = [m for m in ['pathpack-r/fusion', 'pathpack-e', PRIMARY] if m in agg.index]
fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharey=True)
for ax, (kind, title) in zip(axes, [('file', 'Файлы'), ('function', 'Функции')]):
    for i, m in enumerate(show):
        p, k = agg.loc[m, f'{kind}_pool_recall'], agg.loc[m, f'{kind}_packed_recall']
        ax.hlines(i, k, p, color=NEUTRAL, lw=2)
        ax.plot(p, i, 'o', color=NEUTRAL, ms=8, mec=SURFACE, mew=1.5, label='в пуле' if i == 0 else None)
        ax.plot(k, i, 'o', color=BLUE, ms=8, mec=SURFACE, mew=1.5, label='в контексте' if i == 0 else None)
        ax.text(p + 0.02, i, f'{p:.2f}', va='center', fontsize=9); ax.text(k - 0.02, i, f'{k:.2f}', va='center', ha='right', fontsize=9)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([NAMES[m] for m in show])
    ax.set_xlim(0, 1.08); ax.set_title(title, fontsize=10, loc='left'); ax.tick_params(length=0)
axes[0].invert_yaxis(); axes[0].legend(frameon=False, fontsize=8, loc='center left', bbox_to_anchor=(0.0, 0.5))
fig.suptitle('Holdout, 2048: пул кандидатов и упакованный контекст', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
agg.loc[show].rename(index=NAMES).round(3)
''')

md('''
## 6. Неоднородность по репозиториям

Средняя разница Function Recall «основной вариант минус v0.4» по репозиториям holdout. Число задач подписано: в малых репозиториях по 1 to 6 задач, выводы по ним ненадёжны.
''')

code('''
key = 'pathpack-h/nomemory vs pathpack-v4 function_recall'
per = pd.DataFrame({'diff': cluster[key]['per_repo_mean_diff'], 'n': cluster[key]['per_repo_n']}).sort_values('n', ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.axvline(0, color=MUTED, lw=1)
ax.barh(range(len(per)), per['diff'], height=0.6, color=BLUE)
for i, r in enumerate(per.itertuples()):
    ax.text(max(r.diff, 0) + 0.02, i, f'{r.diff:+.2f}  (n={r.n})', va='center', fontsize=9)
ax.set_yticks(range(len(per))); ax.set_yticklabels([i.split('__')[-1] for i in per.index]); ax.invert_yaxis()
ax.set_xlim(-0.45, 0.8); ax.tick_params(length=0); ax.set_xlabel('разница Function Recall')
ax.set_title('Holdout, 2048: H − v0.4 по репозиториям', fontsize=11, loc='left')
fig.tight_layout(); plt.show(); per.round(3)
''')

md('''
## 7. Откуда обучающие примеры PathPack-H

Псевдо-задачи из коммитов, предшествующих самой ранней задаче Verified в каждом репозитории. Метки бенчмарка не использовались.
''')

code('''
plan = json.load(open(ROOT / 'output/pathpack_e/replay_logs/plan.json'))
done = {p.stem: json.load(open(p)) for p in (ROOT / 'output/pathpack_e/train_history').glob('*.done')}
rep = pd.DataFrame([{'репозиторий': i['repo'], 'снимок': i['instance_id'],
                     'граница (UTC)': pd.to_datetime(i['cutoff'], unit='s').date(),
                     'коммитов в логе': done.get(i['repo'].replace('/', '__'), {}).get('commits'),
                     'псевдо-задач': done.get(i['repo'].replace('/', '__'), {}).get('kept'),
                     'фрагментов в снимке': done.get(i['repo'].replace('/', '__'), {}).get('chunks')} for i in plan])
print('всего псевдо-задач:', int(rep['псевдо-задач'].sum()))
print('пакетов логов, не скачанных в частичном клоне:', open(ROOT / 'output/pathpack_e/replay_fetch.log').read().count('failed:'))
rep
''')

md('''
## 8. Один пример: что выбрал алгоритм

Первая задача holdout, где основной вариант покрыл изменённые строки, а v0.4 нет. Показаны выбранные фрагменты с вероятностью правки.
''')

code('''
a = hs[(hs.method == PRIMARY) & (hs.budget == 2048)].set_index('instance_id')
b = hs[(hs.method == 'pathpack-v4') & (hs.budget == 2048)].set_index('instance_id')
better = [i for i in a.index if pd.notna(a.loc[i, 'changed_line_recall']) and a.loc[i, 'changed_line_recall'] > b.loc[i, 'changed_line_recall']]
iid = sorted(better)[0]
issue = next(json.loads(l) for l in open(ROOT / 'output/pathpack_r_splits_v2/final_holdout_100_issues.jsonl') if json.loads(l)['instance_id'] == iid)
print(iid); print(issue['problem_statement'][:700], '...')
print('Changed Line Recall: H =', a.loc[iid, 'changed_line_recall'], '| v0.4 =', b.loc[iid, 'changed_line_recall'])
diag = json.load(gzip.open(HOLD / 'diagnostics' / f"{iid}_{PRIMARY.replace('/', '_')}_2048.json.gz", 'rt'))
cand = pd.DataFrame(diag['candidates'])
sel = cand[cand.decision == 'selected'].sort_values('final_score', ascending=False)
print('кандидатов в пуле:', len(cand), '| выбрано:', len(sel), '| файлов:', sel.path.nunique())
sel[['path', 'function', 'line_start', 'line_end', 'token_cost', 'final_score', 'channels']].head(15).round(3)
''')

md('''
## 9. PathPack-U на втором отложенном наборе

PathPack-U добавляет к модели фрагмента вероятности правки функции и файла и упаковывает контекст, максимизируя сумму ожидаемых File, Function, Complete и Changed Line Recall. Он оценивался на 51 задаче, не входившей в первый holdout (`output/pathpack_holdout2`, заморозка `FREEZE_before_holdout2.json`). Набор мал, интервалы широкие.
''')

code('''
H2 = ROOT / 'output/pathpack_holdout2'
NAMES['pathpack-u'] = 'PathPack-U'; NAMES['pathpack-u/nobundle'] = 'U без хода «функция целиком»'
h2 = summary(H2)
order2 = ['pathpack-u', 'pathpack-u/nobundle', 'pathpack-h/nomemory', 'pathpack-r/fusion', 'pathpack-v4', 'bm25']
at2 = h2[h2.budget == 2048].set_index('method').loc[order2]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
for ax, (key, title) in zip(axes, METRICS):
    vals = at2[key].values
    ax.barh(range(len(order2)), vals, height=0.62, color=[BLUE if m == 'pathpack-u' else NEUTRAL for m in order2])
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_title(title, fontsize=10, loc='left', color=INK)
    ax.set_yticks(range(len(order2))); ax.set_yticklabels([NAMES[m] for m in order2])
    ax.set_xticks([0, 0.5, 1]); ax.tick_params(length=0)
    ax.set_xlabel(f"n = {int(at2[key + '_evaluable'].iloc[0])} задач")
axes[0].invert_yaxis()
fig.suptitle('Второй holdout, 51 задача, бюджет 2048 условных токенов', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
h2[h2.method.isin(order2)].pivot(index='method', columns='budget', values=[k for k, _ in METRICS]).rename(index=NAMES).round(3)
''')

code('''
pj = json.load(open(H2 / 'paired.json'))
rows = []
for base in ('pathpack-v4', 'pathpack-h/nomemory', 'pathpack-r/fusion'):
    for key, title in METRICS:
        v = pj[f'pathpack-u vs {base} {key}']
        rows.append({'сравнение': 'U − ' + NAMES[base].replace(' (основной)', ''), 'метрика': title, 'diff': v['difference'], 'lo': v['ci95'][0], 'hi': v['ci95'][1],
                     'ДИ по репозиториям': [round(x, 3) for x in v['ci95_repo_cluster']], 'n': v['n'],
                     'победы/ничьи/поражения': f"{v['wins']}/{v['ties']}/{v['losses']}"})
cmp2 = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), sharex=True, sharey=True)
for ax, (name, part) in zip(axes, cmp2.groupby('сравнение', sort=False)):
    y = range(len(part))
    ax.axvline(0, color=MUTED, lw=1)
    ax.hlines(y, part.lo, part.hi, color=BLUE, lw=2)
    ax.plot(part['diff'], y, 'o', color=BLUE, ms=7, mec=SURFACE, mew=2)
    for i, r in enumerate(part.itertuples()):
        ax.text(r.hi + 0.015, i, f'{r.diff:+.3f}', va='center', fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(part['метрика'])
    ax.set_title(name, fontsize=10, loc='left'); ax.tick_params(length=0); ax.set_xlim(-0.3, 0.55)
    ax.set_xlabel('разница, 95% ДИ по задачам')
axes[0].invert_yaxis()
fig.suptitle('Второй holdout, бюджет 2048: PathPack-U минус baseline', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
cmp2.round(3)
''')

md('''
## 10. Третий отложенный набор: 125 задач, незнакомые репозитории, проба с Gemma 4

125 задач train-pool, не использовавшихся ни в обучении, ни в оценке (`output/pathpack_holdout3`, заморозка `FREEZE_before_holdout3.json`). «Без своей истории» означает модель, обученную на псевдо-задачах остальных одиннадцати репозиториев (`output/pathpack_holdout3_loro`).
''')

code('''
H3, H3L = ROOT / 'output/pathpack_holdout3', ROOT / 'output/pathpack_holdout3_loro'
NAMES['pathpack-u/noisyor'] = 'U с noisy-OR вместо обученных уровней'
h3 = summary(H3); h3l = summary(H3L); h3l['method'] = 'loro:' + h3l['method']
NAMES['loro:pathpack-u'] = 'PathPack-U без своей истории'; NAMES['loro:pathpack-h/nomemory'] = 'PathPack-H без своей истории'
both = pd.concat([h3, h3l])
order3 = ['pathpack-u', 'loro:pathpack-u', 'pathpack-u/noisyor', 'pathpack-u/nobundle', 'pathpack-h/nomemory', 'loro:pathpack-h/nomemory', 'pathpack-r/fusion', 'pathpack-v4', 'bm25']
at3 = both[both.budget == 2048].set_index('method').loc[order3]
fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
for ax, (key, title) in zip(axes, METRICS):
    vals = at3[key].values
    ax.barh(range(len(order3)), vals, height=0.62, color=[BLUE if m == 'pathpack-u' else NEUTRAL for m in order3])
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_title(title, fontsize=10, loc='left', color=INK)
    ax.set_yticks(range(len(order3))); ax.set_yticklabels([NAMES[m].replace(' (основной)', '') for m in order3])
    ax.set_xticks([0, 0.5, 1]); ax.tick_params(length=0)
    ax.set_xlabel(f"n = {int(at3[key + '_evaluable'].iloc[0])} задач")
axes[0].invert_yaxis()
fig.suptitle('Третий holdout, 125 задач, бюджет 2048 условных токенов', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
at3[[k for k, _ in METRICS]].rename(index=NAMES).round(3)
''')

code('''
pj3 = json.load(open(H3 / 'paired.json'))
groups = [('U − v0.4', 'pathpack-u vs pathpack-v4'), ('U без своей истории − U', 'loro:pathpack-u vs pathpack-u'),
          ('U − U с noisy-OR', 'pathpack-u vs pathpack-u/noisyor')]
rows = []
for label, key0 in groups:
    for key, title in METRICS:
        v = pj3[f'{key0} {key}']
        rows.append({'сравнение': label, 'метрика': title, 'diff': v['difference'], 'lo': v['ci95'][0], 'hi': v['ci95'][1],
                     'ДИ по репозиториям': [round(x, 3) for x in v['ci95_repo_cluster']], 'n': v['n'],
                     'победы/ничьи/поражения': f"{v['wins']}/{v['ties']}/{v['losses']}"})
cmp3 = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), sharex=True, sharey=True)
for ax, (name, part) in zip(axes, cmp3.groupby('сравнение', sort=False)):
    y = range(len(part))
    ax.axvline(0, color=MUTED, lw=1)
    ax.hlines(y, part.lo, part.hi, color=BLUE, lw=2)
    ax.plot(part['diff'], y, 'o', color=BLUE, ms=7, mec=SURFACE, mew=2)
    for i, r in enumerate(part.itertuples()):
        ax.text(r.hi + 0.015, i, f'{r.diff:+.3f}', va='center', fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(part['метрика'])
    ax.set_title(name, fontsize=10, loc='left'); ax.tick_params(length=0); ax.set_xlim(-0.15, 0.45)
    ax.set_xlabel('разница, 95% ДИ по задачам')
axes[0].invert_yaxis()
fig.suptitle('Третий holdout, бюджет 2048: вклад, перенос между репозиториями и абляция', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); plt.show()
cmp3.round(3)
''')

code('''
llm = []
for name, label in [('holdout1', 'первый, 100'), ('holdout2', 'второй, 51'), ('holdout3', 'третий, 125')]:
    d = json.load(open(ROOT / f'output/llm_probe/{name}_summary.json'))
    for m, v in d['summary'].items():
        llm.append({'набор': label, 'method': m, 'файл': v['llm_file_acc'], 'функция': v['llm_function_acc']})
llm = pd.DataFrame(llm)
order_l = ['pathpack-u', 'pathpack-h/nomemory', 'pathpack-r/fusion', 'pathpack-v4', 'bm25']
colors = {'pathpack-u': BLUE, 'pathpack-h/nomemory': ORANGE, 'pathpack-r/fusion': YELLOW, 'pathpack-v4': NEUTRAL, 'bm25': AQUA}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
sets = list(dict.fromkeys(llm['набор']))
for ax, col in zip(axes, ['файл', 'функция']):
    for j, m in enumerate(order_l):
        part = llm[llm.method == m].set_index('набор').reindex(sets)
        xs = [i + (j - 2) * 0.16 for i in range(len(sets))]
        ax.bar(xs, part[col].values, width=0.14, color=colors[m], label=NAMES[m].replace(' (основной)', ''))
        for x, v in zip(xs, part[col].values):
            if pd.notna(v):
                ax.text(x, v + 0.012, f'{v:.2f}', ha='center', fontsize=7, color=INK)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([f'holdout {s}' for s in sets]); ax.set_ylim(0, 1)
    ax.set_title(f'Gemma 4 верно назвала: {col}', fontsize=10, loc='left'); ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, ncol=5, loc='lower left', bbox_to_anchor=(0.03, -0.02))
fig.suptitle('Локализация моделью Gemma 4 E2B по контексту на 2048 токенов', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0.07, 1, 1)); plt.show()
llm.pivot(index='method', columns='набор', values=['файл', 'функция']).rename(index=NAMES).round(3)
''')

md('''
## Что из этого следует и чего не следует

- Основной вариант против v0.4 на holdout выше по Function, Complete и Changed Line; интервалы не включают ноль и при bootstrap по репозиториям. По File разница в пределах интервала.
- Против ручного fusion значим только Function Recall. Против PathPack-E различий нет; E обучалась на 34 задачах, сравнение неравное.
- Память коммитов, признаки истории, графовый раунд и скелеты вклада не показали.
- BM25 остаётся выше по Complete Function Recall.
- PathPack-U на втором holdout выше v0.4 по всем четырём метрикам и выше PathPack-H по File; по Function и Changed Line он не лучше PathPack-H и fusion.
- На третьем наборе (125 задач) PathPack-U выше v0.4 по всем четырём метрикам; модели без истории своего репозитория дают те же числа; обученные уровни функции и файла не лучше noisy-OR.
- Gemma 4 с контекстом PathPack-H и PathPack-U чаще верно называет файл и функцию, чем с контекстом v0.4, на третьем наборе.
- Доля исправленных ошибок не измерялась; токены условные; обучение на истории шло по тем же 12 репозиториям, что и оценка.
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
out = Path('notebooks/11_pathpack_h.ipynb')
nbf.write(nb, out)
print(out)
