"""English versions of the 11 paper figures. Same files, same numbers, same styling as notebooks/11_pathpack_h.ipynb.
Output: docs/figures_en/*.png. Nothing is typed in by hand; every value is read from the saved run files."""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures_en'
OUT.mkdir(parents=True, exist_ok=True)
VAL = ROOT / 'output/pathpack_val_all'
HOLD = ROOT / 'output/pathpack_holdout'
H2 = ROOT / 'output/pathpack_holdout2'
H3 = ROOT / 'output/pathpack_holdout3'
H3L = ROOT / 'output/pathpack_holdout3_loro'
LLM = ROOT / 'output/llm_probe'

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#6b6a64', '#e4e3de'
BLUE, ORANGE, AQUA, YELLOW, NEUTRAL = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#b9b8b1'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                     'text.color': INK, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.grid': False, 'figure.dpi': 110})
METRICS = [('file_recall', 'File Recall'), ('function_recall', 'Function Recall'),
           ('complete_function_recall', 'Complete Function Recall'), ('changed_line_recall', 'Changed Line Recall')]
NAMES = {
    'pathpack-h/nomemory': 'PathPack-H (main)', 'pathpack-h': 'PathPack-H with commit memory',
    'pathpack-h/nohist': 'H without history features', 'pathpack-h/nograph': 'H without graph round',
    'pathpack-h/skeleton': 'H with signature lines', 'pathpack-h/ranked': 'H, ranked packing',
    'pathpack-e': 'PathPack-E (labels, 34 tasks)', 'pathpack-eh': 'PathPack-EH (labels plus history)',
    'pathpack-r/fusion': 'Fusion (hand-tuned)', 'pathpack-r/fusion-epack': 'Fusion plus coverage packing',
    'pathpack-v10e': 'PathPack v1.0e (fixed stop)', 'pathpack-v4': 'PathPack v0.4', 'bm25': 'BM25',
    'pathpack-u': 'PathPack-U', 'pathpack-u/nobundle': 'U without whole-function move',
    'pathpack-u/noisyor': 'U with noisy-OR levels', 'loro:pathpack-u': 'PathPack-U without own repo history',
    'loro:pathpack-h/nomemory': 'PathPack-H without own repo history',
}


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=110)
    plt.close(fig)
    print('wrote', OUT / f'{name}.png')


def summary(run):
    return pd.read_csv(run / 'summary.csv')


def scores(run):
    return pd.DataFrame([json.loads(line) for line in open(run / 'scores.jsonl')])


def paired(df, cand, base, metric, budget=2048, iters=2000, seed=42):
    """Paired bootstrap over tasks, the same procedure as the notebook: mean difference and 95% interval."""
    a = df[(df.method == cand) & (df.budget == budget)].set_index('instance_id')[metric].dropna()
    b = df[(df.method == base) & (df.budget == budget)].set_index('instance_id')[metric].dropna()
    ids = sorted(set(a.index) & set(b.index))
    d = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return dict(diff=sum(d) / len(d), lo=bs[int(0.025 * iters)], hi=bs[int(0.975 * iters) - 1])


def from_json(pj, cand, base, metric):
    v = pj[f'{cand} vs {base} {metric}']
    return dict(diff=v['difference'], lo=v['ci95'][0], hi=v['ci95'][1])


def fig_holdout_summary():
    hold = summary(HOLD)
    order = ['pathpack-h/nomemory', 'pathpack-e', 'pathpack-r/fusion', 'pathpack-v10e', 'pathpack-v4', 'bm25']
    at = hold[hold.budget == 2048].set_index('method').loc[order]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        vals = at[key].values
        ax.barh(range(len(order)), vals, height=0.62, color=[BLUE if m == 'pathpack-h/nomemory' else NEUTRAL for m in order])
        for i, v in enumerate(vals):
            ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, loc='left', color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([NAMES[m] for m in order])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(length=0)
        ax.set_xlabel(f"n = {int(at[key + '_evaluable'].iloc[0])} tasks")
    axes[0].invert_yaxis()
    fig.suptitle('First holdout, 100 tasks, budget 2048 conditional tokens', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    save(fig, 'holdout_summary')


def fig_holdout_budgets():
    hold = summary(HOLD)
    lines = [('pathpack-v4', ORANGE), ('bm25', AQUA), ('pathpack-r/fusion', YELLOW), ('pathpack-h/nomemory', BLUE)]
    pos = {1024: 0, 2048: 1, 4096: 2}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        for method, color in lines:
            part = hold[hold.method == method].sort_values('budget')
            ax.plot([pos[b] for b in part.budget], part[key], color=color, lw=2, marker='o', ms=7,
                    mec=SURFACE, mew=1.5, label=NAMES[method])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['1024', '2048', '4096'])
        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10, loc='left')
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xlabel('budget, conditional tokens')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc='lower left', bbox_to_anchor=(0.03, -0.02))
    fig.suptitle('First holdout: metrics at three budgets', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 'holdout_budgets')


def _diff_panels(panels, title, path_for, xlim, figname):
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 2.8), sharex=True, sharey=True)
    for ax, (label, getter) in zip(axes, panels):
        for i, (key, metric_title) in enumerate(METRICS):
            s = getter(key)
            ax.hlines(i, s['lo'], s['hi'], color=BLUE, lw=2)
            ax.plot(s['diff'], i, 'o', color=BLUE, ms=7, mec=SURFACE, mew=2)
            ax.text(s['hi'] + 0.015, i, f"{s['diff']:+.3f}", va='center', fontsize=9)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels([t for _, t in METRICS])
        ax.set_title(label, fontsize=10, loc='left')
        ax.tick_params(length=0)
        ax.set_xlim(*xlim)
        ax.set_xlabel('difference, 95% CI over tasks')
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    save(fig, figname)


def fig_holdout_differences():
    hs = scores(HOLD)
    panels = [(label, (lambda cand, base: (lambda key: paired(hs, cand, base, key)))(cand, base))
              for label, cand, base in [('PathPack-H minus v0.4', 'pathpack-h/nomemory', 'pathpack-v4'),
                                        ('PathPack-H minus Fusion', 'pathpack-h/nomemory', 'pathpack-r/fusion'),
                                        ('PathPack-H minus PathPack-E', 'pathpack-h/nomemory', 'pathpack-e')]]
    _diff_panels(panels, 'First holdout, budget 2048: PathPack-H minus baseline', None, (-0.2, 0.55),
                 'holdout_differences')


def fig_ablations():
    val = summary(VAL)
    hold = summary(HOLD)
    methods_val = ['pathpack-h/nomemory', 'pathpack-h', 'pathpack-h/nohist', 'pathpack-h/nograph', 'pathpack-h/skeleton',
                   'pathpack-h/ranked', 'pathpack-eh', 'pathpack-e', 'pathpack-r/fusion', 'pathpack-r/fusion-epack',
                   'pathpack-v4']
    methods_hold = [m for m in methods_val if m in set(hold.method)]
    marks = dict(zip([k for k, _ in METRICS], [(BLUE, 'o'), (ORANGE, 's'), (AQUA, '^'), (YELLOW, 'D')]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    for ax, (df, methods, title) in zip(axes, [(val, methods_val, 'Validation, 99 tasks'),
                                               (hold, methods_hold, 'First holdout, 100 tasks')]):
        t = df[df.budget == 2048].set_index('method').loc[methods]
        for i in range(len(methods)):
            ax.axhline(i, color=GRID, lw=0.8, zorder=0)
        for key, label in METRICS:
            color, marker = marks[key]
            ax.plot(t[key], range(len(methods)), marker, color=color, ms=7, mec=SURFACE, mew=1.5, label=label)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([NAMES[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, loc='left')
        ax.tick_params(length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc='lower left', bbox_to_anchor=(0.18, -0.02))
    fig.suptitle('Ablations at budget 2048', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'ablations')


def fig_pool_vs_packed():
    pool = pd.read_csv(HOLD / 'pool_recall.csv')
    pool = pool[pool.budget == 2048]
    agg = pool.groupby('method')[['file_pool_recall', 'file_packed_recall', 'function_pool_recall',
                                  'function_packed_recall']].mean()
    show = [m for m in ['pathpack-r/fusion', 'pathpack-e', 'pathpack-h/nomemory'] if m in agg.index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharey=True)
    for ax, (kind, title) in zip(axes, [('file', 'Files'), ('function', 'Functions')]):
        for i, m in enumerate(show):
            p, k = agg.loc[m, f'{kind}_pool_recall'], agg.loc[m, f'{kind}_packed_recall']
            ax.hlines(i, k, p, color=NEUTRAL, lw=2)
            ax.plot(p, i, 'o', color=NEUTRAL, ms=8, mec=SURFACE, mew=1.5, label='in pool' if i == 0 else None)
            ax.plot(k, i, 'o', color=BLUE, ms=8, mec=SURFACE, mew=1.5, label='in packed context' if i == 0 else None)
            ax.text(p + 0.02, i, f'{p:.2f}', va='center', fontsize=9)
            ax.text(k - 0.02, i, f'{k:.2f}', va='center', ha='right', fontsize=9)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels([NAMES[m] for m in show])
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=10, loc='left')
        ax.tick_params(length=0)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('First holdout, budget 2048: candidate pool and packed context', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    save(fig, 'pool_vs_packed')


def fig_per_repository():
    cluster = json.load(open(HOLD / 'cluster_bootstrap.json'))
    key = 'pathpack-h/nomemory vs pathpack-v4 function_recall'
    per = pd.DataFrame({'diff': cluster[key]['per_repo_mean_diff'], 'n': cluster[key]['per_repo_n']})
    per = per.sort_values('n', ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.axvline(0, color=MUTED, lw=1)
    ax.barh(range(len(per)), per['diff'], height=0.6, color=BLUE)
    for i, r in enumerate(per.itertuples()):
        ax.text(max(r.diff, 0) + 0.02, i, f'{r.diff:+.2f}  (n={r.n})', va='center', fontsize=9)
    ax.set_yticks(range(len(per)))
    ax.set_yticklabels([i.split('__')[-1] for i in per.index])
    ax.invert_yaxis()
    ax.set_xlim(-0.45, 0.8)
    ax.tick_params(length=0)
    ax.set_xlabel('difference in Function Recall')
    ax.set_title('First holdout, budget 2048: PathPack-H minus v0.4 by repository', fontsize=11, loc='left')
    fig.tight_layout()
    save(fig, 'per_repository')


def fig_u_holdout2():
    h2 = summary(H2)
    order = ['pathpack-u', 'pathpack-u/nobundle', 'pathpack-h/nomemory', 'pathpack-r/fusion', 'pathpack-v4', 'bm25']
    at = h2[h2.budget == 2048].set_index('method').loc[order]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        vals = at[key].values
        ax.barh(range(len(order)), vals, height=0.62, color=[BLUE if m == 'pathpack-u' else NEUTRAL for m in order])
        for i, v in enumerate(vals):
            ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, loc='left', color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([NAMES[m] for m in order])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(length=0)
        ax.set_xlabel(f"n = {int(at[key + '_evaluable'].iloc[0])} tasks")
    axes[0].invert_yaxis()
    fig.suptitle('Second holdout, 51 tasks, budget 2048 conditional tokens', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    save(fig, 'u_holdout2')


def fig_u_differences():
    pj = json.load(open(H2 / 'paired.json'))
    panels = [(label, (lambda cand, base: (lambda key: from_json(pj, cand, base, key)))(cand, base))
              for label, cand, base in [('U minus v0.4', 'pathpack-u', 'pathpack-v4'),
                                        ('U minus H', 'pathpack-u', 'pathpack-h/nomemory'),
                                        ('U minus Fusion', 'pathpack-u', 'pathpack-r/fusion')]]
    _diff_panels(panels, 'Second holdout, budget 2048: PathPack-U minus baseline', None, (-0.3, 0.55),
                 'u_differences')


def fig_holdout3_summary():
    both = pd.concat([summary(H3), summary(H3L).assign(method=lambda d: 'loro:' + d.method)])
    order = ['pathpack-u', 'loro:pathpack-u', 'pathpack-u/noisyor', 'pathpack-u/nobundle', 'pathpack-h/nomemory',
             'loro:pathpack-h/nomemory', 'pathpack-r/fusion', 'pathpack-v4', 'bm25']
    at = both[both.budget == 2048].set_index('method').loc[order]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        vals = at[key].values
        ax.barh(range(len(order)), vals, height=0.62, color=[BLUE if m == 'pathpack-u' else NEUTRAL for m in order])
        for i, v in enumerate(vals):
            ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, loc='left', color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([NAMES.get(m, m) for m in order])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(length=0)
        ax.set_xlabel(f"n = {int(at[key + '_evaluable'].iloc[0])} tasks")
    axes[0].invert_yaxis()
    fig.suptitle('Third holdout, 125 tasks, budget 2048 conditional tokens', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    save(fig, 'holdout3_summary')


def fig_holdout3_differences():
    pj = json.load(open(H3 / 'paired.json'))
    panels = [(label, (lambda cand, base: (lambda key: from_json(pj, cand, base, key)))(cand, base))
              for label, cand, base in [('U minus v0.4', 'pathpack-u', 'pathpack-v4'),
                                        ('U without own repo history minus U', 'loro:pathpack-u', 'pathpack-u'),
                                        ('U minus U with noisy-OR', 'pathpack-u', 'pathpack-u/noisyor')]]
    _diff_panels(panels, 'Third holdout, budget 2048: contribution, transfer across repositories, and ablation',
                 None, (-0.15, 0.45), 'holdout3_differences')


def fig_llm_probe():
    rows = []
    for name, label in [('holdout1', 'first holdout, 100'), ('holdout2', 'second holdout, 51'),
                        ('holdout3', 'third holdout, 125')]:
        d = json.load(open(LLM / f'{name}_summary.json'))
        for m, v in d['summary'].items():
            rows.append({'set': label, 'method': m, 'file': v['llm_file_acc'], 'function': v['llm_function_acc']})
    llm = pd.DataFrame(rows)
    order = ['pathpack-u', 'pathpack-h/nomemory', 'pathpack-r/fusion', 'pathpack-v4', 'bm25']
    colors = {'pathpack-u': BLUE, 'pathpack-h/nomemory': ORANGE, 'pathpack-r/fusion': YELLOW,
              'pathpack-v4': NEUTRAL, 'bm25': AQUA}
    sets = list(dict.fromkeys(llm['set']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, col, title in zip(axes, ['file', 'function'],
                              ['Gemma 4 named the correct file', 'Gemma 4 named the correct function']):
        for j, m in enumerate(order):
            part = llm[llm.method == m].set_index('set').reindex(sets)
            xs = [i + (j - 2) * 0.16 for i in range(len(sets))]
            ax.bar(xs, part[col].values, width=0.14, color=colors[m], label=NAMES[m])
            for x, v in zip(xs, part[col].values):
                if pd.notna(v):
                    ax.text(x, v + 0.012, f'{v:.2f}', ha='center', fontsize=7, color=INK)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels(sets)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10, loc='left')
        ax.tick_params(length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=5, loc='lower left', bbox_to_anchor=(0.03, -0.02))
    fig.suptitle('Localization by Gemma 4 E2B from the 2048-token context', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 'llm_probe')


if __name__ == '__main__':
    fig_holdout_summary()
    fig_holdout_budgets()
    fig_holdout_differences()
    fig_ablations()
    fig_pool_vs_packed()
    fig_per_repository()
    fig_u_holdout2()
    fig_u_differences()
    fig_holdout3_summary()
    fig_holdout3_differences()
    fig_llm_probe()
