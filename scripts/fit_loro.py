"""Leave-one-repository-out models: train the chunk model and the function/file models on history replay
of every repository except one. History replay only; no benchmark label is read."""
import glob, json, pickle, sys
from pathlib import Path
from codegraph_rag.pathpack_r import edit_model, hier_model

held = sys.argv[1]                      # e.g. django/django
out = Path('output/pathpack_e/loro') / held.replace('/', '__')
out.mkdir(parents=True, exist_ok=True)
tasks = []
for p in sorted(glob.glob('output/pathpack_e/train_history/*.pkl')):
    with open(p, 'rb') as handle:       # local files written by history_replay
        t = pickle.load(handle)
    if t['repo'] != held:
        tasks.append(t)
assert all(t['supervision'] == 'history_replay' and t['repo'] != held for t in tasks)
meta = {'held_out_repo': held, 'supervision': 'history_replay', 'train_repos': sorted({t['repo'] for t in tasks})}
edit_model.save(edit_model.fit(tasks, meta=meta), out / 'model_history.pkl')
hm = hier_model.fit(tasks, meta=meta)
hier_model.save(hm, out / 'model_hier.pkl')
cfg = Path('configs/pathpack_r.yaml').read_text().rstrip() + f"\nhistory_model: {out / 'model_history.pkl'}\nhier_model: {out / 'model_hier.pkl'}\n"
(out / 'config.yaml').write_text(cfg)
print(json.dumps({'held': held, 'tasks': len(tasks), 'weights': hm.weights}))
