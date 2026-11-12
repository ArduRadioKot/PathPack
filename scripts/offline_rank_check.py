"""Offline sanity check on train-pool tasks only: rank quality of positives. Never touches validation or holdout."""
import pickle, sys, glob
import numpy as np
sys.path.insert(0, 'src')
from codegraph_rag.pathpack_r import edit_model
from codegraph_rag.pathpack_r.edit_features import STAGE1

def load(folder): return [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(folder + '/*.pkl'))]
def mrr_recall(tasks, scorer, k=16):
    rr, rec = [], []
    for t in tasks:
        if t['y'].sum() == 0: continue
        s = scorer(t); order = np.argsort(-s, kind='stable'); y = t['y'][order]
        rr.append(1.0 / (1 + int(np.argmax(y)))); rec.append(y[:k].sum() / y.sum())
    return round(float(np.mean(rr)), 3), round(float(np.mean(rec)), 3), len(rr)
train_h = load(sys.argv[1]); test = load(sys.argv[2])
fus = lambda t: 0.40*t['X'][:, 0] + 0.25*t['X'][:, 2] + 0.20*t['X'][:, 1] + 0.15*t['X'][:, 3]
print('history tasks', len(train_h), 'label tasks', len(test))
print('fusion formula      (MRR, recall@16, n):', mrr_recall(test, fus))
m = edit_model.fit(train_h)
print('history-trained     :', mrr_recall(test, lambda t: m.predict(t['X'], t['struct'])))
print('history, no graph   :', mrr_recall(test, lambda t: m.predict(t['X'], t['struct'], use_graph=False)))
print('history, no memory  :', mrr_recall(test, lambda t: m.predict(t['X'], t['struct'], use_memory=False)))
print('history, no history :', mrr_recall(test, lambda t: m.predict(t['X'], t['struct'], use_history=False)))
