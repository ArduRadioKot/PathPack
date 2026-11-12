"""Signal probe on train-pool tasks only (never validation or holdout): does the test bridge separate edited chunks?"""
import json, sys, glob, os, time
import numpy as np
from sklearn.metrics import roc_auc_score
from codegraph_rag.adaptive_chunker import build_chunk_index
from codegraph_rag.config import V4Config
from codegraph_rag.experiments import load_scoring_dataset, verify_snapshot
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_r.config import PathPackRConfig
from codegraph_rag.pathpack_r.edit_train import label_regions
from codegraph_rag.pathpack_r.retrieve import RepoIndex, build_pool
from codegraph_rag.pathpack_r.test_bridge import TestBridge, FEATURES

built = {os.path.basename(p)[:-4] for p in glob.glob('output/pathpack_e/train/*.pkl')}
rows = [json.loads(l) for l in open('output/pathpack_r_splits_v2/train_110.jsonl') if json.loads(l)['instance_id'] in built]
rows = [r for r in rows if 'sympy' not in r['repo']][:int(sys.argv[1])]
patches = {r['instance_id']: r['patch'] for r in load_scoring_dataset()}
cfg, v4 = PathPackRConfig.load('configs/pathpack_r.yaml'), V4Config.load('configs/pathpack_v4.json')
allv, within = [], []
for row in rows:
    t = time.time()
    repo = verify_snapshot(row, 'data/swebench/snapshots'); graph = scan_repository(repo)
    index = RepoIndex(build_chunk_index(repo, graph, v4), graph, repo)
    regions, *_ = build_pool(index, row['problem_statement'], cfg, ('lexical', 'symbol', 'structural'))
    y = label_regions(regions, patches[row['instance_id']])
    bridge = TestBridge(graph, repo)
    F = np.array(bridge.features(row['problem_statement'], regions)); lex = np.array([r.lexical for r in regions])
    gold_files = {r.chunk.path for r, v in zip(regions, y) if v}
    mask = np.array([r.chunk.path in gold_files for r in regions])
    print(row['instance_id'], 'tests', len(bridge.tests), 'pos', int(y.sum()), 'nonzero share fn/class/file',
          [round(float((F[:, i] > 0).mean()), 2) for i in range(3)], f'{time.time()-t:.0f}s', flush=True)
    if y.sum() and y.sum() < len(y):
        allv.append([roc_auc_score(y, lex)] + [roc_auc_score(y, F[:, i]) for i in range(4)] + [roc_auc_score(y, lex + F[:, 0] + F[:, 1] + F[:, 2])])
    if mask.sum() > y.sum() > 0:
        ym = y[mask]
        within.append([roc_auc_score(ym, lex[mask])] + [roc_auc_score(ym, F[mask, i]) for i in range(4)] + [roc_auc_score(ym, lex[mask] + F[mask, 0] + F[mask, 1])])
names = ['lexical'] + list(FEATURES) + ['lexical+bridge']
print('pool AUC   ', dict(zip(names, np.round(np.mean(allv, 0), 3))), 'n', len(allv))
print('in-file AUC', dict(zip(names, np.round(np.mean(within, 0), 3))), 'n', len(within))
