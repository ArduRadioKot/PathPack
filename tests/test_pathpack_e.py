"""PathPack-E mechanism tests. They do not claim a SWE-bench result."""
import inspect
import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from codegraph_rag.adaptive_chunker import build_chunk_index
from codegraph_rag.context_packer import token_count
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_r import edit_features, edit_model, edit_train
from codegraph_rag.pathpack_r.config import PathPackRConfig
from codegraph_rag.pathpack_r.edit_features import STAGE1, HISTORY_COLUMNS, build_stage1, stage2_features
from codegraph_rag.pathpack_r.edit_pack import pack_edit
from codegraph_rag.pathpack_r.history import parse_log
from codegraph_rag.pathpack_r.retrieve import RepoIndex, build_pool

LOG = (
    '@@\tc3\tFix crash in widgets rendering\nforms/widgets.py\nforms/fields.py\nREADME.rst\n'
    '@@\tc2\tAdd docs\ndocs/a.py\n'
    '@@\tc1\tRefactor widgets\nforms/widgets.py\n'
)


def _repo():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / 'app.py').write_text(
        'def transform(data):\n    return data["value"] + 1\n\n'
        'def handle_request(data):\n    return transform(data)\n\n'
        'def unrelated(data):\n    return 1\n'
    )
    (root / 'util.py').write_text('def helper(x):\n    return x * 2\n')
    return tmp, root


class HistoryTests(unittest.TestCase):
    def test_counts_and_cochange(self):
        h = parse_log(LOG)
        self.assertEqual(h.file_churn('forms/widgets.py'), 2)
        self.assertEqual(h.file_fixes('forms/widgets.py'), 1)
        self.assertEqual(h.file_fixes('docs/a.py'), 0)
        self.assertAlmostEqual(h.cochange_with('forms/fields.py', ['forms/widgets.py']), 0.5)
        self.assertEqual(h.cochange_with('forms/fields.py', ['forms/fields.py']), 0.0)

    def test_memory_votes_for_files_of_similar_commits(self):
        h = parse_log(LOG)
        votes = h.memory_scores('widgets rendering crash on empty value')
        self.assertEqual(max(votes, key=votes.get), 'forms/widgets.py')
        self.assertNotIn('docs/a.py', votes)
        self.assertEqual(h.memory_scores('zzz qqq'), {})

    def test_after_hides_the_commit_itself_and_newer_ones(self):
        h = parse_log(LOG)
        older = h.after('c3')
        self.assertEqual([c[0] for c in older.commits], ['c2', 'c1'])
        self.assertEqual(older.file_fixes('forms/widgets.py'), 0)
        self.assertEqual(h.after('c1').commits, [])


class ReplayTests(unittest.TestCase):
    def test_parse_and_label(self):
        from codegraph_rag.pathpack_r.history_replay import label, parse_patch_log
        text = ('@@@\tabc\t100\tFix transform on empty mapping\tbody line\x00\n'
                'diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n'
                '@@ -2 +2 @@ def transform(data):\n-    old\n+    new\n'
                '@@@\tdef\t90\tDocs\t\x00\n+++ b/docs/x.py\n@@ -1 +1 @@\n')
        commits = parse_patch_log(text)
        self.assertEqual([c['hash'] for c in commits], ['abc', 'def'])
        self.assertEqual(commits[0]['touched'], {'app.py': ['transform']})
        self.assertIn('body line', commits[0]['message'])
        tmp, root = _repo()
        self.addCleanup(tmp.cleanup)
        graph = scan_repository(root)
        index = RepoIndex(build_chunk_index(root, graph), graph, root)
        regions, _, _, _ = build_pool(index, 'transform handle_request unrelated', PathPackRConfig(), ('lexical', 'symbol'))
        y = label(regions, commits[0]['touched'])
        self.assertEqual({r.chunk.qualname for r, v in zip(regions, y) if v}, {'transform'})

    def test_replay_module_never_touches_benchmark_patches(self):
        import ast
        from codegraph_rag.pathpack_r import history_replay
        tree = ast.parse(inspect.getsource(history_replay))
        subscripts = {n.slice.value for n in ast.walk(tree)
                      if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)}
        self.assertFalse({'patch', 'test_patch', 'FAIL_TO_PASS', 'hints_text'} & subscripts)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = _repo()
        self.addCleanup(self.tmp.cleanup)
        self.graph = scan_repository(self.root)
        self.index = RepoIndex(build_chunk_index(self.root, self.graph), self.graph, self.root)
        self.config = PathPackRConfig()
        self.regions, _, _, _ = build_pool(
            self.index, '`handle_request` crashes on empty data', self.config, ('lexical', 'symbol', 'structural'))

    def test_shape_and_determinism(self):
        a, sa = build_stage1(self.regions, self.index, parse_log(LOG))
        b, _ = build_stage1(self.regions, self.index, parse_log(LOG))
        self.assertEqual(a.shape, (len(self.regions), len(STAGE1)))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(sa['callers']), len(self.regions))

    def test_history_switch_zeroes_only_history_columns(self):
        h = parse_log(LOG.replace('forms/widgets.py', 'app.py'))
        with_h, _ = build_stage1(self.regions, self.index, h, use_history=True, query='widgets crash')
        without, _ = build_stage1(self.regions, self.index, h, use_history=False, query='widgets crash')
        other = [i for i in range(len(STAGE1)) if i not in HISTORY_COLUMNS]
        np.testing.assert_array_equal(with_h[:, other], without[:, other])
        self.assertTrue(np.all(without[:, list(HISTORY_COLUMNS)] == 0))
        self.assertTrue(np.any(with_h[:, list(HISTORY_COLUMNS)] != 0))

    def test_callee_sees_caller_probability(self):
        _, struct = build_stage1(self.regions, self.index, None)
        names = [r.chunk.qualname for r in self.regions]
        p0 = np.array([0.9 if name == 'handle_request' else 0.0 for name in names])
        feats = stage2_features(p0, struct)
        callee_rows = [i for i, name in enumerate(names) if name == 'transform']
        self.assertTrue(callee_rows, 'transform must be in the pool through the structural channel')
        self.assertTrue(all(feats[i, 1] > 0.8 for i in callee_rows))  # callers_max_p0

    def test_features_do_not_read_a_patch(self):
        import ast
        for module in (edit_features, edit_model):
            names = set()
            for node in ast.walk(ast.parse(inspect.getsource(module))):
                if isinstance(node, ast.Name):
                    names.add(node.id.lower())
                elif isinstance(node, ast.arg):
                    names.add(node.arg.lower())
                elif isinstance(node, ast.Attribute):
                    names.add(node.attr.lower())
            self.assertFalse({n for n in names if 'patch' in n or 'gold' in n}, module.__name__)


class ModelTests(unittest.TestCase):
    def _tasks(self, n=8, seed=0):
        rng = np.random.default_rng(seed)
        tasks = []
        for i in range(n):
            rows = 120
            X = rng.random((rows, len(STAGE1)))
            y = (X[:, 0] > 0.9).astype(np.int8)
            y[0] = 1
            struct = {'file': np.zeros(rows, dtype=int), 'fn': np.arange(rows), 'callers': [[] for _ in range(rows)],
                      'callees': [[] for _ in range(rows)]}
            tasks.append({'id': f't{i}', 'X': X, 'y': y, 'struct': struct})
        return tasks

    def test_learns_a_planted_signal_and_is_deterministic(self):
        tasks = self._tasks()
        m1 = edit_model.fit(tasks, folds=4)
        m2 = edit_model.fit(tasks, folds=4)
        probe = self._tasks(1, seed=7)[0]
        p1 = m1.predict(probe['X'], probe['struct'])
        p2 = m2.predict(probe['X'], probe['struct'])
        np.testing.assert_allclose(p1, p2)
        self.assertGreater(np.corrcoef(p1, probe['y'])[0, 1], 0.5)

    def test_fit_refuses_overlap_with_evaluation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'train').mkdir()
            for task in self._tasks(6):
                with (tmp / 'train' / f'{task["id"]}.pkl').open('wb') as handle:
                    pickle.dump(task, handle)
            manifest = tmp / 'eval.jsonl'
            manifest.write_text(json.dumps({'instance_id': 't3'}) + '\n')
            args = Namespace(train_dir=str(tmp / 'train'), out=str(tmp / 'm.pkl'), exclude=[str(manifest)])
            with self.assertRaises(SystemExit):
                edit_train.cmd_fit(args)


class PackTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = _repo()
        self.addCleanup(self.tmp.cleanup)
        graph = scan_repository(self.root)
        index = RepoIndex(build_chunk_index(self.root, graph), graph, self.root)
        self.config = PathPackRConfig()
        self.regions, _, _, _ = build_pool(index, 'handle_request unrelated helper', self.config, ('lexical', 'symbol'))
        for region in self.regions:
            region.final = 0.2

    def test_budget_and_no_named_evidence_stop(self):
        context, hits, info = pack_edit(self.regions, 2048, self.config)
        self.assertLessEqual(token_count(context), 2048)
        self.assertFalse(info['named_evidence_stop'])
        self.assertGreaterEqual(len(hits), 2)

    def test_tight_budget_never_exceeded_and_skeleton_marked(self):
        full, _, _ = pack_edit(self.regions, 2048, self.config, skeleton=True)
        budget = max(30, token_count(full) // 3)
        context, _, info = pack_edit(self.regions, budget, self.config, skeleton=True)
        self.assertLessEqual(token_count(context), budget)
        self.assertEqual(info['packer'], 'edit-skeleton')

    def test_zero_probability_is_not_padded_in(self):
        for region in self.regions:
            region.final = 0.0
        context, hits, _ = pack_edit(self.regions, 2048, self.config)
        self.assertEqual(len(hits), 0)


class UtilityTests(unittest.TestCase):
    def _setup(self):
        from types import SimpleNamespace
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        body = ''.join(f'    v{i} = compute_value({i}) + offset_table[{i}] * scale_factor\n' for i in range(60))
        (root / 'big.py').write_text('def big_function(offset_table, scale_factor):\n' + body + '    return v59\n')
        (root / 'other.py').write_text('def small_helper(x):\n    return x * 2\n\ndef second_helper(y):\n    return y + 1\n')
        self.addCleanup(tmp.cleanup)
        graph = scan_repository(root)
        index = RepoIndex(build_chunk_index(root, graph), graph, root)
        regions, _, _, _ = build_pool(index, 'big_function small_helper second_helper compute_value', PathPackRConfig(), ('lexical', 'symbol'))
        X, struct = build_stage1(regions, index, None)
        model = SimpleNamespace(weights={'chunk': 1.0, 'fn': 1.0, 'file': 1.0})
        return index, regions, struct, model

    def test_budget_and_bundle_completes_a_likely_function(self):
        from codegraph_rag.pathpack_r.pack_utility import pack_utility
        index, regions, struct, model = self._setup()
        big = [i for i, r in enumerate(regions) if r.chunk.qualname == 'big_function']
        self.assertGreater(len(index.by_fn[regions[big[0]].chunk.function_id]), 2, 'the test needs a multi-chunk function')
        for r in regions:
            r.final = 0.01
        p_fn = np.full(struct['fn'].max() + 1, 0.01); p_file = np.full(struct['file'].max() + 1, 0.01)
        p_fn[struct['fn'][big[0]]] = 0.9
        context, hits, info = pack_utility(regions, index, 4096, model, p_fn, p_file, struct)
        self.assertLessEqual(token_count(context), 4096)
        self.assertGreaterEqual(info['moves']['bundle'], 1)
        emitted = {h.chunk.id for h in hits}
        self.assertTrue({c.id for c in index.by_fn[regions[big[0]].chunk.function_id]} <= emitted)
        self.assertFalse(info['named_evidence_stop'])
        for r in regions:
            r.decision = ''
        tight, _, _ = pack_utility(regions, index, 60, model, p_fn, p_file, struct)
        self.assertLessEqual(token_count(tight), 60)

    def test_noisy_or(self):
        from codegraph_rag.pathpack_r.pack_utility import noisy_or
        out = noisy_or(np.array([0.5, 0.5, 0.2]), np.array([0, 0, 1]))
        self.assertAlmostEqual(out[0], 0.75); self.assertAlmostEqual(out[1], 0.2)

    def test_aggregate_and_hier_fit_use_history_only(self):
        from codegraph_rag.pathpack_r import hier_model
        p = np.array([0.2, 0.8, 0.5]); X = np.zeros((3, len(STAGE1))); groups = np.array([0, 0, 1])
        agg = hier_model.aggregate(p, X, groups)
        self.assertEqual(agg.shape, (2, len(hier_model.AGG)))
        self.assertAlmostEqual(agg[0, 0], 0.8); self.assertAlmostEqual(agg[0, 2], 1.0); self.assertAlmostEqual(agg[0, 4], 1 - 0.2 * 0.8)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'train').mkdir()
            task = {'id': 'x', 'X': X, 'y': np.array([0, 1, 0]), 'struct': {}, 'supervision': 'benchmark_patch'}
            with (tmp / 'train' / 'x.pkl').open('wb') as handle:
                pickle.dump(task, handle)
            manifest = tmp / 'e.jsonl'; manifest.write_text(json.dumps({'instance_id': 'zzz'}) + '\n')
            with self.assertRaises(SystemExit):
                hier_model.main(['--train-dir', str(tmp / 'train'), '--out', str(tmp / 'm.pkl'), '--exclude', str(manifest)])


class LabelTests(unittest.TestCase):
    def test_label_regions_marks_the_chunk_with_the_removed_line(self):
        tmp, root = _repo()
        self.addCleanup(tmp.cleanup)
        graph = scan_repository(root)
        index = RepoIndex(build_chunk_index(root, graph), graph, root)
        regions, _, _, _ = build_pool(index, 'transform handle_request unrelated', PathPackRConfig(), ('lexical', 'symbol'))
        patch = ('--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def transform(data):\n-    return data["value"] + 1\n+    return 0\n')
        labels = edit_train.label_regions(regions, patch)
        positives = {r.chunk.qualname for r, y in zip(regions, labels) if y}
        self.assertEqual(positives, {'transform'})


if __name__ == '__main__':
    unittest.main()
