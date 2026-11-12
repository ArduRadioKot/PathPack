import json
from pathlib import Path
import tempfile
import unittest

from codegraph_rag.retrieval import (BM25, estimated_tokens, make_chunks,
                                     retrieve, select_context, terms)
from codegraph_rag.parser import scan_repository
from codegraph_rag.evaluate import batch_retrieve, score_retrieval
from codegraph_rag.benchmark import read_jsonl


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "auth.py").write_text(
            "from account import load_record\n"
            "def login(user):\n    return load_record(user)\n", encoding="utf8")
        (self.root / "account.py").write_text(
            "def load_record(user):\n    return user\n", encoding="utf8")
        (self.root / "irrelevant.py").write_text(
            "def login_logs():\n    return 'login logs login logs'\n", encoding="utf8")

    def test_identifier_terms(self):
        self.assertEqual(terms('getUser_name'), ['get', 'user', 'name'])

    def test_budget_and_determinism(self):
        for method in ('bm25', 'graph-expand', 'pathpack'):
            g = scan_repository(self.root)
            chunks = make_chunks(self.root, g)
            a = select_context(chunks, g, 'login error', method=method, budget=90, seeds=1)
            b = select_context(chunks, g, 'login error', method=method, budget=90, seeds=1)
            self.assertEqual([x.chunk.id for x in a], [x.chunk.id for x in b])
            self.assertLessEqual(sum(x.chunk.tokens_est for x in a), 90)
            self.assertEqual(len([x.chunk.id for x in a]), len({x.chunk.id for x in a}))

    def test_graph_path_reason(self):
        graph, hits = retrieve(self.root, 'auth login', method='pathpack',
                               budget=500, seeds=1, hops=2)
        by_id = {x.chunk.id: x for x in hits}
        self.assertIn('account.py::load_record', by_id)
        self.assertIn('path:', by_id['account.py::load_record'].why)
        self.assertIn('calls', by_id['account.py::load_record'].why)

    def test_unverified_snapshot_rejected_by_default(self):
        issues = self.root / 'issues.jsonl'
        snaps = self.root / 'snapshots'
        (snaps / 'demo-1').mkdir(parents=True)
        issues.write_text(json.dumps({'instance_id': 'demo-1', 'split': 'dev',
                          'problem_statement': 'auth login fails',
                          'repo': 'test/demo', 'base_commit': 'abc'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'cannot be verified'):
            batch_retrieve(issues, snaps, self.root / 'out.jsonl',
                           method='bm25', budget=500)

    def test_unsafe_instance_identifier_rejected(self):
        issues = self.root / 'issues.jsonl'
        snaps = self.root / 'snapshots'
        snaps.mkdir()
        issues.write_text(json.dumps({'instance_id': '../escape', 'split': 'dev',
                          'problem_statement': 'hello', 'repo': 'test/demo',
                          'base_commit': 'abc'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            batch_retrieve(issues, snaps, self.root / 'out.jsonl', method='bm25')

    def test_empty_and_tiny_budget(self):
        g = scan_repository(self.root)
        chunks = make_chunks(self.root, g)
        self.assertEqual(select_context(chunks, g, 'somethingnotfound', method='bm25'), [])
        self.assertEqual(select_context(chunks, g, 'login', budget=1), [])
        with self.assertRaises(ValueError):
            select_context(chunks, g, 'login', budget=0)

    def test_batch_labels_isolation(self):
        issues = self.root / 'issues.jsonl'
        gold = self.root / 'gold.jsonl'
        snaps = self.root / 'snapshots'
        import shutil
        one = snaps / 'demo-1'
        one.mkdir(parents=True)
        shutil.copy(self.root / 'auth.py', one / 'auth.py')
        shutil.copy(self.root / 'account.py', one / 'account.py')
        issues.write_text(json.dumps({'instance_id': 'demo-1', 'split': 'dev',
                        'problem_statement': 'auth login fails',
                        'repo': 'test/demo', 'base_commit': 'abc'}) + '\n')
        gold.write_text(json.dumps({'instance_id': 'demo-1', 'gold_python_files': ['account.py']}) + '\n')
        out = self.root / 'results.jsonl'
        result = batch_retrieve(issues, snaps, out, method='pathpack', budget=500, allow_unverified=True)
        self.assertEqual(result['retrieved'], 1)
        preds = read_jsonl(out)
        self.assertNotIn('gold_python_files', preds[0])
        self.assertNotIn('patch', preds[0])
        scored = score_retrieval(out, gold)
        self.assertEqual(scored['evaluated'], 1)
        self.assertEqual(scored['file_recall_macro'], 1.0)

if __name__ == '__main__':
    unittest.main()
