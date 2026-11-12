"""Regression tests for the new candidate policy and gold-only audit."""
import json
from pathlib import Path
import tempfile
import unittest

from codegraph_rag.parser import scan_repository
from codegraph_rag.retrieval import make_chunks, select_context
from codegraph_rag.audit import audit_predictions, paired_bootstrap


class V3SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'routes.py').write_text(
            'from storage import find_user\n'
            'def login(username):\n    return find_user(username)\n', encoding='utf8')
        (self.root/'storage.py').write_text(
            'def find_user(username):\n    return username + "@example.com"\n', encoding='utf8')
        (self.root/'health.py').write_text(
            'def monitor():\n    return "ok"\n', encoding='utf8')
        self.graph = scan_repository(self.root)
        self.chunks = make_chunks(self.root, self.graph)

    def test_budget_nesting_and_repeatability(self):
        prev = set()
        for b in (40, 80, 120, 240, 480):
            a = select_context(self.chunks, self.graph, 'login user',
                               method='pathpack-v3', budget=b, seeds=2)
            again = select_context(self.chunks, self.graph, 'login user',
                                   method='pathpack-v3', budget=b, seeds=2)
            ids = {h.chunk.id for h in a}
            self.assertEqual([h.chunk.id for h in a], [h.chunk.id for h in again])
            self.assertTrue(prev.issubset(ids), f'Budget {b} lost previous result')
            self.assertLessEqual(sum(h.chunk.tokens_est for h in a), b)
            prev = ids

    def test_cross_file_dependency_in_result(self):
        hits = select_context(self.chunks, self.graph, 'login fails for user',
                              method='pathpack-v3', budget=2000, seeds=2)
        ids = {h.chunk.id for h in hits}
        self.assertIn('routes.py::login', ids)
        self.assertIn('storage.py::find_user', ids)

    def test_empty_query_and_nonpositive_budget(self):
        self.assertEqual(select_context(self.chunks, self.graph, 'zzznonexistent',
                                        method='pathpack-v3', budget=80), [])
        with self.assertRaises(ValueError):
            select_context(self.chunks, self.graph, 'login',
                           method='pathpack-v3', budget=-1)


class AuditTests(unittest.TestCase):
    def test_detects_nested_and_token_violations_and_function_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            gold = base/'gold.jsonl'
            gold.write_text(json.dumps({'instance_id':'task1','gold_python_files':['a.py'],
                    'gold_functions':['a.py::target']})+'\n', encoding='utf8')
            small = base/'small.jsonl'
            small.write_text(json.dumps({'instance_id':'task1','method':'pathpack',
                'budget_est':100, 'tokens_est':20,
                'selected':[{'path':'a.py','id':'a.py::helper','tokens_est':20}]})+'\n',encoding='utf8')
            big = base/'big.jsonl'
            big.write_text(json.dumps({'instance_id':'task1','method':'pathpack',
                'budget_est':200,'tokens_est':201,
                'selected':[{'path':'b.py','id':'b.py::foo','tokens_est':201}]})+'\n',encoding='utf8')
            report = audit_predictions([small,big],gold,iterations=100)
            self.assertEqual(len(report['nested_violations']),1)
            self.assertEqual(len(report['budget_violations']),1)
            self.assertEqual(len(report['diagnostic_cases']),1)
            self.assertEqual(report['groups']['pathpack@100']['function_recall'],0.0)
            self.assertEqual(report['groups']['pathpack@100']['file_recall'],1.0)

    def test_pairing_uses_intersection_and_gold_only(self):
        r = paired_bootstrap({'a':1.0,'b':0.4},{'a':0.0,'c':0.7},iterations=100)
        self.assertEqual(r['n'], 1)
        self.assertEqual(r['difference'],1.0)
        self.assertEqual(r['ci95'],[1.0,1.0])

    def test_duplicate_gold_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'gold.jsonl'
            p.write_text(2 * (json.dumps({'instance_id':'duplicate','gold_python_files':[]})+'\n'))
            with self.assertRaisesRegex(ValueError,'Duplicate gold'):
                audit_predictions([],p)


if __name__=='__main__':
    unittest.main()
