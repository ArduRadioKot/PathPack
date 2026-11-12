"""PathPack-R mechanism tests. They do not claim a SWE-bench result."""
import inspect
import tempfile
import unittest
from pathlib import Path

from codegraph_rag.context_packer import token_count
from codegraph_rag.evaluation_v4 import score_prediction
from codegraph_rag.graph import Edge
from codegraph_rag.line_metrics import extra_metrics
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_r.config import PathPackRConfig
from codegraph_rag.pathpack_r.pipeline import execute
from codegraph_rag.pathpack_r.rerank import probe_reranker
from codegraph_rag.pathpack_r.retrieve import RepoIndex
from codegraph_rag.pathpack_r.semantic import SemanticIndex, probe_semantic
from codegraph_rag.pathpack_r.symbols import extract_symbols, matches_any, symbol_match, SymbolQuery
from codegraph_rag.pathpack_r import train
from codegraph_rag.adaptive_chunker import build_chunk_index


def _repo(files):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp, root


class SymbolTests(unittest.TestCase):
    def test_exact_forms_do_not_match_a_shared_piece(self):
        text = 'File "pkg/forms/widgets.py", line 4, in FileInput\n`validate_cookie` and forms.ClearableFileInput'
        found = extract_symbols(text)
        kinds = {(item.kind, item.span) for item in found}
        self.assertIn(('path', 'pkg/forms/widgets.py'), kinds)
        self.assertIn(('symbol', 'FileInput'), kinds)
        self.assertIn(('symbol', 'validate_cookie'), kinds)
        self.assertTrue(any(item.kind == 'qualified' and item.span == 'forms.ClearableFileInput' for item in found))
        self.assertTrue(symbol_match('ClearableFileInput', 'pkg/forms/widgets.py', SymbolQuery('symbol', 'ClearableFileInput', 'camel')))
        self.assertFalse(symbol_match('ClearableFileInput', 'pkg/forms/widgets.py', SymbolQuery('symbol', 'File', 'camel')))
        self.assertFalse(symbol_match('ClearableFileInput', 'pkg/forms/widgets.py', SymbolQuery('symbol', 'Input', 'camel')))
        self.assertTrue(symbol_match('forms.ClearableFileInput', 'pkg/forms/widgets.py', SymbolQuery('qualified', 'forms.ClearableFileInput', 'qualified')))
        self.assertTrue(matches_any('FileInput', 'django/forms/widgets.py', found))
        self.assertFalse(matches_any('unrelated_helper', 'django/http/request.py', found))


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = _repo({
            'app.py': (
                'def transform(data):\n    return data["value"] + 1\n\n'
                'def handle_request(data):\n    return transform(data)\n\n'
                'def unrelated(data):\n    return 1\n'
            ),
        })
        self.addCleanup(self.tmp.cleanup)
        self.graph = scan_repository(self.root)
        self.index = RepoIndex(build_chunk_index(self.root, self.graph), self.graph, self.root)
        self.config = PathPackRConfig()
        self.query = '`handle_request` crashes when the mapping is empty'

    def _run(self):
        return execute(self.index, self.query, self.config, 'heuristic', 2048, instance_id='synthetic')

    def test_unique_callee_without_a_shared_name_is_added(self):
        calls = [edge for edge in self.graph.edges if edge.kind == 'calls']
        self.assertTrue(any(edge.kind == 'calls' for edge in calls))
        outcome = self._run()
        self.assertEqual(outcome['status'], 'ok')
        names = {hit.chunk.qualname for hit in outcome['result'].hits}
        self.assertIn('handle_request', names)
        self.assertIn('transform', names)
        self.assertNotIn('unrelated', names)
        self.assertFalse(outcome['result'].diagnostics['named_evidence_stop'])
        self.assertLessEqual(token_count(outcome['result'].context), 2048)

    def test_two_callees_are_not_added_without_their_own_signal(self):
        tmp, root = _repo({
            'app.py': (
                'def alpha(data):\n    return 1\n\n'
                'def beta(data):\n    return 2\n\n'
                'def handle_request(data):\n    return alpha(data) + beta(data)\n'
            ),
        })
        self.addCleanup(tmp.cleanup)
        graph = scan_repository(root)
        index = RepoIndex(build_chunk_index(root, graph), graph, root)
        outcome = execute(index, self.query, self.config, 'heuristic', 2048)
        names = {hit.chunk.qualname for hit in outcome['result'].hits}
        self.assertIn('handle_request', names)
        self.assertNotIn('alpha', names)
        self.assertNotIn('beta', names)

    def test_graph_relations_are_real_edges(self):
        for function_id in self.index.by_fn:
            for other, kind, direction in self.index.relations(function_id):
                if direction == 'out':
                    self.assertIn(Edge(function_id, other, kind), self.graph.edges)
                else:
                    self.assertIn(Edge(other, function_id, kind), self.graph.edges)

    def test_repeat_is_deterministic_and_spans_do_not_overlap(self):
        first = self._run()['result']
        second = self._run()['result']
        self.assertEqual(first.context, second.context)
        seen = {}
        for hit in first.hits:
            self.assertGreaterEqual(hit.chunk.line_start, 1)
            self.assertGreaterEqual(hit.chunk.line_end, hit.chunk.line_start)
            for line, start, end in hit.chunk.source_spans:
                self.assertGreater(end, start)
                key = (hit.chunk.path, line)
                for previous_start, previous_end in seen.get(key, []):
                    self.assertFalse(max(start, previous_start) < min(end, previous_end))
                seen.setdefault(key, []).append((start, end))


class ModelStatusTests(unittest.TestCase):
    def test_missing_semantic_model_does_not_invent_vectors(self):
        probe = probe_semantic()
        if probe['status'] == 'not_run':
            index = SemanticIndex('not_run', info=probe)
        else:
            tmp, root = _repo({'app.py': 'def alpha():\n    return 1\n'})
            self.addCleanup(tmp.cleanup)
            graph = scan_repository(root)
            chunks = build_chunk_index(root, graph).chunks
            from codegraph_rag.pathpack_r.semantic import load_semantic
            index = load_semantic(chunks, PathPackRConfig(), allow_download=False)
        if index.status != 'available':
            self.assertEqual(index.status, 'not_run')
            self.assertIsNone(index.scores([0.1, 0.2]))
            self.assertIsNone(index.matrix)
        else:
            self.assertIsNotNone(index.matrix)

    def test_reranker_probe_is_explicit(self):
        info = probe_reranker()
        self.assertIn(info['status'], {'not_run', 'library_present', 'available'})
        self.assertFalse(info['code_specialized'])
        if info['status'] == 'not_run':
            self.assertIn('not', info['reason'])


class MetricTests(unittest.TestCase):
    def test_partial_labels_stay_null_and_anchors_stay_separate(self):
        gold = {
            'gold_python_files': ['a.py'], 'functions': {}, 'changed_lines': {},
            'source_lines': {'a.py': ['def alpha():\n', '    return 1\n']},
            'label_completeness': 'partial',
        }
        prediction = {
            'instance_id': 'x', 'method': 'pathpack-r', 'budget_est': 100, 'tokens_est': 4,
            'latency_sec': 0.0,
            'selected': [{'id': 'a', 'path': 'a.py', 'function_id': 'a', 'source_spans': [(1, 0, 12)]}],
        }
        scored = score_prediction(prediction, gold)
        extra = extra_metrics(prediction['selected'], gold)
        self.assertIsNone(scored['changed_line_recall'])
        self.assertIsNone(scored['function_recall'])
        self.assertEqual(scored['file_recall'], 1.0)
        self.assertIsNone(extra['insertion_anchor_recall'])
        self.assertNotIn('changed_line_recall', extra)


class TrainingGuardTests(unittest.TestCase):
    def test_overlap_is_rejected_and_loss_fits_a_separable_pair(self):
        rows = [{'instance_id': 'train-1', 'repo': 'other/other'}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.jsonl'
            path.write_text('{"instance_id":"train-1","repo":"other/other"}\n')
            with self.assertRaises(ValueError):
                train.assert_disjoint(rows, [path])
            path.write_text('{"instance_id":"eval-1","repo":"other/other"}\n')
            with self.assertRaises(ValueError):
                train.assert_disjoint(rows, [path], repository_disjoint=True)
            train.assert_disjoint(rows, [path], repository_disjoint=False)
        positive = [[1.0, 0.0, 0.0]]
        negative = [[0.0, 1.0, 0.0]]
        before = train.pairwise_loss([0.0, 0.0, 0.0], positive, negative)
        weights = train.train_linear([(positive[0], negative[0])], steps=80, lr=0.2, seed=42)
        after = train.pairwise_loss(weights, positive, negative)
        self.assertLess(after, before)
        again = train.train_linear([(positive[0], negative[0])], steps=80, lr=0.2, seed=42)
        self.assertEqual(weights, again)

    def test_retriever_does_not_import_training(self):
        for module_name in ('pipeline', 'retrieve', 'active', 'packing', 'rerank', 'semantic', 'symbols'):
            source = inspect.getsource(__import__(f'codegraph_rag.pathpack_r.{module_name}', fromlist=['*']))
            self.assertNotIn('gold_', source)
            self.assertNotIn('evaluation', source)
            self.assertNotIn('train import', source)
