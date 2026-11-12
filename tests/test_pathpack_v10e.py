"""Mechanism tests for the v1.0e stop rule. The django task is a regression, not a quality claim."""
import inspect
import tempfile
import unittest
from pathlib import Path

from codegraph_rag.config import V4Config
from codegraph_rag.context_packer import token_count
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v10 import retrieve_v10
from codegraph_rag.pathpack_v10e import V10EConfig, retrieve_v10e
from codegraph_rag.pathpack_v4 import retrieve_v4
from codegraph_rag import pathpack_v10e


class StopRuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'auth.py').write_text('def login(token):\n    return token\n')
        (self.root / 'view.py').write_text(
            'def render(value):\n    """request request request request request"""\n    return value\n'
        )
        (self.root / 'noise.py').write_text('def zzzz(value):\n    return value\n')
        self.graph = scan_repository(self.root)
        self.v4 = V4Config(candidate_limit=8, target_chunk_tokens=64, max_chunk_tokens=128,
                           min_chunk_tokens=8, chunk_overlap_lines=0)
        self.query = '`login` rejects a request request request'

    def _names(self, result):
        return {hit.chunk.qualname for hit in result.hits}

    def test_v10_still_stops_when_the_named_symbol_is_covered(self):
        from codegraph_rag.config import V10Config
        covered = retrieve_v10(self.root, self.graph, self.query, budget=2048, config=V10Config(), v4_config=self.v4)
        self.assertIn('login', self._names(covered))
        self.assertNotIn('render', self._names(covered))

    def test_v10e_keeps_an_independent_lexical_candidate(self):
        config = V10EConfig.load(Path('configs/pathpack_v10e.yaml'))
        result = retrieve_v10e(self.root, self.graph, self.query, budget=2048, config=config, v4_config=self.v4)
        names = self._names(result)
        self.assertIn('login', names)
        self.assertIn('render', names)
        self.assertNotIn('zzzz', names)
        self.assertIn(result.diagnostics['stop_reason'], {'min_utility', 'budget', 'exhausted'})
        self.assertFalse(result.diagnostics['selection_stats']['stopped_on_named_evidence'])
        self.assertLessEqual(token_count(result.context), 2048)
        again = retrieve_v10e(self.root, self.graph, self.query, budget=2048, config=config, v4_config=self.v4)
        self.assertEqual(again.context, result.context)

    def test_module_does_not_name_a_reference_answer(self):
        source = inspect.getsource(pathpack_v10e)
        self.assertNotIn('gold_', source)
        self.assertNotIn('patch', source)

    def test_old_v4_and_v10_still_pack(self):
        v4 = retrieve_v4(self.root, self.graph, self.query, budget=2048, config=self.v4)
        v10 = retrieve_v10(self.root, self.graph, self.query, budget=512, v4_config=self.v4)
        self.assertLessEqual(token_count(v4.context), 2048)
        self.assertLessEqual(token_count(v10.context), 512)
        self.assertTrue(v4.hits)


class DjangoStopRegression(unittest.TestCase):
    def test_django_12276_does_not_stop_because_names_are_covered(self):
        root = Path('data/swebench/snapshots/django__django-12276')
        issues = Path('output/benchmark/issues.jsonl')
        if not root.is_dir() or not issues.is_file():
            self.skipTest('snapshot is not present')
        import json
        row = next(json.loads(line) for line in issues.read_text().splitlines() if 'django__django-12276' in line)
        graph = scan_repository(root)
        v4 = V4Config.load(Path('configs/pathpack_v4.json'))
        config = V10EConfig.load(Path('configs/pathpack_v10e.yaml'))
        result = retrieve_v10e(root, graph, row['problem_statement'], budget=2048, config=config, v4_config=v4)
        self.assertIn(result.diagnostics['stop_reason'], {'min_utility', 'budget', 'exhausted'})
        self.assertFalse(result.diagnostics['selection_stats']['stopped_on_named_evidence'])
        self.assertLessEqual(token_count(result.context), 2048)
        self.assertGreater(result.diagnostics['selected_chunks'], 1)
