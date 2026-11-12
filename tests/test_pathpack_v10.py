import inspect, tempfile, unittest
from pathlib import Path

from codegraph_rag.config import V4Config, V10Config, v10_variant
from codegraph_rag.context_packer import token_count
from codegraph_rag.evidence import extract_evidence
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v10 import retrieve_v10
from codegraph_rag.pathpack_v4 import retrieve_v4
from codegraph_rag import evidence, pathpack_v10


class EvidenceTests(unittest.TestCase):
    def test_extracts_only_spans_that_are_present(self):
        text = 'File "pkg/auth.py", line 3, in login\n`validate_cookie` raises ValueError in test_login'
        found = extract_evidence(text)
        kinds = {item.type for item in found}
        spans = {item.query_span for item in found}
        self.assertIn('path', kinds)
        self.assertIn('pkg/auth.py', spans)
        self.assertIn('login', spans)
        self.assertIn('validate_cookie', spans)
        self.assertIn('ValueError', spans)
        self.assertIn('test_login', spans)
        self.assertNotIn('behavior', kinds)
        self.assertTrue(all(item.confidence > 0 for item in found))

    def test_modules_do_not_name_a_reference_answer(self):
        for module in (evidence, pathpack_v10):
            source = inspect.getsource(module)
            self.assertNotIn('gold_', source)
            self.assertNotIn('evaluation', source)
            self.assertNotIn('patch', source)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'noise.py').write_text(
            'def show(value):\n    """request request request request request request"""\n    return value\n')
        (self.root / 'auth.py').write_text(
            'def login(token):\n    return validate_cookie(token)\n\n'
            'def validate_cookie(token):\n    return token\n\n'
            'def unused_helper(value):\n    return value + 1\n')
        self.graph = scan_repository(self.root)
        self.v4 = V4Config(candidate_limit=1, target_chunk_tokens=32, max_chunk_tokens=64,
                           min_chunk_tokens=8, chunk_overlap_lines=0)
        self.cfg = V10Config(mode='coverage', enable_structural=False, beam_width=1)

    def _names(self, result):
        return {hit.chunk.qualname for hit in result.hits}

    def test_named_pair_beats_the_single_top_chunk(self):
        query = '`login` and `validate_cookie` reject a request'
        covered = retrieve_v10(self.root, self.graph, query, budget=2048, config=self.cfg, v4_config=self.v4)
        ranked = retrieve_v4(self.root, self.graph, query, budget=2048, config=self.v4)
        names = self._names(covered)
        self.assertIn('login', names)
        self.assertIn('validate_cookie', names)
        self.assertNotIn('unused_helper', names)
        self.assertLess(len(self._names(ranked) & {'login', 'validate_cookie'}), 2)
        self.assertLess(token_count(covered.context), 2048)
        self.assertFalse(covered.diagnostics['fallback_used'])
        again = retrieve_v10(self.root, self.graph, query, budget=2048, config=self.cfg, v4_config=self.v4)
        self.assertEqual(again.context, covered.context)

    def test_unique_callee_is_a_path_group_and_not_a_coverage_hit(self):
        query = '`login` rejects a request'
        covered = retrieve_v10(self.root, self.graph, query, budget=2048, config=self.cfg, v4_config=self.v4)
        paths = retrieve_v10(self.root, self.graph, query, budget=2048,
                             config=v10_variant(self.cfg, 'paths'), v4_config=self.v4)
        self.assertIn('login', self._names(covered))
        self.assertNotIn('validate_cookie', self._names(covered))
        self.assertIn('login', self._names(paths))
        self.assertIn('validate_cookie', self._names(paths))
        self.assertNotIn('unused_helper', self._names(paths))
        calls = [edge for edge in self.graph.edges if edge.kind == 'calls']
        self.assertTrue(calls)

    def test_nested_is_a_prefix_and_independent_stays_inside_budget(self):
        query = '`login` and `validate_cookie` reject a request'
        nested = v10_variant(V10Config.load(Path('configs/pathpack_v10.yaml')), 'nested')
        small = retrieve_v10(self.root, self.graph, query, budget=80, config=nested, v4_config=self.v4)
        large = retrieve_v10(self.root, self.graph, query, budget=400, config=nested, v4_config=self.v4)
        self.assertLessEqual(token_count(small.context), 80)
        self.assertLessEqual(token_count(large.context), 400)
        self.assertTrue(large.context.startswith(small.context))
        independent = V10Config.load(Path('configs/pathpack_v10.yaml'))
        self.assertFalse(independent.nested)
        self.assertEqual(independent.mode, 'joint')
        self.assertEqual(independent.alpha, 1.0)
        self.assertEqual(independent.repeat_weight, 0.0)
        self.assertEqual(independent.beam_width, 3)
        left = retrieve_v10(self.root, self.graph, query, budget=64, config=independent, v4_config=self.v4)
        right = retrieve_v10(self.root, self.graph, query, budget=400, config=independent, v4_config=self.v4)
        self.assertLessEqual(token_count(left.context), 64)
        self.assertLessEqual(token_count(right.context), 400)
        seen = {}
        for hit in right.hits:
            for line, start, end in hit.chunk.source_spans:
                key = (hit.chunk.path, line)
                for previous_start, previous_end in seen.get(key, []):
                    self.assertFalse(max(start, previous_start) < min(end, previous_end))
                seen.setdefault(key, []).append((start, end))
        with self.assertRaises(ValueError):
            v10_variant(independent, 'hops-3')
