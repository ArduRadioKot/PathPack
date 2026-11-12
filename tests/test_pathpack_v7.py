import inspect, tempfile, unittest
from pathlib import Path
from codegraph_rag.config import V4Config, V7Config, v7_variant
from codegraph_rag.context_packer import token_count
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v4 import retrieve_v4
from codegraph_rag.pathpack_v7 import retrieve_v7
from codegraph_rag import pathpack_v7


class V7Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        filler = ''.join(f'    step_{i} = {i}\n' for i in range(90))
        (self.root / 'auth.py').write_text(
            'def validate_cookie(cookie):\n' + filler +
            "    if cookie == 'expired_session_signature':\n        return False\n    return cookie\n"
            'def tiny_helper(value):\n    return value\n')
        (self.root / 'session.py').write_text(
            'from auth import validate_cookie\ndef login(cookie):\n    return validate_cookie(cookie)\n')
        self.graph = scan_repository(self.root)
        self.v4 = V4Config(target_chunk_tokens=64, max_chunk_tokens=96, min_chunk_tokens=16, chunk_overlap_lines=0)
        self.v7 = V7Config(whole_max_tokens=80)

    def test_slice_beats_the_preamble_and_small_function_stays_whole(self):
        result = retrieve_v7(self.root, self.graph, 'expired session signature invalid cookie', budget=120,
                             config=self.v7, v4_config=self.v4)
        self.assertLessEqual(token_count(result.context), 120)
        self.assertIn('expired_session_signature', result.context)
        kinds = {item['representation'] for item in result.diagnostics['candidates'] if item['selected']}
        self.assertTrue(kinds & {'slice', 'whole', 'v4'})
        helper = retrieve_v7(self.root, self.graph, 'tiny helper value', budget=256, config=self.v7, v4_config=self.v4)
        whole = [item for item in helper.diagnostics['candidates'] if item['representation'] == 'whole' and item['selected']]
        self.assertTrue(any(item['qualname'] == 'tiny_helper' for item in whole))

    def test_no_hierarchy_matches_v4_and_prefixes_nest(self):
        plain = v7_variant(self.v7, 'no-hierarchy')
        query = 'expired session signature'
        v4 = retrieve_v4(self.root, self.graph, query, budget=256, config=self.v4)
        v7 = retrieve_v7(self.root, self.graph, query, budget=256, config=plain, v4_config=self.v4)
        self.assertEqual(v4.context, v7.context)
        full = retrieve_v7(self.root, self.graph, query, budget=64, config=self.v7, v4_config=self.v4)
        larger = retrieve_v7(self.root, self.graph, query, budget=256, config=self.v7, v4_config=self.v4)
        self.assertTrue(larger.context.startswith(full.context))
        self.assertLessEqual(token_count(larger.context), 256)
        ids = [hit.chunk.id for hit in larger.hits]
        self.assertEqual(len(ids), len(set(ids)))
        spans = []
        for hit in larger.hits:
            for line, start, end in hit.chunk.source_spans:
                self.assertLess(start, end)
                spans.append((hit.chunk.path, line, start, end))
        self.assertEqual(len(spans), len(set(spans)))

    def test_flags_and_no_answer_leak(self):
        self.assertEqual(V7Config.load(Path('configs/pathpack_v07.yaml')).whole_max_tokens, 192)
        self.assertFalse(V7Config.load(Path('configs/pathpack_v07.yaml')).cap_to_opening)
        capped = V7Config.load(Path('configs/pathpack_v07_capped.yaml'))
        self.assertTrue(capped.cap_to_opening)
        self.assertFalse(v7_variant(self.v7, 'no-graph').enable_graph)
        self.assertFalse(v7_variant(self.v7, 'no-adaptive').enable_adaptive)
        self.assertFalse(v7_variant(self.v7, 'no-budget').enable_budget)
        self.assertFalse(v7_variant(self.v7, 'no-diversity').enable_diversity)
        source = inspect.getsource(pathpack_v7)
        self.assertNotIn('evaluation', source)
        self.assertNotIn('gold_', source)
        with self.assertRaises(ValueError):
            v7_variant(self.v7, 'hops-3')
