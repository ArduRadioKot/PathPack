import inspect, tempfile, unittest
from pathlib import Path
from codegraph_rag.config import V6Config, v6_variant
from codegraph_rag.context_packer import token_count
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v6 import PreparedQueryV6, make_v6_chunks, retrieve_v6
from codegraph_rag.adaptive_chunker import build_chunk_index
from codegraph_rag import pathpack_v6


class V6Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        filler = ''.join(f'    step_{i} = {i}\n' for i in range(80))
        (self.root / 'auth.py').write_text(
            'def validate_cookie(cookie):\n' + filler +
            "    if cookie == 'expired_session_signature':\n        return False\n    return True\n")
        (self.root / 'session.py').write_text(
            'from auth import validate_cookie\ndef login(cookie):\n    return validate_cookie(cookie)\n')
        self.graph = scan_repository(self.root)
        self.cfg = V6Config(target_chunk_tokens=64, max_chunk_tokens=96, min_chunk_tokens=16, chunk_overlap_lines=0)

    def indexes(self, cfg=None):
        cfg = cfg or self.cfg
        fine = build_chunk_index(self.root, self.graph, cfg.to_v4())
        from dataclasses import replace
        coarse = build_chunk_index(self.root, self.graph, replace(cfg.to_v4(), enable_chunking=False))
        return fine, coarse

    def test_late_slice_outranks_the_preamble(self):
        fine, coarse = self.indexes()
        slices = [c for c in fine.chunks if c.qualname == 'validate_cookie']
        self.assertGreater(len(slices), 1)
        marker = next(c for c in slices if 'expired_session_signature' in c.body)
        prepared = PreparedQueryV6(fine, coarse, self.graph, 'expired session signature invalid cookie', self.cfg)
        result = prepared.pack(marker.tokens_est)
        self.assertLessEqual(token_count(result.context), marker.tokens_est)
        self.assertIn('expired_session_signature', result.context)
        self.assertEqual(result.hits[0].chunk.function_id, marker.function_id)

    def test_budget_nesting_determinism_and_no_duplicates(self):
        fine, coarse = self.indexes()
        query = 'expired session signature invalid cookie'
        first = PreparedQueryV6(fine, coarse, self.graph, query, self.cfg)
        second = PreparedQueryV6(fine, coarse, self.graph, query, self.cfg)
        self.assertEqual([c.chunk.id for c in first.order], [c.chunk.id for c in second.order])
        contexts = []
        for budget in (64, 128, 256):
            packed = first.pack(budget, include_candidates=False)
            self.assertLessEqual(token_count(packed.context), budget)
            ids = [hit.chunk.id for hit in packed.hits]
            self.assertEqual(len(ids), len(set(ids)))
            spans = []
            for hit in packed.hits:
                for line, start, end in hit.chunk.source_spans:
                    self.assertLess(start, end)
                    spans.append((hit.chunk.path, line, start, end))
            self.assertEqual(len(spans), len(set(spans)))
            contexts.append(packed.context)
        self.assertTrue(contexts[1].startswith(contexts[0]))
        self.assertTrue(contexts[2].startswith(contexts[1]))

    def test_no_file_cap_and_graph_fallback(self):
        fine, coarse = self.indexes()
        prepared = PreparedQueryV6(fine, coarse, self.graph, 'login cookie validate', self.cfg)
        paths = {item.chunk.path for item in prepared.order}
        self.assertIn('auth.py', paths)
        self.assertIn('session.py', paths)
        silent = v6_variant(self.cfg, 'no-graph')
        packed = PreparedQueryV6(fine, coarse, self.graph, 'expired session signature', silent).pack(256)
        self.assertIn('expired_session_signature', packed.context)
        self.assertTrue(all(item.graph == 0 for item in PreparedQueryV6(fine, coarse, self.graph, 'login', silent).candidates))

    def test_direction_and_hops_are_configurable(self):
        fine, coarse = self.indexes()
        outward = PreparedQueryV6(fine, coarse, self.graph, 'login', v6_variant(self.cfg, 'callees'))
        inward = PreparedQueryV6(fine, coarse, self.graph, 'login', v6_variant(self.cfg, 'callers'))
        def graph_of(prepared, qualname):
            return max(item.graph for item in prepared.candidates if item.chunk.qualname == qualname)
        self.assertGreater(graph_of(outward, 'validate_cookie'), 0)
        self.assertEqual(graph_of(inward, 'validate_cookie'), 0)
        shallow = PreparedQueryV6(fine, coarse, self.graph, 'login', v6_variant(self.cfg, 'hops-1'))
        self.assertEqual(shallow.config.hops, 1)
        self.assertEqual(V6Config.load(Path('configs/pathpack_v06.yaml')).hops, 3)

    def test_public_entry_and_no_gold_import(self):
        chunks = make_v6_chunks(self.root, self.graph, target_tokens=64, max_chunk_tokens=96)
        self.assertTrue(any('expired_session_signature' in c.body for c in chunks))
        result = retrieve_v6(self.root, self.graph, 'expired session signature', budget=256, config=self.cfg)
        self.assertIn('expired_session_signature', result.context)
        source = inspect.getsource(pathpack_v6)
        self.assertNotIn('evaluation', source)
        self.assertNotIn('gold_', source)
        self.assertNotIn('patch', source)
        with self.assertRaises(ValueError):
            V6Config(graph_direction='sideways')
        with self.assertRaises(ValueError):
            v6_variant(self.cfg, 'not-a-variant')
