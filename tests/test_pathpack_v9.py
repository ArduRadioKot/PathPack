import inspect, tempfile, unittest
from pathlib import Path
from codegraph_rag.config import V4Config, V9Config, v9_variant
from codegraph_rag.context_packer import token_count
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v4 import retrieve_v4
from codegraph_rag.pathpack_v9 import retrieve_v9
from codegraph_rag.retrieval import identifier_terms
from codegraph_rag import pathpack_v9


class V9Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        funcs = '\n'.join(
            f'def alpha_{i}(value):\n    return value + {i}\n' for i in range(6))
        (self.root / 'a.py').write_text(funcs)
        (self.root / 'b.py').write_text('def alpha_beta(value):\n    return value\n')
        self.graph = scan_repository(self.root)
        self.v4 = V4Config(candidate_limit=2, target_chunk_tokens=32, max_chunk_tokens=64, min_chunk_tokens=8, chunk_overlap_lines=0)

    def test_plain_matches_v4_and_cap_admits_the_second_file(self):
        query = 'alpha value'
        plain = retrieve_v9(self.root, self.graph, query, budget=256, config=v9_variant(V9Config(), 'plain'), v4_config=self.v4)
        base = retrieve_v4(self.root, self.graph, query, budget=256, config=self.v4)
        self.assertEqual(plain.context, base.context)
        capped = retrieve_v9(self.root, self.graph, query, budget=256, config=V9Config(per_file_pool=1, coverage_bonus=0), v4_config=self.v4)
        files = {item['path'] for item in capped.diagnostics['candidates'] if item.get('order') is not None}
        self.assertIn('a.py', files)
        self.assertIn('b.py', files)
        counts = {}
        for item in capped.diagnostics['candidates']:
            if item.get('order') is not None:
                counts[item['path']] = counts.get(item['path'], 0) + 1
        self.assertTrue(all(n <= 1 for n in counts.values()))

    def test_budget_prefix_and_no_answer_leak(self):
        query = 'alpha value'
        cfg = V9Config(per_file_pool=2, coverage_bonus=0.25)
        small = retrieve_v9(self.root, self.graph, query, budget=48, config=cfg, v4_config=self.v4)
        large = retrieve_v9(self.root, self.graph, query, budget=256, config=cfg, v4_config=self.v4)
        self.assertTrue(large.context.startswith(small.context))
        self.assertLessEqual(token_count(large.context), 256)
        ids = [hit.chunk.id for hit in large.hits]
        self.assertEqual(len(ids), len(set(ids)))
        loaded = V9Config.load(Path('configs/pathpack_v09.yaml'))
        self.assertEqual(loaded.per_file_pool, 8)
        self.assertEqual(loaded.coverage_bonus, 0.25)
        source = inspect.getsource(pathpack_v9)
        self.assertNotIn('evaluation', source)
        self.assertNotIn('gold_', source)
        self.assertNotIn('patch', source)
        with self.assertRaises(ValueError):
            v9_variant(loaded, 'hops-3')

    def test_joined_identifiers_and_rare_name_opens_the_named_function(self):
        self.assertIn('kmeans', identifier_terms('KMeans'))
        self.assertIn('kmeans', identifier_terms('k_means'))
        self.assertIn('fromstring', identifier_terms('Header.fromstring'))
        (self.root / 'noise.py').write_text(
            'def show(value):\n    """polyval polyval polyval polyval polyval polyval"""\n    return value\n')
        (self.root / 'target.py').write_text('def polyval(x):\n    return x + 1\n')
        graph = scan_repository(self.root)
        query = 'polyval coordinates result'
        plain = retrieve_v9(self.root, graph, query, budget=80, config=v9_variant(V9Config(), 'plain'), v4_config=self.v4)
        named = retrieve_v9(
            self.root, graph, query, budget=80,
            config=V9Config(per_file_pool=0, coverage_bonus=0, name_prior=0.55, name_df_max=0.01),
            v4_config=self.v4)
        self.assertEqual(plain.context, retrieve_v4(self.root, graph, query, budget=80, config=self.v4).context)
        self.assertTrue(named.hits)
        self.assertEqual(named.hits[0].chunk.qualname, 'polyval')
        self.assertEqual(named.hits[0].chunk.path, 'target.py')

    def test_splice_keeps_the_anchor_and_adds_the_long_name(self):
        (self.root / 'noise.py').write_text(
            'def show_0(value):\n    """fromstring fromstring fromstring fromstring fromstring fromstring"""\n    return value\n')
        (self.root / 'target.py').write_text('def parse_fromstring(x):\n' + '    return x + 1\n' * 40)
        graph = scan_repository(self.root)
        query = 'Header.fromstring does not accept bytes'
        base = V4Config(candidate_limit=32, target_chunk_tokens=32, max_chunk_tokens=64, min_chunk_tokens=8, chunk_overlap_lines=0)
        cfg = V9Config(per_file_pool=0, coverage_bonus=0, name_prior=0, name_inserts=1,
                       name_min_len=8, name_df_limit=5, anchor_tokens=120)
        plain = retrieve_v9(self.root, graph, query, budget=500, config=v9_variant(cfg, 'plain'), v4_config=base)
        self.assertEqual(plain.context, retrieve_v4(self.root, graph, query, budget=500, config=base).context)
        from codegraph_rag.context_packer import ContextStream
        from codegraph_rag.hybrid_ranker import HybridIndex
        from codegraph_rag.adaptive_chunker import build_chunk_index
        from codegraph_rag.pathpack_v9 import _name_splice
        index = HybridIndex(build_chunk_index(self.root, graph, base), graph)
        order, cands, _seeds = index.rank(query, base)
        show = next(c for c in cands if c.chunk.qualname == 'show_0')
        forced = [show] + [c for c in order if c.chunk.id != show.chunk.id]
        anchor = ContextStream(forced).pack(5000)[1][0].chunk.tokens_est
        spliced_cfg = V9Config(per_file_pool=0, coverage_bonus=0, name_prior=0, name_inserts=1,
                               name_min_len=8, name_df_limit=5, anchor_tokens=anchor)
        merged = _name_splice(cands, forced, query, spliced_cfg)
        self.assertEqual(merged[0].chunk.qualname, 'show_0')
        self.assertEqual(merged[1].chunk.qualname, 'parse_fromstring')
