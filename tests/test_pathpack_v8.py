import inspect, tempfile, unittest
from pathlib import Path
from codegraph_rag.config import V4Config, V8Config, v8_variant
from codegraph_rag.context_packer import token_count
from codegraph_rag.parser import scan_repository
from codegraph_rag.pathpack_v4 import retrieve_v4
from codegraph_rag.pathpack_v8 import _rewrite, retrieve_v8
from codegraph_rag.retrieval import Chunk
from codegraph_rag import pathpack_v8


def _chunk(cid, path, fid, tokens, line):
    body = f'line {line} {cid}\n'
    return Chunk(cid, path, fid, 'function', body, tokens, fid, line, line, ((line, 0, len(body)),), f'def {fid}', None, body)


class _Item:
    def __init__(self, chunk, lexical):
        self.chunk = chunk
        self.lexical = lexical
        self.score = lexical
        self.symbol = 0.0
        self.graph = 0.0


class V8Tests(unittest.TestCase):
    def test_gap_can_hold_a_later_slice_without_moving_the_next_file(self):
        opener = _chunk('a0', 'a.py', 'alpha', 40, 1)
        tail = _chunk('a1', 'a.py', 'alpha', 100, 2)
        late = _chunk('a9', 'a.py', 'alpha', 80, 9)
        nxt = _chunk('b0', 'b.py', 'beta', 40, 1)
        order = [_Item(opener, 0.2), _Item(tail, 0.1), _Item(nxt, 0.2)]
        chunks = [opener, tail, late, nxt]
        lexical = {chunk.id: 0.9 if chunk.id == 'a9' else 0.2 for chunk in chunks}
        cfg = V8Config(enable_adaptive=False)
        rewritten = _rewrite(order, chunks, lexical, cfg, {})
        self.assertEqual([item.chunk.id for item in rewritten], ['a0', 'a9', 'b0'])
        self.assertEqual([item.kind for item in rewritten], ['anchor', 'slice', 'anchor'])
        before = sum(item.chunk.tokens_est for item in rewritten if item.chunk.path == 'a.py')
        self.assertLessEqual(before, opener.tokens_est + tail.tokens_est)

    def test_no_deepen_matches_v4_and_prefixes_nest(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        filler = ''.join(f'    step_{i} = {i}\n' for i in range(40))
        (root / 'auth.py').write_text('def validate_cookie(cookie):\n' + filler + "    return cookie\n")
        (root / 'session.py').write_text('from auth import validate_cookie\ndef login(cookie):\n    return validate_cookie(cookie)\n')
        graph = scan_repository(root)
        v4cfg = V4Config(target_chunk_tokens=48, max_chunk_tokens=80, min_chunk_tokens=16, chunk_overlap_lines=0)
        query = 'validate cookie login'
        plain = retrieve_v8(root, graph, query, budget=256, config=v8_variant(V8Config(), 'no-deepen'), v4_config=v4cfg)
        base = retrieve_v4(root, graph, query, budget=256, config=v4cfg)
        self.assertEqual(plain.context, base.context)
        full = retrieve_v8(root, graph, query, budget=64, config=V8Config(), v4_config=v4cfg)
        larger = retrieve_v8(root, graph, query, budget=256, config=V8Config(), v4_config=v4cfg)
        self.assertTrue(larger.context.startswith(full.context))
        self.assertLessEqual(token_count(larger.context), 256)
        ids = [hit.chunk.id for hit in larger.hits]
        self.assertEqual(len(ids), len(set(ids)))
        anchors = [item['path'] for item in larger.diagnostics['candidates'] if item['representation'] == 'anchor']
        self.assertEqual(len(anchors), len(set(anchors)))

    def test_config_and_no_answer_leak(self):
        loaded = V8Config.load(Path('configs/pathpack_v08.yaml'))
        self.assertTrue(loaded.enable_deepen)
        self.assertEqual(loaded.whole_max_tokens, 192)
        self.assertFalse(v8_variant(loaded, 'no-graph').enable_graph)
        self.assertFalse(v8_variant(loaded, 'no-adaptive').enable_adaptive)
        source = inspect.getsource(pathpack_v8)
        self.assertNotIn('evaluation', source)
        self.assertNotIn('gold_', source)
        self.assertNotIn('patch', source)
        with self.assertRaises(ValueError):
            v8_variant(loaded, 'hops-3')
