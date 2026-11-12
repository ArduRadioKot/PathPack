"""Static test bridge: existing tests as a natural-language index of production code.

A test function has a descriptive name and a body that exercises some behaviour. The issue text is
matched against test functions with BM25; the score flows along the static edges of the dependency
graph from the test to the production symbols it calls, inherits from or imports. Nothing is
executed and no coverage is collected. The tests are those present at ``base_commit``; the test
patch of the benchmark is never read.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..adaptive_chunker import production_path
from ..retrieval import BM25, Chunk

FEATURES = ('bridge_fn', 'bridge_class', 'bridge_file', 'log_tests_on_fn')
MAX_TEST_CHARS = 2000


class TestBridge:
    def __init__(self, graph, repository):
        root = Path(repository)
        sources = {}

        def text(symbol):
            if symbol.path not in sources:
                try:
                    sources[symbol.path] = (root / symbol.path).read_text(errors='replace').splitlines()
                except OSError:
                    sources[symbol.path] = []
            lines = sources[symbol.path][max(0, symbol.line_start - 1):symbol.line_end]
            return '\n'.join(lines)[:MAX_TEST_CHARS]

        tests = [s for s in graph.nodes.values()
                 if not production_path(s.path) and s.kind in {'function', 'method', 'async_function'}
                 and s.line_end >= s.line_start]
        if not tests:  # kinds differ between parser versions: fall back to every non-module test symbol
            tests = [s for s in graph.nodes.values() if not production_path(s.path) and '__module__' not in s.id
                     and s.kind != 'class']
        self.tests = sorted(tests, key=lambda s: s.id)
        self.bm25 = BM25([Chunk(s.id, s.path, s.qualname, s.kind, text(s), 1) for s in self.tests])
        test_ids = {s.id for s in self.tests}
        test_paths = {s.path for s in self.tests}
        self.to_symbol = defaultdict(set)   # test id -> production symbol ids (calls / inherits)
        self.to_file = defaultdict(set)     # test id or test path -> production paths
        for edge in graph.edges:
            target = graph.nodes.get(edge.target)
            source = graph.nodes.get(edge.source)
            if target is None or source is None or not production_path(target.path) or production_path(source.path):
                continue
            if edge.source in test_ids and edge.kind in {'calls', 'inherits'}:
                self.to_symbol[edge.source].add(edge.target)
                self.to_file[edge.source].add(target.path)
            elif source.path in test_paths and edge.kind == 'imports':
                self.to_file[source.path].add(target.path)
        self.tests_on = defaultdict(int)
        for targets in self.to_symbol.values():
            for target in targets:
                self.tests_on[target] += 1
        self.kind = {sid: s.kind for sid, s in graph.nodes.items()}

    def scores(self, query):
        """Return ``(symbol_score, file_score)`` in [0, 1], the best matching test that reaches each."""
        raw = self.bm25.scores(query) if self.tests else {}
        best = max(raw.values(), default=0.0)
        if best <= 0:
            return {}, {}
        symbol, files = {}, {}
        for test in self.tests:
            value = raw.get(test.id, 0.0) / best
            if value <= 0:
                continue
            for target in self.to_symbol.get(test.id, ()):
                if value > symbol.get(target, 0.0):
                    symbol[target] = value
            for path in self.to_file.get(test.id, ()):
                if value > files.get(path, 0.0):
                    files[path] = value
            for path in self.to_file.get(test.path, ()):   # module-level imports carry half weight
                if 0.5 * value > files.get(path, 0.0):
                    files[path] = 0.5 * value
        return symbol, files

    def features(self, query, regions):
        """One row of ``FEATURES`` per region."""
        import math
        symbol, files = self.scores(query)
        rows = []
        for region in regions:
            fid = region.chunk.function_id or ''
            owner = fid.rsplit('.', 1)[0] if '.' in fid.split('::')[-1] else ''
            rows.append([
                symbol.get(fid, 0.0),
                symbol.get(owner, 0.0) if owner else 0.0,
                files.get(region.chunk.path, 0.0),
                math.log1p(self.tests_on.get(fid, 0)),
            ])
        return rows
