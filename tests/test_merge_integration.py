"""Compatibility checks for features that differ between the two source trees."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codegraph_rag.graph import CodeGraph
from codegraph_rag.parser import scan_repository
from codegraph_rag.retrieval import Chunk, select_context
from codegraph_rag.function_gold import functions_touched


class MergeCompatibilityTests(unittest.TestCase):
    def test_legacy_redundancy_avoids_duplicate_but_v3_rejects_parameter(self):
        chunks = [
            Chunk('a', 'a.py', 'a', 'function', 'login issue', 100),
            Chunk('b', 'b.py', 'b', 'function', 'login issue', 100),
            Chunk('c', 'c.py', 'c', 'function', 'issue billing details', 100),
        ]
        graph = CodeGraph()
        legacy = select_context(chunks, graph, 'login issue', budget=200)
        diverse = select_context(chunks, graph, 'login issue', budget=200,
                                 redundancy_penalty=1.0)
        self.assertEqual([h.chunk.id for h in legacy], ['a', 'b'])
        self.assertEqual([h.chunk.id for h in diverse], ['a', 'c'])
        with self.assertRaisesRegex(ValueError, 'only to legacy pathpack'):
            select_context(chunks, graph, 'login issue', method='pathpack-v3',
                           redundancy_penalty=0.5)

    def test_scoring_reuses_graph_and_does_not_label_unchanged_hunk_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'auth.py').write_text(
                'def login(x):\n    return x\n\ndef other(x):\n    return x\n')
            graph = scan_repository(root)
            diff = ('--- a/auth.py\n+++ b/auth.py\n@@ -1,5 +1,5 @@\n'
                    ' def login(x):\n-    return x\n+    return str(x)\n'
                    ' \n def other(x):\n     return x\n')
            with patch('codegraph_rag.function_gold.scan_repository') as scan:
                self.assertEqual(functions_touched(diff, root, graph=graph),
                                 ['auth.py::login'])
                scan.assert_not_called()
