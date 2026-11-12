"""Visualization tests: extract and render a simple graph."""
import tempfile
import unittest
from pathlib import Path
from codegraph_rag.parser import scan_repository

class VizTests(unittest.TestCase):
    def test_selection_and_render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from codegraph_rag.visualize import select_graph, draw_graph
        root = Path(__file__).resolve().parents[1] / "examples"
        g = scan_repository(root)
        selected = select_graph(g.to_dict(),focus="service.py::login",hops=1)
        self.assertIn("sample_repo/service.py::login",{n["id"] for n in selected["nodes"]})
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"graph.png"
            fig=draw_graph(g.to_dict(),target,title="Smoke test")
            plt.close(fig)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size,4000)
