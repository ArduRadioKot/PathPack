import json
from pathlib import Path
import tempfile
import unittest

from codegraph_rag.parser import scan_repository


class GraphTests(unittest.TestCase):
    def test_cross_file_relative_import_and_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "app"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf8")
            (pkg / "auth.py").write_text("from .store import get_user as fetch\n"
                                                "def login(uid):\n"
                                                "    return fetch(uid)\n", encoding="utf8")
            (pkg / "store.py").write_text("def get_user(uid):\n    return uid\n", encoding="utf8")
            graph = scan_repository(root)
            self.assertIn("app/auth.py::login", graph.nodes)
            self.assertIn("app/store.py::get_user", graph.nodes)
            edges = {(e.source, e.target, e.kind) for e in graph.edges}
            self.assertIn(("app/auth.py::login", "app/store.py::get_user", "calls"), edges)
            self.assertIn(("app/auth.py::__module__", "app/store.py::__module__", "imports"), edges)

    def test_methods_and_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "file.py"
            src.write_text("class Base:\n    def f(self):\n        return 3\n"
                           "class Child(Base):\n    def g(self):\n        return self.f()\n", encoding="utf8")
            graph = scan_repository(tmp)
            edges = {(x.source, x.target, x.kind) for x in graph.edges}
            self.assertIn(("file.py::Child", "file.py::Base", "inherits"), edges)
            self.assertNotIn(("file.py::Child.g", "file.py::Base.f", "calls"), edges)  # no inheritance inference in v0.1

    def test_bad_syntax_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.py").write_text("def :\n", encoding="utf8")
            (root / "good.py").write_text("def good():\n    pass\n", encoding="utf8")
            graph = scan_repository(root)
            self.assertIn("broken.py", graph.errors)
            self.assertIn("good.py::good", graph.nodes)
            dest = root / "g.json"
            graph.save(dest)
            self.assertEqual(len(json.loads(dest.read_text())["nodes"]), 2)

    def test_self_method_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("class A:\n    def a(self):\n        return self.b()\n    def b(self):\n        return 1\n", encoding="utf8")
            graph = scan_repository(root)
            edges = {(e.source, e.target, e.kind) for e in graph.edges}
            self.assertIn(("app.py::A.a", "app.py::A.b", "calls"), edges)


if __name__ == "__main__":
    unittest.main()
