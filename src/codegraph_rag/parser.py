"""Conservative AST-based static graph extraction; does not execute project code.

This is deliberately an approximate call graph, not a complete Python runtime call
resolver. Unknown and dynamic targets are omitted rather than guessed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .graph import CodeGraph, Symbol

EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build"}
MAX_FILE_BYTES = 1_000_000


def module_name(relative_path: Path) -> str:
    parts = list(relative_path.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def expression_parts(expr: ast.AST) -> list[str] | None:
    if isinstance(expr, ast.Name):
        return [expr.id]
    if isinstance(expr, ast.Attribute):
        base = expression_parts(expr.value)
        return base + [expr.attr] if base else None
    return None


@dataclass
class ParsedFile:
    path: str
    module: str
    package: str
    tree: ast.Module
    aliases: dict[str, str]


class Collector(ast.NodeVisitor):
    def __init__(self, graph: CodeGraph, path: str, module: str):
        self.graph = graph
        self.path = path
        self.module = module
        self.scope: list[tuple[str, str]] = []
        self.functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]] = []
        self.classes: list[tuple[str, ast.ClassDef]] = []

    def _add(self, node: ast.AST, kind: str, name: str) -> str:
        qualname = ".".join([item[0] for item in self.scope] + [name])
        symbol_id = f"{self.path}::{qualname}"
        self.graph.add_symbol(Symbol(symbol_id, self.path, self.module, qualname,
                                     kind, getattr(node, "lineno", 1), getattr(node, "end_lineno", getattr(node, "lineno", 1))))
        parent = f"{self.path}::{'.'.join(x[0] for x in self.scope)}" if self.scope else f"{self.path}::__module__"
        self.graph.add_edge(parent, symbol_id, "defines")
        return symbol_id

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol_id = self._add(node, "class", node.name)
        self.classes.append((symbol_id, node))
        self.scope.append((node.name, "class"))
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1][1] == "class" else "function"
        owner_class = ".".join(x[0] for x in self.scope) if kind == "method" else None
        symbol_id = self._add(node, kind, node.name)
        self.functions.append((symbol_id, node, owner_class))
        self.scope.append((node.name, "function"))
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)


class CallVisitor(ast.NodeVisitor):
    """Visits a function body without attributing nested function calls to it."""
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        parts = expression_parts(node.func)
        if parts:
            self.calls.append(parts)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _package(mod: str, rel: Path) -> str:
    if rel.name == "__init__.py":
        return mod
    return mod.rpartition(".")[0]


def _absolute_import(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    prefix = package.split(".") if package else []
    steps_up = node.level - 1
    prefix = prefix[:len(prefix) - steps_up] if steps_up <= len(prefix) else []
    if node.module:
        prefix += node.module.split(".")
    return ".".join(prefix)


def scan_repository(root: str | Path) -> CodeGraph:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    graph = CodeGraph()
    files: dict[str, ParsedFile] = {}
    collectors: dict[str, Collector] = {}
    symbols: dict[tuple[str, str], str] = {}
    module_nodes: dict[str, str] = {}

    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or any(p in EXCLUDED for p in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root)
        name = rel.as_posix()
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                graph.errors[name] = "File exceeds 1MB scan limit"
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            graph.errors[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        mod = module_name(rel)
        # When scanning a package directory itself, preserve a valid name for
        # its root __init__.py and relative imports among its children.
        if (root / "__init__.py").is_file():
            mod = f"{root.name}.{mod}" if mod else root.name
        if not mod:
            graph.errors[name] = "Cannot determine Python module name"
            continue
        module_id = f"{name}::__module__"
        graph.add_symbol(Symbol(module_id, name, mod, "__module__", "module", 1, max((getattr(n, "end_lineno", 1) for n in tree.body), default=1)))
        module_nodes[mod] = module_id
        collector = Collector(graph, name, mod)
        collector.visit(tree)
        for node in graph.nodes.values():
            if node.path == name:
                symbols[(mod, node.qualname)] = node.id
        files[name] = ParsedFile(name, mod, _package(mod, rel), tree, {})
        collectors[name] = collector

    # Phase 2: all symbols are known, so imports and calls can be resolved regardless of file ordering.
    for name, info in files.items():
        module_id = module_nodes[info.module]
        for node in info.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Bare `import a.b` binds name `a`; `import a.b as x` binds `x`.
                    bound = alias.asname or alias.name.split(".")[0]
                    info.aliases[bound] = alias.name if alias.asname else bound
                    if alias.name in module_nodes:
                        graph.add_edge(module_id, module_nodes[alias.name], "imports")
            elif isinstance(node, ast.ImportFrom):
                imported_from = _absolute_import(node, info.package)
                if imported_from in module_nodes:
                    graph.add_edge(module_id, module_nodes[imported_from], "imports")
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    full = ".".join(x for x in (imported_from, alias.name) if x)
                    bound = alias.asname or alias.name
                    info.aliases[bound] = full
                    target = module_nodes.get(full) or symbols.get((imported_from, alias.name))
                    if target:
                        graph.add_edge(module_id, target, "imports")

        def resolve(parts: list[str], owner_class: str | None) -> str | None:
            if not parts:
                return None
            if parts[0] in {"self", "cls"} and owner_class and len(parts) == 2:
                return symbols.get((info.module, f"{owner_class}.{parts[1]}"))
            if len(parts) == 1:
                # A local function wins over an imported symbol.
                local = symbols.get((info.module, parts[0]))
                if local:
                    return local
            if parts[0] in info.aliases:
                qualified = ".".join([info.aliases[parts[0]], *parts[1:]])
            else:
                qualified = ".".join(parts)
            # Try module/function split at each dot.
            for index in range(len(qualified.split(".")) - 1, 0, -1):
                segments = qualified.split(".")
                target_mod = ".".join(segments[:index])
                target_symbol = ".".join(segments[index:])
                match = symbols.get((target_mod, target_symbol))
                if match:
                    return match
            # Local class static-style invocation, e.g. A.run()
            return symbols.get((info.module, qualified))

        collector = collectors[name]
        for symbol_id, node, owner_class in collector.functions:
            visitor = CallVisitor()
            for body_stmt in node.body:
                visitor.visit(body_stmt)
            for call in visitor.calls:
                target = resolve(call, owner_class)
                if target:
                    graph.add_edge(symbol_id, target, "calls")
        for class_id, class_node in collector.classes:
            for base in class_node.bases:
                parts = expression_parts(base)
                if parts:
                    target = resolve(parts, None)
                    if target:
                        graph.add_edge(class_id, target, "inherits")
    return graph
