"""Microbenchmark ONLY: 4 hand-authored scenarios, not SWE-bench evidence."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from codegraph_rag.retrieval import retrieve

SNAPSHOT = Path(__file__).resolve().parent / "snapshot"
ISSUES = [
    ("Login cookie causes failure", "store.py"),
    ("Checkout invoice amount is incorrect", "taxes.py"),
    ("Upload image causes crash", "filesys.py"),
    ("Send notification fails", "packets.py"),
]
METHODS = ["bm25", "graph-expand", "pathpack"]
BUDGETS = [45, 70, 110, 160]


def run() -> list[dict]:
    results = []
    for budget in BUDGETS:
        for query, expected in ISSUES:
            for method in METHODS:
                graph, hits = retrieve(SNAPSHOT, query, method=method,
                                       budget=budget, seeds=1, hops=2)
                results.append({"query": query, "gold_file": expected,
                                "method": method, "budget_est": budget,
                                "hit": int(any(hit.chunk.path == expected for hit in hits)),
                                "tokens_est": sum(hit.chunk.tokens_est for hit in hits),
                                "selected": [h.chunk.path for h in hits]})
    out = ROOT / 'reports' / 'toy_results.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print('TOY DEMO RESULTS. DO NOT CITE AS SWE-BENCH PERFORMANCE.')
    for budget in BUDGETS:
        print('budget', budget, {method: sum(x['hit'] for x in results
                 if x['budget_est'] == budget and x['method'] == method)
                 for method in METHODS}, 'of',len(ISSUES))
    return results

if __name__ == '__main__':
    run()
