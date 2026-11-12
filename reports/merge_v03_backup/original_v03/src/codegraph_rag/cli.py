"""Command line entrypoints for repository scanning and dataset preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import build_manifests, load_verified, read_jsonl
from .parser import scan_repository
from .retrieval import retrieve
from .evaluate import batch_retrieve, score_retrieval
from .audit import audit_predictions
from .function_gold import label_from_raw


def main() -> None:
    parser = argparse.ArgumentParser(prog="codegraph", description="CodeGraph-RAG research starter")
    commands = parser.add_subparsers(dest="cmd", required=True)
    scan = commands.add_parser("scan", help="Generate a static Python dependency graph")
    scan.add_argument("repository", type=Path)
    scan.add_argument("--out", type=Path, default=Path("output/graph.json"))
    prepare = commands.add_parser("prepare", help="Build SWE-bench input + held-out gold labels")
    prepare.add_argument("--input-jsonl", type=Path, help="Read local data instead of downloading from Hugging Face")
    prepare.add_argument("--out", type=Path, default=Path("output/benchmark"))
    prepare.add_argument("--limit", type=int, default=80)
    prepare.add_argument("--dev-count", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=42)
    viz = commands.add_parser("visualize", help="Draw a dependency graph as PNG or SVG")
    viz.add_argument("graph_json", type=Path)
    viz.add_argument("--out", type=Path, default=Path("visualizations/graph.png"))
    viz.add_argument("--focus", type=str, default=None, help="Node name or ID substring")
    viz.add_argument("--hops", type=int, default=1)
    viz.add_argument("--max-nodes", type=int, default=35)
    viz.add_argument("--kind", choices=["calls", "imports", "defines", "inherits"],
                     action="append", default=None)
    retrieve_cmd = commands.add_parser("retrieve", help="Retrieve code for an issue using BM25 or PathPack")
    retrieve_cmd.add_argument("repository", type=Path)
    retrieve_cmd.add_argument("--query", required=True)
    retrieve_cmd.add_argument("--method", choices=["bm25", "graph-expand", "pathpack", "pathpack-v3"], default="pathpack")
    retrieve_cmd.add_argument("--budget", type=int, default=2048)
    retrieve_cmd.add_argument("--seeds", type=int, default=3)
    retrieve_cmd.add_argument("--hops", type=int, default=3)
    retrieve_cmd.add_argument("--out", type=Path)
    batch = commands.add_parser("retrieve-batch", help="Retrieve over prechecked-out SWE-bench snapshots; NO gold labels")
    batch.add_argument("--issues", type=Path, required=True)
    batch.add_argument("--snapshots", type=Path, required=True)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--method", choices=["bm25", "graph-expand", "pathpack", "pathpack-v3"], default="pathpack")
    batch.add_argument("--budget", type=int, default=2048)
    batch.add_argument("--split", choices=["dev", "test"], default=None)
    batch.add_argument("--allow-unverified-snapshots", action="store_true", help="For synthetic demos ONLY")
    score = commands.add_parser("score", help="Evaluate file-level localization against isolated gold labels")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--out", type=Path)
    audit = commands.add_parser('audit', help='Compare saved prediction files and audit per-task budgets (gold used ONLY for scoring)')
    audit.add_argument('--predictions', type=Path, nargs='+', required=True)
    audit.add_argument('--gold', type=Path, required=True)
    audit.add_argument('--out', type=Path, default=Path('reports/audit.json'))
    audit.add_argument('--bootstrap', type=int, default=1000)
    function_labels = commands.add_parser('gold-functions', help='SCORING ONLY: derive function labels using raw gold patches and base snapshots')
    function_labels.add_argument('--raw', type=Path, required=True, help='Full SWE-bench rows with patch, in JSONL')
    function_labels.add_argument('--snapshots', type=Path, required=True)
    function_labels.add_argument('--gold', type=Path, required=True)
    function_labels.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "scan":
        graph = scan_repository(args.repository)
        graph.save(args.out)
        print(json.dumps({"output": str(args.out), **graph.stats()}, ensure_ascii=False, indent=2))
    elif args.cmd == "visualize":
        from .visualize import visualize_file
        import matplotlib.pyplot as plt
        visualize_file(args.graph_json, args.out, focus=args.focus,
                       hops=args.hops, max_nodes=args.max_nodes,
                       edge_kinds=set(args.kind) if args.kind else None,
                       title="CodeGraph-RAG: dependencies")
        plt.close("all")
        print(json.dumps({"output": str(args.out)}, ensure_ascii=False))
    elif args.cmd == "retrieve":
        graph, hits = retrieve(args.repository, args.query, method=args.method,
                               budget=args.budget, seeds=args.seeds, hops=args.hops)
        result = {"method": args.method, "query": args.query,
                  "budget_est": args.budget, "tokens_est": sum(x.chunk.tokens_est for x in hits),
                  "hits": [h.to_dict() for h in hits]}
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"out": str(args.out), "hits": len(hits),
                              "tokens_est": result["tokens_est"]}, ensure_ascii=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.cmd == "retrieve-batch":
        print(json.dumps(batch_retrieve(args.issues, args.snapshots, args.out,
                                        method=args.method, budget=args.budget,
                                        split=args.split,
                                        allow_unverified=args.allow_unverified_snapshots), indent=2))
    elif args.cmd == "score":
        result = score_retrieval(args.predictions, args.gold)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k != "per_task"}, indent=2))
    elif args.cmd == 'gold-functions':
        print(json.dumps(label_from_raw(args.raw, args.snapshots, args.gold, args.out),
                         indent=2, ensure_ascii=False))
    elif args.cmd == 'audit':
        report = audit_predictions(args.predictions, args.gold, iterations=args.bootstrap)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps({'report': str(args.out), 'groups': list(report['groups']),
                          'budget_violations': len(report['budget_violations']),
                          'nested_violations': len(report['nested_violations'])}, ensure_ascii=False))
    elif args.cmd == "prepare":
        rows = read_jsonl(args.input_jsonl) if args.input_jsonl else load_verified()
        result = build_manifests(rows, args.out, limit=args.limit,
                                 dev_count=args.dev_count, seed=args.seed)
        print(json.dumps({"output_dir": str(args.out), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
