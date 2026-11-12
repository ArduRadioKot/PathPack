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
    retrieve_cmd.add_argument("--method", choices=["bm25", "graph-expand", "pathpack", "pathpack-v3", "pathpack-v4", "pathpack-v5", "pathpack-v51", "pathpack-v6", "pathpack-v7", "pathpack-v8", "pathpack-v9", "pathpack-v10"], default="pathpack")
    retrieve_cmd.add_argument("--budget", type=int, default=2048)
    retrieve_cmd.add_argument("--seeds", type=int, default=3)
    retrieve_cmd.add_argument("--hops", type=int, default=3)
    retrieve_cmd.add_argument("--graph-bonus", type=float, default=0.75)
    retrieve_cmd.add_argument("--redundancy-penalty", type=float, default=0.0)
    retrieve_cmd.add_argument("--out", type=Path)
    batch = commands.add_parser("retrieve-batch", help="Retrieve over prechecked-out SWE-bench snapshots; NO gold labels")
    batch.add_argument("--issues", type=Path, required=True)
    batch.add_argument("--snapshots", type=Path, required=True)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--method", choices=["bm25", "graph-expand", "pathpack", "pathpack-v3", "pathpack-v4", "pathpack-v5", "pathpack-v51", "pathpack-v6", "pathpack-v7", "pathpack-v8", "pathpack-v9", "pathpack-v10"], default="pathpack")
    batch.add_argument("--budget", type=int, default=2048)
    batch.add_argument("--split", choices=["dev", "test"], default=None)
    batch.add_argument("--graph-bonus", type=float, default=0.75)
    batch.add_argument("--redundancy-penalty", type=float, default=0.0)
    batch.add_argument("--allow-unverified-snapshots", action="store_true", help="For synthetic demos ONLY")
    score = commands.add_parser("score", help="Evaluate file-level localization against isolated gold labels")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--out", type=Path)
    audit = commands.add_parser('audit', help='Compare saved prediction files and audit per-task budgets (gold used ONLY for scoring)')
    audit.add_argument('--predictions', type=Path, nargs='+')
    audit.add_argument('--run', type=Path, help='v0.4 benchmark run directory')
    audit.add_argument('--gold', type=Path)
    audit.add_argument('--out', type=Path, default=Path('reports/audit.json'))
    audit.add_argument('--bootstrap', type=int, default=1000)
    function_labels = commands.add_parser('gold-functions', help='SCORING ONLY: derive function labels using raw gold patches and base snapshots')
    function_labels.add_argument('--raw', type=Path, required=True, help='Full SWE-bench rows with patch, in JSONL')
    function_labels.add_argument('--snapshots', type=Path, required=True)
    function_labels.add_argument('--gold', type=Path, required=True)
    function_labels.add_argument('--out', type=Path, required=True)
    retrieve_cmd.add_argument('--config', type=Path, help='v0.4 JSON configuration')
    batch.add_argument('--config', type=Path, help='v0.4 JSON configuration')
    for name in ('benchmark','ablation'):
        command = commands.add_parser(name, help='Pinned retrieval experiment; no gold access')
        command.add_argument('--issues', type=Path, default=Path('output/benchmark/issues.jsonl'))
        command.add_argument('--snapshots', type=Path, default=Path('data/swebench/snapshots'))
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--split', choices=['dev','test','holdout'], default='dev')
        command.add_argument('--method', nargs='+', choices=['bm25','graph-expand','pathpack','pathpack-v3','pathpack-v4','pathpack-v5','pathpack-v51','pathpack-v6','pathpack-v7','pathpack-v8','pathpack-v9','pathpack-v10','embedding'], default=['pathpack-v4'] if name=='ablation' else ['bm25','graph-expand','pathpack','pathpack-v3','pathpack-v4'])
        command.add_argument('--budgets', nargs='+', type=int, default=[1024,2048,4096])
        command.add_argument('--config', type=Path)
        command.add_argument('--v5-config', type=Path)
        command.add_argument('--v51-config', type=Path)
        command.add_argument('--v6-config', type=Path)
        command.add_argument('--v7-config', type=Path)
        command.add_argument('--v8-config', type=Path)
        command.add_argument('--v9-config', type=Path)
        command.add_argument('--v10-config', type=Path)
        command.add_argument('--include-ablations', action='store_true')
        command.add_argument('--v5-ablations', action='store_true')
        command.add_argument('--v51-ablations', action='store_true')
        command.add_argument('--v6-ablations', action='store_true')
        command.add_argument('--v7-ablations', action='store_true')
        command.add_argument('--v8-ablations', action='store_true')
        command.add_argument('--v9-ablations', action='store_true')
        command.add_argument('--v10-ablations', action='store_true')
        command.add_argument('--limit', type=int)
    scoring = commands.add_parser('score-run', help='GOLD ONLY: score a completed benchmark')
    scoring.add_argument('--run', type=Path, required=True)
    scoring.add_argument('--raw', type=Path, help='Optional local raw SWE-bench JSONL; default offline dataset cache')
    holdout = commands.add_parser('prepare-holdout', help='Freeze unseen instance ids without evaluating them')
    holdout.add_argument('--issues', type=Path, default=Path('output/benchmark/issues.jsonl'))
    holdout.add_argument('--out', type=Path, required=True)
    holdout.add_argument('--raw', type=Path)
    holdout.add_argument('--count', type=int, default=60)
    holdout.add_argument('--seed', type=int, default=43)
    localize = commands.add_parser('audit-localization', help='GOLD ONLY: case audit of a finished retrieval run')
    localize.add_argument('--run', type=Path, required=True)
    localize.add_argument('--out', type=Path, required=True)
    localize.add_argument('--method', default='pathpack-v4')
    localize.add_argument('--budget', type=int, default=2048)
    localize.add_argument('--limit', type=int, default=10)
    localize.add_argument('--raw', type=Path)
    figures = commands.add_parser('visualize-run', help='Plot actual graph paths, selected chunks and measured scores')
    figures.add_argument('--run', type=Path, required=True)
    figures.add_argument('--out', type=Path)
    figures.add_argument('--instance-id')
    figures.add_argument('--budget', type=int, default=2048)
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
    elif args.cmd == "retrieve" and args.method == "pathpack-v10":
        from .config import V4Config, V10Config
        from .pathpack_v10 import retrieve_v10
        cfg = V10Config.load(args.config) if args.config else V10Config()
        graph = scan_repository(args.repository)
        result = retrieve_v10(args.repository, graph, args.query, budget=args.budget, config=cfg, v4_config=V4Config()).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve" and args.method == "pathpack-v9":
        from .config import V4Config, V9Config
        from .pathpack_v9 import retrieve_v9
        cfg = V9Config.load(args.config) if args.config else V9Config()
        graph = scan_repository(args.repository)
        result = retrieve_v9(args.repository, graph, args.query, budget=args.budget, config=cfg, v4_config=V4Config()).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve" and args.method == "pathpack-v8":
        from .config import V4Config, V8Config
        from .pathpack_v8 import retrieve_v8
        cfg = V8Config.load(args.config) if args.config else V8Config()
        graph = scan_repository(args.repository)
        result = retrieve_v8(args.repository, graph, args.query, budget=args.budget, config=cfg, v4_config=V4Config()).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve" and args.method == "pathpack-v7":
        from .config import V4Config, V7Config
        from .pathpack_v7 import retrieve_v7
        cfg = V7Config.load(args.config) if args.config else V7Config()
        graph = scan_repository(args.repository)
        result = retrieve_v7(args.repository, graph, args.query, budget=args.budget, config=cfg, v4_config=V4Config()).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve" and args.method == "pathpack-v6":
        from .config import V6Config
        from .pathpack_v6 import retrieve_v6
        cfg = V6Config.load(args.config) if args.config else V6Config()
        graph = scan_repository(args.repository)
        result = retrieve_v6(args.repository, graph, args.query, budget=args.budget, config=cfg).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve" and args.method in {"pathpack-v5", "pathpack-v51"}:
        from .config import V5Config
        from .pathpack_v5 import retrieve_v5
        cfg = V5Config.load(args.config) if args.config else V5Config()
        graph = scan_repository(args.repository)
        result = retrieve_v5(args.repository, graph, args.query, budget=args.budget, config=cfg).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve" and args.method == "pathpack-v4":
        from .config import V4Config
        from .pathpack_v4 import retrieve_v4
        cfg = V4Config.load(args.config) if args.config else V4Config()
        graph = scan_repository(args.repository)
        result = retrieve_v4(args.repository, graph, args.query, budget=args.budget, config=cfg).to_dict()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'output':str(args.out),'tokens_est':result['tokens_est'],'chunks':len(result['selected'])}))
        else: print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd == "retrieve":
        graph, hits = retrieve(args.repository, args.query, method=args.method,
                               budget=args.budget, seeds=args.seeds, hops=args.hops,
                               graph_bonus=args.graph_bonus,
                               redundancy_penalty=args.redundancy_penalty)
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
        from .config import V4Config
        cfg = V4Config.load(args.config) if args.config else None
        print(json.dumps(batch_retrieve(args.issues, args.snapshots, args.out,
                                        method=args.method, budget=args.budget,
                                        split=args.split,
                                        allow_unverified=args.allow_unverified_snapshots,
                                        graph_bonus=args.graph_bonus,
                                        redundancy_penalty=args.redundancy_penalty, config=cfg), indent=2))
    elif args.cmd == "score":
        result = score_retrieval(args.predictions, args.gold)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k != "per_task"}, indent=2))
    elif args.cmd == 'gold-functions':
        print(json.dumps(label_from_raw(args.raw, args.snapshots, args.gold, args.out),
                         indent=2, ensure_ascii=False))
    elif args.cmd == 'audit' and args.run:
        from .experiments import audit_run
        report=audit_run(args.run)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if not report['ok']: raise SystemExit(1)
    elif args.cmd == 'audit':
        if not args.predictions or not args.gold: parser.error('audit requires --run or both --predictions and --gold')
        report = audit_predictions(args.predictions, args.gold, iterations=args.bootstrap)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps({'report': str(args.out), 'groups': list(report['groups']),
                          'budget_violations': len(report['budget_violations']),
                          'nested_violations': len(report['nested_violations'])}, ensure_ascii=False))
    elif args.cmd in {'benchmark','ablation'}:
        from .config import V4Config, V5Config, V6Config, V7Config, V8Config, V9Config, V10Config
        from .experiments import run_benchmark
        cfg = V4Config.load(args.config) if args.config else V4Config()
        v5_cfg = V5Config.load(args.v5_config) if args.v5_config else None
        v51_cfg = V5Config.load(args.v51_config) if args.v51_config else None
        v6_cfg = V6Config.load(args.v6_config) if args.v6_config else None
        v7_cfg = V7Config.load(args.v7_config) if args.v7_config else None
        v8_cfg = V8Config.load(args.v8_config) if args.v8_config else None
        v9_cfg = V9Config.load(args.v9_config) if args.v9_config else None
        v10_cfg = V10Config.load(args.v10_config) if args.v10_config else None
        print(json.dumps(run_benchmark(args.issues,args.snapshots,args.out,split=args.split,methods=args.method,budgets=args.budgets,config=cfg,include_ablations=args.include_ablations or (args.cmd=='ablation' and 'pathpack-v4' in args.method),limit=args.limit,v5_config=v5_cfg,include_v5_ablations=args.v5_ablations or (args.cmd=='ablation' and 'pathpack-v5' in args.method),v51_config=v51_cfg,include_v51_ablations=args.v51_ablations or (args.cmd=='ablation' and 'pathpack-v51' in args.method),v6_config=v6_cfg,include_v6_ablations=args.v6_ablations or (args.cmd=='ablation' and 'pathpack-v6' in args.method),v7_config=v7_cfg,include_v7_ablations=args.v7_ablations or (args.cmd=='ablation' and 'pathpack-v7' in args.method),v8_config=v8_cfg,include_v8_ablations=args.v8_ablations or (args.cmd=='ablation' and 'pathpack-v8' in args.method),v9_config=v9_cfg,include_v9_ablations=args.v9_ablations or (args.cmd=='ablation' and 'pathpack-v9' in args.method),v10_config=v10_cfg,include_v10_ablations=args.v10_ablations or (args.cmd=='ablation' and 'pathpack-v10' in args.method)),indent=2))
    elif args.cmd == 'score-run':
        from .experiments import score_run
        report=score_run(args.run,raw=args.raw)
        print(json.dumps(report,indent=2))
        if not report['audit']: raise SystemExit(1)
    elif args.cmd == 'visualize-run':
        from .research_visuals import visualize_run
        print(json.dumps(visualize_run(args.run,args.out,instance_id=args.instance_id,budget=args.budget),indent=2))
    elif args.cmd == 'audit-localization':
        from .error_localization_audit import audit_run
        print(json.dumps(audit_run(args.run,args.out,method=args.method,budget=args.budget,limit=args.limit,raw=args.raw),ensure_ascii=False,indent=2))
    elif args.cmd == 'prepare-holdout':
        from .experiments import prepare_holdout
        print(json.dumps(prepare_holdout(args.issues,args.out,raw=args.raw,count=args.count,seed=args.seed),indent=2))
    elif args.cmd == "prepare":
        rows = read_jsonl(args.input_jsonl) if args.input_jsonl else load_verified()
        result = build_manifests(rows, args.out, limit=args.limit,
                                 dev_count=args.dev_count, seed=args.seed)
        print(json.dumps({"output_dir": str(args.out), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
