# CodeGraph-RAG / PathPack v0.2

**Research prototype, 1 November 2026.** Python static graph and a first comparison of budget-limited issue-to-code retrieval. No model fine-tuning, no real SWE-bench score, no claim of research novelty yet.

## Open these first

- [Russian progress report](docs/PROGRESS_2026-11-01_RU.md)
- [Updated research brief DOCX](docs/CodeGraph-RAG_Research_Brief_RU.docx)
- [Executed PathPack notebook](notebooks/02_pathpack_experiments.ipynb)
- [SWE-bench Verified experiment notebook](notebooks/03_swebench_pathpack.ipynb)
- [Visualization: selected graph path](visualizations/pathpack_selected.png)
- [Visualization: synthetic methods](visualizations/toy_methods.png)
- [Previous AST exploration](notebooks/01_code_graph_research.ipynb)

## Quickstart (macOS / Linux)

Requires Python 3.10+ and Git.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[viz]'
python -m unittest discover -s tests -v
python examples/issue_demo/run_demo.py
codegraph retrieve examples/issue_demo/snapshot \
  --query 'login cookie causes failure' \
  --method pathpack --budget 110 --seeds 1 --hops 2
pip install notebook
jupyter notebook notebooks/02_pathpack_experiments.ipynb
```

## Methods

`bm25` = lexical issue to code retrieval. `graph-expand` = BM25 seed plus relation-aware neighboring source units. `pathpack` = same seed with confidence-weighted paths, semantic-free lexical score, cost-aware packing. The graph is a **conservative static approximation**; path scores are heuristic and have not been validated on real issues. Supported edge types: `calls`, `imports`, `defines`, `inherits`.

Test modules (`tests/`, `test_*.py`) are currently omitted from retrieval chunks by default for fix-localization.

`tokens_est` is a **rough UTF8 byte estimate** (`ceil(bytes/4)`), *not* the Gemma tokenizer. The retrieval methods use exactly the same estimated budget, making this an engineering smoke test rather than a token-precise research result.

## Toy experiment (not SWE-bench)

4 manually constructed issue-to-target-file pairs in `examples/issue_demo` and four proxy budgets:

| budget_est | BM25 hits | Graph-expand hits | PathPack hits |
|---:|---:|---:|---:|
| 45 | 0/4 | 0/4 | 0/4 |
| 70 | 0/4 | 1/4 | 0/4 |
| 110 | 0/4 | 4/4 | 4/4 |
| 160 | 1/4 | 4/4 | 4/4 |

Results can be regenerated with `python examples/issue_demo/run_demo.py`. **PathPack has no demonstrated advantage over simple graph expansion on this tiny example; at budget 70 it is worse after preserving the lexical anchor.** These results must not be used as a paper benchmark.

## Real SWE-bench Verified protocol (not run yet)

The official 500 verified issue dataset includes `instance_id`, `repo`, `base_commit`, `problem_statement` and gold `patch` fields. The latter must be inaccessible to the retriever. See [official documentation](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md).

```bash
pip install -e '.[benchmark,viz]'
codegraph prepare --out output/benchmark --limit 80 --dev-count 20 --seed 42
# Checkout each issue's exact Git commit into snapshots/<instance_id>/
# Each snapshot must be a Git checkout whose HEAD == base_commit.
codegraph retrieve-batch \
  --issues output/benchmark/issues.jsonl --snapshots snapshots \
  --split dev --method pathpack --budget 2048 --out output/dev_pathpack.jsonl
codegraph score --predictions output/dev_pathpack.jsonl \
  --gold output/benchmark/gold_labels.jsonl --out output/dev_scores.json
```

`retrieve-batch` does **not** load the gold patch. It rejects snapshots whose git HEAD differs from expected `base_commit`. For **synthetic demos only**, use `--allow-unverified-snapshots` when there is no Git repo.

Metrics are file-level localization recall and issue hit rate, **not** patch-test success. The scorer excludes cases without retrievable pre-fix Python targets, reporting their count.

## Limitations / next steps

- Validate on genuine SWE-bench snapshots; compare BM25, graph-expand, PathPack and eventually an embedding-based method.
- Improve chunking to preserve imports/top-level module code and class definitions.
- Replace rough estimated tokens with the target model tokenizer before real budget conclusions.
- Run ablations: graph weight 0, path length 1/2/3, seed count.
- Investigate whether graph search offers a meaningful advantage over [GraphCoder](https://arxiv.org/abs/2406.07003) and related methods.
- Only then test Gemma 4 patch creation in controlled evaluation.

## Main files

- `src/codegraph_rag/parser.py` — AST parser and graph builder.
- `src/codegraph_rag/graph.py` — graph data model.
- `src/codegraph_rag/retrieval.py` — new retrieval algorithms.
- `src/codegraph_rag/evaluate.py` — new leak-resistant batch retrieval/scoring separation.
- `src/codegraph_rag/cli.py` — CLI.
- `tests/` — unit/integration tests.
- `reports/toy_results.json` — complete toy predictions and scores.

MIT licensed. This is research infrastructure, not a validated SWE agent.

## Real-code engineering smoke test (not quality evaluation)

The installed NetworkX `algorithms/shortest_paths` source tree was scanned: 242 nodes, 288 edges, 56 retrieval chunks, zero parser errors, approximately 0.06 seconds for the static graph scan in this environment. No gold bugs/patches were used; **this does not show localization accuracy**. Reproducibility details: `reports/networkx_smoke.json`.
