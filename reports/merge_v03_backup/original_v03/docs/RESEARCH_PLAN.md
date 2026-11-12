# Research scope and falsifiable claims (draft, 2026-11-01)

## Question

Can connected dependency-path selection improve retrieval of code relevant to a real issue at a **fixed prompt token budget**, relative to text retrieval and a naive graph-neighborhood expansion?

## Proposed hypotheses

- H1: PathPack yields higher gold-file recall under the same token budget.
- H2: Gains concentrate in issues whose modified files are graph-distant from issue-matching entry points.
- H3: PathPack has a measurable runtime and parsing overhead; report it rather than ignore it.

## Negative outcome is valid

PathPack may not outperform BM25/vector search or may improve file recall without improving patch success. Do not claim agent reliability from localization alone.

## First baseline comparisons (November 3 onward)

- BM25 over identical function chunks.
- Embedding similarity over identical function chunks.
- BM25 plus 1-hop / 2-hop graph neighbors.
- PathPack (to be implemented).

Token budgets: 1024, 2048, 4096. Use one tokenizer for all, and count serialized context, not merely AST source sizes.

## Data leakage controls

- `issues.jsonl` contains only public issue metadata + problem statement.
- `gold_labels.jsonl` stays in scoring process, never ranking, prompting or retrieval.
- Git checkout is always pinned to pre-fix `base_commit`.
- Tune weights and expansion rules on dev split only.
- Report evaluation on held-out tasks once parameters are frozen.

## Threats to validity

- Git patches can change files which are not the only files needed to understand a defect.
- Static call graphs miss polymorphism and dynamic calls.
- Similar tasks from the same repo may be correlated.
- Selected issues may differ systematically from the full 500-item benchmark.
- Better retrieval recall may not improve actual code repair.
