# PathPack: edit localization under a token budget

**English** | [Русский](README.ru.md)

PathPack is a research code base for selecting repository code that a small local coding agent can read within a fixed token budget. The main question is where a fix will edit, not which files look similar to the task text.

The project contains:

- a Python code graph and retrieval pipeline (PathPack v0.4 and its successors);
- PathPack-H, which learns edit probabilities from the history of repositories (history replay, no benchmark labels);
- PathPack-U, which packs context to maximize a weighted sum of expected File, Function, Complete Function and Changed Line Recall;
- evaluation code with frozen splits, paired bootstrap statistics, freeze files and run summaries;
- a probe of localization by a local Gemma 4 E2B model.

## Documents

- Full research paper, English: [docs/PAPER_EN.md](docs/PAPER_EN.md), PDF: [docs/pdf/PathPack_Research_Paper.pdf](docs/pdf/PathPack_Research_Paper.pdf)
- Full research paper, Russian: [docs/PAPER_RU.md](docs/PAPER_RU.md), PDF: [docs/pdf/PathPack_Research_Paper_RU.pdf](docs/pdf/PathPack_Research_Paper_RU.pdf)
- Competition write-up, English (3000-word limit): [docs/WRITEUP_EN.md](docs/WRITEUP_EN.md)
- Competition write-up, Russian: [docs/WRITEUP_RU.md](docs/WRITEUP_RU.md)
- Figures: [docs/figures_en](docs/figures_en) (English) and [docs/figures](docs/figures) (Russian)
- Notebook with tables and figures for PathPack-H and PathPack-U: [notebooks/11_pathpack_h.ipynb](notebooks/11_pathpack_h.ipynb)

## Main results

All numbers are at a budget of 2048 conditional tokens (`ceil(UTF-8 bytes / 4)` over the serialized context). They are not Gemma tokenizer counts.

On 100 held-out SWE-bench Verified tasks, PathPack-H against PathPack v0.4:

| Metric | v0.4 | PathPack-H |
|---|---:|---:|
| File Recall | 0.847 | 0.803 |
| Function Recall | 0.192 | 0.529 |
| Complete Function Recall | 0.094 | 0.303 |
| Changed Line Recall | 0.169 | 0.445 |

On two further held-out sets (51 and 125 tasks), PathPack-U beats v0.4 on all four metrics. Models trained without the history of the target repository give the same results. The probe with Gemma 4 E2B is a localization measure only: no patches are generated and no tests are run.

## Layout

- `src/codegraph_rag/`: graph parser, chunking, retrieval, PathPack versions and the PathPack-R/E/H/U pipeline (`src/codegraph_rag/pathpack_r/`).
- `scripts/`: experiment drivers, split construction, figure and notebook generation, the Gemma probe.
- `tests/`: unit tests for graph parsing, retrieval, leakage guards, budget accounting and the training rules.
- `configs/`: frozen configuration files.
- `output/`: run summaries (CSV, JSON) of the experiments reported in the papers. Per-task predictions, model files and caches are not versioned.
- `docs/`: papers, write-ups, figures and reports.

## Setup

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"
```

The benchmark data are not in this repository. SWE-bench Verified is read from the Hugging Face dataset `SWE-bench/SWE-bench_Verified`. Repository snapshots at the exact `base_commit` are created by `scripts/fetch_snapshots.py` from local clones. Set the snapshot directory in the scripts (default `data/swebench/snapshots`).

## Reproducing the main results

Unit tests (no benchmark data needed for most of them):

```bash
PYTHONPATH=src python -m unittest tests.test_pathpack_e tests.test_pathpack_r tests.test_pathpack_v10e
```

Summaries of the frozen held-out runs are already in `output/`. The notebook rebuilds the tables and figures from them:

```bash
python scripts/write_pathpack_h_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/11_pathpack_h.ipynb
```

Re-running the held-out experiments requires the snapshots and the history replay cache. The commands and the freeze files that were written before each run are listed in the reproducibility section of [docs/PAPER_EN.md](docs/PAPER_EN.md).

Freeze files (hashes of the source tree and models, hypotheses written before each run): [output/pathpack_e](output/pathpack_e) and the manifests in [output/pathpack_r_splits_v2](output/pathpack_r_splits_v2).

## Limitations

- No patch generation and no test execution. The paper measures localization and context coverage, not the share of fixed bugs.
- The held-out sets are small (51 to 125 tasks) and dominated by django in the largest set.
- History replay reads code from the snapshot at the earliest base commit, not from the tree at each historical commit.
- Only Python repositories from SWE-bench were evaluated.

## Citing and license

The text of the papers and the code are under the license in [LICENSE](LICENSE).

## Acknowledgements

The benchmark is SWE-bench (Jimenez et al., ICLR 2024). Related localization work that this project compares against is listed in the references of [docs/PAPER_EN.md](docs/PAPER_EN.md).
