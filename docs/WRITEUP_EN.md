# History Replay and Expected-Coverage Packing for Edit Localization Under a Token Budget

*Learning where to edit from repository history, without benchmark labels, for small local coding agents*

## Abstract

A small local language model runs with a limited context budget, so retrieval must fit the functions and lines that a fix will change into that budget. On SWE-bench Verified we find that a common heuristic packer finds the right files but loses the right functions, and that most of the loss happens when fragments are chosen from the candidate pool. We propose PathPack-H, which estimates the probability that each fragment will be edited with a model trained only on the past commits of the repositories themselves, and packs the context by expected coverage per token. On 100 held-out tasks at a budget of 2048 conditional tokens, PathPack-H raises Function Recall from 0.192 to 0.529 and Changed Line Recall from 0.169 to 0.445 relative to our previous method v0.4, while File Recall drops from 0.847 to 0.803. We then propose PathPack-U, which maximizes a weighted sum of expected File, Function, Complete Function and Changed Line Recall, with weights derived from history. On two further held-out sets (51 and 125 tasks) PathPack-U beats v0.4 on all four metrics. Models trained without the history of the target repository perform the same. A probe with a local Gemma 4 E2B model shows that the PathPack-H context raises the share of tasks where the model names the correct function from 0.364 to 0.505 on the first held-out set. We did not generate patches or run tests, so the share of fixed bugs is not measured.

## 1. Introduction

A coding agent that fixes a bug selects a few thousand tokens out of a repository of hundreds of thousands. We treat the task as edit localization under a budget. Given the task text and a snapshot at the base commit, select a set of fragments that contains the lines the reference fix changes. Quality is measured by coverage of the changed files, functions and lines, not by textual similarity to the task.

Our diagnosis is that retrieval loses information in two places. First, the candidate pool contains most of the needed functions (0.80 on held-out tasks), but the packed context keeps only 0.50 to 0.60 of them. Second, a stop rule that halts when all symbols named in the task are covered drops needed code, because the text of a task names only some of the places a fix touches.

Contributions:

1. A diagnosis of the packing loss, with a measured candidate pool and packed context reported separately.
2. PathPack-H, which learns an edit-probability model from history replay: past commits serve as pseudo-tasks, with guards against leakage from the benchmark's fix commits.
3. PathPack-U, a packing objective equal to the sum of expected recalls, with weights derived from history rather than tuned.
4. A transfer test on held-out repositories and an ablation of the components.
5. A probe of localization by a local language model on the packed contexts.

We do not claim that these methods make agents fix more bugs. We claim that they select code that is more often the code a fix needs, under a fixed budget, and we state which claims the data support.

## 2. Related Work

Learning a localizer from commit history is not new at the file level. CommitCompass trains a gradient-boosting ranker on pairs of commit messages and changed files, from the same twelve SWE-bench repositories, and reports file-level Acc@1 of 47.7 on SWE-bench Lite [CommitCompass]. Gandhi, Gao and Callan use BM25 over commit messages and CodeBERT rerankers trained on commit triplets, evaluated at the file level on seven repositories [arXiv:2502.07067]. Wang et al. give an agent BM25 over the messages of 7000 earlier commits as a memory tool, reporting Acc@1 of 68.6 on SWE-bench Verified at the file level [arXiv:2510.01003]. Classical bug localization already used change history [Zhou et al., ICSE 2012; Wen et al., ASE 2016].

Agent-based localization is another line. LocAgent traverses a code graph with an LLM and fine-tunes a 7B and a 32B model on successful trajectories [arXiv:2503.09089]. CoSIL searches the call graph iteratively with an LLM and reports Top-1 accuracy of 43.3% on Lite [arXiv:2503.22424]. OrcaLoca schedules LLM actions with context pruning by distance [arXiv:2502.00350]. Agentless uses a three-phase pipeline without an agent loop [arXiv:2407.01489]. SweRank trains a bi-encoder and a listwise reranker on issue-to-function pairs from repositories outside SWE-bench [arXiv:2505.07849]. IssueExec uses tests as a bridge between issues and code through execution traces [arXiv:2607.17286], and Issue-Test-Localizer maps tests to source lines by coverage [IBM]. We checked these sources against their arXiv and repository pages; for IssueExec and Issue-Test-Localizer were checked only through search excerpts. we read only search excerpts.

Our differences are the function and line levels, the fixed token budget, history replay with temporal guards, and packing weights derived from history. Our numbers are not comparable with the Acc@k values above.

Evaluating the context that retrieval gives agents under a budget is itself an active topic. ContextBench measures context recall, precision, and efficiency along agent trajectories on 1,136 issue-resolution tasks [arXiv:2602.05892]. SWE-Explore asks explorers for ranked code regions under a fixed line budget [arXiv:2606.07297]. Agent Retrieval Bench reports budgeted context yield at 8K tokens for file-level retrieval [arXiv:2607.24882]. We do not claim that budgeted evaluation of context is new. Our contribution is a learned edit-probability model trained by history replay, and a packing objective optimized under a token budget, evaluated at function and line level on held-out SWE-bench Verified tasks. Our numbers are not comparable with the Acc@k values above, and we did not run a direct comparison.

## 3. Method

### 3.1 Setting and metrics

The input is a repository at `base_commit`, a task text and a budget $B$ in conditional tokens, computed as $\lceil \text{UTF-8 bytes} / 4 \rceil$ over the serialized context with headers and paths. Metrics: File Recall (changed files covered); Function Recall (changed functions with at least 20% of body lines covered); Complete Function Recall (changed functions covered in full); Changed Line Recall (deleted or replaced base lines); and Insertion Anchor Recall for pure insertions.

### 3.2 Baseline and fusion

PathPack v0.4 scores AST chunks with BM25, propagates relevance over the dependency graph for two hops, and packs with a diversity term and a cost term. The fusion variant merges four channels (lexical BM25, exact identifier matches, UniXcoder-base embeddings, and one-hop structural neighbors) into a fixed weighted score, and packs greedily.

### 3.3 PathPack-H: learning from history

Each fragment has 33 features: channel scores and ranks, sizes, positions, percentiles over the file and the pool, and six history features computed from commits strictly before `base_commit`. A gradient-boosting model with a second stage over call edges outputs the edit probability.

Training data come from history replay. For each repository we take the earliest `base_commit` among its benchmark tasks. Earlier commits become pseudo-tasks: the commit message is the query, and fragments of changed non-test files whose function or class appears in a hunk header are positives. Three guards prevent leakage. Only ancestors of the earliest `base_commit` are read. Commits must predate the earliest task creation date in the repository. History features are built only from strictly older commits. The module never reads the benchmark patch field, and a test checks this. The result is 1295 pseudo-tasks from twelve repositories.

Packing is greedy by expected coverage per token, with a bonus for the first fragment of a new file.

### 3.4 PathPack-U: packing for expected metrics

PathPack-U adds two models: one estimates the probability that a function is edited, and one estimates the probability that a file is edited. Both are trained on aggregates of the fragment probabilities, out-of-fold by task. The packing objective is

$$U(S) = w_{\text{file}} \sum_{f} P(f)\,[S \text{ touches } f] + w_{\text{fn}} \sum_{g} P(g)\,[\mathrm{cov}_g(S) \ge 0.20] + w_{\text{fn}} \sum_{g} P(g)\,[\mathrm{cov}_g(S) = 1] + w_{\text{chunk}} \sum_{c} P(c)\,[c \in S]$$

Each term is a probability-weighted surrogate for expected coverage, not an exact expected recall: a fragment's probability is not its exact contribution to Changed Line Recall. During optimization, function coverage is counted over tokens; in evaluation, it is counted over lines. The weights are the reciprocals of the mean number of edited files, functions and fragments per pseudo-task, which gives $w_{\text{file}} = 0.815$, $w_{\text{fn}} = 0.384$ and $w_{\text{chunk}} = 0.105$. Nothing is tuned. Selection is greedy per token, with two move types: adding one fragment, or adding all remaining fragments of one function. The second move matters because the completion term is never earned by a single fragment.

### 3.5 Probe with a local language model

A local Gemma 4 E2B model receives the task text and a packed context of 2048 conditional tokens, and names one function to edit in JSON. A file counts as correct if it is in the reference patch, and a function counts as correct if the patch changes it. The comparison plan was fixed before answers were scored.

## 4. Experimental Protocol

Data are SWE-bench Verified: 500 tasks at exact base commits. The diagnostic set (20 tasks) was used for early development and is not independent. Validation (99 tasks) was used to choose the PathPack-H variant. The first held-out set (100 tasks) was used once to evaluate it. The second held-out set (51 tasks) and the third (125 tasks, drawn from the train pool and never retrieved before) were used once each for PathPack-U. Each held-out run had a frozen manifest, a recorded hash of the source tree, and a stated list of hypotheses written before the run. Intervals are paired bootstrap intervals over tasks (2000 iterations, seed 42), and for the main comparisons a bootstrap over repositories (4000 iterations).

Validation was used twice. The second and third sets are the independent checks for PathPack-U. The third set was drawn from the train pool, but the 45 tasks used to build PathPack-E training data were excluded. None of the 125 tasks was used to train the final H or U models, to tune parameters, or to choose the architecture. Disjointness was checked by `instance_id`.

## 5. Results

### 5.1 First held-out set, budget 2048 (100 tasks)

| Method        |  File | Function | Complete | Changed Line |
| ------------- | ----: | -------: | -------: | -----------: |
| BM25          | 0.602 |    0.353 |    0.353 |        0.367 |
| PathPack v0.4 | 0.847 |    0.192 |    0.094 |        0.169 |
| Fusion        | 0.745 |    0.433 |    0.298 |        0.392 |
| PathPack-H    | 0.803 |    0.529 |    0.303 |        0.445 |

PathPack-H minus v0.4, with 95% intervals over tasks: Function +0.337 [+0.235, +0.442], Complete +0.209 [+0.122, +0.301], Changed Line +0.277 [+0.168, +0.386], File −0.043 [−0.102, +0.017]. Against fusion, only Function is significant (+0.097 [+0.029, +0.176]).

### 5.2 Second and third held-out sets, budget 2048

| Set       | Method     |  File | Function | Complete | Changed Line |
| --------- | ---------- | ----: | -------: | -------: | -----------: |
| 51 tasks  | v0.4       | 0.735 |    0.252 |    0.145 |        0.229 |
|           | PathPack-U | 0.863 |    0.508 |    0.417 |        0.414 |
| 125 tasks | v0.4       | 0.649 |    0.186 |    0.115 |        0.156 |
|           | PathPack-U | 0.765 |    0.419 |    0.340 |        0.352 |

On the 125-task set, PathPack-U minus v0.4 gives File +0.116 [+0.056, +0.180], Function +0.233 [+0.143, +0.319], Complete +0.226 [+0.156, +0.301] and Changed Line +0.196 [+0.112, +0.285]. On the 51-task set, PathPack-U minus v0.4 gives File +0.127 [+0.029, +0.235], Function +0.256 [+0.141, +0.381], Complete +0.272 [+0.153, +0.403] and Changed Line +0.185 [+0.071, +0.308]. Against PathPack-H on the 125-task set, PathPack-U is higher on File (+0.130 [+0.078, +0.187]) and Complete (+0.085 [+0.025, +0.149]), and not distinguishable on Function and Changed Line.

### 5.3 Transfer and ablations

Models trained without the history of the target repository, using pseudo-tasks from the other eleven repositories only, match the models that include it: the difference in every metric is at most 0.016 in absolute value, and all intervals include zero. Replacing the two trained function and file models by a noisy-OR of fragment probabilities changes no metric beyond noise. The whole-function move contributes +0.057 [+0.018, +0.104] to Complete Function Recall. A static test-based bridge showed no signal (AUC 0.50 against a lexical AUC of 0.73 to 0.81 on 22 training tasks).

### 5.4 Probe with Gemma 4 E2B

Share of tasks where the model names the correct file and function, at budget 2048. Denominators: first set 100 tasks (file) and 99 (function); second set 51 and 42; third set 125 and 106. Model: `gemma4:e2b-mlx` (Ollama digest `f4a8449d1781`). Decoding: temperature 0, seed 0, JSON output, thinking disabled, 8192-token window, at most 120 generated tokens. The prompt text is in `scripts/llm_localize.py`.

| Set        | Method     |  File | Function |
| ---------- | ---------- | ----: | -------: |
| first, 100 | v0.4       | 0.660 |    0.364 |
|            | PathPack-H | 0.710 |    0.505 |
| second, 51 | v0.4       | 0.647 |    0.500 |
|            | PathPack-U | 0.804 |    0.524 |
| third, 125 | v0.4       | 0.432 |    0.264 |
|            | PathPack-H | 0.544 |    0.406 |
|            | PathPack-U | 0.568 |    0.377 |

Paired differences with 95% intervals: PathPack-H minus v0.4 on Function gives +0.141 [+0.040, +0.242] on the first set (36 of 99 against 50 of 99 tasks) and +0.142 [+0.047, +0.236] on the third (28 of 106 against 43 of 106). PathPack-U minus v0.4 on File gives +0.157 [+0.059, +0.255] on the second set and +0.136 [+0.064, +0.216] on the third. The function gain did not replicate on the second set, where v0.4 already scores 0.500.

## 6. Analysis and Limitations

PathPack-H recovers much of the function and line coverage that v0.4 loses, at a small cost in file coverage. PathPack-U is the first variant we built that beats v0.4 on all four metrics on two independent sets. Learning from history transfers to repositories the model has not seen.

Some claims do not hold. A learned model is not clearly better than a hand-tuned fusion on Function and Changed Line. Commit memory and history features do not add signal beyond history as training data. Within a file, the models still do not separate the edited function from its neighbors. The probe shows a benefit for a small local model on localization, not on repair.

The limitations are these. History replay reads code from the snapshot at the earliest base commit, not from the tree at each historical commit, so a pseudo-task can describe code that changed after an earlier fix. The temporal guards remove benchmark fix commits from training, but they do not remove this effect. We did not generate patches or run tests, so we do not measure the share of fixed bugs, and we make no claim about agent performance. The held-out sets are small, and the third set is dominated by django (84 of 125 tasks). Transfer was tested on five repositories, all in Python and all from SWE-bench. Token counts are conditional and are not Gemma token counts. The probe uses one model and one prompt. The validation set was reused, and the held-out sets were used once each.

## 7. Conclusion

Learning edit probabilities from the history of the repository itself, with explicit leakage guards, and packing for expected coverage, improves what a small model sees under a fixed budget. On two independent sets of 51 and 125 tasks, PathPack-U beats our earlier best method on all four coverage metrics. The next question is whether this coverage changes what a small agent repairs, and we leave that for a repair experiment with fixed test environments.

## Reproducibility

Code, manifests, freeze files and run summaries: https://github.com/ArduRadioKot/PathPack (commit [SHA to be inserted]). Full paper: https://github.com/ArduRadioKot/PathPack/blob/main/docs/PAPER_EN.md.

## References

- [CommitCompass] Repository [github.com/VijayChamakuri/commitcompass](https://github.com/VijayChamakuri/commitcompass) (no paper, no peer review; viewed 11 November 2026).
- Chen Z., Tang X., Deng G., Wu F., Wu J., Jiang Z., Prasanna V., Cohan A., Wang X. LocAgent: Graph-Guided LLM Agents for Code Localization. [arXiv:2503.09089](https://arxiv.org/abs/2503.09089), 2025.
- Gandhi S., Gao L., Callan J. Repository-level Code Search with Neural Retrieval Methods. [arXiv:2502.07067](https://arxiv.org/abs/2502.07067), 2025.
- IssueExec: A Test-Driven Approach for Localizing Software Engineering Issues. [arXiv:2607.17286](https://arxiv.org/abs/2607.17286), 2026.
- Issue-Test-Localizer. IBM, [github.com/IBM/Issue-Test-Localizer](https://github.com/IBM/Issue-Test-Localizer).
- Jiang Z., Ren X., Yan M., Jiang W., Li Y., Liu Z. Issue Localization via LLM-Driven Iterative Code Graph Searching. [arXiv:2503.22424](https://arxiv.org/abs/2503.22424), ASE 2025.
- Jimenez C. E., Yang J., Wettig A., Yao S., Pei K., Press O., Narasimhan K. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? [arXiv:2310.06770](https://arxiv.org/abs/2310.06770), ICLR 2024.
- Reddy R. G., Suresh T., Doo J., Liu Y., Nguyen X. P., Zhou Y., Yavuz S., Xiong C., Ji H., Joty S. SweRank: Software Issue Localization with Code Ranking. [arXiv:2505.07849](https://arxiv.org/abs/2505.07849), ICLR 2026.
- Wang B., Xu W., Li Y., Gao M., Xie Y., Sun H., Chen D. Improving Code Localization with Repository Memory. [arXiv:2510.01003](https://arxiv.org/abs/2510.01003), ICLR 2026.
- Wen M., Wu R., Cheung S.-C. Locus: Locating Bugs from Software Changes. ASE 2016. [Preprint](https://cse.hkust.edu.hk/faculty/scc/publ/ase16-Locus-preprint.pdf).
- Xia C. S., Deng Y., Dunn S., Zhang L. Agentless: Demystifying LLM-based Software Engineering Agents. [arXiv:2407.01489](https://arxiv.org/abs/2407.01489), 2024.
- Yu Z., Zhang H., Zhao Y., Huang H., Yao M., Ding K., Zhao J. OrcaLoca: An LLM Agent Framework for Software Issue Localization. [arXiv:2502.00350](https://arxiv.org/abs/2502.00350), ICML 2025.
- Zhou J., Zhang H., Lo D. Where Should the Bugs Be Fixed? More Accurate Information Retrieval-Based Bug Localization Based on Bug Report. ICSE 2012.
- Li H., Zhu L., Zhang B., Feng R., Wang J., Pan Y., Barr E. T., Sarro F., Chu Z., Ye H. ContextBench: A Benchmark for Context Retrieval in Coding Agents. [arXiv:2602.05892](https://arxiv.org/abs/2602.05892), 2026.
- Zhang S., Wang Y., Liang J., Shi Y., Zeng W., Wang M., He S., Xu N., Ye S., Cai K., Gu X. SWE-Explore: Benchmarking How Coding Agents Explore Repositories. [arXiv:2606.07297](https://arxiv.org/abs/2606.07297), 2026.
- Qin B., Xie Y. Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents. [arXiv:2607.24882](https://arxiv.org/abs/2607.24882), 2026.
