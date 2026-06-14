# Benchmark subsets

This directory contains the fixed data inputs for the SearchSwarm cost-controlled
comparison. Each component has 20 questions and `all_80.jsonl` is their ordered
concatenation. These are experiment-specific subsets, not separate official
"mini" editions of the upstream benchmarks.

The harness loader in `run_multi_react.py` accepts JSON or JSONL. It reads
`task_question` (or alias `question`) and maps `ground_truth` to `answer`.
These files also retain a unique `task_id`, a unique identifier in `file_name`,
and provenance in `metadata`.

## Files and sources

| File | Upstream source | Upstream size | Selection |
|---|---|---:|---|
| `browsecomp_subset_20.jsonl` | OpenAI BrowseComp from `openai/simple-evals` | 1,266 rows | Proportional stratification by `problem_topic`, then seeded sampling |
| `gaia_subset_20.jsonl` | `gaia-benchmark/GAIA`, public 2023 validation/dev split | The official release contains more than 450 total tasks; the gated dev row count is printed after authorized download | Remove rows with a non-empty attachment `file_name`, then target Level 1/2/3 = 8/8/4 |
| `xbench_deepsearch_subset_20.jsonl` | `xbench/DeepSearch` on Hugging Face (`DeepSearch.csv`, the 2505 release) | 100 rows | No category/difficulty fields exist, so seeded random sampling |
| `self_deepsearch_qa_20.jsonl` | Hand-authored for this experiment | 20 rows | Five questions in each of four planned categories |
| `all_80.jsonl` | Ordered concatenation of the four files above | 80 rows | BrowseComp, GAIA, xBench, then self QA |

Official links:

- BrowseComp: <https://github.com/openai/simple-evals> and
  <https://openai.com/index/browsecomp/>
- GAIA: <https://huggingface.co/datasets/gaia-benchmark/GAIA> and
  <https://arxiv.org/abs/2311.12983>
- xBench-DeepSearch: <https://huggingface.co/datasets/xbench/DeepSearch> and
  <https://github.com/xbench-ai/xbench-evals>

## Fixed sampling rules

All random operations use `seed=42`.

### BrowseComp

The script downloads the original 1,266-row BrowseComp test set used by
OpenAI's `simple-evals`. Because the source has `problem_topic`, it computes
20 proportional quotas with the largest-remainder method and samples within
each topic. The selected rows are finally ordered by their original row index.

`browsecomp_subset_20.jsonl` is a cost-controlled sample created for this
experiment. It is not an official independent BrowseComp mini dataset.

### GAIA

The script requests `2023/validation/metadata.parquet` from the official gated
Hugging Face repository. It removes every row whose upstream `file_name` is
non-empty, because this harness sends only text questions and does not attach
local images, audio, spreadsheets, archives, PDFs, or other files.

It then attempts a stratified allocation of Level 1 = 8, Level 2 = 8, and
Level 3 = 4. If an attachment-free level has fewer rows than requested, the
script takes all available rows in that level and fills the remaining slots
from the other attachment-free levels with the same seeded RNG. The script
prints both the eligible pool and actual final level distribution.

GAIA added an access gate to prevent automated scraping. To prepare it:

1. Open <https://huggingface.co/datasets/gaia-benchmark/GAIA>, sign in, and
   accept the dataset access conditions.
2. Authenticate locally with `hf auth login`, or set `HF_TOKEN` only in the
   shell/session.
3. Rerun `python scripts/prepare_benchmark_subsets.py`.

Do not put a Hugging Face token in this README, a JSONL file, `.env.example`,
or git history.

### xBench-DeepSearch

The requested official `xbench/DeepSearch` Hugging Face dataset contains 100
rows in `DeepSearch.csv`. Its schema is `id`, `prompt`, `answer`,
`reference_steps`, and `canary`; it has no category or difficulty field.
The script therefore uses ordinary `random.Random(42).sample(..., 20)`.
Questions and answers are decrypted with the algorithm in the official
`xbench-evals` repository. Reference steps are not copied into the prompt or
ground truth.

### Self deep-search QA

The self QA file is manually maintained, not randomly generated. Its
distribution is exactly:

- `多约束实体识别`: 5
- `时间线追踪`: 5
- `候选比较排除`: 5
- `学术 / 技术检索`: 5

Every question has a unique short answer, at least two source/evidence hints,
and an expected reasoning type. Questions use multiple constraints, timeline
disambiguation, candidate elimination, or paper/specification linkage rather
than open-ended summarization.

## JSONL schema

Each line is one UTF-8 JSON object:

```json
{
  "task_id": "unique-stable-id",
  "task_question": "Question shown to the agent",
  "file_name": "unique-stable-id",
  "ground_truth": "Gold answer used only for evaluation",
  "metadata": {
    "dataset": "dataset name",
    "category": "category or unspecified",
    "difficulty": "level/difficulty or null"
  }
}
```

Official subset metadata also records `sample_seed`, `sampling_method`,
`original_dataset`, `original_split`, `original_id`, source revision/hash, and
any available original category or level. The answer is never appended to
`task_question`.

## Rebuild and validate

Use the same conda environment as the harness:

```bash
cd /home/electrolyte/workspace/SearchSwarm/harness
conda activate searchswarm-harness
python scripts/prepare_benchmark_subsets.py
python scripts/validate_benchmark_data.py
```

The preparation script downloads only small metadata/CSV inputs into the
ignored `eval_data/benchmark/.cache/` directory. It does not run inference,
invoke a model, or call a model API.

The validator checks JSON parsing, required fields, exact 20/80 counts,
non-empty questions and answers, unique `file_name` values,
`metadata.dataset`, obvious verbatim answer leakage, component ordering, GAIA
levels, and category distributions.

## Harness configuration

Set one path at a time in the local `.env`:

```bash
DATASET=eval_data/benchmark/browsecomp_subset_20.jsonl
# DATASET=eval_data/benchmark/gaia_subset_20.jsonl
# DATASET=eval_data/benchmark/xbench_deepsearch_subset_20.jsonl
# DATASET=eval_data/benchmark/self_deepsearch_qa_20.jsonl
# DATASET=eval_data/benchmark/all_80.jsonl
```

For a one-question smoke test without editing the fixed subset, create a
temporary one-line file outside version control:

```bash
head -n 1 eval_data/benchmark/browsecomp_subset_20.jsonl > /tmp/searchswarm-smoke.jsonl
```

Then set `DATASET=/tmp/searchswarm-smoke.jsonl`. Set `ENABLE_SUB_AGENT=0` for
Single Agent or `ENABLE_SUB_AGENT=1` for SearchSwarm. Both experiment
conditions must use byte-identical fixed dataset files for a fair comparison.

The benchmark wrappers automate the same setup while preserving `.env` model
and API credentials:

```bash
python scripts/make_smoke_dataset.py \
  --input eval_data/benchmark/browsecomp_subset_20.jsonl \
  --output /tmp/browsecomp_subset_20_smoke_1.jsonl --n 1
bash run_single_agent_benchmark.sh /tmp/browsecomp_subset_20_smoke_1.jsonl
bash run_searchswarm_benchmark.sh /tmp/browsecomp_subset_20_smoke_1.jsonl
python scripts/summarize_benchmark_results.py
```

Each invocation creates a new timestamped directory below
`results/benchmark/<dataset>/<single|swarm>/`. Both variants force
`TOOL_TYPE=four`, so search, visit, Google Scholar, and Python remain available;
the only capability difference is `call_sub_agent`.

One-row SearchSwarm smoke runs default `BENCHMARK_REQUIRE_SUB_AGENT_CALL=1`.
This adds a smoke-only requirement to dispatch at least one meaningful
sub-agent and makes the wrapper fail if no usable sub-agent trajectory is
produced. Multi-row runs default it to `0`, preserving the model's natural
delegation policy; either behavior can be selected explicitly with the same
variable. Single-agent runs always disable the requirement and fail validation
if a sub-agent trajectory appears. Tool-call format retries default to three
and can be changed with `BENCHMARK_MAX_TOOL_FORMAT_RETRIES`.

When a sub-agent reaches a force-answer boundary,
`SUB_AGENT_FORCE_ANSWER_ATTEMPTS` controls how many no-tools delivery attempts
are allowed (default: `2`). Text-encoded DSML/XML tool calls are rejected and
retried once by default. If no valid `<report>` is produced, the harness gives
the main agent a clearly marked evidence fallback, while strict smoke
validation still fails with a fallback-only status.

## Redistribution and git

BrowseComp and xBench ship encrypted data with canaries stating that plaintext
must not appear online. GAIA also asks users not to reshare validation/test
data in crawlable form. Consequently, decrypted official subset files,
`all_80.jsonl`, and raw caches are intentionally ignored by git even though
they are generated locally for evaluation.

The `openai/simple-evals` repository and the `xbench/DeepSearch` dataset card
declare MIT licensing, but their anti-contamination plaintext warnings still
apply. GAIA access is governed by the terms accepted through its Hugging Face
gate; no GAIA records should be redistributed from this workspace.

It is appropriate to commit this README, both scripts, the hand-authored
`self_deepsearch_qa_20.jsonl`, dependency changes, and `.gitignore`. Do not
commit `.env`, API keys, Hugging Face tokens, raw benchmark caches, decrypted
official benchmark JSONL, or model outputs.
