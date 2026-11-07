# Evals

Three suites, one command, one gate.

```bash
python -m evals.run                                          # all suites with the offline model (CI)
python -m evals.run --suite agent --model configured --judge  # real model + LLM judge
```

Results print as a table and are saved to `evals/reports/eval-<time>.json`. If any metric falls below
`evals/thresholds.json`, the command exits with code 1 and CI fails.

## Suites

| Suite | Data | Metrics | Needs a model |
|---|---|---|---|
| Retrieval | `datasets/retrieval.jsonl`: 36 questions with the review IDs that answer them | recall@5, MRR | No |
| Guardrails | `datasets/guardrails.jsonl`: 20 attacks, 20 look-alike benign questions, 5 PII inputs, 5 poisoned reviews | attack block rate, benign pass rate, precision, PII recall, untrusted-content recall | No |
| Agent | `datasets/agent.jsonl`: 18 end-to-end cases, incl. multi-turn memory | task success, tool selection, citation validity, refusals, safety, p50/p95 latency, cost, tokens | Yes (offline stand-in in CI) |

Each agent case lists only the checks that matter for it: expected tools, must cite, cited reviews must
belong to the right products, words that must or must not appear (for example, the injected
"best CRM ever made" or a reviewer's phone number), the exact average rating from the database, and
profile keys saved by memory.

## Offline vs online evaluation

- **Offline** (before release): the suites above, in CI on every push.
- **Online** (in production): every request is checked live (citation validity, guardrail flags) and stored
  in `agent_runs`. Users can rate answers through `/v1/feedback`. `GET /v1/admin/runs?flagged_only=true`
  is the review queue.

The loop: review flagged runs and thumbs-down answers, label the real failures, add them to the datasets,
fix, and let the gate prove the fix holds.

## LLM judge

`evals/judge.py` grades groundedness and relevance from 1 to 5 against the cited evidence, with structured
output. A judge can drift, so check it against human labels from time to time: label 20-30 answers
yourself, compare with the judge's scores, and tighten the rubric where they disagree.

## Reading the current numbers honestly

The sample data is synthetic and small, so near-perfect retrieval scores mean the pipeline works, not that
search is solved. The offline agent numbers test the plumbing (tools, citations, memory, safety checks);
model quality comes from the `--model configured --judge` run.
