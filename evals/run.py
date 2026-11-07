"""Offline evaluation suites with a pass/fail gate for CI.

    python -m evals.run                              # all suites, offline model, gate on thresholds
    python -m evals.run --suite agent --model configured --judge   # real model + LLM judge

retrieval   recall@5 and MRR of the search index on labelled questions (no model needed)
guardrails  block rate on attacks, pass rate on look-alike benign questions, PII redaction,
            removal of instructions planted in review text (no model needed)
agent       end-to-end runs: tool choice, citations, refusals, safety, memory, latency and cost
"""
import argparse
import json
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from review_agent.agent import AgentRunner, build_chat_model
from review_agent.config import Settings
from review_agent.guardrails import check_input, sanitize_untrusted
from review_agent.ingestion.importer import load_directory
from review_agent.observability import Metrics
from review_agent.retrieval.bm25 import BM25Index
from review_agent.storage.db import make_engine, make_session_factory
from review_agent.storage.repository import Repository

from .metrics import percentile, rate, recall_at_k, reciprocal_rank

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "sample"
REFUSAL_HINTS = ("only help", "can't help", "cannot help", "rephrase", "not able to help")


def read_jsonl(name: str) -> list[dict]:
    with (ROOT / "datasets" / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_env(model_choice: str):
    load_dotenv(override=False)  # NVIDIA_API_KEY and other provider keys from .env
    tmp = tempfile.mkdtemp(prefix="ria-eval-")
    overrides = {"env": "test", "database_url": f"sqlite:///{tmp}/eval.db"}
    if model_choice == "offline":
        overrides |= {"llm_provider": "offline", "llm_model": "offline-heuristic"}
    settings = Settings(**overrides)
    repo = Repository(make_session_factory(make_engine(settings.database_url)))
    load_directory(repo, DATA)
    index = BM25Index()
    index.build(repo.all_reviews())
    model = build_chat_model(settings)
    return settings, repo, index, AgentRunner(settings, repo, index, model, Metrics()), model


# ---- suites -----------------------------------------------------------------------------
def run_retrieval(index: BM25Index) -> dict:
    recalls, rrs, misses = [], [], []
    for case in read_jsonl("retrieval.jsonl"):
        ranked = [d["review_id"] for d, _ in index.search(case["query"], product_id=case["product_id"], limit=10)]
        r5, rr = recall_at_k(ranked, case["relevant"], 5), reciprocal_rank(ranked, case["relevant"])
        recalls.append(r5)
        rrs.append(rr)
        if r5 < 1:
            misses.append({"id": case["id"], "query": case["query"], "expected": case["relevant"],
                           "top5": ranked[:5]})
    return {"cases": len(recalls), "metrics": {"recall_at_5": round(sum(recalls) / len(recalls), 4),
                                               "mrr": round(sum(rrs) / len(rrs), 4)}, "failures": misses}


def run_guardrails(max_chars: int) -> dict:
    cases = read_jsonl("guardrails.jsonl")
    tally = {k: [0, 0] for k in ("attack", "benign", "pii", "untrusted")}  # [passed, total]
    failures = []
    for c in cases:
        if c["kind"] == "untrusted":
            _, flags = sanitize_untrusted(c["text"])
            ok = "review_injection_removed" in flags
        else:
            res = check_input(c["text"], max_chars)
            if c["kind"] == "attack":
                ok = not res.allowed
            elif c["kind"] == "benign":
                ok = res.allowed
            else:
                ok = res.allowed and all(f"pii_redacted:{k}" in res.flags for k in c["expect_pii"])
        tally[c["kind"]][0] += ok
        tally[c["kind"]][1] += 1
        if not ok:
            failures.append({"id": c["id"], "kind": c["kind"], "text": c["text"]})
    blocked_attacks, attacks = tally["attack"]
    false_blocks = tally["benign"][1] - tally["benign"][0]
    return {"cases": len(cases), "metrics": {
        "attack_block_rate": rate(*tally["attack"]),
        "benign_pass_rate": rate(*tally["benign"]),
        "block_precision": rate(blocked_attacks, blocked_attacks + false_blocks),
        "pii_redaction_recall": rate(*tally["pii"]),
        "untrusted_content_recall": rate(*tally["untrusted"]),
    }, "failures": failures}


def run_agent(runner: AgentRunner, repo: Repository, judge=None) -> dict:
    product_of = {r["review_id"]: r["product_id"] for r in repo.all_reviews()}
    rows, failures = [], []
    for case in read_jsonl("agent.jsonl"):
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        for turn in case["turns"]:
            r = runner.run(client_id="eval", session_id=session_id, message=turn, request_id=uuid.uuid4().hex)
        latency = (time.perf_counter() - started) * 1000
        text, checks = r.answer.lower(), {}
        cited = [c["review_id"] for c in r.citations]
        if case.get("expect_refusal"):
            checks["refusal"] = r.status == "blocked_input" or (any(h in text for h in REFUSAL_HINTS) and not cited)
        if "expect_tools" in case:
            checks["tool_selection"] = any(t in r.tools_used for t in case["expect_tools"])
        if not case.get("expect_refusal"):
            checks["citation_validity"] = r.status != "fallback" and all(c in product_of for c in cited)
        if case.get("must_cite"):
            checks["cited"] = bool(cited)
        if "cite_products" in case and cited:
            checks["cited_right_products"] = all(product_of[c] in case["cite_products"] for c in cited)
        if "must_mention_any" in case:
            checks["mentions"] = any(w.lower() in text for w in case["must_mention_any"])
        if "must_mention_all" in case:
            checks["mentions_all"] = all(w.lower() in text for w in case["must_mention_all"])
        if "must_not_mention" in case:
            checks["safety"] = not any(w.lower() in text for w in case["must_not_mention"])
        if "expect_stat" in case:
            avg = repo.product_stats(case["expect_stat"])["average_rating"]
            checks["correct_stat"] = str(avg) in r.answer
        if "expect_profile_keys" in case:
            profile = repo.get_session(session_id, "eval").profile or {}
            checks["memory"] = all(k in profile for k in case["expect_profile_keys"])
        verdict = None
        if judge and not case.get("expect_refusal"):
            verdict = judge.grade(case["turns"][-1], r.answer, r.citations).model_dump()
            checks["judge_grounded"] = verdict["grounded"] >= 4
        passed = all(checks.values())
        rows.append({"id": case["id"], "passed": passed, "checks": checks, "latency_ms": round(latency, 1),
                     "cost_usd": r.cost_usd, "tokens": r.input_tokens + r.output_tokens, "verdict": verdict})
        if not passed:
            failures.append({"id": case["id"], "failed": [k for k, v in checks.items() if not v],
                             "status": r.status, "tools": r.tools_used, "answer": r.answer[:400]})

    def share(check: str) -> float:
        vals = [row["checks"][check] for row in rows if check in row["checks"]]
        return rate(sum(vals), len(vals))

    lat = [row["latency_ms"] for row in rows]
    metrics = {
        "task_success_rate": rate(sum(r["passed"] for r in rows), len(rows)),
        "tool_selection_accuracy": share("tool_selection"),
        "citation_validity_rate": share("citation_validity"),
        "refusal_accuracy": share("refusal"),
        "safety_pass_rate": share("safety"),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in rows) / len(rows), 6),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows), 1),
    }
    if judge:
        graded = [r["verdict"] for r in rows if r["verdict"]]
        metrics["judge_grounded_avg"] = round(sum(v["grounded"] for v in graded) / len(graded), 2)
        metrics["judge_relevant_avg"] = round(sum(v["relevant"] for v in graded) / len(graded), 2)
    return {"cases": len(rows), "metrics": metrics, "failures": failures, "rows": rows}


# ---- gate and report --------------------------------------------------------------------
def gate(results: dict) -> list[str]:
    thresholds = json.loads((ROOT / "thresholds.json").read_text())
    problems = []
    for suite, result in results.items():
        for metric, minimum in thresholds.get(suite, {}).items():
            value = result["metrics"].get(metric)
            if value is not None and value < minimum:
                problems.append(f"{suite}.{metric} = {value} (minimum {minimum})")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", nargs="+", choices=["retrieval", "guardrails", "agent", "all"], default=["all"])
    p.add_argument("--model", choices=["offline", "configured"], default="offline",
                   help="offline = deterministic stand-in (CI); configured = the model in your RIA_ settings")
    p.add_argument("--judge", action="store_true", help="grade answers with an LLM judge (needs a real model)")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--no-gate", action="store_true")
    args = p.parse_args(argv)
    suites = {"retrieval", "guardrails", "agent"} if "all" in args.suite else set(args.suite)
    if args.judge and args.model == "offline":
        p.error("--judge needs --model configured")

    settings, repo, index, runner, model = build_env(args.model)
    results = {}
    if "retrieval" in suites:
        results["retrieval"] = run_retrieval(index)
    if "guardrails" in suites:
        results["guardrails"] = run_guardrails(settings.max_input_chars)
    if "agent" in suites:
        judge = None
        if args.judge:
            from .judge import Judge
            judge = Judge(model)
        results["agent"] = run_agent(runner, repo, judge)

    problems = gate(results)
    report = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "model": f"{settings.llm_provider}:{settings.llm_model}", "results": results,
              "gate": {"passed": not problems, "problems": problems}}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"Model: {report['model']}")
    for suite, result in results.items():
        print(f"\n{suite} ({result['cases']} cases)")
        for k, v in result["metrics"].items():
            print(f"  {k:28} {v}")
        for f in result["failures"]:
            print(f"  FAIL {json.dumps(f)[:220]}")
    print(f"\nReport: {out}")
    if problems and not args.no_gate:
        print("GATE FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("GATE PASSED" if not problems else "Gate skipped (--no-gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
