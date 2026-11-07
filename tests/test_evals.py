from evals.metrics import percentile, recall_at_k, reciprocal_rank
from evals.run import gate, main


def test_metric_functions():
    assert recall_at_k(["a", "b", "c"], ["c", "z"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c", "z"], 3) == 0.5
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert percentile([10.0], 95) == 10.0


def test_gate_flags_metrics_below_threshold():
    problems = gate({"retrieval": {"metrics": {"recall_at_5": 0.5, "mrr": 0.9}}})
    assert problems == ["retrieval.recall_at_5 = 0.5 (minimum 0.85)"]


def test_eval_suites_pass_the_gate(tmp_path):
    assert main(["--suite", "all", "--report-dir", str(tmp_path)]) == 0
    assert list(tmp_path.glob("eval-*.json"))
