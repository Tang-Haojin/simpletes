import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletes.engine.core import (
    _counts_toward_valid_budget,
    _retryable_evaluation_delay,
    _retryable_evaluation_diagnostic,
)
from simpletes.engine.scheduler import SchedulerMixin
from simpletes.node import Node


def test_only_explicit_generated_candidates_count_as_valid():
    initial = Node(id="initial", code="x", metrics={"valid_candidate": 1})
    retryable = Node(
        id="retryable",
        code="x",
        gen_id=1,
        metrics={"valid_candidate": 0, "retryable": True},
    )
    candidate = Node(
        id="candidate",
        code="x",
        gen_id=2,
        metrics={"valid_candidate": 1},
    )

    assert not _counts_toward_valid_budget(initial)
    assert not _counts_toward_valid_budget(retryable)
    assert _counts_toward_valid_budget(candidate)


def test_scheduler_stops_immediately_at_valid_candidate_limit():
    scheduler = SchedulerMixin()
    scheduler.config = SimpleNamespace(max_valid_evaluations=8)
    scheduler.valid_evaluations = 8
    scheduler._stop_event = asyncio.Event()

    assert not asyncio.run(scheduler._schedule_generation())
    assert asyncio.run(scheduler._is_run_complete())


def test_retryable_infrastructure_metrics_get_a_bounded_delay():
    assert _retryable_evaluation_delay({"infrastructure_retry": 1}) == 30.0
    assert _retryable_evaluation_delay({"retryable_infra": True, "retry_after_s": 5}) == 5.0
    assert _retryable_evaluation_delay({"retryable": True, "retry_after_s": 0}) == 1.0
    assert _retryable_evaluation_delay({"retryable": True, "retry_after_s": 9999}) == 300.0
    assert _retryable_evaluation_delay({"valid_candidate": 1}) is None


def test_retryable_diagnostic_requires_explicit_scrubbed_field_and_is_bounded():
    assert _retryable_evaluation_diagnostic({"error": "not explicitly safe"}) is None
    assert _retryable_evaluation_diagnostic({"retry_diagnostic": "  quiet\n CCD  "}) == (
        "quiet CCD"
    )
    diagnostic = _retryable_evaluation_diagnostic(
        {"retry_diagnostic": "x" * 2_000}
    )
    assert diagnostic is not None
    assert len(diagnostic) == 512
    assert diagnostic.endswith("...")
