from evalscope.agent_api.contracts import BenchmarkResult, MetricResult
from evalscope.agent_api.results import artifact_refs, sample_summary


def test_artifact_refs_preserve_evalscope_native_layout():
    refs = artifact_refs('/tmp/run-1')

    assert refs.predictions == '/tmp/run-1/predictions'
    assert refs.reviews == '/tmp/run-1/reviews'
    assert refs.reports == '/tmp/run-1/reports'


def test_scored_count_uses_primary_metric_not_instruction_count():
    result = BenchmarkResult(
        benchmark_id='ifeval', model_id='m', requested_samples=20,
        succeeded_samples=20, errored_samples=0,
        metrics=[
            MetricResult(name='prompt_level_strict', score=0.9, num=20, is_primary=True),
            MetricResult(name='inst_level_strict', score=0.9, num=30),
        ],
    )

    summary = sample_summary([result])

    assert summary.planned == 20
    assert summary.processed == 20
    assert summary.successful == 20
    assert summary.scored == 20


def test_cancelled_processed_samples_are_not_reported_as_skipped():
    summary = sample_summary([
        BenchmarkResult(
            benchmark_id='gsm8k', model_id='m', status='cancelled',
            requested_samples=10, processed_samples=3, incomplete=True,
        ),
        BenchmarkResult(
            benchmark_id='ifeval', model_id='m', requested_samples=3,
            processed_samples=3, succeeded_samples=3, errored_samples=0,
        ),
    ])

    assert summary.planned == 13
    assert summary.processed == 6
    assert summary.successful == 3
    assert summary.skipped == 7
