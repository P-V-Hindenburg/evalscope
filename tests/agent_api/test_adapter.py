import pytest

from evalscope.agent_api import (
    EvaluationRequest,
    EvaluationValidationError,
    JudgeProfile,
    ModelTarget,
    build_task_configs,
    resolve_credentials,
    validate_evaluation,
)
from evalscope.agent_api.adapter import _has_all_expected_results, _merge_result_snapshots, _stop_process
from evalscope.agent_api.contracts import BenchmarkResult, MetricResult
from evalscope.agent_api.results import sample_summary


def _target(modality='multimodal'):
    return ModelTarget(
        model_id='test-model',
        api_url='https://example.test/v1',
        credential_ref='target-key',
        modality=modality,
    )


def test_validation_requires_judge_for_judge_benchmark():
    request = EvaluationRequest(target=_target(), benchmarks=['mt_bench'])

    with pytest.raises(EvaluationValidationError, match='judge profile required'):
        validate_evaluation(request)


def test_validation_rejects_text_target_for_vlm():
    request = EvaluationRequest(target=_target('text'), benchmarks=['pope'])

    with pytest.raises(EvaluationValidationError, match='multimodal target required'):
        validate_evaluation(request)


def test_task_configs_split_mixed_dataset_hubs_and_enable_sandbox():
    request = EvaluationRequest(
        target=_target(),
        benchmarks=['gsm8k', 'humaneval', 'mvbench'],
        judge=JudgeProfile(
            model_id='judge-model',
            api_url='https://judge.example.test/v1',
            credential_ref='judge-key',
        ),
    )
    plan = validate_evaluation(request)
    credentials = resolve_credentials(request, lambda ref: f'secret-for-{ref}')
    configs = build_task_configs(plan, credentials, '/tmp/agent-api-test')

    assert len(configs) == 2
    by_hub = {config.dataset_hub: config for config in configs}
    assert by_hub['modelscope'].datasets == ['gsm8k', 'humaneval']
    assert by_hub['huggingface'].datasets == ['mvbench']
    assert by_hub['modelscope'].sandbox.enabled is True
    assert by_hub['huggingface'].sandbox.enabled is False
    assert by_hub['modelscope'].api_key.get_secret_value() == 'secret-for-target-key'
    assert by_hub['modelscope'].judge.strategy == 'auto'


def test_limit_name_is_explicitly_per_subset():
    request = EvaluationRequest(target=_target(), benchmarks=['pope'], limit_per_subset=10)
    plan = validate_evaluation(request)
    credentials = resolve_credentials(request, lambda _: 'secret')

    config = build_task_configs(plan, credentials, '/tmp/agent-api-test')[0]

    assert config.limit == 10


def test_generation_retry_controls_are_forwarded():
    request = EvaluationRequest(
        target=_target(), benchmarks=['pope'],
        generation={'timeout': 60, 'retries': 1, 'retry_interval': 1},
    )
    plan = validate_evaluation(request)
    credentials = resolve_credentials(request, lambda _: 'secret')

    config = build_task_configs(plan, credentials, '/tmp/agent-api-test')[0]

    assert config.generation_config.timeout == 60
    assert config.generation_config.retries == 1
    assert config.generation_config.retry_interval == 1


def test_sample_summary_preserves_partial_success_counts():
    summary = sample_summary([BenchmarkResult(
        benchmark_id='docvqa', model_id='m',
        metrics=[MetricResult(name='anls', score=0.8, num=9)],
        requested_samples=10, succeeded_samples=9, errored_samples=1, incomplete=True,
    )])
    assert summary.planned == 10
    assert summary.successful == 9
    assert summary.failed == 1
    assert summary.scored == 9
    assert summary.error_counts == {'sample_error': 1}


def test_stop_process_escalates_when_worker_ignores_terminate():
    class StubbornProcess:
        def __init__(self):
            self.alive = True
            self.terminated = 0
            self.killed = 0

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1
            self.alive = False

        def join(self, timeout=None):
            return None

    process = StubbornProcess()

    assert _stop_process(process, cleanup_timeout=0.01) is True
    assert process.terminated == 1
    assert process.killed == 1
    assert process.is_alive() is False


def test_recovery_requires_every_expected_benchmark():
    gsm8k = BenchmarkResult(benchmark_id='gsm8k', model_id='m', metrics=[])
    hellaswag = BenchmarkResult(benchmark_id='hellaswag', model_id='m', metrics=[])

    assert not _has_all_expected_results([gsm8k], ['gsm8k', 'hellaswag'])
    assert _has_all_expected_results([gsm8k, hellaswag], ['gsm8k', 'hellaswag'])


def test_latest_recovery_snapshot_replaces_stale_result():
    stale = BenchmarkResult(
        benchmark_id='gsm8k', model_id='m', metrics=[], succeeded_samples=1,
    )
    final = BenchmarkResult(
        benchmark_id='gsm8k', model_id='m', metrics=[], succeeded_samples=20,
    )
    other = BenchmarkResult(benchmark_id='hellaswag', model_id='m', metrics=[])

    merged = _merge_result_snapshots([stale], [final, other])

    assert {item.benchmark_id for item in merged} == {'gsm8k', 'hellaswag'}
    assert next(item for item in merged if item.benchmark_id == 'gsm8k').succeeded_samples == 20
