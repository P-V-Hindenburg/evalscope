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
from evalscope.agent_api.results import sample_summary
from evalscope.agent_api.contracts import BenchmarkResult, MetricResult


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
