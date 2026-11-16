from evalscope.agent_api import EvaluationRequest, ModelTarget, preflight_evaluation


def test_preflight_can_run_without_network_calls():
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='key'),
        benchmarks=['gsm8k'],
        runtime={'minimum_free_disk_bytes': 0},
    )
    result = preflight_evaluation(request, lambda _: 'secret', check_network=False)
    assert result.ok is True
    assert {item.name for item in result.checks} >= {'request', 'credentials', 'disk_space'}
