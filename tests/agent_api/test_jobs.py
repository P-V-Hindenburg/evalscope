import time

from evalscope.agent_api import EvaluationRequest, ModelTarget
from evalscope.agent_api.jobs import EvaluationJobManager


def test_cancel_queued_job():
    manager = EvaluationJobManager(max_workers=1)
    blocker = manager._executor.submit(time.sleep, 0.2)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k'],
    )
    job_id = manager.submit(request, lambda _: 'secret')
    assert manager.cancel(job_id)
    blocker.result()
    assert manager.status(job_id).status == 'cancelled'
