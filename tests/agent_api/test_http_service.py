from datetime import datetime, timezone

from fastapi.testclient import TestClient

from evalscope.agent_api.jobs import EvaluationJob, JobProgress
from evalscope.agent_api.events import EventPage, RunEvent
from evalscope.agent_api.service.app import create_app
from evalscope.agent_api.service.credentials import CredentialReferenceError
from evalscope.agent_api.service.settings import ServiceSettings


class FakeJobManager:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.job = EvaluationJob(
            job_id='eval_test',
            status='queued',
            created_at=now,
            updated_at=now,
            progress=JobProgress(),
        )

    def submit(self, request, resolver):
        assert request.benchmarks == ['gsm8k']
        assert resolver('deepseek') == 'secret'
        return self.job.job_id

    def status(self, job_id):
        if job_id != self.job.job_id:
            raise KeyError(job_id)
        return self.job

    def cancel(self, job_id, benchmarks=None):
        if job_id != self.job.job_id:
            raise KeyError(job_id)
        return True

    def events(self, job_id, after_cursor=0, limit=100):
        if job_id != self.job.job_id:
            raise KeyError(job_id)
        return EventPage(
            job_id=job_id,
            events=[RunEvent(
                sequence=1,
                timestamp=datetime.now(timezone.utc),
                type='job.queued',
                message='queued',
            )] if after_cursor == 0 else [],
            next_cursor=1 if after_cursor == 0 else after_cursor,
            has_more=False,
        )


def _client(credential_resolver=None):
    settings = ServiceSettings(
        service_token='service-secret',
        allowed_endpoint_hosts=frozenset({'example.test'}),
    )
    app = create_app(
        settings=settings,
        credential_resolver=credential_resolver or (lambda _: 'secret'),
        job_manager=FakeJobManager(),
    )
    return TestClient(app)


def _headers():
    return {'Authorization': 'Bearer service-secret'}


def _payload():
    return {
        'target': {
            'model_id': 'test-model',
            'api_url': 'https://example.test/v1',
            'credential_ref': 'deepseek',
            'modality': 'text',
        },
        'benchmarks': ['gsm8k'],
        'limit_per_subset': 3,
    }


def test_health_does_not_require_authentication():
    response = _client().get('/health')

    assert response.status_code == 200
    assert response.json()['service'] == 'evalscope-agent-api'


def test_agent_routes_require_service_token():
    response = _client().get('/v1/benchmarks')

    assert response.status_code == 401
    assert response.json()['error']['code'] == 'AUTHENTICATION_REQUIRED'

    response = _client().get('/v1/benchmarks', headers={'Authorization': 'Bearer wrong'})
    assert response.status_code == 401
    assert response.json()['error']['code'] == 'INVALID_SERVICE_TOKEN'


def test_benchmark_detail_and_unknown_benchmark():
    client = _client()

    existing = client.get('/v1/benchmarks/gsm8k', headers=_headers())
    missing = client.get('/v1/benchmarks/not-a-benchmark', headers=_headers())

    assert existing.status_code == 200
    assert existing.json()['benchmark_id'] == 'gsm8k'
    assert missing.status_code == 404
    assert missing.json()['error']['code'] == 'BENCHMARK_NOT_FOUND'


def test_submit_returns_job_id_without_accepting_server_paths():
    response = _client().post('/v1/evaluations', headers=_headers(), json=_payload())

    assert response.status_code == 202
    assert response.json() == {'job_id': 'eval_test', 'status': 'queued'}

    unsafe = _payload() | {'work_dir': '/tmp/client-controlled'}
    response = _client().post('/v1/evaluations', headers=_headers(), json=unsafe)
    assert response.status_code == 422

    unsafe = _payload() | {'dataset_args': {'gsm8k': {'local_path': '/tmp/client-controlled'}}}
    response = _client().post('/v1/evaluations', headers=_headers(), json=unsafe)
    assert response.status_code == 422


def test_http_accepts_safe_per_benchmark_limit_and_subset_overrides():
    payload = _payload()
    payload['target']['modality'] = 'multimodal'
    payload['benchmarks'] = ['pope']
    payload['benchmark_options'] = {
        'pope': {
            'limit_per_subset': 2,
            'subsets': ['popular', 'random'],
        }
    }

    class CapturingJobManager(FakeJobManager):
        def submit(self, request, resolver):
            assert request.benchmark_options['pope'].limit_per_subset == 2
            assert request.benchmark_options['pope'].subsets == ['popular', 'random']
            return self.job.job_id

    settings = ServiceSettings(
        service_token='service-secret',
        allowed_endpoint_hosts=frozenset({'example.test'}),
    )
    client = TestClient(create_app(
        settings=settings,
        credential_resolver=lambda _: 'secret',
        job_manager=CapturingJobManager(),
    ))

    response = client.post('/v1/evaluations', headers=_headers(), json=payload)

    assert response.status_code == 202
    assert response.json()['job_id'] == 'eval_test'


def test_status_and_pending_result_are_separate():
    client = _client()

    status_response = client.get('/v1/evaluations/eval_test', headers=_headers())
    result_response = client.get('/v1/evaluations/eval_test/result', headers=_headers())

    assert status_response.status_code == 200
    assert status_response.json()['result_available'] is False
    assert result_response.status_code == 202
    assert result_response.json()['status'] == 'queued'


def test_unknown_job_and_disallowed_endpoint_are_structured_errors():
    client = _client()
    missing = client.get('/v1/evaluations/missing', headers=_headers())

    payload = _payload()
    payload['target']['api_url'] = 'https://not-allowed.test/v1'
    disallowed = client.post('/v1/evaluations', headers=_headers(), json=payload)

    assert missing.status_code == 404
    assert missing.json()['error']['code'] == 'JOB_NOT_FOUND'
    assert disallowed.status_code == 400
    assert disallowed.json()['error']['code'] == 'ENDPOINT_NOT_ALLOWED'


def test_cancel_returns_acknowledgement():
    response = _client().delete('/v1/evaluations/eval_test', headers=_headers())

    assert response.status_code == 202
    assert response.json()['accepted'] is True


def test_partial_cancel_and_incremental_events():
    client = _client()

    cancelled = client.post(
        '/v1/evaluations/eval_test/cancellations',
        headers=_headers(),
        json={'benchmarks': ['gsm8k']},
    )
    events = client.get(
        '/v1/evaluations/eval_test/events?after_cursor=0&limit=10',
        headers=_headers(),
    )
    no_new_events = client.get(
        '/v1/evaluations/eval_test/events?after_cursor=1&limit=10',
        headers=_headers(),
    )

    assert cancelled.status_code == 202
    assert events.status_code == 200
    assert events.json()['events'][0]['sequence'] == 1
    assert events.json()['next_cursor'] == 1
    assert no_new_events.json()['events'] == []
    assert no_new_events.json()['next_cursor'] == 1


def test_partial_cancel_rejects_duplicate_benchmark_ids():
    response = _client().post(
        '/v1/evaluations/eval_test/cancellations',
        headers=_headers(),
        json={'benchmarks': ['gsm8k', 'gsm8k']},
    )

    assert response.status_code == 422


def test_unknown_credential_reference_fails_before_queueing():
    def unknown_credential(_):
        raise CredentialReferenceError('unknown credential_ref')

    response = _client(unknown_credential).post('/v1/evaluations', headers=_headers(), json=_payload())

    assert response.status_code == 400
    assert response.json()['error']['code'] == 'CREDENTIAL_REFERENCE_UNKNOWN'
