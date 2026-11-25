# Copyright (c) Alibaba, Inc. and its affiliates.

from pathlib import Path
from typing import Any, Generator
from unittest import mock

import pytest
from flask.testing import FlaskClient

from evalscope.service.app import create_app

from .test_agent_manifests import write_manifests


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[FlaskClient, None, None]:
    catalog = write_manifests(tmp_path)
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')
    app = create_app(outputs=str(tmp_path / 'outputs'))
    app.config.update(
        TESTING=True,
        AGENT_MODELS_MANIFEST=str(catalog.models_path),
        AGENT_DATASETS_MANIFEST=str(catalog.datasets_path),
        AGENT_SCENARIOS_MANIFEST=str(catalog.scenarios_path),
        AGENT_JOB_DB=str(tmp_path / 'state' / 'jobs.db'),
        AGENT_OUTPUT_ROOT=str(tmp_path / 'outputs'),
        AGENT_SOURCE_COMMIT='abc123',
    )
    with mock.patch('evalscope.service.job_runner.EvaluationJobRunner.submit'), mock.patch(
        'evalscope.service.job_runner.EvaluationJobRunner.submit_load_test'
    ):
        yield app.test_client()
    service = app.extensions.get('evalscope_agent_service')
    if service is not None:
        service.runner.shutdown()


def evaluation_payload(**updates: Any) -> dict[str, Any]:
    payload = {
        'idempotency_key': 'request-1',
        'task_type': 'evaluation',
        'model_ref': 'demo-model',
        'dataset_ref': 'demo-5',
        'limit': 3,
        'generation_config': {'max_tokens': 64},
        'tags': {'suite': 'smoke'},
    }
    payload.update(updates)
    return payload


def load_test_payload(**updates: Any) -> dict[str, Any]:
    payload = {
        'idempotency_key': 'load-request-1',
        'task_type': 'load_test',
        'model_ref': 'demo-model',
        'scenario_ref': 'smoke',
        'overrides': {'number': 8, 'parallel': 4},
        'tags': {'suite': 'perf-smoke'},
    }
    payload.update(updates)
    return payload


def test_submit_query_catalogs_and_cancel(client: FlaskClient) -> None:
    created = client.post('/v1/evaluations', json=evaluation_payload())

    assert created.status_code == 202
    body = created.get_json()
    assert body['created'] is True
    assert body['status'] == 'queued'
    assert 'secret-value' not in str(body)
    job_id = body['job_id']

    duplicate = client.post('/v1/evaluations', json=evaluation_payload())
    assert duplicate.status_code == 200
    assert duplicate.get_json()['job_id'] == job_id
    assert duplicate.get_json()['created'] is False

    status = client.get(f'/v1/jobs/{job_id}')
    assert status.status_code == 200
    assert status.get_json()['request']['model_ref'] == 'demo-model'

    result = client.get(f'/v1/jobs/{job_id}/results')
    assert result.status_code == 409

    logs = client.get(f'/v1/jobs/{job_id}/logs')
    assert logs.status_code == 200
    assert logs.get_json()['total_lines'] == 0

    models = client.get('/v1/models').get_json()['models']
    assert models[0]['ref'] == 'demo-model'
    assert models[0]['load_test_allowed'] is True
    assert 'api_key_env' not in models[0]
    assert client.get('/v1/datasets').get_json()['datasets'][0]['ref'] == 'demo-5'
    assert client.get('/v1/scenarios').get_json()['scenarios'][0]['ref'] == 'smoke'
    assert client.get('/v1/capabilities').get_json()['load_tests'] is True
    assert client.get('/healthz').status_code == 200
    assert client.get('/readyz').status_code == 200

    cancelled = client.post(f'/v1/jobs/{job_id}/cancel')
    assert cancelled.status_code == 200
    assert cancelled.get_json()['status'] == 'cancelled'


def test_idempotency_conflict_and_manifest_limits_are_rejected(client: FlaskClient) -> None:
    assert client.post('/v1/evaluations', json=evaluation_payload()).status_code == 202

    conflict = client.post('/v1/evaluations', json=evaluation_payload(limit=2))
    assert conflict.status_code == 409

    excessive = client.post('/v1/evaluations', json=evaluation_payload(idempotency_key='request-2', limit=6))
    assert excessive.status_code == 400

    unlimited = client.post('/v1/evaluations', json=evaluation_payload(idempotency_key='request-4', limit=0))
    assert unlimited.status_code == 422

    unknown = client.post(
        '/v1/evaluations',
        json=evaluation_payload(idempotency_key='request-3', model_ref='unregistered'),
    )
    assert unknown.status_code == 400


def test_submit_load_test_and_reject_excessive_override(client: FlaskClient) -> None:
    created = client.post('/v1/load-tests', json=load_test_payload())

    assert created.status_code == 202
    body = created.get_json()
    assert body['kind'] == 'load_test'
    assert body['status'] == 'queued'
    assert 'secret-value' not in str(body)

    duplicate = client.post('/v1/load-tests', json=load_test_payload())
    assert duplicate.status_code == 200
    assert duplicate.get_json()['job_id'] == body['job_id']

    excessive = client.post(
        '/v1/load-tests',
        json=load_test_payload(idempotency_key='load-request-2', overrides={'number': 1001}),
    )
    assert excessive.status_code == 422


def test_load_test_requires_explicit_model_approval(client: FlaskClient) -> None:
    assert client.get('/v1/models').status_code == 200
    service = client.application.extensions['evalscope_agent_service']
    models_path = Path(service.catalog.models_path)
    models_path.write_text(
        models_path.read_text(encoding='utf-8').replace('allow_load_test: true', 'allow_load_test: false'),
        encoding='utf-8',
    )

    response = client.post('/v1/load-tests', json=load_test_payload(idempotency_key='blocked-request'))

    assert response.status_code == 403
    assert 'not approved' in response.get_json()['error']
