# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent-facing asynchronous evaluation and job endpoints."""

import json
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request
from pydantic import ValidationError

from evalscope.service.agent_manifests import (
    AgentManifestCatalog,
    LoadTestNotAllowedError,
    ManifestError,
    MissingSecretError,
)
from evalscope.service.agent_service import AgentEvaluationService
from evalscope.service.api_models import LogResponse
from evalscope.service.api_models.jobs import (
    CapabilitiesResponse,
    DatasetCatalogResponse,
    EvaluationCreateRequest,
    JobResponse,
    JobResultResponse,
    LoadTestCreateRequest,
    ModelCatalogResponse,
    ScenariosResponse,
)
from evalscope.service.job_store import IdempotencyConflictError, JobRecord, JobStore

from ..responses import json_response

bp_jobs = Blueprint('agent_jobs', __name__)

_EXTENSION_KEY = 'evalscope_agent_service'
_SERVICE_LOCK = threading.Lock()


def _get_service() -> AgentEvaluationService:
    service = current_app.extensions.get(_EXTENSION_KEY)
    if service is not None:
        return service
    with _SERVICE_LOCK:
        service = current_app.extensions.get(_EXTENSION_KEY)
        if service is not None:
            return service
        catalog = AgentManifestCatalog(
            current_app.config['AGENT_MODELS_MANIFEST'],
            current_app.config['AGENT_DATASETS_MANIFEST'],
            current_app.config['AGENT_SCENARIOS_MANIFEST'],
        )
        service = AgentEvaluationService(
            catalog=catalog,
            store=JobStore(current_app.config['AGENT_JOB_DB']),
            output_root=current_app.config['AGENT_OUTPUT_ROOT'],
            max_workers=current_app.config['AGENT_MAX_WORKERS'],
            source_commit=current_app.config.get('AGENT_SOURCE_COMMIT'),
        )
        current_app.extensions[_EXTENSION_KEY] = service
        return service


def _load_progress(job: JobRecord) -> dict[str, Any] | None:
    if not job.output_dir:
        return None
    progress_path = (
        Path(job.output_dir) / 'perf' / 'progress.json'
        if job.kind == 'load_test'
        else Path(job.output_dir) / 'progress.json'
    )
    try:
        with progress_path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and 'percent' in payload else None


def _job_payload(job: JobRecord, created: bool | None = None) -> dict[str, Any]:
    payload = {
        'job_id': job.job_id,
        'kind': job.kind,
        'status': job.status,
        'request': job.request,
        'output_dir': job.output_dir,
        'error': job.error,
        'created_at': job.created_at,
        'started_at': job.started_at,
        'finished_at': job.finished_at,
        'progress': _load_progress(job),
    }
    if created is not None:
        payload['created'] = created
    return payload


def _read_log(path: Path, start_line: int | None, page: int) -> dict[str, Any]:
    if page < 1 or (start_line is not None and start_line < 0):
        raise ValueError('page must be positive and start_line cannot be negative')
    if not path.is_file():
        return {'text': '', 'head_line': 0, 'tail_line': 0, 'total_lines': 0}
    lines: deque[str] | list[str] = deque(maxlen=page) if start_line is None else []
    total_lines = 0
    with path.open('r', encoding='utf-8', errors='replace') as file:
        for line in file:
            total_lines += 1
            if start_line is None:
                lines.append(line)
            elif total_lines > start_line and len(lines) < page:
                lines.append(line)
    head_line = max(0, total_lines - page) if start_line is None else min(start_line, total_lines)
    return {
        'text': ''.join(lines),
        'head_line': head_line,
        'tail_line': head_line + len(lines),
        'total_lines': total_lines,
    }


@bp_jobs.route('/v1/evaluations', methods=['POST'])
def create_evaluation() -> Response | tuple[Response, int]:
    """Validate and enqueue an allowlisted asynchronous evaluation."""
    try:
        evaluation_request = EvaluationCreateRequest.model_validate(request.get_json(silent=True))
        result = _get_service().submit_evaluation(evaluation_request)
        return json_response(
            JobResponse, _job_payload(result.job, result.created), status=202 if result.created else 200
        )
    except ValidationError as error:
        return jsonify(
            {
                'error': 'Invalid evaluation request',
                'details': error.errors(include_url=False, include_context=False),
            }
        ), 422
    except IdempotencyConflictError as error:
        return jsonify({'error': str(error)}), 409
    except MissingSecretError as error:
        return jsonify({'error': str(error)}), 503
    except ManifestError as error:
        return jsonify({'error': str(error)}), 400


@bp_jobs.route('/v1/load-tests', methods=['POST'])
def create_load_test() -> Response | tuple[Response, int]:
    """Validate and enqueue an allowlisted asynchronous load test."""
    try:
        load_test_request = LoadTestCreateRequest.model_validate(request.get_json(silent=True))
        result = _get_service().submit_load_test(load_test_request)
        return json_response(
            JobResponse, _job_payload(result.job, result.created), status=202 if result.created else 200
        )
    except ValidationError as error:
        return jsonify(
            {
                'error': 'Invalid load-test request',
                'details': error.errors(include_url=False, include_context=False),
            }
        ), 422
    except IdempotencyConflictError as error:
        return jsonify({'error': str(error)}), 409
    except LoadTestNotAllowedError as error:
        return jsonify({'error': str(error)}), 403
    except MissingSecretError as error:
        return jsonify({'error': str(error)}), 503
    except ManifestError as error:
        return jsonify({'error': str(error)}), 400


@bp_jobs.route('/v1/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str) -> Response | tuple[Response, int]:
    """Return persisted job status and progress."""
    job = _get_service().store.get_job(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404
    return json_response(JobResponse, _job_payload(job))


@bp_jobs.route('/v1/jobs/<job_id>/results', methods=['GET'])
def get_job_result(job_id: str) -> Response | tuple[Response, int]:
    """Return a completed job's persisted structured result."""
    job = _get_service().store.get_job(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404
    if job.status != 'completed':
        return jsonify({'error': 'Job has not completed', 'status': job.status}), 409
    return json_response(JobResultResponse, {'job_id': job.job_id, 'status': job.status, 'result': job.result})


@bp_jobs.route('/v1/jobs/<job_id>/logs', methods=['GET'])
def get_job_logs(job_id: str) -> Response | tuple[Response, int]:
    """Return a page of the evaluation log."""
    job = _get_service().store.get_job(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404
    start_line = request.args.get('start_line', type=int)
    page = request.args.get('page', default=500, type=int)
    try:
        if job.output_dir is None:
            return json_response(LogResponse, {'text': '', 'head_line': 0, 'tail_line': 0, 'total_lines': 0})
        log_path = (
            Path(job.output_dir) / 'perf' / 'benchmark.log'
            if job.kind == 'load_test'
            else Path(job.output_dir) / 'logs' / 'eval_log.log'
        )
        return json_response(LogResponse, _read_log(log_path, start_line, page))
    except ValueError as error:
        return jsonify({'error': str(error)}), 400


@bp_jobs.route('/v1/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id: str) -> Response | tuple[Response, int]:
    """Request cancellation of a queued or running job."""
    job = _get_service().store.get_job(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404
    cancelled = _get_service().runner.cancel(job_id)
    return json_response(JobResponse, _job_payload(cancelled))


@bp_jobs.route('/v1/models', methods=['GET'])
def list_models() -> Response | tuple[Response, int]:
    """List registered models without exposing secret environment names or values."""
    try:
        models = [
            {
                'ref': reference,
                'model': entry.model,
                'model_id': entry.model_id,
                'eval_type': entry.eval_type,
                'api_url': entry.api_url,
                'load_test_allowed': entry.allow_load_test,
            }
            for reference, entry in _get_service().catalog.models().items()
        ]
        return json_response(ModelCatalogResponse, {'models': models})
    except ManifestError as error:
        return jsonify({'error': str(error)}), 503


@bp_jobs.route('/v1/datasets', methods=['GET'])
def list_datasets() -> Response | tuple[Response, int]:
    """List registered dataset configurations."""
    try:
        datasets = [
            {
                'ref': reference,
                'datasets': entry.datasets,
                'limit': entry.limit,
                'eval_batch_size': entry.eval_batch_size,
                'repeats': entry.repeats,
            }
            for reference, entry in _get_service().catalog.datasets().items()
        ]
        return json_response(DatasetCatalogResponse, {'datasets': datasets})
    except ManifestError as error:
        return jsonify({'error': str(error)}), 503


@bp_jobs.route('/v1/scenarios', methods=['GET'])
def list_scenarios() -> Response | tuple[Response, int]:
    """List registered bounded load-test scenarios."""
    try:
        scenarios = [
            {'ref': reference, **entry.model_dump(mode='json', exclude={'dataset_path', 'prompt', 'tokenizer_path'})}
            for reference, entry in _get_service().catalog.scenarios().items()
        ]
        return json_response(ScenariosResponse, {'scenarios': scenarios})
    except ManifestError as error:
        return jsonify({'error': str(error)}), 503


@bp_jobs.route('/v1/capabilities', methods=['GET'])
def get_capabilities() -> Response | tuple[Response, int]:
    """Describe the currently enabled agent-service features."""
    try:
        return json_response(
            CapabilitiesResponse,
            {
                'evaluations': True,
                'load_tests': bool(_get_service().catalog.scenarios()),
                'idempotency': True,
                'cancellation': True,
                'persistence': 'sqlite',
                'max_workers': current_app.config['AGENT_MAX_WORKERS'],
            },
        )
    except ManifestError as error:
        return jsonify({'error': str(error)}), 503


@bp_jobs.route('/healthz', methods=['GET'])
def healthz() -> Response:
    """Return process liveness without touching external dependencies."""
    return jsonify({'status': 'ok', 'service': 'evalscope-agent'})


@bp_jobs.route('/readyz', methods=['GET'])
def readyz() -> Response | tuple[Response, int]:
    """Validate manifests, persistent storage and required secret references."""
    try:
        service = _get_service()
        models = service.catalog.models()
        service.catalog.datasets()
        service.catalog.scenarios()
        missing = sorted(entry.api_key_env for entry in models.values() if entry.api_key_env not in os.environ)
        if missing:
            return jsonify({'status': 'not_ready', 'missing_secret_env': missing}), 503
        return jsonify({'status': 'ready'})
    except (ManifestError, OSError) as error:
        return jsonify({'status': 'not_ready', 'error': str(error)}), 503
