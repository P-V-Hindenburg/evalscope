# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evalscope.service.job_store import (
    CreateJobResult,
    IdempotencyConflictError,
    InvalidJobTransitionError,
    JobAlreadyExistsError,
    JobNotFoundError,
    JobStore,
)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / 'jobs.db')


def create_job(
    store: JobStore, job_id: str = 'job-1', idempotency_key: str = 'request-1'
) -> CreateJobResult:
    return store.create_job(
        job_id=job_id,
        idempotency_key=idempotency_key,
        kind='eval',
        request={'model': 'mock', 'datasets': ['gsm8k']},
    )


def test_create_and_query_job(store: JobStore) -> None:
    result = store.create_job(
        job_id='job-1',
        idempotency_key='request-1',
        kind='eval',
        request={'model': 'mock', 'datasets': ['gsm8k']},
        output_dir='/tmp/job-1',
    )

    assert result.created is True
    assert result.job.status == 'queued'
    assert result.job.request == {'datasets': ['gsm8k'], 'model': 'mock'}
    assert result.job.created_at
    assert result.job.output_dir == '/tmp/job-1'
    assert store.get_job('job-1') == result.job
    assert store.get_by_idempotency_key('request-1') == result.job
    assert store.get_job('missing') is None


def test_same_idempotency_key_and_request_returns_existing_job(store: JobStore) -> None:
    first = create_job(store)
    second = store.create_job(
        job_id='ignored-job-id',
        idempotency_key='request-1',
        kind='eval',
        request={'datasets': ['gsm8k'], 'model': 'mock'},
    )

    assert second.created is False
    assert second.job == first.job


@pytest.mark.parametrize(
    ('kind', 'payload'),
    [
        ('eval', {'model': 'different'}),
        ('perf', {'model': 'mock', 'datasets': ['gsm8k']}),
    ],
)
def test_reused_idempotency_key_for_different_operation_conflicts(
    store: JobStore, kind: str, payload: dict[str, object]
) -> None:
    create_job(store)

    with pytest.raises(IdempotencyConflictError):
        store.create_job('job-2', 'request-1', kind, payload)


def test_duplicate_job_id_is_rejected(store: JobStore) -> None:
    create_job(store)

    with pytest.raises(JobAlreadyExistsError):
        create_job(store, job_id='job-1', idempotency_key='request-2')


def test_status_transitions_set_timestamps(store: JobStore) -> None:
    create_job(store)

    running = store.update_status('job-1', 'running')
    cancelling = store.update_status('job-1', 'cancelling')
    cancelled = store.update_status('job-1', 'cancelled')

    assert running.started_at is not None
    assert running.finished_at is None
    assert cancelling.started_at == running.started_at
    assert cancelled.started_at == running.started_at
    assert cancelled.finished_at is not None


def test_invalid_status_transition_is_rejected(store: JobStore) -> None:
    create_job(store)

    with pytest.raises(InvalidJobTransitionError):
        store.update_status('job-1', 'completed')

    assert store.get_job('job-1').status == 'queued'


def test_missing_job_update_is_rejected(store: JobStore) -> None:
    with pytest.raises(JobNotFoundError):
        store.update_status('missing', 'running')


def test_completion_persists_json_result_and_output_directory(store: JobStore) -> None:
    create_job(store)
    store.update_status('job-1', 'running')

    completed = store.complete_job('job-1', {'score': 0.75, 'metrics': ['accuracy']}, '/tmp/output')

    assert completed.status == 'completed'
    assert completed.result == {'score': 0.75, 'metrics': ['accuracy']}
    assert completed.output_dir == '/tmp/output'
    assert completed.error is None
    assert completed.finished_at is not None


@pytest.mark.parametrize('initial_status', ['queued', 'running'])
def test_failure_persists_error(store: JobStore, initial_status: str) -> None:
    create_job(store)
    if initial_status == 'running':
        store.update_status('job-1', 'running')

    failed = store.fail_job('job-1', 'worker exited')

    assert failed.status == 'error'
    assert failed.error == 'worker exited'
    assert failed.finished_at is not None


def test_reopening_database_preserves_records(tmp_path: Path) -> None:
    db_path = tmp_path / 'jobs.db'
    first_store = JobStore(db_path)
    create_job(first_store)
    first_store.update_status('job-1', 'running')
    first_store.complete_job('job-1', {'score': 1})

    reopened_store = JobStore(db_path)

    assert reopened_store.get_job('job-1').result == {'score': 1}
    assert reopened_store.get_job('job-1').status == 'completed'


def test_recovery_marks_only_running_jobs_interrupted(store: JobStore) -> None:
    create_job(store, 'running-job', 'running-request')
    create_job(store, 'queued-job', 'queued-request')
    store.update_status('running-job', 'running')

    count = store.mark_running_jobs_interrupted()

    interrupted = store.get_job('running-job')
    assert count == 1
    assert interrupted.status == 'interrupted'
    assert interrupted.finished_at is not None
    assert store.get_job('queued-job').status == 'queued'
    assert store.mark_running_jobs_interrupted() == 0


def test_concurrent_idempotent_creation_creates_exactly_one_job(store: JobStore) -> None:
    def submit(index: int) -> CreateJobResult:
        return store.create_job(
            job_id=f'job-{index}',
            idempotency_key='shared-request',
            kind='eval',
            request={'model': 'mock'},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(submit, range(16)))

    assert sum(result.created for result in results) == 1
    assert len({result.job.job_id for result in results}) == 1


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions only')
def test_database_file_is_owner_only(tmp_path: Path) -> None:
    db_path = tmp_path / 'jobs.db'

    JobStore(db_path)

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
