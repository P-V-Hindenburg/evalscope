import time
import json
import threading
from datetime import datetime, timezone

from evalscope.agent_api import EvaluationRequest, ModelTarget
from evalscope.agent_api.jobs import EvaluationJob, EvaluationJobManager, JobProgress


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
    deadline = time.monotonic() + 2
    while manager.status(job_id).status != 'cancelled' and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.status(job_id).status == 'cancelled'
    result = manager.result(job_id)
    assert result is not None
    assert result.status == 'cancelled'
    assert [item.status for item in result.benchmarks] == ['cancelled']


def test_partial_cancel_marks_only_selected_benchmark(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    blocker = manager._executor.submit(time.sleep, 0.2)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k', 'hellaswag'],
        work_dir=str(tmp_path),
    )
    job_id = manager.submit(request, lambda _: 'secret')

    assert manager.cancel(job_id, benchmarks=['hellaswag'])
    job = manager.status(job_id, refresh=False)
    states = {state.benchmark_id: state.status for state in job.benchmarks}
    assert states == {'gsm8k': 'queued', 'hellaswag': 'cancellation_requested'}

    events = manager.events(job_id, after_cursor=0, limit=100)
    assert any(event.type == 'benchmark.cancellation_requested' for event in events.events)
    manager.cancel(job_id)
    blocker.result()


def test_job_events_are_incremental_and_read_only(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    blocker = manager._executor.submit(time.sleep, 0.2)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k'],
        work_dir=str(tmp_path),
    )
    job_id = manager.submit(request, lambda _: 'secret')

    first = manager.events(job_id, after_cursor=0, limit=1)
    repeated = manager.events(job_id, after_cursor=0, limit=1)
    assert first == repeated
    assert manager.status(job_id, refresh=False).status == 'queued'

    manager.cancel(job_id)
    blocker.result()


def test_progress_uses_native_progress_tree(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k'],
        work_dir=str(tmp_path),
    )
    job_id = 'eval_progress_test'
    now = datetime.now(timezone.utc)
    manager._jobs[job_id] = EvaluationJob(
        job_id=job_id, status='loading_dataset', created_at=now, updated_at=now,
    )
    manager._requests[job_id] = request
    progress = {
        'status': 'running',
        'processed_count': 7,
        'total_count': 20,
        'stage': {
            'name': 'Evaluating', 'label': 'gsm8k', 'status': 'running',
            'children': [{'name': 'Predicting', 'status': 'running'}],
        },
    }
    (tmp_path / 'progress.json').write_text(json.dumps(progress), encoding='utf-8')

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)

    assert job.status == 'running_model'
    assert job.progress.current == 7
    assert job.progress.total == 20
    assert job.progress.percent == 35.0
    assert job.progress.current_benchmark == 'gsm8k'


def test_progress_refresh_preserves_job_cancellation_acknowledgement(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k'], work_dir=str(tmp_path),
    )
    job_id = 'eval_cancelling_progress'
    now = datetime.now(timezone.utc)
    manager._jobs[job_id] = EvaluationJob(
        job_id=job_id, status='cancellation_requested', created_at=now, updated_at=now,
    )
    manager._requests[job_id] = request
    manager._cancel_events[job_id] = threading.Event()
    manager._cancel_events[job_id].set()
    (tmp_path / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 3, 'total_count': 10,
        'stage': {'name': 'Predicting', 'status': 'running'},
    }), encoding='utf-8')

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)

    assert job.status == 'cancellation_requested'
    assert job.progress.stage == 'cancellation_requested'
    assert job.progress.current == 3
    assert job.progress.percent == 30.0


def test_multi_group_progress_keeps_planned_total_stable(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['pope', 'gsm8k', 'hellaswag'],
        work_dir=str(tmp_path),
    )
    job_id = 'eval_multi_group_progress'
    now = datetime.now(timezone.utc)
    manager._jobs[job_id] = EvaluationJob(
        job_id=job_id,
        status='running_model',
        created_at=now,
        updated_at=now,
        progress=JobProgress(
            total=80,
            total_is_estimate=True,
            total_groups=2,
            current_group=1,
            current_group_benchmarks=['pope'],
        ),
    )
    manager._requests[job_id] = request
    first = tmp_path / 'first'
    first.mkdir()
    (first / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 15, 'total_count': 30,
    }), encoding='utf-8')

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)
    assert job.progress.current == 15
    assert job.progress.total == 80
    assert job.progress.percent == 18.8
    assert job.progress.total_is_estimate
    assert job.progress.total_groups == 2
    assert job.progress.current_group == 1

    second = tmp_path / 'second'
    second.mkdir()
    (first / 'progress.json').write_text(json.dumps({
        'status': 'finished', 'processed_count': 30, 'total_count': 30,
    }), encoding='utf-8')
    (second / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 10, 'total_count': 50,
    }), encoding='utf-8')
    manager._apply_progress_event(job_id, {
        'event': 'group_completed', 'completed_groups': 1,
    })
    manager._apply_progress_event(job_id, {
        'event': 'group_started', 'group_index': 2, 'benchmarks': ['gsm8k', 'hellaswag'],
    })

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)
    assert job.progress.current == 40
    assert job.progress.total == 80
    assert job.progress.percent == 50.0
    assert job.progress.completed_groups == 1
    assert job.progress.current_group == 2
    assert job.progress.current_group_benchmarks == ['gsm8k', 'hellaswag']


def test_unknown_multi_group_total_stays_null_until_all_groups_exist(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k', 'hellaswag'],
        work_dir=str(tmp_path),
    )
    job_id = 'eval_unknown_total'
    now = datetime.now(timezone.utc)
    manager._jobs[job_id] = EvaluationJob(
        job_id=job_id,
        status='running_model',
        created_at=now,
        updated_at=now,
        progress=JobProgress(total_groups=2),
    )
    manager._requests[job_id] = request
    first = tmp_path / 'first'
    first.mkdir()
    (first / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 4, 'total_count': 10,
    }), encoding='utf-8')

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)
    assert job.progress.current == 4
    assert job.progress.total is None
    assert job.progress.percent == 0.0

    second = tmp_path / 'second'
    second.mkdir()
    (second / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 5, 'total_count': 10,
    }), encoding='utf-8')
    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)
    assert job.progress.current == 9
    assert job.progress.total == 20
    assert job.progress.percent == 45.0


def test_progress_falls_back_to_latest_log_event(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k', 'ifeval'],
        work_dir=str(tmp_path),
    )
    job_id = 'eval_old_progress_test'
    now = datetime.now(timezone.utc)
    manager._jobs[job_id] = EvaluationJob(
        job_id=job_id, status='loading_dataset', created_at=now, updated_at=now,
    )
    manager._requests[job_id] = request
    (tmp_path / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 25, 'total_count': 40,
    }), encoding='utf-8')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'eval_log.log').write_text(
        'Generating report...\n'
        'Benchmark gsm8k evaluation finished.\n'
        'Start loading benchmark dataset: ifeval\n'
        'Evaluating[ifeval] 25%| 5/20\n',
        encoding='utf-8',
    )

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)

    assert job.status == 'running_model'
    assert job.progress.current_benchmark == 'ifeval'
    assert job.progress.current == 25
    assert job.progress.total == 40


def test_old_progress_detects_generation_after_loading_event(tmp_path):
    manager = EvaluationJobManager(max_workers=1)
    request = EvaluationRequest(
        target=ModelTarget(model_id='m', api_url='https://example.test/v1', credential_ref='k'),
        benchmarks=['gsm8k'], work_dir=str(tmp_path),
    )
    job_id = 'eval_generating_test'
    now = datetime.now(timezone.utc)
    manager._jobs[job_id] = EvaluationJob(
        job_id=job_id, status='loading_dataset', created_at=now, updated_at=now,
    )
    manager._requests[job_id] = request
    (tmp_path / 'progress.json').write_text(json.dumps({
        'status': 'running', 'processed_count': 12, 'total_count': 20,
    }), encoding='utf-8')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'eval_log.log').write_text(
        'Start loading benchmark dataset: gsm8k\n', encoding='utf-8',
    )

    manager._refresh_progress(job_id)
    job = manager.status(job_id, refresh=False)

    assert job.status == 'running_model'
    assert job.progress.current_benchmark == 'gsm8k'
