import time
import json
from datetime import datetime, timezone

from evalscope.agent_api import EvaluationRequest, ModelTarget
from evalscope.agent_api.jobs import EvaluationJob, EvaluationJobManager


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
