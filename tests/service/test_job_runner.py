# Copyright (c) Alibaba, Inc. and its affiliates.

import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.service.agent_manifests import LoadTestScenarioEntry
from evalscope.service.job_runner import EvaluationJobRunner
from evalscope.service.job_store import JobRecord, JobStore
from evalscope.service.utils import TaskStoppedError


def wait_for_terminal(store: JobStore, job_id: str, timeout: float = 5.0) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get_job(job_id)
        if job is not None and job.status in {'completed', 'error', 'cancelled'}:
            return job
        time.sleep(0.01)
    raise AssertionError(f'Job {job_id} did not finish')


def test_runner_resolves_secret_and_persists_structured_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / 'jobs.db')
    output_dir = str(tmp_path / 'outputs' / 'job-1')
    store.create_job('job-1', 'request-1', 'evaluation', {'model_ref': 'demo'}, output_dir=output_dir)
    config = TaskConfig(
        model='demo',
        model_id='demo',
        eval_type='openai_api',
        api_url='https://example.test/v1',
        datasets=['general_fc'],
        limit=1,
        work_dir=output_dir,
        no_timestamp=True,
    )
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')
    runner = EvaluationJobRunner(store, source_commit='abc123')

    with mock.patch('evalscope.service.job_runner.run_in_subprocess', return_value={'general_fc': {'score': 1}}):
        runner.submit('job-1', config, 'DEMO_API_KEY', 'demo-model', 'demo-1', {'suite': 'smoke'})
        completed = wait_for_terminal(store, 'job-1')
    runner.shutdown()

    assert completed.status == 'completed'
    assert completed.result['summary'] == {'general_fc': {'score': 1}}
    assert completed.result['provenance']['source_commit'] == 'abc123'
    assert completed.result['provenance']['tags'] == {'suite': 'smoke'}
    assert 'secret-value' not in str(completed.request)


def test_runner_persists_safe_error_without_exception_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / 'jobs.db')
    output_dir = str(tmp_path / 'outputs' / 'job-1')
    store.create_job('job-1', 'request-1', 'evaluation', {'model_ref': 'demo'}, output_dir=output_dir)
    config = TaskConfig(
        model='demo',
        eval_type='openai_api',
        api_url='https://example.test/v1',
        datasets=['general_fc'],
        work_dir=output_dir,
        no_timestamp=True,
    )
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')
    runner = EvaluationJobRunner(store)

    with mock.patch('evalscope.service.job_runner.run_in_subprocess', side_effect=RuntimeError('secret-value')):
        runner.submit('job-1', config, 'DEMO_API_KEY', 'demo-model', 'demo-1', {})
        failed = wait_for_terminal(store, 'job-1')
    runner.shutdown()

    assert failed.status == 'error'
    assert failed.error == 'Evaluation failed; inspect the job log for details'
    assert 'secret-value' not in failed.error


def test_runner_cancels_a_running_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JobStore(tmp_path / 'jobs.db')
    output_dir = str(tmp_path / 'outputs' / 'job-1')
    store.create_job('job-1', 'request-1', 'evaluation', {'model_ref': 'demo'}, output_dir=output_dir)
    config = TaskConfig(
        model='demo',
        eval_type='openai_api',
        api_url='https://example.test/v1',
        datasets=['general_fc'],
        work_dir=output_dir,
        no_timestamp=True,
    )
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')
    entered = threading.Event()

    def wait_for_cancel(_func: object, _config: TaskConfig, **kwargs: Any) -> None:
        cancel_event = kwargs['cancel_event']
        entered.set()
        assert cancel_event.wait(timeout=2)
        raise TaskStoppedError('job-1')

    runner = EvaluationJobRunner(store)
    with mock.patch('evalscope.service.job_runner.run_in_subprocess', side_effect=wait_for_cancel), mock.patch(
        'evalscope.service.job_runner.stop_process', return_value=False
    ):
        runner.submit('job-1', config, 'DEMO_API_KEY', 'demo-model', 'demo-1', {})
        assert entered.wait(timeout=2)
        cancelling = runner.cancel('job-1')
        cancelled = wait_for_terminal(store, 'job-1')
    runner.shutdown()

    assert cancelling.status == 'cancelling'
    assert cancelled.status == 'cancelled'


def test_runner_resolves_load_test_secret_and_persists_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / 'jobs.db')
    output_dir = str(tmp_path / 'outputs' / 'load-1')
    store.create_job('load-1', 'load-request-1', 'load_test', {'model_ref': 'demo'}, output_dir=output_dir)
    arguments = PerfArguments(
        model='demo',
        url='https://example.test/v1',
        api='openai',
        dataset='openqa',
        number=2,
        parallel=1,
        outputs_dir=output_dir,
        name='perf',
        no_timestamp=True,
    )
    scenario = LoadTestScenarioEntry(number=2, parallel=1)
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')
    runner = EvaluationJobRunner(store, source_commit='abc123')

    def run_perf(_func: object, perf_arguments: PerfArguments, **_kwargs: Any) -> dict[str, Any]:
        assert perf_arguments.api_key is not None
        assert perf_arguments.api_key.get_secret_value() == 'secret-value'
        assert perf_arguments.headers['Authorization'].get_secret_value() == 'Bearer secret-value'
        return {'parallel_1_number_2': {'success_rate': 1.0}}

    with mock.patch('evalscope.service.job_runner.run_in_subprocess', side_effect=run_perf):
        runner.submit_load_test(
            'load-1', arguments, 'DEMO_API_KEY', 'demo-model', 'smoke', scenario, {'suite': 'perf'}
        )
        completed = wait_for_terminal(store, 'load-1')
    runner.shutdown()

    assert completed.status == 'completed'
    assert completed.result['summary']['parallel_1_number_2']['success_rate'] == 1.0
    assert completed.result['provenance']['scenario_ref'] == 'smoke'
    assert completed.result['artifacts']['run_dir'] == str(Path(output_dir) / 'perf')
    assert 'secret-value' not in str(completed.result)
    assert 'secret-value' not in str(completed.request)


def test_runner_cancels_a_running_load_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JobStore(tmp_path / 'jobs.db')
    output_dir = str(tmp_path / 'outputs' / 'load-1')
    store.create_job('load-1', 'load-request-1', 'load_test', {'model_ref': 'demo'}, output_dir=output_dir)
    arguments = PerfArguments(
        model='demo',
        url='https://example.test/v1',
        number=2,
        parallel=1,
        outputs_dir=output_dir,
    )
    scenario = LoadTestScenarioEntry(number=2, parallel=1)
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')
    entered = threading.Event()

    def wait_for_cancel(_func: object, _arguments: PerfArguments, **kwargs: Any) -> None:
        entered.set()
        assert kwargs['cancel_event'].wait(timeout=2)
        raise TaskStoppedError('load-1')

    runner = EvaluationJobRunner(store)
    with mock.patch('evalscope.service.job_runner.run_in_subprocess', side_effect=wait_for_cancel), mock.patch(
        'evalscope.service.job_runner.stop_process', return_value=False
    ):
        runner.submit_load_test('load-1', arguments, 'DEMO_API_KEY', 'demo-model', 'smoke', scenario, {})
        assert entered.wait(timeout=2)
        cancelling = runner.cancel('load-1')
        cancelled = wait_for_terminal(store, 'load-1')
    runner.shutdown()

    assert cancelling.status == 'cancelling'
    assert cancelled.status == 'cancelled'
