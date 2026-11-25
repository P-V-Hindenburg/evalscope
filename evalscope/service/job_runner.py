# Copyright (c) Alibaba, Inc. and its affiliates.
"""Single-host asynchronous evaluation runner backed by JobStore."""

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.service.agent_manifests import LoadTestScenarioEntry
from evalscope.service.api_models.common import ActiveTaskInfo
from evalscope.service.job_store import InvalidJobTransitionError, JobNotFoundError, JobRecord, JobStore
from evalscope.service.utils import (
    TaskStoppedError,
    run_eval_wrapper,
    run_in_subprocess,
    run_perf_wrapper,
    serialize_result,
    stop_process,
)
from evalscope.utils.logger import get_logger
from evalscope.version import __version__

logger = get_logger()


class EvaluationJobRunner:
    """Execute queued evaluation jobs with a bounded local thread pool."""

    def __init__(self, store: JobStore, max_workers: int = 1, source_commit: str | None = None) -> None:
        if max_workers < 1:
            raise ValueError('max_workers must be positive')
        self.store = store
        self.max_workers = max_workers
        self.source_commit = source_commit or os.environ.get('EVALSCOPE_SOURCE_COMMIT')
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='evalscope-job')
        self._lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def submit(
        self,
        job_id: str,
        task_config: TaskConfig,
        api_key_env: str,
        model_ref: str,
        dataset_ref: str,
        tags: dict[str, str],
    ) -> None:
        """Queue one already-persisted evaluation job for execution."""
        cancel_event = threading.Event()
        with self._lock:
            if job_id in self._futures:
                raise ValueError(f'Job {job_id!r} is already submitted')
            self._cancel_events[job_id] = cancel_event
            future = self._executor.submit(
                self._execute,
                job_id,
                task_config,
                api_key_env,
                model_ref,
                dataset_ref,
                tags,
                cancel_event,
            )
            self._futures[job_id] = future

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel a queued or running job and return its latest persisted state."""
        job = self.store.get_job(job_id)
        if job is None:
            raise JobNotFoundError(f'Job {job_id!r} was not found')
        if job.status in {'completed', 'error', 'cancelled', 'interrupted'}:
            return job

        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
            future = self._futures.get(job_id)
        if cancel_event is not None:
            cancel_event.set()

        if job.status == 'queued':
            if future is not None and future.cancel():
                self._forget(job_id)
            return self.store.update_status(job_id, 'cancelled')

        if job.status == 'running':
            self.store.update_status(job_id, 'cancelling')
        stop_process(job_id)
        return self.store.get_job(job_id) or job

    def submit_load_test(
        self,
        job_id: str,
        perf_arguments: PerfArguments,
        api_key_env: str,
        model_ref: str,
        scenario_ref: str,
        scenario: LoadTestScenarioEntry,
        tags: dict[str, str],
    ) -> None:
        """Queue one already-persisted load-test job for execution."""
        cancel_event = threading.Event()
        with self._lock:
            if job_id in self._futures:
                raise ValueError(f'Job {job_id!r} is already submitted')
            self._cancel_events[job_id] = cancel_event
            future = self._executor.submit(
                self._execute_load_test,
                job_id,
                perf_arguments,
                api_key_env,
                model_ref,
                scenario_ref,
                scenario,
                tags,
                cancel_event,
            )
            self._futures[job_id] = future

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work and release runner threads."""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(
        self,
        job_id: str,
        task_config: TaskConfig,
        api_key_env: str,
        model_ref: str,
        dataset_ref: str,
        tags: dict[str, str],
        cancel_event: threading.Event,
    ) -> None:
        try:
            initial = self.store.get_job(job_id)
            if initial is None or initial.status != 'queued' or cancel_event.is_set():
                return
            self.store.update_status(job_id, 'running')
            if cancel_event.is_set():
                self._mark_cancelled(job_id)
                return

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f'Required secret environment variable is not configured: {api_key_env}')
            task_config.api_key = SecretStr(api_key)

            Path(task_config.work_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
            task_meta = ActiveTaskInfo(
                task_id=job_id,
                kind='eval',
                model=str(task_config.model),
                alias=model_ref,
                status='preparing',
            )
            result = run_in_subprocess(
                run_eval_wrapper,
                task_config,
                task_id=job_id,
                task_meta=task_meta,
                cancel_event=cancel_event,
            )
            payload = self._result_payload(result, task_config, model_ref, dataset_ref, tags)
            self.store.complete_job(job_id, payload, output_dir=task_config.work_dir)
        except TaskStoppedError:
            self._mark_cancelled(job_id)
        except InvalidJobTransitionError:
            current = self.store.get_job(job_id)
            if current is None or current.status not in {'cancelled', 'cancelling'}:
                logger.exception(f'Job {job_id} reached an invalid state transition')
            else:
                self._mark_cancelled(job_id)
        except Exception as error:
            logger.error(f'Asynchronous evaluation job {job_id} failed: {type(error).__name__}')
            current = self.store.get_job(job_id)
            if current is not None and current.status in {'queued', 'running', 'cancelling'}:
                self.store.fail_job(job_id, 'Evaluation failed; inspect the job log for details')
        finally:
            self._forget(job_id)

    def _execute_load_test(
        self,
        job_id: str,
        perf_arguments: PerfArguments,
        api_key_env: str,
        model_ref: str,
        scenario_ref: str,
        scenario: LoadTestScenarioEntry,
        tags: dict[str, str],
        cancel_event: threading.Event,
    ) -> None:
        try:
            initial = self.store.get_job(job_id)
            if initial is None or initial.status != 'queued' or cancel_event.is_set():
                return
            self.store.update_status(job_id, 'running')
            if cancel_event.is_set():
                self._mark_cancelled(job_id)
                return

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f'Required secret environment variable is not configured: {api_key_env}')
            perf_arguments.api_key = SecretStr(api_key)
            perf_arguments.headers = {
                **perf_arguments.headers,
                'Authorization': SecretStr(f'Bearer {api_key}'),
            }

            Path(perf_arguments.outputs_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
            task_meta = ActiveTaskInfo(
                task_id=job_id,
                kind='perf',
                model=str(perf_arguments.model),
                endpoint=urlsplit(perf_arguments.url).netloc,
                alias=model_ref,
                status='preparing',
            )
            result = run_in_subprocess(
                run_perf_wrapper,
                perf_arguments,
                task_id=job_id,
                task_meta=task_meta,
                cancel_event=cancel_event,
            )
            payload = self._perf_result_payload(
                result,
                perf_arguments,
                model_ref,
                scenario_ref,
                scenario,
                tags,
            )
            self.store.complete_job(job_id, payload, output_dir=perf_arguments.outputs_dir)
        except TaskStoppedError:
            self._mark_cancelled(job_id)
        except InvalidJobTransitionError:
            current = self.store.get_job(job_id)
            if current is None or current.status not in {'cancelled', 'cancelling'}:
                logger.exception(f'Job {job_id} reached an invalid state transition')
            else:
                self._mark_cancelled(job_id)
        except Exception as error:
            logger.error(f'Asynchronous load-test job {job_id} failed: {type(error).__name__}')
            current = self.store.get_job(job_id)
            if current is not None and current.status in {'queued', 'running', 'cancelling'}:
                self.store.fail_job(job_id, 'Load test failed; inspect the job log for details')
        finally:
            self._forget(job_id)

    def _mark_cancelled(self, job_id: str) -> None:
        current = self.store.get_job(job_id)
        if current is None or current.status == 'cancelled':
            return
        if current.status in {'queued', 'running', 'cancelling'}:
            self.store.update_status(job_id, 'cancelled')

    def _forget(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

    def _result_payload(
        self,
        result: Any,
        task_config: TaskConfig,
        model_ref: str,
        dataset_ref: str,
        tags: dict[str, str],
    ) -> dict[str, Any]:
        return {
            'summary': serialize_result(result),
            'artifacts': {'output_dir': task_config.work_dir},
            'provenance': {
                'evalscope_version': __version__,
                'source_commit': self.source_commit,
                'model_ref': model_ref,
                'dataset_ref': dataset_ref,
                'tags': tags,
            },
        }

    def _perf_result_payload(
        self,
        result: Any,
        perf_arguments: PerfArguments,
        model_ref: str,
        scenario_ref: str,
        scenario: LoadTestScenarioEntry,
        tags: dict[str, str],
    ) -> dict[str, Any]:
        return {
            'summary': serialize_result(result),
            'artifacts': {
                'output_dir': perf_arguments.outputs_dir,
                'run_dir': str(Path(perf_arguments.outputs_dir) / 'perf'),
            },
            'provenance': {
                'evalscope_version': __version__,
                'source_commit': self.source_commit,
                'model_ref': model_ref,
                'scenario_ref': scenario_ref,
                'scenario': scenario.model_dump(mode='json'),
                'tags': tags,
            },
        }
