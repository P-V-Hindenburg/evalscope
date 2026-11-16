"""In-process asynchronous facade for LangChain/LangGraph tool calls."""

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .adapter import CredentialResolver, EvaluationCancelledError, run_evaluation
from .contracts import EvaluationRequest, EvaluationResult

JobStatus = Literal[
    'queued', 'validating', 'loading_dataset', 'running_model', 'running_judge',
    'scoring', 'generating_report', 'completed', 'partial_success', 'failed', 'cancelled'
]


class JobProgress(BaseModel):
    current: int = 0
    total: Optional[int] = None
    percent: float = 0.0
    current_benchmark: Optional[str] = None
    stage: JobStatus = 'queued'


class EvaluationJob(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: JobProgress = Field(default_factory=JobProgress)
    result: Optional[EvaluationResult] = None
    message: Optional[str] = None


class EvaluationJobManager:
    """Thread-safe registry; EvalScope execution stays in cancellable child processes."""

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='evalscope-agent')
        self._lock = threading.Lock()
        self._jobs: Dict[str, EvaluationJob] = {}
        self._futures: Dict[str, Future] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._requests: Dict[str, EvaluationRequest] = {}

    def submit(self, request: EvaluationRequest, resolver: CredentialResolver) -> str:
        job_id = f'eval_{uuid4().hex}'
        now = datetime.now(timezone.utc)
        with self._lock:
            self._jobs[job_id] = EvaluationJob(job_id=job_id, status='queued', created_at=now, updated_at=now)
            self._cancel_events[job_id] = threading.Event()
            self._requests[job_id] = request
        self._futures[job_id] = self._executor.submit(self._run, job_id, request, resolver)
        return job_id

    def _set_stage(self, job_id: str, stage: JobStatus) -> None:
        job = self.status(job_id, refresh=False)
        self._update(job_id, status=stage, progress=job.progress.model_copy(update={'stage': stage}))

    def _run(self, job_id: str, request: EvaluationRequest, resolver: CredentialResolver) -> None:
        try:
            result = run_evaluation(
                request, resolver, run_id=job_id,
                cancel_event=self._cancel_events[job_id],
                stage_callback=lambda stage: self._set_stage(job_id, stage),
            )
            self._update(job_id, status=result.status, result=result)
        except EvaluationCancelledError:
            self._update(job_id, status='cancelled', message='cancelled by caller')
        except Exception as exc:
            self._update(job_id, status='failed', message=str(exc))

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(update={**changes, 'updated_at': datetime.now(timezone.utc)})

    def _refresh_progress(self, job_id: str) -> None:
        request = self._requests[job_id]
        root = Path(request.work_dir or Path('outputs') / 'agent-api' / job_id)
        files = list(root.glob('**/progress.json'))
        if not files:
            return
        processed = total = 0
        for path in files:
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                processed += int(data.get('processed_count') or 0)
                total += int(data.get('total_count') or 0)
            except (OSError, ValueError):
                continue
        job = self.status(job_id, refresh=False)
        percent = round(processed * 100 / total, 1) if total else 0.0
        stage = job.status
        logs = []
        for path in root.glob('**/eval_log.log'):
            try:
                logs.append(path.read_text(encoding='utf-8', errors='replace')[-12000:])
            except OSError:
                pass
        log_text = '\n'.join(logs)
        if 'Generating report' in log_text:
            stage = 'generating_report'
        elif 'aggregating scores' in log_text:
            stage = 'scoring'
        elif request.judge and f'Creating model {request.judge.model_id}' in log_text:
            stage = 'running_judge'
        elif 'Loading model for prediction' in log_text:
            stage = 'running_model'
        self._update(job_id, status=stage, progress=JobProgress(
            current=processed, total=total or None, percent=percent, stage=stage,
        ))

    def status(self, job_id: str, refresh: bool = True) -> EvaluationJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f'unknown job_id: {job_id}')
            terminal = self._jobs[job_id].status in {'completed', 'partial_success', 'failed', 'cancelled'}
        if refresh and not terminal:
            self._refresh_progress(job_id)
        with self._lock:
            return self._jobs[job_id].model_copy(deep=True)

    def result(self, job_id: str) -> Optional[EvaluationResult]:
        return self.status(job_id).result

    def cancel(self, job_id: str) -> bool:
        if job_id not in self._cancel_events:
            raise KeyError(f'unknown job_id: {job_id}')
        future = self._futures[job_id]
        if future.done():
            return False
        self._cancel_events[job_id].set()
        if future.cancel():
            self._update(job_id, status='cancelled', message='cancelled before execution')
        return True


_DEFAULT_MANAGER = EvaluationJobManager()


def submit_evaluation(request: EvaluationRequest, resolver: CredentialResolver) -> str:
    return _DEFAULT_MANAGER.submit(request, resolver)


def get_evaluation_status(job_id: str) -> EvaluationJob:
    return _DEFAULT_MANAGER.status(job_id)


def get_evaluation_result(job_id: str) -> Optional[EvaluationResult]:
    return _DEFAULT_MANAGER.result(job_id)


def cancel_evaluation(job_id: str) -> bool:
    return _DEFAULT_MANAGER.cancel(job_id)
