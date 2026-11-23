"""In-process asynchronous facade for LangChain/LangGraph tool calls."""

import json
import re
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
    total_is_estimate: bool = False
    current_benchmark: Optional[str] = None
    completed_groups: int = 0
    total_groups: Optional[int] = None
    current_group: Optional[int] = None
    current_group_benchmarks: list[str] = Field(default_factory=list)
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
                progress_callback=lambda event: self._apply_progress_event(job_id, event),
            )
            self._refresh_progress(job_id)
            job = self.status(job_id, refresh=False)
            total = result.sample_summary.planned or job.progress.total
            terminal_progress = job.progress.model_copy(update={
                'current': total if total is not None else job.progress.current,
                'total': total,
                'percent': 100.0,
                'total_is_estimate': False,
                'current_benchmark': None,
                'completed_groups': job.progress.total_groups or job.progress.completed_groups,
                'current_group': None,
                'current_group_benchmarks': [],
                'stage': result.status,
            })
            self._update(job_id, status=result.status, result=result, progress=terminal_progress)
        except EvaluationCancelledError:
            self._set_terminal(job_id, 'cancelled', message='cancelled by caller')
        except Exception as exc:
            self._set_terminal(job_id, 'failed', message=str(exc))

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(update={**changes, 'updated_at': datetime.now(timezone.utc)})

    def _update_progress(self, job_id: str, **changes) -> None:
        """Atomically merge progress events without losing concurrent polling updates."""

        with self._lock:
            current = self._jobs[job_id]
            progress = current.progress.model_copy(update=changes)
            self._jobs[job_id] = current.model_copy(update={
                'progress': progress,
                'updated_at': datetime.now(timezone.utc),
            })

    def _apply_progress_event(self, job_id: str, event: dict) -> None:
        event_name = event.get('event')
        if event_name == 'plan':
            planned = event.get('planned_samples')
            self._update_progress(
                job_id,
                total=planned,
                total_is_estimate=planned is not None,
                total_groups=event.get('total_groups'),
            )
        elif event_name == 'group_started':
            self._update_progress(
                job_id,
                current_group=event.get('group_index'),
                current_group_benchmarks=list(event.get('benchmarks') or []),
            )
        elif event_name == 'group_completed':
            self._update_progress(
                job_id,
                completed_groups=int(event.get('completed_groups') or 0),
                current_group=None,
                current_group_benchmarks=[],
            )

    def _set_terminal(self, job_id: str, status: JobStatus, message: Optional[str] = None) -> None:
        self._refresh_progress(job_id)
        job = self.status(job_id, refresh=False)
        progress = job.progress.model_copy(update={
            'stage': status,
            'total_is_estimate': False,
            'current_benchmark': None,
            'current_group': None,
            'current_group_benchmarks': [],
        })
        self._update(job_id, status=status, message=message, progress=progress)

    @staticmethod
    def _progress_stage(data: dict, request: EvaluationRequest, fallback: JobStatus):
        stage_data = data.get('stage') or {}
        nodes = [stage_data]
        nodes.extend(stage_data.get('children') or [])
        running = next((node for node in reversed(nodes) if node.get('status') == 'running'), stage_data)
        name = str(running.get('name') or '').lower()
        if 'predict' in name:
            stage = 'running_model'
        elif 'review' in name:
            stage = 'running_judge' if request.judge else 'scoring'
        elif 'score' in name or 'aggregat' in name:
            stage = 'scoring'
        elif 'report' in name:
            stage = 'generating_report'
        else:
            stage = fallback
        return stage, stage_data.get('label')

    @staticmethod
    def _log_progress_fallback(root: Path, fallback: JobStatus):
        """Infer the current stage from the newest event on older EvalScope versions."""

        logs = sorted(root.glob('**/eval_log.log'), key=lambda path: path.stat().st_mtime, reverse=True)
        if not logs:
            return fallback, None
        try:
            lines = logs[0].read_text(encoding='utf-8', errors='replace').splitlines()[-300:]
        except OSError:
            return fallback, None

        current_benchmark = None
        for line in reversed(lines):
            match = re.search(r'Start loading benchmark dataset:\s*([^\s]+)', line)
            if not match:
                match = re.search(r'Evaluating\[([^\]]+)\]', line)
            if match:
                current_benchmark = match.group(1).lower()
                break

        # Read backwards so an old "Generating report" line cannot override a
        # newer benchmark's loading or prediction event.
        for line in reversed(lines):
            if 'Start loading benchmark dataset:' in line:
                return 'loading_dataset', current_benchmark
            if re.search(r'Evaluating\[[^\]]+\]', line):
                return 'running_model', current_benchmark
            if 'predictions complete, aggregating scores' in line or 'Aggregating scores for subset:' in line:
                return 'scoring', current_benchmark
            if 'Generating report' in line:
                return 'generating_report', current_benchmark
        return fallback, current_benchmark

    @staticmethod
    def _completed_sample_count(root: Path) -> int:
        """Count samples covered by reports that have already been finalized."""

        completed = 0
        for path in root.glob('**/reports/*/*.json'):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                execution = report.get('execution_summary') or {}
                completed += int(execution.get('requested') or report.get('num') or 0)
            except (OSError, ValueError, TypeError):
                continue
        return completed

    def _refresh_progress(self, job_id: str) -> None:
        request = self._requests[job_id]
        root = Path(request.work_dir or Path('outputs') / 'agent-api' / job_id)
        files = list(root.glob('**/progress.json'))
        if not files:
            return
        processed = total = 0
        snapshots = []
        for path in files:
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                processed += int(data.get('processed_count') or 0)
                total += int(data.get('total_count') or 0)
                snapshots.append((path.stat().st_mtime, data))
            except (OSError, ValueError):
                continue
        if not snapshots:
            return
        job = self.status(job_id, refresh=False)
        # A multi-group task must not advertise the first group's total as the
        # whole task total. Prefer the stable plan total. If it is unknowable
        # (full/fractional datasets), expose no total until all groups exist.
        if job.progress.total is not None:
            display_total = job.progress.total
            total_is_estimate = job.progress.total_is_estimate
        elif job.progress.total_groups is not None and len(snapshots) < job.progress.total_groups:
            display_total = None
            total_is_estimate = False
        else:
            display_total = total or None
            total_is_estimate = False
        percent = round(processed * 100 / display_total, 1) if display_total else 0.0
        if total_is_estimate:
            percent = min(percent, 99.9)
        _, latest = max(snapshots, key=lambda item: item[0])
        if latest.get('stage'):
            stage, current_benchmark = self._progress_stage(latest, request, job.status)
        else:
            stage, current_benchmark = self._log_progress_fallback(root, job.status)
            # Older EvalScope releases only persist the loading event while
            # prediction is in progress. Once the global processed count moves
            # beyond finalized reports, the current benchmark is generating.
            if stage == 'loading_dataset' and processed > self._completed_sample_count(root):
                stage = 'running_model'
        self._update(job_id, status=stage, progress=job.progress.model_copy(update={
            'current': processed,
            'total': display_total,
            'percent': percent,
            'total_is_estimate': total_is_estimate,
            'current_benchmark': current_benchmark,
            'stage': stage,
        }))

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
