"""In-process asynchronous facade for LangChain/LangGraph tool calls."""

import json
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .adapter import CredentialResolver, EvaluationCancelledError, run_evaluation
from .contracts import EvaluationRequest, EvaluationResult
from .events import EventPage, JobEventStore

JobStatus = Literal[
    'queued', 'validating', 'loading_dataset', 'running_model', 'running_judge',
    'scoring', 'generating_report', 'cancellation_requested',
    'completed', 'partial_success', 'failed', 'cancelled'
]
BenchmarkJobStatus = Literal[
    'queued', 'running', 'cancellation_requested', 'completed', 'partial_success', 'failed', 'cancelled'
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


class BenchmarkJobState(BaseModel):
    benchmark_id: str
    status: BenchmarkJobStatus = 'queued'
    current: int = 0
    total: Optional[int] = None
    percent: float = 0.0
    result_available: bool = False
    message: Optional[str] = None


class EvaluationJob(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: JobProgress = Field(default_factory=JobProgress)
    benchmarks: List[BenchmarkJobState] = Field(default_factory=list)
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
        self._benchmark_cancel_events: Dict[str, Dict[str, threading.Event]] = {}
        self._requests: Dict[str, EvaluationRequest] = {}
        self._event_stores: Dict[str, JobEventStore] = {}

    def submit(self, request: EvaluationRequest, resolver: CredentialResolver) -> str:
        job_id = f'eval_{uuid4().hex}'
        now = datetime.now(timezone.utc)
        with self._lock:
            benchmark_states = [BenchmarkJobState(benchmark_id=name) for name in request.benchmarks]
            self._jobs[job_id] = EvaluationJob(
                job_id=job_id,
                status='queued',
                created_at=now,
                updated_at=now,
                benchmarks=benchmark_states,
            )
            self._cancel_events[job_id] = threading.Event()
            self._benchmark_cancel_events[job_id] = {
                name: threading.Event() for name in request.benchmarks
            }
            self._requests[job_id] = request
            run_root = request.work_dir or str(Path('outputs') / 'agent-api' / job_id)
            self._event_stores[job_id] = JobEventStore(job_id, run_root)
        self._emit(job_id, 'job.queued', 'evaluation queued')
        for name in request.benchmarks:
            self._emit(
                job_id,
                'benchmark.queued',
                f'benchmark {name} queued',
                data={'benchmark_id': name},
            )
        self._futures[job_id] = self._executor.submit(self._run, job_id, request, resolver)
        return job_id

    def _set_stage(self, job_id: str, stage: JobStatus) -> None:
        job = self.status(job_id, refresh=False)
        if self._cancel_events[job_id].is_set():
            return
        if job.status != stage:
            self._emit(
                job_id,
                'job.stage_changed',
                f'evaluation stage changed to {stage}',
                data={'previous': job.status, 'status': stage},
            )
        self._update(job_id, status=stage, progress=job.progress.model_copy(update={'stage': stage}))

    def _run(self, job_id: str, request: EvaluationRequest, resolver: CredentialResolver) -> None:
        try:
            result = run_evaluation(
                request, resolver, run_id=job_id,
                cancel_event=self._cancel_events[job_id],
                benchmark_cancel_events=self._benchmark_cancel_events[job_id],
                stage_callback=lambda stage: self._set_stage(job_id, stage),
                progress_callback=lambda event: self._apply_progress_event(job_id, event),
            )
            self._refresh_progress(job_id)
            job = self.status(job_id, refresh=False)
            total = result.sample_summary.planned or job.progress.total
            processed = result.sample_summary.processed
            percent = round(processed * 100 / total, 1) if total else job.progress.percent
            terminal_progress = job.progress.model_copy(update={
                'current': processed,
                'total': total,
                'percent': percent,
                'total_is_estimate': False,
                'current_benchmark': None,
                'completed_groups': job.progress.total_groups or job.progress.completed_groups,
                'current_group': None,
                'current_group_benchmarks': [],
                'stage': result.status,
            })
            self._update(job_id, status=result.status, result=result, progress=terminal_progress)
            self._sync_benchmark_results(job_id, result)
            self._emit(
                job_id,
                f'job.{result.status}',
                f'evaluation {result.status}',
                level='warning' if result.status in {'partial_success', 'cancelled'} else 'info',
                data={'status': result.status},
            )
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

    def _emit(self, job_id: str, event_type: str, message: str, **kwargs) -> None:
        self._event_stores[job_id].append(event_type, message, **kwargs)

    def _update_benchmark(self, job_id: str, benchmark_id: str, **changes) -> None:
        with self._lock:
            current = self._jobs[job_id]
            states = []
            found = False
            for state in current.benchmarks:
                if state.benchmark_id == benchmark_id:
                    states.append(state.model_copy(update=changes))
                    found = True
                else:
                    states.append(state)
            if not found:
                raise KeyError(f'benchmark {benchmark_id!r} is not part of job {job_id}')
            self._jobs[job_id] = current.model_copy(update={
                'benchmarks': states,
                'updated_at': datetime.now(timezone.utc),
            })

    def _sync_benchmark_results(self, job_id: str, result: EvaluationResult) -> None:
        for item in result.benchmarks:
            current = item.processed_samples
            if current is None:
                current = (item.succeeded_samples or 0) + (item.errored_samples or 0)
            total = item.requested_samples
            percent = round(current * 100 / total, 2) if total else 0.0
            self._update_benchmark(
                job_id,
                item.benchmark_id,
                status=item.status,
                current=current,
                total=total,
                percent=percent,
                result_available=True,
                message=', '.join(item.error_codes) or None,
            )

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
        elif event_name == 'benchmark_started':
            benchmark_id = event['benchmark_id']
            self._update_benchmark(job_id, benchmark_id, status='running')
            self._emit(
                job_id,
                'benchmark.started',
                f'benchmark {benchmark_id} started',
                data={'benchmark_id': benchmark_id},
            )
        elif event_name == 'benchmark_progress':
            benchmark_id = event['benchmark_id']
            current = int(event.get('current') or 0)
            total = event.get('total')
            percent = float(event.get('percent') or 0.0)
            self._update_benchmark(
                job_id, benchmark_id,
                current=current, total=total, percent=percent,
            )
            self._emit(
                job_id,
                'benchmark.progress_updated',
                f'benchmark {benchmark_id} progress updated',
                data={
                    'benchmark_id': benchmark_id,
                    'completed_samples': current,
                    'total_samples': total,
                    'percent': percent,
                },
            )
        elif event_name == 'benchmark_completed':
            benchmark_id = event['benchmark_id']
            self._update_benchmark(job_id, benchmark_id, status='completed', result_available=True)
            self._emit(
                job_id,
                'benchmark.completed',
                f'benchmark {benchmark_id} completed',
                data={'benchmark_id': benchmark_id},
            )
        elif event_name == 'benchmark_failed':
            benchmark_id = event['benchmark_id']
            error = event.get('error') or {}
            self._update_benchmark(
                job_id, benchmark_id, status='failed', result_available=True,
                message=error.get('message'),
            )
            self._emit(
                job_id,
                'benchmark.failed',
                f'benchmark {benchmark_id} failed',
                level='error',
                data={'benchmark_id': benchmark_id, 'error': error},
            )
        elif event_name == 'benchmark_cancelled':
            benchmark_id = event['benchmark_id']
            self._update_benchmark(
                job_id, benchmark_id, status='cancelled', result_available=True,
                message=event.get('reason'),
            )
            self._emit(
                job_id,
                'benchmark.cancelled',
                f'benchmark {benchmark_id} cancelled',
                level='warning',
                data={'benchmark_id': benchmark_id, 'reason': event.get('reason')},
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
        self._emit(
            job_id,
            f'job.{status}',
            message or f'evaluation {status}',
            level='error' if status == 'failed' else 'warning',
            data={'status': status},
        )

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
        # A progress refresh may race with a whole-job cancellation request.
        # Keep the acknowledgement visible until the worker has actually
        # stopped and publishes the terminal ``cancelled`` state.
        cancel_event = self._cancel_events.get(job_id)
        if cancel_event is not None and cancel_event.is_set():
            stage = 'cancellation_requested'
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

    def cancel(self, job_id: str, benchmarks: Optional[List[str]] = None) -> bool:
        if job_id not in self._cancel_events:
            raise KeyError(f'unknown job_id: {job_id}')
        future = self._futures[job_id]
        if future.done():
            return False
        requested = list(benchmarks or [])
        if requested:
            unknown = sorted(set(requested) - set(self._benchmark_cancel_events[job_id]))
            if unknown:
                raise KeyError(f'benchmarks not part of job {job_id}: {", ".join(unknown)}')
            accepted = False
            job = self.status(job_id, refresh=False)
            state_by_id = {state.benchmark_id: state for state in job.benchmarks}
            for benchmark_id in requested:
                if state_by_id[benchmark_id].status in {'completed', 'partial_success', 'failed', 'cancelled'}:
                    continue
                self._benchmark_cancel_events[job_id][benchmark_id].set()
                self._update_benchmark(job_id, benchmark_id, status='cancellation_requested')
                self._emit(
                    job_id,
                    'benchmark.cancellation_requested',
                    f'cancellation requested for benchmark {benchmark_id}',
                    level='warning',
                    data={'benchmark_id': benchmark_id},
                )
                accepted = True
            return accepted

        self._cancel_events[job_id].set()
        self._emit(job_id, 'job.cancellation_requested', 'evaluation cancellation requested', level='warning')
        job = self.status(job_id, refresh=False)
        self._update(job_id, status='cancellation_requested', message='cancellation requested')
        for state in job.benchmarks:
            if state.status in {'queued', 'running'}:
                self._update_benchmark(job_id, state.benchmark_id, status='cancellation_requested')
        # Do not cancel the executor Future itself. Let the worker observe the
        # cancellation event and create a complete result containing every
        # requested benchmark, including items cancelled before they started.
        return True

    def events(self, job_id: str, after_cursor: int = 0, limit: int = 100) -> EventPage:
        if job_id not in self._event_stores:
            raise KeyError(f'unknown job_id: {job_id}')
        return self._event_stores[job_id].page(after_cursor=after_cursor, limit=limit)


_DEFAULT_MANAGER = EvaluationJobManager()


def submit_evaluation(request: EvaluationRequest, resolver: CredentialResolver) -> str:
    return _DEFAULT_MANAGER.submit(request, resolver)


def get_evaluation_status(job_id: str) -> EvaluationJob:
    return _DEFAULT_MANAGER.status(job_id)


def get_evaluation_result(job_id: str) -> Optional[EvaluationResult]:
    return _DEFAULT_MANAGER.result(job_id)


def cancel_evaluation(job_id: str, benchmarks: Optional[List[str]] = None) -> bool:
    return _DEFAULT_MANAGER.cancel(job_id, benchmarks=benchmarks)


def get_evaluation_events(job_id: str, after_cursor: int = 0, limit: int = 100) -> EventPage:
    return _DEFAULT_MANAGER.events(job_id, after_cursor=after_cursor, limit=limit)
