"""Thin Python adapter from agent contracts to EvalScope TaskConfig and run_task."""

import multiprocessing as mp
import os
import queue
import re
import socket
import time
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from pydantic import SecretStr

from evalscope import TaskConfig, run_task
from evalscope.config import JudgeConfig, JudgeModelConfig, SandboxTaskConfig
from evalscope.utils.uri_utils import validate_http_endpoint

from .cache import dataset_cache_available
from .catalog import BenchmarkNotFoundError, get_benchmark
from .contracts import (
    ArtifactRefs,
    BenchmarkResult,
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    ResolvedCredentials,
    ResolvedEvaluationPlan,
)
from .results import artifact_refs, merge_results, normalize_run_result, recover_results, sample_summary

CredentialResolver = Callable[[str], str | SecretStr]


class EvaluationValidationError(ValueError):
    """Raised when a request cannot be mapped safely to EvalScope."""


class EvaluationCancelledError(RuntimeError):
    """Raised when an asynchronous caller requests cancellation."""


def validate_evaluation(request: EvaluationRequest) -> ResolvedEvaluationPlan:
    """Validate benchmark existence, modality, judge requirements, and hub routing."""

    try:
        validate_http_endpoint(request.target.api_url)
        if request.judge:
            validate_http_endpoint(request.judge.api_url)
    except ValueError as exc:
        raise EvaluationValidationError(str(exc)) from exc

    try:
        benchmarks = [get_benchmark(name) for name in request.benchmarks]
    except BenchmarkNotFoundError as exc:
        raise EvaluationValidationError(str(exc)) from exc

    vlm_benchmarks = [item.benchmark_id for item in benchmarks if item.category == 'vlm']
    if vlm_benchmarks and request.target.modality != 'multimodal':
        names = ', '.join(vlm_benchmarks)
        raise EvaluationValidationError(f'multimodal target required for benchmarks: {names}')

    judge_benchmarks = [item.benchmark_id for item in benchmarks if item.requires_judge]
    if judge_benchmarks and request.judge is None:
        names = ', '.join(judge_benchmarks)
        raise EvaluationValidationError(f'judge profile required for benchmarks: {names}')

    benchmark_by_id = {item.benchmark_id: item for item in benchmarks}
    for benchmark_id, options in request.benchmark_options.items():
        if options.subsets is None:
            continue
        available = benchmark_by_id[benchmark_id].subsets
        unknown = sorted(set(options.subsets) - set(available))
        if unknown:
            raise EvaluationValidationError(
                f'unknown subsets for benchmark {benchmark_id!r}: {", ".join(unknown)}; '
                f'available subsets: {", ".join(available)}'
            )

    benchmark_hubs = {item.benchmark_id: request.dataset_hub or item.default_dataset_hub for item in benchmarks}
    return ResolvedEvaluationPlan(
        request=request,
        benchmarks=benchmarks,
        benchmark_hubs=benchmark_hubs,
        sandbox_enabled=any(item.requires_sandbox for item in benchmarks),
    )


def _secret(value: str | SecretStr) -> SecretStr:
    return value if isinstance(value, SecretStr) else SecretStr(value)


def resolve_credentials(request: EvaluationRequest, resolver: CredentialResolver) -> ResolvedCredentials:
    """Resolve secret references outside the agent/model context."""

    target_api_key = _secret(resolver(request.target.credential_ref))
    judge_api_key = _secret(resolver(request.judge.credential_ref)) if request.judge else None
    if not target_api_key.get_secret_value():
        raise EvaluationValidationError('target credential resolved to an empty value')
    if request.judge and (judge_api_key is None or not judge_api_key.get_secret_value()):
        raise EvaluationValidationError('judge credential resolved to an empty value')
    return ResolvedCredentials(target_api_key=target_api_key, judge_api_key=judge_api_key)


def _judge_config(request: EvaluationRequest, credentials: ResolvedCredentials) -> JudgeConfig:
    if request.judge is None:
        return JudgeConfig()
    judge = request.judge
    return JudgeConfig(
        strategy='auto',
        models=[
            JudgeModelConfig(
                model_id=judge.model_id,
                api_url=judge.api_url,
                api_key=credentials.judge_api_key,
                eval_type=judge.eval_type,
                generation_config=judge.generation_config,
            )
        ],
    )


def _safe_group_name(
    dataset_hub: str,
    limit: Optional[Union[int, float]],
    *,
    include_limit: bool,
) -> str:
    limit_name = 'full' if limit is None else str(limit).replace('.', '_')
    raw = f'{dataset_hub}-limit-{limit_name}' if include_limit else dataset_hub
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', raw).strip('-') or 'default'


def _effective_limit(plan: ResolvedEvaluationPlan, benchmark_id: str) -> Optional[Union[int, float]]:
    options = plan.request.benchmark_options.get(benchmark_id)
    if options is not None and options.limit_per_subset is not None:
        return options.limit_per_subset
    return plan.request.limit_per_subset


def _effective_dataset_args(plan: ResolvedEvaluationPlan, benchmark_id: str) -> Dict:
    args = dict(plan.request.dataset_args.get(benchmark_id, {}))
    options = plan.request.benchmark_options.get(benchmark_id)
    if options is not None and options.subsets is not None:
        args['subset_list'] = list(options.subsets)
    return args


def build_task_configs(
    plan: ResolvedEvaluationPlan,
    credentials: ResolvedCredentials,
    run_root: str,
) -> List[TaskConfig]:
    """Build native configs grouped by dataset hub and effective per-subset limit.

    Grouping is an execution detail only. Each benchmark first resolves its
    request-wide default and optional benchmark-specific override.
    """

    GroupKey = Tuple[str, Optional[Union[int, float]]]
    grouped: Dict[GroupKey, List[str]] = defaultdict(list)
    for benchmark_id, dataset_hub in plan.benchmark_hubs.items():
        grouped[(dataset_hub, _effective_limit(plan, benchmark_id))].append(benchmark_id)

    multiple_groups = len(grouped) > 1
    groups_per_hub = {
        hub: sum(1 for grouped_hub, _ in grouped if grouped_hub == hub)
        for hub, _ in grouped
    }
    configs = []
    ordered_groups = sorted(grouped.items(), key=lambda item: (item[0][0], repr(item[0][1])))
    for (dataset_hub, limit), benchmarks in ordered_groups:
        group_name = _safe_group_name(
            dataset_hub,
            limit,
            include_limit=groups_per_hub[dataset_hub] > 1,
        )
        work_dir = os.path.join(run_root, group_name) if multiple_groups else run_root
        use_cache = plan.request.use_cache
        if use_cache and multiple_groups:
            use_cache = os.path.join(use_cache, group_name)
        sandbox_enabled = any(item.requires_sandbox for item in plan.benchmarks if item.benchmark_id in benchmarks)
        dataset_args = {}
        for name in benchmarks:
            args = _effective_dataset_args(plan, name)
            if args:
                dataset_args[name] = args
        for item in plan.benchmarks:
            if item.benchmark_id in benchmarks and item.requires_sandbox:
                dataset_args.setdefault(item.benchmark_id, {})['review_timeout'] = (
                    plan.request.runtime.sandbox_execution_timeout_seconds
                )
        configs.append(
            TaskConfig(
                model=plan.request.target.model_id,
                model_id=plan.request.target.model_id,
                api_url=plan.request.target.api_url,
                api_key=credentials.target_api_key,
                eval_type=plan.request.target.eval_type,
                datasets=benchmarks,
                dataset_args=dataset_args,
                dataset_hub=dataset_hub,
                limit=limit,
                generation_config=plan.request.generation.model_dump(exclude_none=True),
                eval_batch_size=plan.request.eval_batch_size,
                ignore_errors=plan.request.ignore_errors,
                seed=plan.request.seed,
                work_dir=work_dir,
                no_timestamp=True,
                enable_progress_tracker=True,
                use_cache=use_cache,
                rerun_review=plan.request.rerun_review,
                judge=_judge_config(plan.request, credentials),
                sandbox=SandboxTaskConfig(enabled=True) if sandbox_enabled else None,
            )
        )
    return configs


def _planned_sample_total(plan: ResolvedEvaluationPlan, configs: List[TaskConfig]) -> Optional[int]:
    """Return a stable upper-bound sample total when every group uses an integer limit.

    EvalScope resolves the exact count after loading each dataset. Until then,
    an integer limit and the selected subset list provide a deterministic plan.
    Full-dataset and fractional runs deliberately remain unknown.
    """

    benchmark_by_id = {item.benchmark_id: item for item in plan.benchmarks}
    total = 0
    for config in configs:
        if not isinstance(config.limit, int) or isinstance(config.limit, bool):
            return None
        for benchmark_id in config.datasets:
            configured_subsets = config.dataset_args.get(benchmark_id, {}).get('subset_list')
            subsets = configured_subsets or benchmark_by_id[benchmark_id].subsets or ['default']
            total += config.limit * len(subsets)
    return total


def _run_id() -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{timestamp}_{uuid4().hex[:8]}'


def _redact(message: str, secrets: ResolvedCredentials) -> str:
    redacted = message
    values = [secrets.target_api_key, secrets.judge_api_key]
    for value in values:
        if value is not None:
            raw = value.get_secret_value()
            if raw:
                redacted = redacted.replace(raw, '**********')
    return redacted


def _classify_error(exc: Exception, benchmarks: List[str]) -> EvaluationError:
    message = str(exc)
    lowered = message.lower()
    if 'cancelled' in lowered:
        return EvaluationError(
            code='CANCELLED', stage='execution', category='cancelled', message=message, benchmarks=benchmarks
        )
    if 'maximum allowed total pixels' in lowered or 'image exceeds' in lowered:
        return EvaluationError(
            code='IMAGE_LIMIT_EXCEEDED',
            stage='model',
            category='image_limit_exceeded',
            message=message,
            benchmarks=benchmarks,
        )
    if 'network is unreachable' in lowered or 'connection refused' in lowered:
        return EvaluationError(
            code='NETWORK_UNREACHABLE',
            stage='execution',
            category='network',
            message=message,
            benchmarks=benchmarks,
            retryable=True,
        )
    if 'timed out' in lowered or 'timeout' in lowered or 'exceeded' in lowered:
        return EvaluationError(
            code='TASK_TIMEOUT',
            stage='execution',
            category='timeout',
            message=message,
            benchmarks=benchmarks,
            retryable=True,
        )
    if 'judge' in lowered and ('parse' in lowered or 'format' in lowered):
        return EvaluationError(
            code='JUDGE_PARSE_FAILED',
            stage='judge',
            category='judge_parse_failed',
            message=message,
            benchmarks=benchmarks,
            retryable=True,
        )
    if 'sandbox' in lowered or 'code execution' in lowered:
        return EvaluationError(
            code='SANDBOX_EXECUTION_FAILED',
            stage='sandbox',
            category='sandbox_execution_failed',
            message=message,
            benchmarks=benchmarks,
        )
    if 'dataset' in lowered or 'huggingface' in lowered or 'modelscope' in lowered:
        return EvaluationError(
            code='DATASET_LOAD_FAILED',
            stage='dataset_load',
            category='dataset',
            message=message,
            benchmarks=benchmarks,
            retryable=True,
        )
    return EvaluationError(
        code='EVALUATION_EXECUTION_FAILED',
        stage='execution',
        category='unknown',
        message=message,
        benchmarks=benchmarks,
    )


def _run_task_worker(config: TaskConfig, output: mp.Queue) -> None:
    try:
        if config.dataset_hub == 'huggingface':
            infos = [get_benchmark(name) for name in config.datasets]
            cached = dataset_cache_available(infos, 'huggingface')
            reachable = True
            try:
                with socket.create_connection(('huggingface.co', 443), timeout=2.0):
                    pass
            except OSError:
                reachable = False
            if cached and not reachable:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
        normalized = normalize_run_result(run_task(config))
        output.put({'ok': True, 'results': [item.model_dump(mode='json') for item in normalized]})
    except BaseException as exc:
        output.put({'ok': False, 'error_type': type(exc).__name__, 'message': str(exc)})


def _stop_process(process: mp.Process, cleanup_timeout: float) -> bool:
    """Stop a worker, escalating from SIGTERM to SIGKILL when necessary.

    Network clients and sandbox cleanup handlers may delay or ignore a normal
    termination signal. Returning ``True`` means cleanup required escalation.
    """

    if not process.is_alive():
        return False
    process.terminate()
    process.join(timeout=cleanup_timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout=cleanup_timeout)
    return True


def _has_all_expected_results(results: List[BenchmarkResult], expected: List[str]) -> bool:
    """Return whether a recovered snapshot covers every requested benchmark."""

    return set(expected).issubset({item.benchmark_id for item in results})


def _merge_result_snapshots(*snapshots: List[BenchmarkResult]) -> List[BenchmarkResult]:
    """Merge repeated snapshots, preferring the latest value for each benchmark."""

    merged: Dict[str, BenchmarkResult] = {}
    for snapshot in snapshots:
        for item in snapshot:
            merged[item.benchmark_id] = item
    return list(merged.values())


def _run_config_isolated(config: TaskConfig, task_timeout: float, cleanup_timeout: float, cancel_event=None):
    """Run EvalScope in a child process so sandbox/background resources cannot hang the API."""

    started = time.time()
    # Jobs are launched from a ThreadPoolExecutor. Forking a multi-threaded
    # server can inherit a locked mutex and deadlock before EvalScope starts;
    # spawn creates a clean interpreter and is the safe cross-platform model.
    context = mp.get_context('spawn')
    output = context.Queue()
    process = context.Process(target=_run_task_worker, args=(config, output), daemon=True)
    process.start()
    deadline = started + task_timeout
    payload = None
    recovered = []
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            _stop_process(process, cleanup_timeout)
            raise EvaluationCancelledError('evaluation cancelled by caller')
        try:
            payload = output.get(timeout=min(1.0, max(0.01, deadline - time.time())))
            break
        except queue.Empty:
            recovered = recover_results(config.work_dir, minimum_mtime=started - 1.0)
            # A multi-benchmark run is not complete when only its first report
            # appears. Disk reports are a fallback for workers that finish the
            # evaluation but hang during cleanup (not the primary result channel).
            if _has_all_expected_results(recovered, list(config.datasets)):
                break

            # A worker normally reports success or failure through the queue.
            # If it exits without doing so, stop waiting and let the final disk
            # scan below decide whether anything can be recovered.
            if not process.is_alive():
                try:
                    payload = output.get(timeout=0.5)
                except queue.Empty:
                    pass
                break

    timed_out = payload is None and time.time() >= deadline
    if timed_out:
        _stop_process(process, cleanup_timeout)

    process.join(timeout=cleanup_timeout)
    cleanup_timed_out = process.is_alive()
    if cleanup_timed_out:
        _stop_process(process, cleanup_timeout)

    if payload is not None:
        if not payload['ok']:
            raise RuntimeError(f'{payload["error_type"]}: {payload["message"]}')
        results = [BenchmarkResult.model_validate(item) for item in payload['results']]
    else:
        results = recovered

    # Reports can be written between the last polling pass and process cleanup.
    # Always rescan once and merge by benchmark id so the caller never receives
    # a stale first-report-only snapshot.
    final_recovered = recover_results(config.work_dir, minimum_mtime=started - 1.0)
    results = _merge_result_snapshots(results, final_recovered)

    if timed_out and not _has_all_expected_results(results, list(config.datasets)):
        raise TimeoutError(f'evaluation exceeded {task_timeout} seconds')
    if payload is None and not _has_all_expected_results(results, list(config.datasets)):
        raise RuntimeError(
            f'evaluation worker exited without all expected results (exit code {process.exitcode})'
        )
    return results, cleanup_timed_out


def run_evaluation(
    request: EvaluationRequest,
    credential_resolver: CredentialResolver,
    run_id: Optional[str] = None,
    cancel_event=None,
    stage_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> EvaluationResult:
    """Run a logical evaluation synchronously and return normalized, agent-safe results."""

    if stage_callback:
        stage_callback('validating')
    plan = validate_evaluation(request)
    credentials = resolve_credentials(request, credential_resolver)
    resolved_run_id = run_id or _run_id()
    run_root = request.work_dir or os.path.join('outputs', 'agent-api', resolved_run_id)
    configs = build_task_configs(plan, credentials, run_root)
    if progress_callback:
        progress_callback({
            'event': 'plan',
            'total_groups': len(configs),
            'planned_samples': _planned_sample_total(plan, configs),
        })

    result_groups = []
    artifacts: List[ArtifactRefs] = []
    errors = []
    for group_index, config in enumerate(configs, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise EvaluationCancelledError('evaluation cancelled by caller')
        if progress_callback:
            progress_callback({
                'event': 'group_started',
                'group_index': group_index,
                'benchmarks': list(config.datasets),
            })
        artifacts.append(artifact_refs(config.work_dir))
        try:
            if stage_callback:
                stage_callback('loading_dataset')
            group, cleanup_timed_out = _run_config_isolated(
                config,
                task_timeout=request.runtime.task_timeout_seconds,
                cleanup_timeout=request.runtime.cleanup_timeout_seconds,
                cancel_event=cancel_event,
            )
            result_groups.append(group)
            if cleanup_timed_out:
                errors.append(
                    EvaluationError(
                        code='CLEANUP_TIMEOUT_RECOVERED',
                        stage='cleanup',
                        category='cleanup_timeout',
                        message='evaluation report was recovered, but worker cleanup exceeded its deadline',
                        benchmarks=list(config.datasets),
                        retryable=False,
                    )
                )
        except EvaluationCancelledError:
            raise
        except Exception as exc:
            error = _classify_error(exc, list(config.datasets))
            error.message = _redact(error.message, credentials)
            errors.append(error)
            recovered = recover_results(config.work_dir)
            if recovered:
                result_groups.append(recovered)
        if progress_callback:
            progress_callback({
                'event': 'group_completed',
                'completed_groups': group_index,
            })

    benchmarks = merge_results(result_groups)
    substantive_errors = [error for error in errors if error.stage != 'cleanup']
    incomplete = any(item.incomplete or (item.errored_samples or 0) > 0 for item in benchmarks)
    if substantive_errors and benchmarks or incomplete:
        status = 'partial_success'
    elif substantive_errors:
        status = 'failed'
    else:
        status = 'completed'
    error_counts = {}
    for error in errors:
        if error.stage != 'cleanup':
            error_counts[error.category] = error_counts.get(error.category, 0) + 1
    if stage_callback:
        stage_callback('generating_report')
    return EvaluationResult(
        run_id=resolved_run_id,
        status=status,
        benchmarks=benchmarks,
        sample_summary=sample_summary(benchmarks, error_counts),
        artifacts=artifacts,
        errors=errors,
    )
