"""Normalization of EvalScope Report objects into agent-facing results."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from evalscope.report import Report

from .contracts import ArtifactRefs, BenchmarkResult, MetricResult, SampleSummary


def artifact_refs(work_dir: str) -> ArtifactRefs:
    """Build predictable references to EvalScope's native output structure."""

    root = Path(work_dir)
    return ArtifactRefs(
        root=str(root),
        configs=str(root / 'configs'),
        logs=str(root / 'logs'),
        predictions=str(root / 'predictions'),
        reviews=str(root / 'reviews'),
        reports=str(root / 'reports'),
    )


def _metric_result(report: Report, metric) -> MetricResult:
    semantics = metric.semantics
    return MetricResult(
        name=metric.identity.key,
        score=metric.score,
        num=metric.num,
        is_primary=metric.identity == report.primary_metric_identity,
        direction=getattr(semantics, 'direction', None),
        unit=getattr(semantics, 'unit', None),
    )


def normalize_report(benchmark_id: str, report: Report) -> BenchmarkResult:
    """Normalize one EvalScope report without discarding its native artifact."""

    execution = report.execution_summary
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        model_id=report.model_name,
        metrics=[_metric_result(report, metric) for metric in report.metrics],
        requested_samples=execution.requested if execution else None,
        processed_samples=(execution.succeeded + execution.errored) if execution else report.num,
        succeeded_samples=execution.succeeded if execution else report.num,
        errored_samples=execution.errored if execution else None,
        incomplete=execution.incomplete if execution else False,
        judge_summary=report.judge_summary.model_dump(mode='json') if report.judge_summary else None,
    )


def normalize_run_result(result: object) -> List[BenchmarkResult]:
    """Normalize the mapping returned by EvalScope's native run_task function."""

    if isinstance(result, Report):
        return [normalize_report(result.dataset_name, result)]
    if not isinstance(result, dict):
        return []

    normalized = []
    for benchmark_id, value in result.items():
        if isinstance(value, Report):
            normalized.append(normalize_report(str(benchmark_id), value))
    return normalized


def merge_results(groups: Iterable[List[BenchmarkResult]]) -> List[BenchmarkResult]:
    """Flatten split dataset-hub runs while rejecting duplicate benchmark results."""

    merged: Dict[str, BenchmarkResult] = {}
    for group in groups:
        for item in group:
            if item.benchmark_id in merged:
                raise ValueError(f'duplicate result for benchmark {item.benchmark_id!r}')
            merged[item.benchmark_id] = item
    return list(merged.values())


def recover_results(work_dir: str, minimum_mtime: Optional[float] = None) -> List[BenchmarkResult]:
    """Recover normalized results from reports already persisted by EvalScope."""

    reports_root = Path(work_dir) / 'reports'
    recovered = []
    if not reports_root.exists():
        return recovered
    for path in sorted(reports_root.glob('*/*.json')):
        if minimum_mtime is not None and path.stat().st_mtime < minimum_mtime:
            continue
        try:
            report = Report.model_validate(json.loads(path.read_text(encoding='utf-8')))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        recovered.append(normalize_report(report.dataset_name, report))
    return recovered


def sample_summary(benchmarks: Iterable[BenchmarkResult], error_counts: Optional[Dict[str, int]] = None) -> SampleSummary:
    """Aggregate report coverage without pretending unknown counts are zero."""

    items = list(benchmarks)
    requested = [item.requested_samples for item in items]
    planned = sum(value for value in requested if value is not None) if any(value is not None for value in requested) else None
    successful = sum(item.succeeded_samples or 0 for item in items)
    failed = sum(item.errored_samples or 0 for item in items)
    processed = sum(
        item.processed_samples
        if item.processed_samples is not None
        else (item.succeeded_samples or 0) + (item.errored_samples or 0)
        for item in items
    )
    scored = 0
    for item in items:
        primary = next((metric.num for metric in item.metrics if metric.is_primary), None)
        if primary is not None:
            scored += primary
        elif item.succeeded_samples is not None:
            scored += item.succeeded_samples
        else:
            scored += max((metric.num for metric in item.metrics), default=0)
    skipped = max((planned or 0) - processed, 0)
    counts = dict(error_counts or {})
    classified = sum(counts.values())
    if failed > classified:
        counts['sample_error'] = failed - classified
    return SampleSummary(
        planned=planned, processed=processed, successful=successful, failed=failed,
        skipped=skipped, scored=scored, error_counts=counts,
    )
