"""Cheap readiness checks performed before dataset loading or paid inference."""

import importlib.util
import shutil
import socket
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from .adapter import CredentialResolver, resolve_credentials, validate_evaluation
from .cache import dataset_cache_available
from .contracts import EvaluationError, EvaluationRequest, PreflightCheck, PreflightResult


def _tcp_check(url: str, timeout: float) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise OSError(f'invalid endpoint: {url}')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    with socket.create_connection((host, port), timeout=timeout):
        pass


def preflight_evaluation(
    request: EvaluationRequest,
    credential_resolver: CredentialResolver,
    *,
    check_network: bool = True,
) -> PreflightResult:
    """Validate routing, credentials, dependencies, disk, and endpoint reachability."""

    checks: List[PreflightCheck] = []
    errors: List[EvaluationError] = []
    try:
        plan = validate_evaluation(request)
        checks.append(PreflightCheck(name='request', ok=True, detail='benchmark and modality validation passed'))
    except Exception as exc:
        return PreflightResult(ok=False, checks=[PreflightCheck(name='request', ok=False, detail=str(exc))], errors=[
            EvaluationError(code='PREFLIGHT_VALIDATION_FAILED', stage='preflight', category='invalid_request', message=str(exc))
        ])

    try:
        resolve_credentials(request, credential_resolver)
        checks.append(PreflightCheck(name='credentials', ok=True, detail='credential references resolved'))
    except Exception as exc:
        errors.append(EvaluationError(code='CREDENTIAL_RESOLUTION_FAILED', stage='preflight', category='credential', message=str(exc)))
        checks.append(PreflightCheck(name='credentials', ok=False, detail=str(exc)))

    if plan.sandbox_enabled:
        available = importlib.util.find_spec('ms_enclave') is not None
        checks.append(PreflightCheck(name='sandbox_dependency', ok=available, detail='ms_enclave available' if available else 'ms_enclave missing'))
        if not available:
            errors.append(EvaluationError(code='SANDBOX_DEPENDENCY_MISSING', stage='preflight', category='dependency', message='ms_enclave is required', benchmarks=[item.benchmark_id for item in plan.benchmarks if item.requires_sandbox]))

    root = Path(request.work_dir or 'outputs')
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    disk_ok = free >= request.runtime.minimum_free_disk_bytes
    checks.append(PreflightCheck(name='disk_space', ok=disk_ok, detail=f'{free} bytes free'))
    if not disk_ok:
        errors.append(EvaluationError(code='INSUFFICIENT_DISK_SPACE', stage='preflight', category='resource', message=f'{free} bytes free', retryable=True))

    if check_network:
        endpoints = [('target_endpoint', request.target.api_url, list(request.benchmarks))]
        if request.judge:
            endpoints.append(('judge_endpoint', request.judge.api_url, [item.benchmark_id for item in plan.benchmarks if item.requires_judge]))
        hubs = sorted(set(plan.benchmark_hubs.values()))
        hub_urls = {'modelscope': 'https://modelscope.cn', 'huggingface': 'https://huggingface.co'}
        endpoints.extend((f'dataset_hub:{hub}', hub_urls[hub], [name for name, value in plan.benchmark_hubs.items() if value == hub]) for hub in hubs)
        for name, url, benchmarks in endpoints:
            try:
                _tcp_check(url, request.runtime.preflight_timeout_seconds)
                checks.append(PreflightCheck(name=name, ok=True, detail=f'{urlparse(url).hostname} reachable'))
            except OSError as exc:
                category = 'dataset_hub_unreachable' if name.startswith('dataset_hub:') else 'endpoint_unreachable'
                matching = [item for item in plan.benchmarks if item.benchmark_id in benchmarks]
                if name.startswith('dataset_hub:') and dataset_cache_available(matching, name.split(':', 1)[1]):
                    checks.append(PreflightCheck(name=name, ok=True, detail=f'network unavailable; local cache found for {benchmarks}'))
                    continue
                errors.append(EvaluationError(code=category.upper(), stage='preflight', category=category, message=str(exc), benchmarks=benchmarks, retryable=True))
                checks.append(PreflightCheck(name=name, ok=False, detail=str(exc)))

    return PreflightResult(ok=not errors, checks=checks, errors=errors)
