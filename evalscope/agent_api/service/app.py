"""FastAPI application exposing the agent-facing evaluation API."""

from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from evalscope.version import __version__

from ..adapter import EvaluationValidationError, resolve_credentials, validate_evaluation
from ..catalog import BenchmarkNotFoundError, get_benchmark, list_benchmarks
from ..contracts import BenchmarkInfo, EvaluationRequest, EvaluationResult, PreflightResult
from ..jobs import EvaluationJobManager
from ..events import EventCursorAheadError, EventCursorExpiredError
from ..preflight import preflight_evaluation
from .auth import build_auth_dependency
from .credentials import (
    CredentialReferenceError,
    CredentialUnavailableError,
    EnvironmentCredentialResolver,
)
from .schemas import (
    ApiError,
    ApiErrorResponse,
    CancellationResponse,
    EvaluationStatusResponse,
    EventPageResponse,
    HealthResponse,
    HttpEvaluationRequest,
    PendingResultResponse,
    PartialCancellationRequest,
    SubmitEvaluationResponse,
)
from .settings import ServiceSettings


def _error_content(code: str, message: str, *, retryable: bool = False, details=None) -> dict:
    return ApiErrorResponse(
        error=ApiError(code=code, message=message, retryable=retryable, details=details)
    ).model_dump(mode='json', exclude_none=True)


def _raise_http_error(status_code: int, code: str, message: str, *, retryable: bool = False) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={'code': code, 'message': message, 'retryable': retryable},
    )


def _validate_endpoint_host(url: str, settings: ServiceSettings) -> None:
    if not settings.allowed_endpoint_hosts:
        _raise_http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            'ENDPOINT_ALLOWLIST_NOT_CONFIGURED',
            'EVALSCOPE_ALLOWED_ENDPOINT_HOSTS is not configured',
        )
    hostname = (urlparse(url).hostname or '').lower()
    if hostname not in settings.allowed_endpoint_hosts:
        _raise_http_error(
            status.HTTP_400_BAD_REQUEST,
            'ENDPOINT_NOT_ALLOWED',
            f'endpoint host {hostname!r} is not allowed',
        )


def _to_domain_request(payload: HttpEvaluationRequest, settings: ServiceSettings) -> EvaluationRequest:
    _validate_endpoint_host(payload.target.api_url, settings)
    if payload.judge:
        _validate_endpoint_host(payload.judge.api_url, settings)
    try:
        return payload.to_domain()
    except ValidationError as exc:
        _raise_http_error(
            422,
            'REQUEST_VALIDATION_FAILED',
            'evaluation request validation failed',
            retryable=False,
        )
        raise exc  # pragma: no cover


def create_app(
    *,
    settings: Optional[ServiceSettings] = None,
    credential_resolver=None,
    job_manager: Optional[EvaluationJobManager] = None,
) -> FastAPI:
    """Create an isolated HTTP application around the existing Python API."""

    resolved_settings = settings or ServiceSettings.from_environment()
    resolver = credential_resolver or EnvironmentCredentialResolver.from_environment()
    manager = job_manager or EvaluationJobManager()
    authenticate = build_auth_dependency(resolved_settings)

    application = FastAPI(
        title='EvalScope Agent Evaluation API',
        version='0.1.0',
        description='Asynchronous LLM/VLM evaluation service for upper-level agents.',
    )
    router = APIRouter(prefix='/v1', dependencies=[Depends(authenticate)])

    @application.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        content = _error_content(
            code=str(detail.get('code') or 'HTTP_ERROR'),
            message=str(detail.get('message') or exc.detail),
            retryable=bool(detail.get('retryable', False)),
        )
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_content(
                'REQUEST_VALIDATION_FAILED',
                'HTTP request validation failed',
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @application.get('/health', response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @router.get('/benchmarks', response_model=list[BenchmarkInfo])
    async def benchmarks(
        capability: Optional[str] = Query(default=None),
        modality: Optional[str] = Query(default=None),
        verified_only: bool = Query(default=False),
    ) -> list[BenchmarkInfo]:
        return list_benchmarks(capability=capability, modality=modality, verified_only=verified_only)

    @router.get('/benchmarks/{benchmark_id}', response_model=BenchmarkInfo)
    async def benchmark_detail(benchmark_id: str) -> BenchmarkInfo:
        try:
            return get_benchmark(benchmark_id)
        except BenchmarkNotFoundError as exc:
            _raise_http_error(status.HTTP_404_NOT_FOUND, 'BENCHMARK_NOT_FOUND', str(exc))
            raise exc  # pragma: no cover

    @router.post('/evaluations/preflight', response_model=PreflightResult)
    async def evaluation_preflight(payload: HttpEvaluationRequest) -> PreflightResult:
        request = _to_domain_request(payload, resolved_settings)
        return preflight_evaluation(request, resolver)

    @router.post(
        '/evaluations',
        response_model=SubmitEvaluationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit(payload: HttpEvaluationRequest) -> SubmitEvaluationResponse:
        request = _to_domain_request(payload, resolved_settings)
        try:
            validate_evaluation(request)
            resolve_credentials(request, resolver)
        except CredentialReferenceError as exc:
            _raise_http_error(status.HTTP_400_BAD_REQUEST, 'CREDENTIAL_REFERENCE_UNKNOWN', str(exc))
        except CredentialUnavailableError as exc:
            _raise_http_error(status.HTTP_503_SERVICE_UNAVAILABLE, 'CREDENTIAL_UNAVAILABLE', str(exc))
        except EvaluationValidationError as exc:
            _raise_http_error(status.HTTP_400_BAD_REQUEST, 'INVALID_EVALUATION_REQUEST', str(exc))
        job_id = manager.submit(request, resolver)
        return SubmitEvaluationResponse(job_id=job_id)

    @router.get('/evaluations/{job_id}', response_model=EvaluationStatusResponse)
    async def evaluation_status(job_id: str) -> EvaluationStatusResponse:
        try:
            job = manager.status(job_id)
        except KeyError:
            _raise_http_error(status.HTTP_404_NOT_FOUND, 'JOB_NOT_FOUND', f'unknown job_id: {job_id}')
        return EvaluationStatusResponse.from_job(job)

    @router.get(
        '/evaluations/{job_id}/result',
        response_model=EvaluationResult | PendingResultResponse,
    )
    async def evaluation_result(job_id: str):
        try:
            job = manager.status(job_id)
        except KeyError:
            _raise_http_error(status.HTTP_404_NOT_FOUND, 'JOB_NOT_FOUND', f'unknown job_id: {job_id}')
        if job.result is not None:
            return job.result
        if job.status in {'failed', 'cancelled'}:
            _raise_http_error(
                status.HTTP_409_CONFLICT,
                'JOB_TERMINATED_WITHOUT_RESULT',
                job.message or f'evaluation {job.status}',
            )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=PendingResultResponse(job_id=job_id, status=job.status).model_dump(mode='json'),
        )

    @router.get('/evaluations/{job_id}/events', response_model=EventPageResponse)
    async def evaluation_events(
        job_id: str,
        after_cursor: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> EventPageResponse:
        try:
            page = manager.events(job_id, after_cursor=after_cursor, limit=limit)
            return EventPageResponse.model_validate(page.model_dump())
        except KeyError:
            _raise_http_error(status.HTTP_404_NOT_FOUND, 'JOB_NOT_FOUND', f'unknown job_id: {job_id}')
        except EventCursorExpiredError as exc:
            _raise_http_error(status.HTTP_410_GONE, 'EVENT_CURSOR_EXPIRED', str(exc))
        except EventCursorAheadError as exc:
            _raise_http_error(status.HTTP_409_CONFLICT, 'EVENT_CURSOR_AHEAD', str(exc))

    @router.delete(
        '/evaluations/{job_id}',
        response_model=CancellationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel(job_id: str) -> CancellationResponse:
        try:
            accepted = manager.cancel(job_id)
        except KeyError:
            _raise_http_error(status.HTTP_404_NOT_FOUND, 'JOB_NOT_FOUND', f'unknown job_id: {job_id}')
        if not accepted:
            _raise_http_error(
                status.HTTP_409_CONFLICT,
                'JOB_ALREADY_TERMINAL',
                f'job {job_id} is already complete',
            )
        return CancellationResponse(job_id=job_id, accepted=True)

    @router.post(
        '/evaluations/{job_id}/cancellations',
        response_model=CancellationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_benchmarks(
        job_id: str,
        payload: PartialCancellationRequest,
    ) -> CancellationResponse:
        try:
            accepted = manager.cancel(job_id, benchmarks=payload.benchmarks)
        except KeyError as exc:
            message = str(exc)
            code = 'BENCHMARK_NOT_IN_JOB' if 'benchmarks not part' in message else 'JOB_NOT_FOUND'
            _raise_http_error(status.HTTP_404_NOT_FOUND, code, message)
        if not accepted:
            _raise_http_error(
                status.HTTP_409_CONFLICT,
                'BENCHMARKS_ALREADY_TERMINAL',
                'all requested benchmarks are already terminal',
            )
        return CancellationResponse(job_id=job_id, accepted=True)

    application.include_router(router)
    return application


app = create_app()
