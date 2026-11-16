"""HTTP-only request and response contracts."""

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import (
    EvaluationRequest,
    GenerationOptions,
    JudgeProfile,
    ModelTarget,
    RuntimeOptions,
)
from ..jobs import EvaluationJob, JobProgress, JobStatus


class HttpApiModel(BaseModel):
    """Strict base model for the cross-service HTTP contract."""

    model_config = ConfigDict(extra='forbid')


class HttpEvaluationRequest(HttpApiModel):
    """Remote-safe subset of EvaluationRequest.

    Server filesystem fields are intentionally absent. The service owns work
    directories and cache paths.
    """

    target: ModelTarget
    benchmarks: List[str] = Field(min_length=1)
    limit_per_subset: Optional[int | float] = None
    dataset_hub: Optional[Literal['modelscope', 'huggingface']] = None
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)
    judge: Optional[JudgeProfile] = None
    eval_batch_size: int = Field(default=1, ge=1)
    ignore_errors: bool = True
    seed: int = 42

    def to_domain(self) -> EvaluationRequest:
        """Validate with the canonical domain contract."""

        return EvaluationRequest.model_validate(self.model_dump())


class HealthResponse(HttpApiModel):
    status: Literal['ok'] = 'ok'
    service: str = 'evalscope-agent-api'
    version: str


class SubmitEvaluationResponse(HttpApiModel):
    job_id: str
    status: JobStatus = 'queued'


class EvaluationStatusResponse(HttpApiModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: JobProgress
    result_available: bool = False
    message: Optional[str] = None

    @classmethod
    def from_job(cls, job: EvaluationJob) -> 'EvaluationStatusResponse':
        """Hide the potentially large result from status polling responses."""

        return cls(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            progress=job.progress,
            result_available=job.result is not None,
            message=job.message,
        )


class PendingResultResponse(HttpApiModel):
    job_id: str
    status: JobStatus
    detail: str = 'evaluation result is not ready'


class CancellationResponse(HttpApiModel):
    job_id: str
    accepted: bool
    status: Literal['cancellation_requested'] = 'cancellation_requested'


class ApiError(HttpApiModel):
    code: str
    message: str
    retryable: bool = False
    details: Optional[Any] = None


class ApiErrorResponse(HttpApiModel):
    error: ApiError
