# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pydantic contracts for the agent-facing asynchronous job API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalscope.service.job_store import JobStatus

from .common import ApiResponseModel, ProgressResponse


class EvaluationCreateRequest(BaseModel):
    """Allowlisted asynchronous evaluation request."""

    model_config = ConfigDict(extra='forbid')

    idempotency_key: str = Field(min_length=1, max_length=200)
    task_type: Literal['evaluation'] = 'evaluation'
    model_ref: str = Field(min_length=1, max_length=200)
    dataset_ref: str = Field(min_length=1, max_length=200)
    limit: int | float | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator('idempotency_key', 'model_ref', 'dataset_ref')
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator('limit')
    @classmethod
    def _validate_limit(cls, value: int | float | None) -> int | float | None:
        if value is None:
            return value
        if isinstance(value, bool) or value <= 0:
            raise ValueError('limit must be positive')
        if isinstance(value, float) and value > 1:
            raise ValueError('A fractional limit cannot exceed 1')
        return value


class LoadTestOverrides(BaseModel):
    """Restricted per-request overrides for a registered load-test scenario."""

    model_config = ConfigDict(extra='forbid')

    number: int | list[int] | None = None
    parallel: int | list[int] | None = None
    rate: float | list[float] | None = None
    duration: float | None = None
    stream: bool | None = None


class LoadTestCreateRequest(BaseModel):
    """Allowlisted asynchronous model load-test request."""

    model_config = ConfigDict(extra='forbid')

    idempotency_key: str = Field(min_length=1, max_length=200)
    task_type: Literal['load_test'] = 'load_test'
    model_ref: str = Field(min_length=1, max_length=200)
    scenario_ref: str = Field(min_length=1, max_length=200)
    overrides: LoadTestOverrides = Field(default_factory=LoadTestOverrides)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator('idempotency_key', 'model_ref', 'scenario_ref')
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class JobResponse(ApiResponseModel):
    """Public job state without credentials or internal request hashes."""

    job_id: str
    kind: str
    status: JobStatus
    created: bool | None = None
    request: dict[str, Any]
    output_dir: str | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: ProgressResponse | None = None


class JobResultResponse(ApiResponseModel):
    """Persisted result for a completed asynchronous job."""

    job_id: str
    status: JobStatus
    result: Any


class ModelCatalogItem(ApiResponseModel):
    """Safe model registry entry exposed to callers."""

    ref: str
    model: str
    model_id: str | None = None
    eval_type: str
    api_url: str
    load_test_allowed: bool


class ModelCatalogResponse(ApiResponseModel):
    models: list[ModelCatalogItem]


class DatasetCatalogItem(ApiResponseModel):
    """Dataset registry entry exposed to callers."""

    ref: str
    datasets: list[str]
    limit: int | float | None = None
    eval_batch_size: int
    repeats: int


class DatasetCatalogResponse(ApiResponseModel):
    datasets: list[DatasetCatalogItem]


class CapabilitiesResponse(ApiResponseModel):
    evaluations: bool
    load_tests: bool
    idempotency: bool
    cancellation: bool
    persistence: Literal['sqlite']
    max_workers: int


class LoadTestScenarioItem(ApiResponseModel):
    """Safe registered load-test scenario with enforced bounds."""

    ref: str
    dataset: str
    number: int | list[int]
    parallel: int | list[int]
    rate: float | list[float]
    open_loop: bool
    duration: float | None = None
    warmup_num: int | float
    stream: bool
    max_tokens: int
    min_tokens: int | None = None
    min_prompt_length: int
    max_prompt_length: int
    temperature: float


class ScenariosResponse(ApiResponseModel):
    scenarios: list[LoadTestScenarioItem]
