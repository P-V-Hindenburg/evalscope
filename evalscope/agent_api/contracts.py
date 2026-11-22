"""Typed contracts for the agent-facing evaluation adapter."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class AgentApiModel(BaseModel):
    """Strict base model shared by the agent-facing API contracts."""

    model_config = ConfigDict(extra='forbid')


class BenchmarkInfo(AgentApiModel):
    """Stable benchmark metadata exposed to an evaluation-planning agent."""

    benchmark_id: str
    pretty_name: str
    category: Literal['llm', 'vlm']
    modalities: List[str]
    capabilities: List[str]
    tags: List[str]
    subsets: List[str]
    metrics: List[str]
    dataset_id: str
    default_dataset_hub: Literal['modelscope', 'huggingface'] = 'modelscope'
    requires_judge: bool = False
    requires_sandbox: bool = False
    verified: bool = False
    notes: List[str] = Field(default_factory=list)


class ModelTarget(AgentApiModel):
    """Reference to the model endpoint under evaluation."""

    model_id: str
    api_url: str
    credential_ref: str
    eval_type: str = 'openai_api'
    modality: Literal['text', 'multimodal'] = 'text'


class JudgeProfile(AgentApiModel):
    """Reference to a separately configured judge model."""

    model_id: str
    api_url: str
    credential_ref: str
    eval_type: str = 'openai_api'
    generation_config: Dict[str, Any] = Field(default_factory=dict)


class GenerationOptions(AgentApiModel):
    """Common generation settings forwarded to EvalScope."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    timeout: Optional[float] = Field(default=None, gt=0.0)
    retries: int = Field(default=3, ge=1, le=10)
    retry_interval: int = Field(default=0, ge=0)
    stream: Optional[bool] = None


class RuntimeOptions(AgentApiModel):
    """Deadlines for one logical evaluation and its cleanup."""

    task_timeout_seconds: float = Field(default=3600.0, gt=0.0)
    cleanup_timeout_seconds: float = Field(default=15.0, gt=0.0)
    sandbox_execution_timeout_seconds: float = Field(default=4.0, gt=0.0)
    preflight_timeout_seconds: float = Field(default=5.0, gt=0.0)
    minimum_free_disk_bytes: int = Field(default=1_073_741_824, ge=0)


class EvaluationRequest(AgentApiModel):
    """Agent-facing request for one logical evaluation."""

    target: ModelTarget
    benchmarks: List[str] = Field(min_length=1)
    limit_per_subset: Optional[int | float] = None
    dataset_hub: Optional[Literal['modelscope', 'huggingface']] = None
    dataset_args: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)
    judge: Optional[JudgeProfile] = None
    eval_batch_size: int = Field(default=1, ge=1)
    ignore_errors: bool = True
    seed: int = 42
    work_dir: Optional[str] = None
    use_cache: Optional[str] = None
    rerun_review: bool = False

    @field_validator('benchmarks')
    @classmethod
    def _unique_benchmarks(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError('at least one benchmark is required')
        if len(cleaned) != len(set(cleaned)):
            raise ValueError('benchmarks must not contain duplicates')
        return cleaned

    @field_validator('limit_per_subset')
    @classmethod
    def _validate_limit(cls, value: Optional[int | float]) -> Optional[int | float]:
        if value is None:
            return value
        if isinstance(value, bool) or value <= 0:
            raise ValueError('limit_per_subset must be positive')
        if isinstance(value, float) and value > 1:
            raise ValueError('a float limit_per_subset must be a fraction in (0, 1]')
        return value


class ResolvedEvaluationPlan(AgentApiModel):
    """Validated, deterministic plan before conversion to TaskConfig objects."""

    request: EvaluationRequest
    benchmarks: List[BenchmarkInfo]
    benchmark_hubs: Dict[str, Literal['modelscope', 'huggingface']]
    sandbox_enabled: bool


class MetricResult(AgentApiModel):
    """Normalized aggregate metric returned to an upper-level agent."""

    name: str
    score: float
    num: int
    is_primary: bool = False
    direction: Optional[str] = None
    unit: Optional[str] = None


class BenchmarkResult(AgentApiModel):
    """Normalized result for one benchmark."""

    benchmark_id: str
    model_id: str
    metrics: List[MetricResult]
    requested_samples: Optional[int] = None
    succeeded_samples: Optional[int] = None
    errored_samples: Optional[int] = None
    incomplete: bool = False
    judge_summary: Optional[Dict[str, Any]] = None


class SampleSummary(AgentApiModel):
    """Sample coverage across all normalized benchmark reports."""

    planned: Optional[int] = None
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    scored: int = 0
    error_counts: Dict[str, int] = Field(default_factory=dict)


class ArtifactRefs(AgentApiModel):
    """Paths to EvalScope-native artifacts retained for audit and debugging."""

    root: str
    configs: str
    logs: str
    predictions: str
    reviews: str
    reports: str


class EvaluationError(AgentApiModel):
    """Structured execution error safe for an upper-level agent to inspect."""

    code: str
    stage: Literal['preflight', 'validation', 'dataset_load', 'model', 'judge', 'sandbox', 'scoring', 'report', 'cleanup', 'execution', 'result']
    category: str = 'unknown'
    message: str
    benchmarks: List[str] = Field(default_factory=list)
    retryable: bool = False


class EvaluationResult(AgentApiModel):
    """Unified result for a logical evaluation, including split backend runs."""

    run_id: str
    status: Literal['completed', 'partial_success', 'failed']
    benchmarks: List[BenchmarkResult]
    sample_summary: SampleSummary = Field(default_factory=SampleSummary)
    artifacts: List[ArtifactRefs]
    errors: List[EvaluationError] = Field(default_factory=list)


class PreflightCheck(AgentApiModel):
    """One deterministic validation performed before paid model calls."""

    name: str
    ok: bool
    detail: str


class PreflightResult(AgentApiModel):
    """Readiness result returned without starting an evaluation."""

    ok: bool
    checks: List[PreflightCheck]
    errors: List[EvaluationError] = Field(default_factory=list)


class ResolvedCredentials(AgentApiModel):
    """Internal secret values resolved outside the model context."""

    target_api_key: SecretStr
    judge_api_key: Optional[SecretStr] = None
