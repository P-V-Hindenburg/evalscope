"""Agent-facing Python API for deterministic EvalScope evaluation."""

from .adapter import (
    CredentialResolver,
    EvaluationValidationError,
    build_task_configs,
    resolve_credentials,
    run_evaluation,
    validate_evaluation,
)
from .catalog import BenchmarkNotFoundError, get_benchmark, list_benchmarks
from .jobs import (
    EvaluationJob,
    EvaluationJobManager,
    cancel_evaluation,
    get_evaluation_result,
    get_evaluation_status,
    submit_evaluation,
)
from .preflight import preflight_evaluation
from .contracts import (
    BenchmarkInfo,
    BenchmarkResult,
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    GenerationOptions,
    JudgeProfile,
    MetricResult,
    ModelTarget,
    PreflightResult,
    ResolvedEvaluationPlan,
    RuntimeOptions,
    SampleSummary,
)

__all__ = [
    'BenchmarkInfo',
    'BenchmarkNotFoundError',
    'BenchmarkResult',
    'CredentialResolver',
    'EvaluationError',
    'EvaluationRequest',
    'EvaluationResult',
    'EvaluationValidationError',
    'EvaluationJob',
    'EvaluationJobManager',
    'cancel_evaluation',
    'get_evaluation_result',
    'get_evaluation_status',
    'GenerationOptions',
    'JudgeProfile',
    'MetricResult',
    'ModelTarget',
    'PreflightResult',
    'ResolvedEvaluationPlan',
    'RuntimeOptions',
    'SampleSummary',
    'build_task_configs',
    'get_benchmark',
    'list_benchmarks',
    'preflight_evaluation',
    'resolve_credentials',
    'run_evaluation',
    'submit_evaluation',
    'validate_evaluation',
]
