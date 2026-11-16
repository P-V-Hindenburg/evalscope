"""Small, explicit compatibility rules learned from verified evaluations."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkCompatibility(BaseModel):
    """Project-specific defaults that are not universal EvalScope semantics."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    default_dataset_hub: Optional[Literal['modelscope', 'huggingface']] = None
    requires_judge: Optional[bool] = None
    requires_sandbox: Optional[bool] = None
    verified: bool = False
    notes: List[str] = Field(default_factory=list)


_VERIFIED_DEFAULT = BenchmarkCompatibility(verified=True)

BENCHMARK_COMPATIBILITY: Dict[str, BenchmarkCompatibility] = {
    'gsm8k': _VERIFIED_DEFAULT,
    'hellaswag': _VERIFIED_DEFAULT,
    'ifeval': BenchmarkCompatibility(verified=True, notes=['Requires the IFEval optional dependencies.']),
    'pope': _VERIFIED_DEFAULT,
    'mm_bench': _VERIFIED_DEFAULT,
    'docvqa': BenchmarkCompatibility(
        verified=True,
        notes=['Provider image limits may exclude oversized document images when ignore_errors is enabled.'],
    ),
    'humaneval': BenchmarkCompatibility(
        verified=True,
        requires_sandbox=True,
        notes=['Generated code must run in an isolated execution environment.'],
    ),
    'longbench_v2': _VERIFIED_DEFAULT,
    'mt_bench': BenchmarkCompatibility(verified=True, requires_judge=True),
    'mia_bench': BenchmarkCompatibility(verified=True, requires_judge=True),
    'mmmu': _VERIFIED_DEFAULT,
    'mvbench': BenchmarkCompatibility(
        verified=True,
        default_dataset_hub='huggingface',
        notes=['The verified dataset path uses the Hugging Face subset layout.'],
    ),
}


def get_compatibility(benchmark_id: str) -> BenchmarkCompatibility:
    """Return project compatibility defaults for a benchmark."""

    return BENCHMARK_COMPATIBILITY.get(benchmark_id, BenchmarkCompatibility())
