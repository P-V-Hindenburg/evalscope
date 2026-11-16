"""Read-only benchmark catalogue for upper-level evaluation agents."""

import re
from typing import Any, Dict, List, Optional

from evalscope.utils.resource_utils import load_benchmark_data

from .compatibility import get_compatibility
from .contracts import BenchmarkInfo


class BenchmarkNotFoundError(ValueError):
    """Raised when a benchmark is absent from EvalScope's generated metadata catalogue."""


def _capability_name(tag: str) -> str:
    if tag.isupper():
        return tag.lower()
    value = re.sub(r'(?<!^)(?=[A-Z])', '_', tag).replace('-', '_').replace(' ', '_')
    return re.sub(r'_+', '_', value).strip('_').lower()


def _metric_names(raw_metrics: List[Any]) -> List[str]:
    return [next(iter(item)) if isinstance(item, dict) and item else str(item) for item in raw_metrics]


def _modalities(category: str, tags: List[str]) -> List[str]:
    if category == 'llm':
        return ['text']
    if 'Video' in tags:
        return ['text', 'video']
    return ['text', 'image']


def _build_info(benchmark_id: str, entry: Dict[str, Any]) -> BenchmarkInfo:
    meta = entry.get('meta') or {}
    category = meta.get('category', 'llm')
    if category not in ('llm', 'vlm'):
        raise ValueError(f'benchmark {benchmark_id!r} is not an LLM or VLM benchmark')

    tags = [str(tag) for tag in meta.get('tags', [])]
    compatibility = get_compatibility(benchmark_id)
    requires_judge = bool(meta.get('requires_judge', False))
    if compatibility.requires_judge is not None:
        requires_judge = compatibility.requires_judge
    requires_sandbox = bool(meta.get('sandbox_config'))
    if compatibility.requires_sandbox is not None:
        requires_sandbox = compatibility.requires_sandbox

    return BenchmarkInfo(
        benchmark_id=benchmark_id,
        pretty_name=meta.get('pretty_name') or benchmark_id,
        category=category,
        modalities=_modalities(category, tags),
        capabilities=[_capability_name(tag) for tag in tags],
        tags=tags,
        subsets=list(meta.get('subset_list') or []),
        metrics=_metric_names(list(meta.get('metrics') or [])),
        dataset_id=meta.get('dataset_id') or '',
        default_dataset_hub=compatibility.default_dataset_hub or 'modelscope',
        requires_judge=requires_judge,
        requires_sandbox=requires_sandbox,
        verified=compatibility.verified,
        notes=compatibility.notes,
    )


def get_benchmark(benchmark_id: str) -> BenchmarkInfo:
    """Return stable metadata for one LLM or VLM benchmark."""

    entry = load_benchmark_data(benchmark_id).get(benchmark_id, {})
    if not entry.get('meta'):
        raise BenchmarkNotFoundError(f'benchmark {benchmark_id!r} was not found')
    return _build_info(benchmark_id, entry)


def list_benchmarks(
    capability: Optional[str] = None,
    modality: Optional[str] = None,
    verified_only: bool = False,
) -> List[BenchmarkInfo]:
    """List LLM/VLM benchmarks, optionally filtered for agent planning."""

    capability_filter = _capability_name(capability) if capability else None
    result = []
    for benchmark_id, entry in load_benchmark_data().items():
        category = (entry.get('meta') or {}).get('category', 'llm')
        if category not in ('llm', 'vlm'):
            continue
        info = _build_info(benchmark_id, entry)
        if capability_filter and capability_filter not in info.capabilities:
            continue
        if modality and modality not in info.modalities:
            continue
        if verified_only and not info.verified:
            continue
        result.append(info)
    return sorted(result, key=lambda item: item.benchmark_id)
