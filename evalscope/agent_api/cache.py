"""Dataset cache discovery shared by preflight and execution routing."""

import os
import re
from pathlib import Path
from typing import Iterable

from .contracts import BenchmarkInfo


def _cache_token(value: str) -> str:
    """Normalize hub naming differences such as MVBench vs mv_bench."""

    return re.sub(r'[^a-z0-9]+', '', value.lower())


def dataset_cache_available(
    benchmarks: Iterable[BenchmarkInfo],
    hub: str,
) -> bool:
    """Return true when every requested dataset has a matching local cache directory."""

    roots = {
        'modelscope': Path(os.path.expanduser('~/.cache/modelscope/hub/datasets')),
        'huggingface': Path(os.path.expanduser('~/.cache/huggingface/datasets')),
    }
    root = roots.get(hub)
    if root is None or not root.exists():
        return False
    cached_names = [_cache_token(path.name) for path in root.iterdir()]
    for benchmark in benchmarks:
        candidates = {
            _cache_token(benchmark.benchmark_id),
            _cache_token(benchmark.dataset_id.split('/')[-1]),
        }
        if not any(any(token and token in cached for cached in cached_names) for token in candidates):
            return False
    return True
