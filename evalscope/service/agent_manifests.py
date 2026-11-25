# Copyright (c) Alibaba, Inc. and its affiliates.
"""Validated model and dataset manifests for the agent-facing service."""

import copy
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.utils.io_utils import yaml_to_dict
from evalscope.utils.uri_utils import validate_http_endpoint

_ENV_NAME_PATTERN = re.compile(r'^[A-Z_][A-Z0-9_]*$')
_ManifestEntry = TypeVar('_ManifestEntry', bound=BaseModel)


class ManifestError(ValueError):
    """Base exception for invalid or unavailable service manifests."""


class ManifestReferenceError(ManifestError):
    """Raised when a request refers to an unregistered manifest entry."""


class MissingSecretError(ManifestError):
    """Raised when a registered model's secret environment variable is absent."""


class LoadTestNotAllowedError(ManifestError):
    """Raised when a model is not explicitly approved for load testing."""


class ModelManifestEntry(BaseModel):
    """Allowlisted model configuration without any secret value."""

    model_config = ConfigDict(extra='forbid')

    model: str
    model_id: str | None = None
    eval_type: str = 'openai_api'
    api_url: str
    api_key_env: str
    generation_config: dict[str, Any] = Field(default_factory=dict)
    allow_load_test: bool = False
    load_test_api: str = 'openai'

    @field_validator('api_url')
    @classmethod
    def _validate_api_url(cls, value: str) -> str:
        validate_http_endpoint(value)
        return value.strip()

    @field_validator('api_key_env')
    @classmethod
    def _validate_api_key_env(cls, value: str) -> str:
        if not _ENV_NAME_PATTERN.fullmatch(value):
            raise ValueError('api_key_env must be an uppercase environment variable name')
        return value


class DatasetManifestEntry(BaseModel):
    """Allowlisted dataset configuration used to construct TaskConfig."""

    model_config = ConfigDict(extra='forbid')

    datasets: list[str] = Field(min_length=1)
    limit: int | float | None = None
    eval_batch_size: int = Field(default=1, ge=1)
    repeats: int = Field(default=1, ge=1)
    dataset_args: dict[str, dict[str, Any]] = Field(default_factory=dict)


class LoadTestScenarioEntry(BaseModel):
    """Bounded load-test scenario expanded into EvalScope Perf arguments."""

    model_config = ConfigDict(extra='forbid')

    dataset: str = 'openqa'
    dataset_path: str | None = None
    number: int | list[int] = 20
    parallel: int | list[int] = 1
    rate: float | list[float] = -1.0
    open_loop: bool = False
    duration: float | None = None
    warmup_num: int | float = 0
    stream: bool = True
    max_tokens: int = 256
    min_tokens: int | None = None
    min_prompt_length: int = 0
    max_prompt_length: int = 1024
    prompt: str | None = None
    tokenizer_path: str | None = None
    temperature: float = 0.0

    @model_validator(mode='after')
    def _validate_safety_limits(self) -> 'LoadTestScenarioEntry':
        numbers = self.number if isinstance(self.number, list) else [self.number]
        parallels = self.parallel if isinstance(self.parallel, list) else [self.parallel]
        rates = self.rate if isinstance(self.rate, list) else [self.rate]
        if not numbers or any(value < 1 or value > 1000 for value in numbers):
            raise ValueError('number must contain values between 1 and 1000')
        if not parallels or any(value < 1 or value > 32 for value in parallels):
            raise ValueError('parallel must contain values between 1 and 32')
        if self.open_loop and (not rates or any(value <= 0 or value > 100 for value in rates)):
            raise ValueError('open-loop rate must contain values between 0 and 100')
        if self.duration is not None and (self.duration <= 0 or self.duration > 1800):
            raise ValueError('duration must be between 0 and 1800 seconds')
        if self.max_tokens < 1 or self.max_tokens > 2048:
            raise ValueError('max_tokens must be between 1 and 2048')
        if self.min_tokens is not None and (self.min_tokens < 0 or self.min_tokens > self.max_tokens):
            raise ValueError('min_tokens must be between 0 and max_tokens')
        if self.min_prompt_length < 0 or self.max_prompt_length < self.min_prompt_length:
            raise ValueError('Invalid prompt length range')
        if self.max_prompt_length > 16384:
            raise ValueError('max_prompt_length cannot exceed 16384')
        if not 0 <= self.temperature <= 2:
            raise ValueError('temperature must be between 0 and 2')
        return self


class AgentManifestCatalog:
    """Load and validate the allowlisted model and dataset YAML files."""

    def __init__(
        self,
        models_path: str | os.PathLike[str],
        datasets_path: str | os.PathLike[str],
        scenarios_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.models_path = Path(models_path).expanduser().resolve()
        self.datasets_path = Path(datasets_path).expanduser().resolve()
        self.scenarios_path = Path(scenarios_path).expanduser().resolve() if scenarios_path is not None else None

    @staticmethod
    def _load(path: Path, model_type: type[_ManifestEntry]) -> dict[str, _ManifestEntry]:
        if not path.is_file():
            raise ManifestError(f'Manifest file does not exist: {path}')
        try:
            payload = yaml_to_dict(str(path))
        except Exception as error:
            raise ManifestError(f'Unable to load manifest: {path}') from error
        if not isinstance(payload, dict):
            raise ManifestError(f'Manifest must contain a mapping: {path}')
        result = {}
        for reference, entry in payload.items():
            if not isinstance(reference, str) or not reference.strip():
                raise ManifestError(f'Manifest contains an invalid reference: {path}')
            try:
                result[reference] = model_type.model_validate(entry)
            except ValidationError as error:
                raise ManifestError(f'Invalid manifest entry {reference!r}: {path}') from error
        return result

    def models(self) -> dict[str, ModelManifestEntry]:
        """Return the validated model registry keyed by public reference."""
        return self._load(self.models_path, ModelManifestEntry)

    def datasets(self) -> dict[str, DatasetManifestEntry]:
        """Return the validated dataset registry keyed by public reference."""
        return self._load(self.datasets_path, DatasetManifestEntry)

    def scenarios(self) -> dict[str, LoadTestScenarioEntry]:
        """Return the load-test scenario registry, or an empty registry when disabled."""
        if self.scenarios_path is None or not self.scenarios_path.is_file():
            return {}
        return self._load(self.scenarios_path, LoadTestScenarioEntry)

    def get_model(self, reference: str) -> ModelManifestEntry:
        """Resolve a model reference or raise a safe validation error."""
        entry = self.models().get(reference)
        if entry is None:
            raise ManifestReferenceError(f'Unknown model_ref: {reference}')
        return entry

    def get_dataset(self, reference: str) -> DatasetManifestEntry:
        """Resolve a dataset reference or raise a safe validation error."""
        entry = self.datasets().get(reference)
        if entry is None:
            raise ManifestReferenceError(f'Unknown dataset_ref: {reference}')
        return entry

    def get_scenario(self, reference: str) -> LoadTestScenarioEntry:
        """Resolve a load-test scenario reference or raise a safe validation error."""
        entry = self.scenarios().get(reference)
        if entry is None:
            raise ManifestReferenceError(f'Unknown scenario_ref: {reference}')
        return entry

    def build_task_config(
        self,
        model_ref: str,
        dataset_ref: str,
        work_dir: str,
        limit: int | float | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> tuple[TaskConfig, str]:
        """Build a typed task config while leaving the API secret unresolved."""
        model = self.get_model(model_ref)
        dataset = self.get_dataset(dataset_ref)
        if model.api_key_env not in os.environ:
            raise MissingSecretError(f'Required secret environment variable is not configured: {model.api_key_env}')

        effective_limit = self._validated_limit_override(dataset.limit, limit)
        effective_generation = self._deep_merge(model.generation_config, generation_config or {})
        try:
            task_config = TaskConfig.from_dict(
                {
                    'model': model.model,
                    'model_id': model.model_id,
                    'eval_type': model.eval_type,
                    'api_url': model.api_url,
                    'datasets': dataset.datasets,
                    'dataset_args': dataset.dataset_args,
                    'limit': effective_limit,
                    'eval_batch_size': dataset.eval_batch_size,
                    'repeats': dataset.repeats,
                    'generation_config': effective_generation,
                    'work_dir': work_dir,
                    'no_timestamp': True,
                    'enable_progress_tracker': True,
                    'analysis_report': True,
                }
            )
        except (ImportError, TypeError, ValueError) as error:
            raise ManifestError('Registered model or dataset configuration is invalid') from error
        return task_config, model.api_key_env

    def build_perf_arguments(
        self,
        model_ref: str,
        scenario_ref: str,
        output_dir: str,
        number: int | list[int] | None = None,
        parallel: int | list[int] | None = None,
        rate: float | list[float] | None = None,
        duration: float | None = None,
        stream: bool | None = None,
    ) -> tuple[PerfArguments, str, LoadTestScenarioEntry]:
        """Build bounded Perf arguments while leaving the API secret unresolved."""
        model = self.get_model(model_ref)
        if not model.allow_load_test:
            raise LoadTestNotAllowedError(f'Model {model_ref!r} is not approved for load testing')
        if model.api_key_env not in os.environ:
            raise MissingSecretError(f'Required secret environment variable is not configured: {model.api_key_env}')
        scenario = self.get_scenario(scenario_ref)
        overrides = {
            key: value
            for key, value in {
                'number': number,
                'parallel': parallel,
                'rate': rate,
                'duration': duration,
                'stream': stream,
            }.items()
            if value is not None
        }
        scenario = LoadTestScenarioEntry.model_validate({**scenario.model_dump(), **overrides})
        try:
            arguments = PerfArguments.from_dict(
                {
                    'model': model.model,
                    'model_id': model.model_id,
                    'api': model.load_test_api,
                    'url': model.api_url,
                    'dataset': scenario.dataset,
                    'dataset_path': scenario.dataset_path,
                    'number': scenario.number,
                    'parallel': scenario.parallel,
                    'rate': scenario.rate,
                    'open_loop': scenario.open_loop,
                    'duration': scenario.duration,
                    'warmup_num': scenario.warmup_num,
                    'stream': scenario.stream,
                    'max_tokens': scenario.max_tokens,
                    'min_tokens': scenario.min_tokens,
                    'min_prompt_length': scenario.min_prompt_length,
                    'max_prompt_length': scenario.max_prompt_length,
                    'prompt': scenario.prompt,
                    'tokenizer_path': scenario.tokenizer_path,
                    'temperature': scenario.temperature,
                    'outputs_dir': output_dir,
                    'name': 'perf',
                    'alias': model_ref,
                    'no_timestamp': True,
                    'enable_progress_tracker': True,
                    'no_test_connection': True,
                }
            )
        except (ImportError, TypeError, ValueError) as error:
            raise ManifestError('Registered load-test configuration is invalid') from error
        return arguments, model.api_key_env, scenario

    @staticmethod
    def _validated_limit_override(configured: int | float | None, requested: int | float | None) -> int | float | None:
        if requested is None:
            return configured
        if configured is None:
            raise ManifestError('This dataset manifest does not permit a limit override')
        if type(requested) is not type(configured) or requested > configured:
            raise ManifestError(f'limit must use the registered type and cannot exceed {configured}')
        return requested

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AgentManifestCatalog._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
