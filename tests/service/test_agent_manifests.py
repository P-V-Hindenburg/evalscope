# Copyright (c) Alibaba, Inc. and its affiliates.

from pathlib import Path

import pytest
from pydantic import ValidationError

from evalscope.service.agent_manifests import AgentManifestCatalog, ManifestError, MissingSecretError


def write_manifests(tmp_path: Path) -> AgentManifestCatalog:
    models_path = tmp_path / 'models.yaml'
    models_path.write_text(
        """
demo-model:
  model: demo
  model_id: demo-model
  eval_type: openai_api
  api_url: https://example.test/v1
  api_key_env: DEMO_API_KEY
  allow_load_test: true
  generation_config:
    temperature: 0
    extra_body:
      thinking:
        type: disabled
""".strip(),
        encoding='utf-8',
    )
    datasets_path = tmp_path / 'datasets.yaml'
    datasets_path.write_text(
        """
demo-5:
  datasets:
    - general_fc
  limit: 5
  eval_batch_size: 2
""".strip(),
        encoding='utf-8',
    )
    scenarios_path = tmp_path / 'scenarios.yaml'
    scenarios_path.write_text(
        """
smoke:
  dataset: openqa
  number: 4
  parallel: 2
  max_tokens: 32
  max_prompt_length: 256
""".strip(),
        encoding='utf-8',
    )
    return AgentManifestCatalog(models_path, datasets_path, scenarios_path)


def test_build_task_config_uses_allowlisted_values_without_resolving_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = write_manifests(tmp_path)
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')

    config, api_key_env = catalog.build_task_config(
        'demo-model',
        'demo-5',
        str(tmp_path / 'output'),
        limit=3,
        generation_config={'max_tokens': 64},
    )

    assert api_key_env == 'DEMO_API_KEY'
    assert config.model == 'demo'
    assert config.datasets == ['general_fc']
    assert config.limit == 3
    assert config.api_key.get_secret_value() == 'EMPTY'
    assert config.generation_config.max_tokens == 64
    assert config.generation_config.extra_body == {'thinking': {'type': 'disabled'}}


def test_limit_cannot_exceed_registered_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = write_manifests(tmp_path)
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')

    with pytest.raises(ManifestError, match='cannot exceed 5'):
        catalog.build_task_config('demo-model', 'demo-5', str(tmp_path / 'output'), limit=6)


def test_missing_secret_reference_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = write_manifests(tmp_path)
    monkeypatch.delenv('DEMO_API_KEY', raising=False)

    with pytest.raises(MissingSecretError, match='DEMO_API_KEY'):
        catalog.build_task_config('demo-model', 'demo-5', str(tmp_path / 'output'))


def test_build_perf_arguments_uses_registered_scenario_and_enforces_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = write_manifests(tmp_path)
    monkeypatch.setenv('DEMO_API_KEY', 'secret-value')

    arguments, api_key_env, scenario = catalog.build_perf_arguments(
        'demo-model',
        'smoke',
        str(tmp_path / 'output'),
        number=8,
        parallel=4,
    )

    assert api_key_env == 'DEMO_API_KEY'
    assert arguments.number == [8]
    assert arguments.parallel == [4]
    assert arguments.api_key is None
    assert arguments.url == 'https://example.test/v1/chat/completions'
    assert scenario.number == 8

    with pytest.raises(ValidationError, match='number must contain values between 1 and 1000'):
        catalog.build_perf_arguments('demo-model', 'smoke', str(tmp_path / 'output'), number=1001)
