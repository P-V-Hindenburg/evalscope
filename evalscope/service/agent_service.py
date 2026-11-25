# Copyright (c) Alibaba, Inc. and its affiliates.
"""Application service coordinating manifests, persistent jobs and execution."""

import uuid
from pathlib import Path

from evalscope.service.agent_manifests import AgentManifestCatalog
from evalscope.service.api_models.jobs import EvaluationCreateRequest, LoadTestCreateRequest
from evalscope.service.job_runner import EvaluationJobRunner
from evalscope.service.job_store import CreateJobResult, JobStore


class AgentEvaluationService:
    """Submit allowlisted evaluation requests to the persistent local runner."""

    def __init__(
        self,
        catalog: AgentManifestCatalog,
        store: JobStore,
        output_root: str,
        max_workers: int = 1,
        source_commit: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.store = store
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.runner = EvaluationJobRunner(store, max_workers=max_workers, source_commit=source_commit)
        self.store.mark_running_jobs_interrupted()

    def submit_evaluation(self, request: EvaluationCreateRequest) -> CreateJobResult:
        """Validate, persist and enqueue an asynchronous evaluation."""
        request_payload = request.model_dump(mode='json')
        existing = self.store.get_by_idempotency_key(request.idempotency_key)
        if existing is not None:
            return self.store.create_job(
                job_id=f'eval_{uuid.uuid4().hex}',
                idempotency_key=request.idempotency_key,
                kind='evaluation',
                request=request_payload,
            )

        job_id = f'eval_{uuid.uuid4().hex}'
        output_dir = str(self.output_root / job_id)
        task_config, api_key_env = self.catalog.build_task_config(
            model_ref=request.model_ref,
            dataset_ref=request.dataset_ref,
            work_dir=output_dir,
            limit=request.limit,
            generation_config=request.generation_config,
        )
        created = self.store.create_job(
            job_id=job_id,
            idempotency_key=request.idempotency_key,
            kind='evaluation',
            request=request_payload,
            output_dir=output_dir,
        )
        if created.created:
            self.runner.submit(
                job_id=job_id,
                task_config=task_config,
                api_key_env=api_key_env,
                model_ref=request.model_ref,
                dataset_ref=request.dataset_ref,
                tags=request.tags,
            )
        return created

    def submit_load_test(self, request: LoadTestCreateRequest) -> CreateJobResult:
        """Validate, persist and enqueue an asynchronous load test."""
        request_payload = request.model_dump(mode='json')
        existing = self.store.get_by_idempotency_key(request.idempotency_key)
        if existing is not None:
            return self.store.create_job(
                job_id=f'load_{uuid.uuid4().hex}',
                idempotency_key=request.idempotency_key,
                kind='load_test',
                request=request_payload,
            )

        job_id = f'load_{uuid.uuid4().hex}'
        output_dir = str(self.output_root / job_id)
        overrides = request.overrides
        perf_arguments, api_key_env, scenario = self.catalog.build_perf_arguments(
            model_ref=request.model_ref,
            scenario_ref=request.scenario_ref,
            output_dir=output_dir,
            number=overrides.number,
            parallel=overrides.parallel,
            rate=overrides.rate,
            duration=overrides.duration,
            stream=overrides.stream,
        )
        created = self.store.create_job(
            job_id=job_id,
            idempotency_key=request.idempotency_key,
            kind='load_test',
            request=request_payload,
            output_dir=output_dir,
        )
        if created.created:
            self.runner.submit_load_test(
                job_id=job_id,
                perf_arguments=perf_arguments,
                api_key_env=api_key_env,
                model_ref=request.model_ref,
                scenario_ref=request.scenario_ref,
                scenario=scenario,
                tags=request.tags,
            )
        return created
