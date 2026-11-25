# Copyright (c) Alibaba, Inc. and its affiliates.
"""Blueprint modules for EvalScope service."""

from .eval import bp_eval
from .jobs import bp_jobs
from .perf import bp_perf
from .reports import bp_reports
from .tasks import bp_tasks

__all__ = ['bp_eval', 'bp_jobs', 'bp_perf', 'bp_reports', 'bp_tasks']
