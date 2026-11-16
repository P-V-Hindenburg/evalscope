"""Environment-backed settings for the agent-facing HTTP service."""

import os
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ServiceSettings:
    """Security-sensitive settings loaded once when the service starts."""

    service_token: str
    allowed_endpoint_hosts: FrozenSet[str]

    @classmethod
    def from_environment(cls) -> 'ServiceSettings':
        """Build settings without exposing secret values in logs or responses."""

        hosts = {
            item.strip().lower()
            for item in os.getenv('EVALSCOPE_ALLOWED_ENDPOINT_HOSTS', '').split(',')
            if item.strip()
        }
        return cls(
            service_token=os.getenv('EVALSCOPE_SERVICE_TOKEN', ''),
            allowed_endpoint_hosts=frozenset(hosts),
        )
