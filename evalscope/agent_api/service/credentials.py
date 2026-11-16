"""Safe credential-reference resolution for the HTTP service."""

import json
import os
from typing import Dict, Mapping, Optional


class CredentialReferenceError(ValueError):
    """Raised when a request uses an unknown credential reference."""


class CredentialUnavailableError(RuntimeError):
    """Raised when a known credential has no configured secret value."""


DEFAULT_CREDENTIAL_ENV_MAP = {
    'deepseek': 'DEEPSEEK_API_KEY',
    'doubao': 'ARK_API_KEY',
}


class EnvironmentCredentialResolver:
    """Resolve allow-listed logical names to server-side environment variables."""

    def __init__(self, mapping: Optional[Mapping[str, str]] = None):
        self._mapping = dict(mapping or DEFAULT_CREDENTIAL_ENV_MAP)

    @classmethod
    def from_environment(cls) -> 'EnvironmentCredentialResolver':
        """Load an optional JSON mapping without reading the secret values yet."""

        mapping: Dict[str, str] = dict(DEFAULT_CREDENTIAL_ENV_MAP)
        configured = os.getenv('EVALSCOPE_CREDENTIAL_MAP')
        if configured:
            try:
                payload = json.loads(configured)
            except json.JSONDecodeError as exc:
                raise CredentialUnavailableError('EVALSCOPE_CREDENTIAL_MAP must be valid JSON') from exc
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str) and key and value for key, value in payload.items()
            ):
                raise CredentialUnavailableError('EVALSCOPE_CREDENTIAL_MAP must map names to environment variables')
            mapping.update(payload)
        return cls(mapping)

    def __call__(self, credential_ref: str) -> str:
        env_name = self._mapping.get(credential_ref)
        if env_name is None:
            raise CredentialReferenceError(f'unknown credential_ref: {credential_ref!r}')
        value = os.getenv(env_name)
        if not value:
            raise CredentialUnavailableError(f'credential {credential_ref!r} is not configured on the server')
        return value
