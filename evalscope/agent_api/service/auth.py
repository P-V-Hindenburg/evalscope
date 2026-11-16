"""Bearer-token authentication for cross-service calls."""

from hmac import compare_digest
from typing import Callable, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .settings import ServiceSettings

_BEARER = HTTPBearer(auto_error=False)


def build_auth_dependency(settings: ServiceSettings) -> Callable:
    """Create a FastAPI dependency bound to one configured service token."""

    async def authenticate(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(_BEARER),
    ) -> None:
        if not settings.service_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    'code': 'SERVICE_TOKEN_NOT_CONFIGURED',
                    'message': 'EVALSCOPE_SERVICE_TOKEN is not configured',
                },
            )
        if credentials is None or credentials.scheme.lower() != 'bearer':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={'code': 'AUTHENTICATION_REQUIRED', 'message': 'Bearer token required'},
                headers={'WWW-Authenticate': 'Bearer'},
            )
        if not compare_digest(credentials.credentials, settings.service_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={'code': 'INVALID_SERVICE_TOKEN', 'message': 'Invalid service token'},
                headers={'WWW-Authenticate': 'Bearer'},
            )

    return authenticate
