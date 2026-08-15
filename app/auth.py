"""
API key authentication dependency for FastAPI routes.

Usage
-----
Add ``dependencies=[Depends(require_api_key)]`` to any router decorator:

    @router.post("/chat", dependencies=[Depends(require_api_key)])
    def chat(req: ChatRequest): ...

Dev mode
--------
If ``API_ACCESS_KEY`` is not set (empty string in .env), the check is
bypassed entirely so local development works without any configuration.
"""
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app import config

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """
    Validate the ``X-API-Key`` request header against ``API_ACCESS_KEY``
    in the environment configuration.

    Raises ``401 Unauthorized`` when authentication fails.
    Passes silently when ``API_ACCESS_KEY`` is empty (dev mode).
    """
    if not config.API_ACCESS_KEY:
        return

    if api_key != config.API_ACCESS_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def get_current_user_id(
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> str:
    """
    Extract the caller's user identity from the ``X-User-ID`` request header.

    Returns ``"default_user"`` when the header is absent so that local
    development and single-user deployments work without any extra config.
    All Qdrant operations are scoped to this value so documents uploaded by
    one user are never visible to another.
    """
    if not x_user_id or not x_user_id.strip():
        return "default_user"
    return x_user_id.strip()
