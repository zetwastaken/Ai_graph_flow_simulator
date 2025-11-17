"""Authentication helpers for the FastAPI layer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from jose import JWTError, jwt


class JWTAuthenticator:
    """Validates JWT tokens using a provided key or leaves endpoints open."""

    def __init__(self, public_key: Optional[str], audience: Optional[str] = None, algorithms=None):
        self.public_key = public_key
        self.audience = audience
        self.algorithms = algorithms or ["RS256", "HS256"]

    def verify(self, token: Optional[str]) -> Dict[str, Any]:
        if not self.public_key:
            return {}
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=self.algorithms,
                audience=self.audience,
                options={"verify_aud": bool(self.audience)},
            )
            return payload
        except JWTError as exc:  # pragma: no cover - depends on tokens
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
