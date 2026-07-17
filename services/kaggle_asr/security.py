from __future__ import annotations

import hmac

from fastapi import HTTPException, status


def verify_bearer_token(
    authorization: str | None,
    *,
    expected_token: str,
) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
        )
