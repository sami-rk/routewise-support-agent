"""Long-term memory endpoints.

Reading and erasing a customer's memory is treated as staff-only, because the
summary is built from their conversations.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from ..memory.long_term import clear_memory, get_memory
from .deps import require_admin

router = APIRouter(tags=["memory"], dependencies=[Depends(require_admin)])


@router.get("/customers/{customer_id}/memory")
def read_memory(customer_id: str) -> dict[str, Any]:
    """What the agent remembers about a customer."""
    from ..tools.customers import get_customer

    if get_customer(customer_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No customer {customer_id!r}."
        )
    return {"customer_id": customer_id, "summary": get_memory(customer_id)}


@router.delete("/customers/{customer_id}/memory")
def erase_memory(customer_id: str) -> dict[str, Any]:
    """Forget a customer. Returns whether there was anything to forget."""
    return {"customer_id": customer_id, "erased": clear_memory(customer_id)}
