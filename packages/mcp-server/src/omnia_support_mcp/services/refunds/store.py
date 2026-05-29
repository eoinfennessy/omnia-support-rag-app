"""In-memory mock data store for refunds."""

from enum import StrEnum

from pydantic import BaseModel


class RefundStatus(StrEnum):
    """Lifecycle status of a refund."""

    PENDING = "pending"
    COMPLETED = "completed"
    REJECTED = "rejected"


class Refund(BaseModel):
    """A refund for an order."""

    refund_id: str
    order_id: str
    status: RefundStatus
    amount: float
    reason: str
    created_date: str


REFUNDS: dict[str, Refund] = {
    "REF-2024-001": Refund(
        refund_id="REF-2024-001",
        order_id="ORD-2024-005",
        status=RefundStatus.COMPLETED,
        amount=219.98,
        reason="Customer requested cancellation",
        created_date="2024-10-25",
    ),
}

_refund_counter = len(REFUNDS)
