"""CRUD operations for refunds."""

import datetime

from omnia_support_mcp.services.orders.service import get_order, update_order
from omnia_support_mcp.services.orders.store import OrderStatus
from omnia_support_mcp.services.refunds.store import REFUNDS, Refund, RefundStatus, _refund_counter


def get_refund(refund_id: str) -> Refund | None:
    """Get a refund by ID, or None if not found."""
    return REFUNDS.get(refund_id)


def get_refund_by_order(order_id: str) -> Refund | None:
    """Get the refund for an order, or None if none exists."""
    return next((r for r in REFUNDS.values() if r.order_id == order_id), None)


def list_refunds() -> list[Refund]:
    """List all refunds."""
    return list(REFUNDS.values())


def create_refund(order_id: str, reason: str) -> Refund:
    """Create a refund for an order, cancelling the order in the process."""
    global _refund_counter
    order = get_order(order_id)
    if not order:
        raise ValueError(f"Order {order_id} not found")
    if order.status == "cancelled":
        existing = next((r for r in REFUNDS.values() if r.order_id == order_id), None)
        if existing:
            raise ValueError(f"Refund {existing.refund_id} already exists for this order")
    if order.status == "processing":
        raise ValueError("Cannot refund an order that is still processing. Please cancel it first.")

    update_order(order_id, {"status": OrderStatus.CANCELLED})

    _refund_counter += 1
    refund_id = f"REF-2024-{_refund_counter:03d}"
    refund = Refund(
        refund_id=refund_id,
        order_id=order_id,
        status=RefundStatus.PENDING,
        amount=order.total,
        reason=reason,
        created_date=datetime.date.today().isoformat(),
    )
    REFUNDS[refund_id] = refund
    return refund


def update_refund(refund_id: str, updates: dict) -> Refund | None:
    """Update a refund's fields, or return None if not found."""
    refund = REFUNDS.get(refund_id)
    if not refund:
        return None
    updated = refund.model_copy(update=updates)
    REFUNDS[refund_id] = updated
    return updated


def delete_refund(refund_id: str) -> bool:
    """Delete a refund by ID. Returns True if it existed."""
    return REFUNDS.pop(refund_id, None) is not None
