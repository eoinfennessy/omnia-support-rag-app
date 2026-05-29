"""Refund tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnia_support_mcp.services.refunds import service

if TYPE_CHECKING:
    from omnia_support_mcp.services.refunds.store import Refund


def process_refund(order_id: str, reason: str) -> Refund:
    """Process a refund request for an order.

    Args:
        order_id: The order identifier to refund (e.g. ORD-2024-001).
        reason: The reason for the refund request.

    Returns:
        Refund details.

    Raises:
        ValueError: If the order is not found or is not eligible for a refund.
    """
    return service.create_refund(order_id, reason)


def get_refund(order_id: str) -> Refund:
    """Get the refund for an order.

    Args:
        order_id: The order identifier (e.g. ORD-2024-001).

    Returns:
        Refund details.

    Raises:
        ValueError: If no refund exists for the order.
    """
    refund = service.get_refund_by_order(order_id)
    if not refund:
        raise ValueError(f"No refund found for order '{order_id}'.")
    return refund
