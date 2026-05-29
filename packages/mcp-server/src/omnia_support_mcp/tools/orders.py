"""Order tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnia_support_mcp.services.orders import service

if TYPE_CHECKING:
    from omnia_support_mcp.services.orders.store import Order


def get_order(order_id: str) -> Order:
    """Get an order by its ID.

    Args:
        order_id: The order identifier (e.g. ORD-2024-001).

    Returns:
        Order details.

    Raises:
        ValueError: If the order ID is not found.
    """
    order = service.get_order(order_id)
    if not order:
        raise ValueError(f"Order '{order_id}' not found. Please check the order ID and try again.")
    return order


def list_customer_orders(customer_name: str) -> list[Order]:
    """List all orders for a customer by name (case-insensitive).

    Args:
        customer_name: The customer name to search for.

    Returns:
        List of matching orders.

    Raises:
        ValueError: If no orders are found for the customer.
    """
    orders = service.get_orders_by_customer(customer_name)
    if not orders:
        raise ValueError(f"No orders found for customer '{customer_name}'.")
    return orders
