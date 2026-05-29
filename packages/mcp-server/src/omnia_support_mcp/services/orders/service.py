"""CRUD operations for orders."""

from omnia_support_mcp.services.orders.store import ORDERS, Order, _order_counter


def get_order(order_id: str) -> Order | None:
    """Get an order by ID, or None if not found."""
    return ORDERS.get(order_id)


def get_orders_by_customer(customer_name: str) -> list[Order]:
    """Get all orders matching a customer name (case-insensitive)."""
    name_lower = customer_name.lower()
    return [o for o in ORDERS.values() if name_lower in o.customer_name.lower()]


def list_orders() -> list[Order]:
    """List all orders."""
    return list(ORDERS.values())


def create_order(order: Order) -> Order:
    """Create a new order and assign it an ID."""
    global _order_counter
    _order_counter += 1
    order_id = f"ORD-2024-{_order_counter:03d}"
    order = order.model_copy(update={"order_id": order_id})
    ORDERS[order_id] = order
    return order


def update_order(order_id: str, updates: dict) -> Order | None:
    """Update an order's fields, or return None if not found."""
    order = ORDERS.get(order_id)
    if not order:
        return None
    updated = order.model_copy(update=updates)
    ORDERS[order_id] = updated
    return updated


def delete_order(order_id: str) -> bool:
    """Delete an order by ID. Returns True if it existed."""
    return ORDERS.pop(order_id, None) is not None
