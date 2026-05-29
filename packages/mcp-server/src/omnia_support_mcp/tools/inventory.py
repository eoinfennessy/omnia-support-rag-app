"""Inventory tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnia_support_mcp.services.inventory import service

if TYPE_CHECKING:
    from omnia_support_mcp.services.inventory.store import Product


def get_product(product_id: str) -> Product:
    """Get a product by its ID.

    Args:
        product_id: The product identifier (e.g. PROD-001).

    Returns:
        Product details.

    Raises:
        ValueError: If the product is not found.
    """
    product = service.get_product(product_id)
    if not product:
        raise ValueError(f"Product '{product_id}' not found.")
    return product


def search_product(product_name: str) -> Product:
    """Search for a product by name (case-insensitive).

    Args:
        product_name: The product name to search for.

    Returns:
        Product details.

    Raises:
        ValueError: If no product matches the name.
    """
    product = service.get_product_by_name(product_name)
    if not product:
        raise ValueError(f"No product matching '{product_name}' found.")
    return product
