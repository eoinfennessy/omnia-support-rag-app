"""CRUD operations for inventory."""

from omnia_support_mcp.services.inventory.store import (
    PRODUCT_NAME_INDEX,
    PRODUCTS,
    Product,
    _product_counter,
)


def get_product(product_id: str) -> Product | None:
    """Get a product by ID, or None if not found."""
    return PRODUCTS.get(product_id)


def get_product_by_name(name: str) -> Product | None:
    """Get a product by exact name (case-insensitive), or None if not found."""
    product_id = PRODUCT_NAME_INDEX.get(name.lower())
    if product_id:
        return PRODUCTS[product_id]
    return None


def list_products() -> list[Product]:
    """List all products."""
    return list(PRODUCTS.values())


def create_product(product: Product) -> Product:
    """Create a new product and assign it an ID."""
    global _product_counter
    _product_counter += 1
    product_id = f"PROD-{_product_counter:03d}"
    product = product.model_copy(update={"product_id": product_id})
    PRODUCTS[product_id] = product
    PRODUCT_NAME_INDEX[product.name.lower()] = product_id
    return product


def update_product(product_id: str, updates: dict) -> Product | None:
    """Update a product's fields, or return None if not found."""
    product = PRODUCTS.get(product_id)
    if not product:
        return None
    old_name = product.name.lower()
    updated = product.model_copy(update=updates)
    PRODUCTS[product_id] = updated
    if "name" in updates:
        del PRODUCT_NAME_INDEX[old_name]
        PRODUCT_NAME_INDEX[updated.name.lower()] = product_id
    return updated


def delete_product(product_id: str) -> bool:
    """Delete a product by ID. Returns True if it existed."""
    product = PRODUCTS.pop(product_id, None)
    if not product:
        return False
    PRODUCT_NAME_INDEX.pop(product.name.lower(), None)
    return True
