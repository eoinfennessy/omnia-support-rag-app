"""In-memory mock data store for inventory."""

from enum import StrEnum

from pydantic import BaseModel, computed_field

from omnia_support_models import ProductCategory


class WarehouseLocation(StrEnum):
    """Warehouse locations."""

    DUBLIN = "Dublin"
    CORK = "Cork"


class Product(BaseModel):
    """A product in the inventory."""

    product_id: str
    name: str
    category: ProductCategory
    price: float
    stock_quantity: int
    warehouse_location: WarehouseLocation

    @computed_field
    @property
    def in_stock(self) -> bool:
        """Whether the product is currently in stock."""
        return self.stock_quantity > 0


PRODUCTS: dict[str, Product] = {
    "PROD-001": Product(
        product_id="PROD-001",
        name="ProBook 15 Laptop",
        category=ProductCategory.LAPTOPS,
        price=899.99,
        stock_quantity=23,
        warehouse_location=WarehouseLocation.DUBLIN,
    ),
    "PROD-002": Product(
        product_id="PROD-002",
        name="ProBook 13 Laptop",
        category=ProductCategory.LAPTOPS,
        price=1199.99,
        stock_quantity=8,
        warehouse_location=WarehouseLocation.DUBLIN,
    ),
    "PROD-003": Product(
        product_id="PROD-003",
        name='UltraWide 34" Monitor',
        category=ProductCategory.MONITORS,
        price=549.99,
        stock_quantity=15,
        warehouse_location=WarehouseLocation.CORK,
    ),
    "PROD-004": Product(
        product_id="PROD-004",
        name='24" Full HD Monitor',
        category=ProductCategory.MONITORS,
        price=279.99,
        stock_quantity=0,
        warehouse_location=WarehouseLocation.DUBLIN,
    ),
    "PROD-005": Product(
        product_id="PROD-005",
        name="USB-C Docking Station",
        category=ProductCategory.ACCESSORIES,
        price=149.99,
        stock_quantity=42,
        warehouse_location=WarehouseLocation.DUBLIN,
    ),
    "PROD-006": Product(
        product_id="PROD-006",
        name="1TB Portable SSD",
        category=ProductCategory.STORAGE,
        price=109.99,
        stock_quantity=67,
        warehouse_location=WarehouseLocation.CORK,
    ),
    "PROD-007": Product(
        product_id="PROD-007",
        name="Wireless Keyboard & Mouse Combo",
        category=ProductCategory.ACCESSORIES,
        price=79.99,
        stock_quantity=31,
        warehouse_location=WarehouseLocation.DUBLIN,
    ),
}

PRODUCT_NAME_INDEX: dict[str, str] = {p.name.lower(): pid for pid, p in PRODUCTS.items()}

_product_counter = len(PRODUCTS)
