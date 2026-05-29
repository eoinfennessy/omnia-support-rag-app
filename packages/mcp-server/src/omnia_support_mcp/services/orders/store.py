"""In-memory mock data store for orders."""

from enum import StrEnum

from pydantic import BaseModel


class OrderStatus(StrEnum):
    """Lifecycle status of an order."""

    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    """A line item within an order."""

    product_id: str
    name: str
    quantity: int
    price: float


class Order(BaseModel):
    """A customer order."""

    order_id: str
    customer_name: str
    status: OrderStatus
    tracking_number: str | None
    estimated_delivery: str | None
    delivered_date: str | None
    items: list[OrderItem]
    total: float


ORDERS: dict[str, Order] = {
    "ORD-2024-001": Order(
        order_id="ORD-2024-001",
        customer_name="Sarah Murphy",
        status=OrderStatus.DELIVERED,
        tracking_number="IE2024TRK00142",
        estimated_delivery="2024-11-15",
        delivered_date="2024-11-14",
        items=[
            OrderItem(product_id="PROD-001", name="ProBook 15 Laptop", quantity=1, price=899.99),
            OrderItem(product_id="PROD-005", name="USB-C Docking Station", quantity=1, price=149.99),
        ],
        total=1049.98,
    ),
    "ORD-2024-002": Order(
        order_id="ORD-2024-002",
        customer_name="Liam O'Brien",
        status=OrderStatus.SHIPPED,
        tracking_number="IE2024TRK00198",
        estimated_delivery="2024-12-03",
        delivered_date=None,
        items=[
            OrderItem(product_id="PROD-003", name='UltraWide 34" Monitor', quantity=1, price=549.99),
        ],
        total=549.99,
    ),
    "ORD-2024-003": Order(
        order_id="ORD-2024-003",
        customer_name="Aoife Byrne",
        status=OrderStatus.PROCESSING,
        tracking_number=None,
        estimated_delivery="2024-12-10",
        delivered_date=None,
        items=[
            OrderItem(product_id="PROD-002", name="ProBook 13 Laptop", quantity=1, price=1199.99),
            OrderItem(product_id="PROD-007", name="Wireless Keyboard & Mouse Combo", quantity=1, price=79.99),
        ],
        total=1279.98,
    ),
    "ORD-2024-004": Order(
        order_id="ORD-2024-004",
        customer_name="Cian Doyle",
        status=OrderStatus.DELIVERED,
        tracking_number="IE2024TRK00076",
        estimated_delivery="2024-10-20",
        delivered_date="2024-10-19",
        items=[
            OrderItem(product_id="PROD-004", name='24" Full HD Monitor', quantity=2, price=279.99),
        ],
        total=559.98,
    ),
    "ORD-2024-005": Order(
        order_id="ORD-2024-005",
        customer_name="Niamh Kelly",
        status=OrderStatus.CANCELLED,
        tracking_number=None,
        estimated_delivery=None,
        delivered_date=None,
        items=[
            OrderItem(product_id="PROD-006", name="1TB Portable SSD", quantity=2, price=109.99),
        ],
        total=219.98,
    ),
}

_order_counter = len(ORDERS)
