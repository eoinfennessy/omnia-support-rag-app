"""Unit tests for MCP tool handlers (called directly, not via MCP transport)."""

from __future__ import annotations

import pytest

from omnia_support_mcp.tools.inventory import get_product, search_product
from omnia_support_mcp.tools.orders import get_order, list_customer_orders
from omnia_support_mcp.tools.refunds import get_refund, process_refund


class TestGetOrder:
    def test_returns_order_details(self):
        order = get_order("ORD-2024-001")
        assert order.order_id == "ORD-2024-001"
        assert order.status == "delivered"
        assert order.tracking_number == "IE2024TRK00142"
        assert len(order.items) == 2

    def test_shipped_order_has_tracking(self):
        order = get_order("ORD-2024-002")
        assert order.status == "shipped"
        assert order.tracking_number is not None
        assert order.delivered_date is None

    def test_processing_order_has_no_tracking(self):
        order = get_order("ORD-2024-003")
        assert order.status == "processing"
        assert order.tracking_number is None

    def test_unknown_order_raises(self):
        with pytest.raises(ValueError, match="ORD-FAKE-999"):
            get_order("ORD-FAKE-999")


class TestListCustomerOrders:
    def test_finds_orders_by_customer_name(self):
        orders = list_customer_orders("Sarah Murphy")
        assert len(orders) == 1
        assert orders[0].order_id == "ORD-2024-001"

    def test_case_insensitive(self):
        orders = list_customer_orders("sarah")
        assert len(orders) == 1

    def test_unknown_customer_raises(self):
        with pytest.raises(ValueError, match="No orders found"):
            list_customer_orders("Nobody")


class TestProcessRefund:
    def test_creates_refund_for_delivered_order(self):
        refund = process_refund("ORD-2024-001", "Item arrived damaged")
        assert refund.refund_id.startswith("REF-")
        assert refund.status == "pending"
        assert refund.amount == 1049.98

    def test_rejects_refund_for_unknown_order(self):
        with pytest.raises(ValueError, match="not found"):
            process_refund("ORD-FAKE-999", "want refund")

    def test_rejects_refund_for_processing_order(self):
        with pytest.raises(ValueError, match="processing"):
            process_refund("ORD-2024-003", "changed mind")

    def test_rejects_duplicate_refund_for_cancelled_order(self):
        with pytest.raises(ValueError, match="already exists"):
            process_refund("ORD-2024-005", "already cancelled")


class TestGetRefund:
    def test_finds_existing_refund(self):
        refund = get_refund("ORD-2024-005")
        assert refund.refund_id == "REF-2024-001"
        assert refund.status == "completed"

    def test_unknown_order_raises(self):
        with pytest.raises(ValueError, match="No refund found"):
            get_refund("ORD-FAKE-999")


class TestGetStock:
    def test_lookup_by_product_id(self):
        product = get_product("PROD-001")
        assert product.name == "ProBook 15 Laptop"
        assert product.in_stock is True
        assert product.stock_quantity == 23

    def test_out_of_stock_product(self):
        product = get_product("PROD-004")
        assert product.in_stock is False
        assert product.stock_quantity == 0

    def test_unknown_product_id_raises(self):
        with pytest.raises(ValueError, match="PROD-999"):
            get_product("PROD-999")


class TestSearchProduct:
    def test_lookup_by_name_exact(self):
        product = search_product("1TB Portable SSD")
        assert product.product_id == "PROD-006"
        assert product.category == "storage"

    def test_lookup_by_name_case_insensitive(self):
        product = search_product("PROBOOK 15 LAPTOP")
        assert product.product_id == "PROD-001"

    def test_unknown_product_name_raises(self):
        with pytest.raises(ValueError, match="Quantum Computer"):
            search_product("Quantum Computer")
