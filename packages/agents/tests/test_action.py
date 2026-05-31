"""Tests for the action agent utilities."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from omnia_support_agents.action import extract_tool_calls
from omnia_support_agents.models import ToolCallResultErr, ToolCallResultOk


def _make_run_result(*, final_output: str, new_items: list | None = None) -> MagicMock:
    result = MagicMock()
    result.final_output = final_output
    result.new_items = new_items or []
    return result


def _make_tool_call_item(call_id: str, tool_name: str, arguments: dict | None = None) -> MagicMock:
    from agents.items import ToolCallItem

    item = MagicMock(spec=ToolCallItem)
    item.call_id = call_id
    item.tool_name = tool_name
    item.raw_item = MagicMock()
    item.raw_item.arguments = json.dumps(arguments or {})
    return item


def _make_tool_output_item(call_id: str, output: str | dict) -> MagicMock:
    from agents.items import ToolCallOutputItem

    item = MagicMock(spec=ToolCallOutputItem)
    item.call_id = call_id
    item.output = output
    return item


class TestExtractToolCalls:
    def test_extracts_single_tool_call(self):
        call_item = _make_tool_call_item("call_1", "get_order", {"order_id": "ORD-2024-001"})
        output_item = _make_tool_output_item(
            "call_1",
            json.dumps({"order_id": "ORD-2024-001", "status": "delivered"}),
        )

        result = _make_run_result(final_output="ok", new_items=[call_item, output_item])
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "get_order"
        assert tool_calls[0].arguments == {"order_id": "ORD-2024-001"}
        assert isinstance(tool_calls[0].result, ToolCallResultOk)
        assert tool_calls[0].result.data == {"order_id": "ORD-2024-001", "status": "delivered"}

    def test_extracts_multiple_tool_calls(self):
        call_1 = _make_tool_call_item("call_1", "get_order", {"order_id": "ORD-2024-001"})
        call_2 = _make_tool_call_item("call_2", "process_refund", {"order_id": "ORD-2024-001", "reason": "damaged"})
        output_1 = _make_tool_output_item("call_1", json.dumps({"order_id": "ORD-2024-001"}))
        output_2 = _make_tool_output_item("call_2", json.dumps({"refund_id": "REF-2024-001"}))

        result = _make_run_result(
            final_output="ok",
            new_items=[call_1, call_2, output_1, output_2],
        )
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 2
        assert tool_calls[0].tool_name == "get_order"
        assert tool_calls[0].arguments == {"order_id": "ORD-2024-001"}
        assert tool_calls[1].tool_name == "process_refund"
        assert tool_calls[1].arguments == {"order_id": "ORD-2024-001", "reason": "damaged"}

    def test_handles_plain_text_output(self):
        call_item = _make_tool_call_item("call_1", "get_order", {"order_id": "ORD-999"})
        output_item = _make_tool_output_item("call_1", "Order not found")

        result = _make_run_result(final_output="ok", new_items=[call_item, output_item])
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {"order_id": "ORD-999"}
        assert isinstance(tool_calls[0].result, ToolCallResultOk)
        assert tool_calls[0].result.data == "Order not found"

    def test_handles_dict_output_directly(self):
        call_item = _make_tool_call_item("call_1", "get_product", {"product_id": "PROD-001"})
        output_item = _make_tool_output_item("call_1", {"product_id": "PROD-001", "name": "UltraBook Pro 15"})

        result = _make_run_result(final_output="ok", new_items=[call_item, output_item])
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0].result, ToolCallResultOk)
        assert tool_calls[0].result.data == {"product_id": "PROD-001", "name": "UltraBook Pro 15"}

    def test_empty_new_items(self):
        result = _make_run_result(final_output="ok", new_items=[])
        tool_calls = extract_tool_calls(result)
        assert tool_calls == []

    def test_failed_mcp_call_with_error(self):
        call_item = _make_tool_call_item("call_1", "process_refund", {"order_id": "ORD-999"})
        call_item.raw_item.error = "MCP server unreachable"
        call_item.raw_item.status = "failed"

        result = _make_run_result(final_output="ok", new_items=[call_item])
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "process_refund"
        assert isinstance(tool_calls[0].result, ToolCallResultErr)
        assert tool_calls[0].result.error == "MCP server unreachable"

    def test_failed_mcp_call_without_error_message(self):
        call_item = _make_tool_call_item("call_1", "get_order", {"order_id": "ORD-999"})
        call_item.raw_item.error = None
        call_item.raw_item.status = "failed"

        result = _make_run_result(final_output="ok", new_items=[call_item])
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0].result, ToolCallResultErr)
        assert tool_calls[0].result.error == "Tool call failed"

    def test_unknown_tool_name_when_no_matching_call(self):
        output_item = _make_tool_output_item("call_orphan", json.dumps({"data": "value"}))

        result = _make_run_result(final_output="ok", new_items=[output_item])
        tool_calls = extract_tool_calls(result)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "unknown"
        assert tool_calls[0].arguments == {}
