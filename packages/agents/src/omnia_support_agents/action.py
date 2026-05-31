"""Utilities for building ActionResult from an Agents SDK RunResult."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

from .models import ToolCall, ToolCallResultErr, ToolCallResultOk

if TYPE_CHECKING:
    from agents.result import RunResult
    from pydantic import JsonValue


class _CallInfo(NamedTuple):
    name: str
    arguments: dict[str, JsonValue]
    raw_item: Any = None


def _parse_arguments(raw_item: Any) -> dict[str, JsonValue]:
    raw = getattr(raw_item, "arguments", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


def _parse_output(output: Any) -> JsonValue:
    """Parse JSON string output into a Python object, or pass through as-is."""
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    return output


def extract_tool_calls(result: RunResult) -> list[ToolCall]:
    """Extract ToolCall objects from an Agents SDK RunResult."""
    from agents.items import ToolCallItem, ToolCallOutputItem

    call_info: dict[str, _CallInfo] = {}
    completed: set[str] = set()
    tool_calls: list[ToolCall] = []

    for item in result.new_items:
        if isinstance(item, ToolCallItem) and item.tool_name and item.call_id:
            call_info[item.call_id] = _CallInfo(item.tool_name, _parse_arguments(item.raw_item), item.raw_item)
        elif isinstance(item, ToolCallOutputItem) and item.call_id:
            completed.add(item.call_id)
            info = call_info.get(item.call_id, _CallInfo("unknown", {}))
            tool_calls.append(
                ToolCall(
                    tool_name=info.name,
                    arguments=info.arguments,
                    result=ToolCallResultOk(data=_parse_output(item.output)),
                )
            )

    for call_id, info in call_info.items():
        if call_id in completed:
            continue
        error = getattr(info.raw_item, "error", None)
        tool_calls.append(
            ToolCall(
                tool_name=info.name,
                arguments=info.arguments,
                result=ToolCallResultErr(error=error or "Tool call failed"),
            )
        )

    return tool_calls
