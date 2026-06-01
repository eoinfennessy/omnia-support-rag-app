"""Tests for aggregator agent evaluation: gold set loading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from omnia_support_eval.aggregator.gold_set import (
    AggregatorGoldSetEntry,
    load_gold_set,
)

if TYPE_CHECKING:
    from pathlib import Path


_MINIMAL_GOLD_SET = {
    "aggregator_queries": [
        {
            "query_id": "AGG-001",
            "query": "Where is my order and can I return it?",
            "expected_answer": "Your order is in transit. Returns are accepted within 30 days.",
            "information_inputs": [
                {
                    "query": "Can I return items?",
                    "answer": "Returns accepted within 30 days.",
                    "context": [
                        {"page_content": "Standard Return Period: 30 days from delivery date"},
                    ],
                },
            ],
            "action_inputs": [
                {
                    "query": "Where is my order?",
                    "answer": "Your order is in transit.",
                    "tool_calls": [
                        {
                            "tool_name": "get_order",
                            "arguments": {"order_id": "ORD-2024-002"},
                            "result": {"data": {"order_id": "ORD-2024-002", "status": "shipped"}},
                        },
                    ],
                },
            ],
            "category": "multi-mixed",
        },
    ],
}


class TestAggregatorGoldSet:
    def test_loads_entries(self, tmp_path: Path):
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps(_MINIMAL_GOLD_SET))

        entries = load_gold_set(gold_set_path)

        assert len(entries) == 1
        assert isinstance(entries[0], AggregatorGoldSetEntry)

    def test_entry_has_correct_fields(self, tmp_path: Path):
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps(_MINIMAL_GOLD_SET))

        entry = load_gold_set(gold_set_path)[0]

        assert entry.query_id == "AGG-001"
        assert entry.query == "Where is my order and can I return it?"
        assert entry.expected_answer == "Your order is in transit. Returns are accepted within 30 days."
        assert len(entry.information_inputs) == 1
        assert len(entry.action_inputs) == 1
        assert entry.category == "multi-mixed"

    def test_information_input_has_context(self, tmp_path: Path):
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps(_MINIMAL_GOLD_SET))

        entry = load_gold_set(gold_set_path)[0]
        info = entry.information_inputs[0]

        assert info.query == "Can I return items?"
        assert len(info.context) == 1
        assert info.context[0].page_content == "Standard Return Period: 30 days from delivery date"

    def test_action_input_has_tool_calls(self, tmp_path: Path):
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps(_MINIMAL_GOLD_SET))

        entry = load_gold_set(gold_set_path)[0]
        action = entry.action_inputs[0]

        assert action.query == "Where is my order?"
        assert len(action.tool_calls) == 1
        assert action.tool_calls[0].tool_name == "get_order"
        assert action.tool_calls[0].arguments == {"order_id": "ORD-2024-002"}

    def test_raises_system_exit_when_file_missing(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="gold set file not found"):
            load_gold_set(tmp_path / "missing.json")

    def test_raises_system_exit_when_no_entries(self, tmp_path: Path):
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps({"aggregator_queries": []}))

        with pytest.raises(SystemExit, match="no aggregator_queries"):
            load_gold_set(gold_set_path)

    def test_loads_multiple_entries(self, tmp_path: Path):
        data = {
            "aggregator_queries": [
                {
                    "query_id": "AGG-001",
                    "query": "Q1",
                    "expected_answer": "A1",
                    "information_inputs": [{"query": "q", "answer": "a"}],
                    "action_inputs": [],
                },
                {
                    "query_id": "AGG-002",
                    "query": "Q2",
                    "expected_answer": "A2",
                    "information_inputs": [],
                    "action_inputs": [{"query": "q", "answer": "a"}],
                },
            ],
        }
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps(data))

        entries = load_gold_set(gold_set_path)

        assert len(entries) == 2
        assert entries[0].query_id == "AGG-001"
        assert entries[1].query_id == "AGG-002"

    def test_defaults_context_and_tool_calls_to_empty(self, tmp_path: Path):
        data = {
            "aggregator_queries": [
                {
                    "query_id": "AGG-001",
                    "query": "Q1",
                    "expected_answer": "A1",
                    "information_inputs": [{"query": "q", "answer": "a"}],
                    "action_inputs": [{"query": "q", "answer": "a"}],
                },
            ],
        }
        gold_set_path = tmp_path / "gold_set.json"
        gold_set_path.write_text(json.dumps(data))

        entry = load_gold_set(gold_set_path)[0]

        assert entry.information_inputs[0].context == []
        assert entry.action_inputs[0].tool_calls == []
