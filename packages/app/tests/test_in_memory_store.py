"""Tests for the in-memory session store."""

from __future__ import annotations

from omnia_support_app.session import InMemorySessionStore


class TestGetHistory:
    def test_returns_empty_list_for_unknown_session(self):
        store = InMemorySessionStore()

        assert store.get_history("unknown") == []

    def test_returns_copy_not_reference(self):
        store = InMemorySessionStore()
        store.add_messages("s1", [{"role": "user", "content": "hi"}])

        history = store.get_history("s1")
        history.append({"role": "assistant", "content": "injected"})

        assert len(store.get_history("s1")) == 1


class TestAddMessages:
    def test_single_message(self):
        store = InMemorySessionStore()

        store.add_messages("s1", [{"role": "user", "content": "hello"}])

        history = store.get_history("s1")
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "hello"}

    def test_multiple_messages_in_one_call(self):
        store = InMemorySessionStore()

        store.add_messages(
            "s1",
            [
                {"role": "developer", "content": "context"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )

        assert len(store.get_history("s1")) == 3

    def test_accumulates_across_calls(self):
        store = InMemorySessionStore()

        store.add_messages("s1", [{"role": "user", "content": "first"}])
        store.add_messages("s1", [{"role": "user", "content": "second"}])

        assert len(store.get_history("s1")) == 2

    def test_sessions_are_isolated(self):
        store = InMemorySessionStore()

        store.add_messages("s1", [{"role": "user", "content": "from s1"}])
        store.add_messages("s2", [{"role": "user", "content": "from s2"}])

        assert len(store.get_history("s1")) == 1
        assert len(store.get_history("s2")) == 1
        assert store.get_history("s1")[0]["content"] == "from s1"


class TestClear:
    def test_clears_existing_session(self):
        store = InMemorySessionStore()
        store.add_messages("s1", [{"role": "user", "content": "hi"}])

        store.clear("s1")

        assert store.get_history("s1") == []

    def test_clear_unknown_session_is_noop(self):
        store = InMemorySessionStore()

        store.clear("nonexistent")

    def test_clear_does_not_affect_other_sessions(self):
        store = InMemorySessionStore()
        store.add_messages("s1", [{"role": "user", "content": "hi"}])
        store.add_messages("s2", [{"role": "user", "content": "hi"}])

        store.clear("s1")

        assert store.get_history("s1") == []
        assert len(store.get_history("s2")) == 1
