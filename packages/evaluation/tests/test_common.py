"""Tests for shared evaluation infrastructure."""

from __future__ import annotations

import hashlib

from omnia_support_eval.common import content_hash


class TestContentHash:
    def test_returns_sha256_hex_digest(self):
        expected = hashlib.sha256(b"hello").hexdigest()

        assert content_hash("hello") == expected

    def test_different_inputs_produce_different_hashes(self):
        assert content_hash("a") != content_hash("b")

    def test_deterministic(self):
        assert content_hash("test") == content_hash("test")
