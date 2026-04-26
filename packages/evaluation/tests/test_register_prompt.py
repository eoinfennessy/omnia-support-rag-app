"""Tests for the prompt registration module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from omnia_support_eval.register_prompt import _templates_equal, register

if TYPE_CHECKING:
    from pathlib import Path

_VALID_TEMPLATE = [
    {"role": "system", "content": "You are a helper."},
    {"role": "user", "content": "{{question}}"},
]

_VALID_YAML = "- role: system\n  content: You are a helper.\n- role: user\n  content: '{{question}}'\n"


class TestTemplatesEqual:
    def test_equal_templates(self):
        a = [{"role": "system", "content": "hello"}]
        b = [{"role": "system", "content": "hello"}]

        assert _templates_equal(a, b) is True

    def test_different_templates(self):
        a = [{"role": "system", "content": "hello"}]
        b = [{"role": "system", "content": "goodbye"}]

        assert _templates_equal(a, b) is False

    def test_key_order_irrelevant(self):
        a = [{"content": "x", "role": "user"}]
        b = [{"role": "user", "content": "x"}]

        assert _templates_equal(a, b) is True

    def test_different_number_of_messages(self):
        a = [{"role": "system", "content": "a"}]
        b = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]

        assert _templates_equal(a, b) is False


class TestRegisterValidation:
    def test_raises_system_exit_for_missing_file(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="prompt file not found"):
            register(tmp_path / "missing.yaml", "test", alias=None)

    def test_raises_system_exit_for_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("")

        with pytest.raises(SystemExit, match="prompt file is empty"):
            register(path, "test", alias=None)

    def test_raises_system_exit_for_invalid_structure(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("key: value\n")

        with pytest.raises(SystemExit, match="must be a list of messages"):
            register(path, "test", alias=None)

    def test_raises_system_exit_for_messages_missing_role(self, tmp_path: Path):
        path = tmp_path / "no_role.yaml"
        path.write_text("- content: hello\n")

        with pytest.raises(SystemExit, match="must be a list of messages"):
            register(path, "test", alias=None)


@patch("omnia_support_eval.register_prompt.mlflow.genai.set_prompt_alias")
@patch("omnia_support_eval.register_prompt.mlflow.genai.register_prompt")
@patch("omnia_support_eval.register_prompt._load_prompt")
class TestRegister:
    def test_registers_new_prompt(self, mock_load, mock_register, mock_alias, tmp_path: Path):
        path = tmp_path / "prompt.yaml"
        path.write_text(_VALID_YAML)
        mock_load.return_value = None
        registered = MagicMock()
        registered.version = 1
        mock_register.return_value = registered

        register(path, "test_prompt", alias="prod")

        mock_register.assert_called_once()
        call_kwargs = mock_register.call_args
        assert call_kwargs.kwargs["name"] == "test_prompt"
        assert isinstance(call_kwargs.kwargs["template"], list)
        mock_alias.assert_called_once_with(name="test_prompt", alias="prod", version=1)

    def test_skips_registration_when_template_unchanged(self, mock_load, mock_register, mock_alias, tmp_path: Path):
        path = tmp_path / "prompt.yaml"
        path.write_text(_VALID_YAML)
        existing = MagicMock()
        existing.template = _VALID_TEMPLATE
        existing.version = 3

        def load_side_effect(_name, *, alias=None):
            if alias == "prod":
                different_version = MagicMock()
                different_version.version = 2
                return different_version
            return existing

        mock_load.side_effect = load_side_effect

        register(path, "test_prompt", alias="prod")

        mock_register.assert_not_called()
        mock_alias.assert_called_once_with(name="test_prompt", alias="prod", version=3)

    def test_skips_alias_when_already_set(self, mock_load, mock_register, mock_alias, tmp_path: Path):
        path = tmp_path / "prompt.yaml"
        path.write_text(_VALID_YAML)
        existing = MagicMock()
        existing.template = _VALID_TEMPLATE
        existing.version = 3

        mock_load.return_value = existing

        register(path, "test_prompt", alias="prod")

        mock_register.assert_not_called()
        mock_alias.assert_not_called()

    def test_no_alias_when_alias_is_none(self, mock_load, mock_register, mock_alias, tmp_path: Path):
        path = tmp_path / "prompt.yaml"
        path.write_text(_VALID_YAML)
        mock_load.return_value = None
        registered = MagicMock()
        registered.version = 1
        mock_register.return_value = registered

        register(path, "test_prompt", alias=None)

        mock_register.assert_called_once()
        mock_alias.assert_not_called()
