"""Tests for mnemix.bypass: rule-based classifier and test doubles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from mnemix.bypass import (
    DEFAULT_PATTERNS,
    AlwaysBypass,
    BypassClassifier,
    NeverBypass,
    RuleBasedBypass,
)

TIME_SENSITIVE_QUERIES = [
    "what's the weather today?",
    "Give me today's weather forecast",
    "Latest news about the election",
    "what is the current stock price of TSLA?",
    "AAPL share price right now",
    "what time is it in Tokyo",
    "tell me about breaking news",
    "what's trending on twitter",
    "how is MSFT currently doing",
    "weather tomorrow in SF",
    "this week's top stories",
]

STATIC_QUERIES = [
    "explain quantum mechanics",
    "write a Python function to reverse a linked list",
    "what is the capital of France",
    "summarize the plot of Hamlet",
    "derive the quadratic formula",
    "translate 'hello' to Spanish",
    "what is 2 plus 2",
]


class TestRuleBasedBypass:
    def test_conforms_to_protocol(self) -> None:
        assert isinstance(RuleBasedBypass(), BypassClassifier)

    def test_default_patterns_loaded(self) -> None:
        bp = RuleBasedBypass()
        assert bp.patterns == DEFAULT_PATTERNS
        assert len(bp.patterns) > 0

    @pytest.mark.parametrize("query", TIME_SENSITIVE_QUERIES)
    def test_time_sensitive_queries_bypassed(self, query: str) -> None:
        assert RuleBasedBypass().should_bypass(query) is True, query

    @pytest.mark.parametrize("query", STATIC_QUERIES)
    def test_static_queries_not_bypassed(self, query: str) -> None:
        assert RuleBasedBypass().should_bypass(query) is False, query

    def test_case_insensitive(self) -> None:
        bp = RuleBasedBypass()
        assert bp.should_bypass("WHAT'S THE WEATHER TODAY?") is True
        assert bp.should_bypass("weather") is True
        assert bp.should_bypass("WEATHER") is True

    def test_custom_patterns_replace_defaults(self) -> None:
        bp = RuleBasedBypass([r"\bsecret\b"])
        # Custom pattern matches
        assert bp.should_bypass("tell me the secret") is True
        # Default patterns no longer apply
        assert bp.should_bypass("what's the weather today?") is False

    def test_empty_patterns_never_bypasses(self) -> None:
        bp = RuleBasedBypass([])
        assert bp.should_bypass("whatever you want") is False
        assert bp.should_bypass("weather") is False

    def test_invalid_regex_raises_on_construction(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — re.error, Exception is fine here
            RuleBasedBypass(["[unclosed"])


class TestRuleBasedBypassFromToml:
    def _write(self, tmp_path: Path, body: str) -> Path:
        f = tmp_path / "bypass.toml"
        f.write_text(body, encoding="utf-8")
        return f

    def test_loads_patterns_from_toml(self, tmp_path: Path) -> None:
        config = self._write(
            tmp_path,
            'patterns = ["\\\\bzebra\\\\b", "\\\\bquokka\\\\b"]\n',
        )
        bp = RuleBasedBypass.from_toml(config)
        assert bp.should_bypass("look at that zebra") is True
        assert bp.should_bypass("the quokka is smiling") is True
        assert bp.should_bypass("a regular cat") is False

    def test_loaded_patterns_replace_defaults(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, 'patterns = ["\\\\bonly_this\\\\b"]\n')
        bp = RuleBasedBypass.from_toml(config)
        # Default 'weather' pattern should not apply
        assert bp.should_bypass("weather tomorrow") is False
        assert bp.should_bypass("only_this") is True

    def test_accepts_path_as_string(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, 'patterns = ["\\\\bfoo\\\\b"]\n')
        bp = RuleBasedBypass.from_toml(str(config))
        assert bp.should_bypass("foo") is True

    def test_missing_patterns_key_raises(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "other = 42\n")
        with pytest.raises(ValueError, match="missing top-level 'patterns'"):
            RuleBasedBypass.from_toml(config)

    def test_patterns_wrong_type_raises(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, 'patterns = "not a list"\n')
        with pytest.raises(ValueError, match="must be an array"):
            RuleBasedBypass.from_toml(config)

    def test_patterns_nonstring_element_raises(self, tmp_path: Path) -> None:
        config = self._write(tmp_path, "patterns = [123, 456]\n")
        with pytest.raises(ValueError, match="must be a string"):
            RuleBasedBypass.from_toml(config)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            RuleBasedBypass.from_toml(tmp_path / "nope.toml")


class TestNeverBypass:
    def test_conforms_to_protocol(self) -> None:
        assert isinstance(NeverBypass(), BypassClassifier)

    @pytest.mark.parametrize("query", [*TIME_SENSITIVE_QUERIES, *STATIC_QUERIES, ""])
    def test_always_returns_false(self, query: str) -> None:
        assert NeverBypass().should_bypass(query) is False


class TestAlwaysBypass:
    def test_conforms_to_protocol(self) -> None:
        assert isinstance(AlwaysBypass(), BypassClassifier)

    @pytest.mark.parametrize("query", [*TIME_SENSITIVE_QUERIES, *STATIC_QUERIES, ""])
    def test_always_returns_true(self, query: str) -> None:
        assert AlwaysBypass().should_bypass(query) is True
