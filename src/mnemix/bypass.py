"""Bypass classification for Mnemix.

A bypass classifier decides whether a query is too time-sensitive (or
otherwise unsafe) to serve from or store in the cache. When
``should_bypass(query)`` is True, the proxy forwards the request upstream
and does *not* write the response back to the cache.

The default :class:`RuleBasedBypass` ships with regex patterns for common
time-sensitive queries (weather, news, stock prices, "right now", etc.) and
can be extended via a TOML config file.

Example:
    Short-circuit time-sensitive queries::

        >>> from mnemix.bypass import RuleBasedBypass
        >>> bp = RuleBasedBypass()
        >>> bp.should_bypass("what's the weather today?")
        True
        >>> bp.should_bypass("explain big-O notation")
        False
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Protocol, runtime_checkable

DEFAULT_PATTERNS: tuple[str, ...] = (
    r"\bweather\b",
    r"\bforecast\b",
    r"\bnews\b",
    r"\b(?:latest|breaking)\b",
    r"\bstock\s*prices?\b",
    r"\bshare\s+price\b",
    r"\bright\s+now\b",
    r"\btoday(?:'s)?\b",
    r"\btonight\b",
    r"\byesterday\b",
    r"\btomorrow\b",
    r"\bcurrent(?:ly)?\b",
    r"\bthis\s+(?:morning|afternoon|evening|week|month|year)\b",
    r"\bwhat\s+time\b",
    r"\bnow\s+playing\b",
    r"\btrending\b",
)


@runtime_checkable
class BypassClassifier(Protocol):
    """Protocol for deciding whether a query should skip the cache.

    Returning True from :meth:`should_bypass` means the proxy will forward
    the request upstream without checking the cache and will not store the
    response. Returning False means the request is eligible for caching.
    """

    def should_bypass(self, query: str) -> bool:
        """Return True when ``query`` should bypass the cache."""
        ...


class RuleBasedBypass:
    r"""Regex-based classifier for time-sensitive queries.

    Patterns are matched case-insensitively against the raw query text.
    A single match is enough to bypass.

    Attributes:
        patterns: The raw pattern strings in use (read-only view).

    Example:
        >>> bp = RuleBasedBypass([r"\bprice\b"])
        >>> bp.should_bypass("What's the price of oil?")
        True
        >>> bp.should_bypass("Explain recursion.")
        False
    """

    def __init__(self, patterns: list[str] | tuple[str, ...] | None = None) -> None:
        """Compile patterns at construction time.

        Args:
            patterns: Regex strings. If ``None``, uses :data:`DEFAULT_PATTERNS`.

        Raises:
            re.error: If any pattern fails to compile.
        """
        source = tuple(patterns) if patterns is not None else DEFAULT_PATTERNS
        self._patterns: tuple[str, ...] = source
        self._compiled: tuple[re.Pattern[str], ...] = tuple(
            re.compile(p, re.IGNORECASE) for p in source
        )

    @property
    def patterns(self) -> tuple[str, ...]:
        """Return the raw pattern strings in use."""
        return self._patterns

    def should_bypass(self, query: str) -> bool:
        """Return True when any pattern matches ``query``."""
        return any(p.search(query) is not None for p in self._compiled)

    @classmethod
    def from_toml(cls, path: str | Path) -> RuleBasedBypass:
        r"""Load patterns from a TOML config file.

        The file must contain a top-level ``patterns`` array of strings::

            patterns = [
                "\\bweather\\b",
                "\\bstock\\s*price\\b",
            ]

        Args:
            path: Path to the TOML file.

        Returns:
            A new :class:`RuleBasedBypass` with the loaded patterns.

        Raises:
            ValueError: If ``patterns`` is missing, not a list, or contains
                non-string elements.
            re.error: If any loaded pattern fails to compile.
        """
        file_path = Path(path)
        with file_path.open("rb") as f:
            data = tomllib.load(f)
        patterns = data.get("patterns")
        if patterns is None:
            msg = f"{file_path}: missing top-level 'patterns' array"
            raise ValueError(msg)
        if not isinstance(patterns, list):
            msg = f"{file_path}: 'patterns' must be an array, got {type(patterns).__name__}"
            raise ValueError(msg)
        if not all(isinstance(p, str) for p in patterns):
            msg = f"{file_path}: every entry in 'patterns' must be a string"
            raise ValueError(msg)
        return cls(patterns)


class NeverBypass:
    """Classifier that never bypasses. Useful as a test double.

    Example:
        >>> NeverBypass().should_bypass("anything at all")
        False
    """

    def should_bypass(self, query: str) -> bool:
        """Always return False."""
        return False


class AlwaysBypass:
    """Classifier that always bypasses. Useful as a test double.

    Example:
        >>> AlwaysBypass().should_bypass("anything at all")
        True
    """

    def should_bypass(self, query: str) -> bool:
        """Always return True."""
        return True
