"""Smoke test to confirm package imports and version is exposed."""

import mnemix


def test_version_exposed() -> None:
    assert isinstance(mnemix.__version__, str)
    assert mnemix.__version__ == "0.1.0"
