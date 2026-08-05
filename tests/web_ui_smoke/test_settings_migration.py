"""Regression tests for dashboard settings migrations.

These tests run without a live dashboard server. They import the server
module directly and exercise private I/O helpers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from web_ui import server


@pytest.fixture
def settings_path() -> Path:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "web_ui_settings.json"
        # Point the server module at the temporary file for this test.
        original = server.SETTINGS_PATH
        server.SETTINGS_PATH = path
        yield path
        server.SETTINGS_PATH = original


def test_read_settings_migrates_websocket_to_raw_websocket(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({"mode": "websocket"}), encoding="utf-8")

    read = server._read_settings()

    assert read is not None
    assert read["mode"] == "raw-websocket"
    # Migration must be persisted so it only happens once.
    assert json.loads(settings_path.read_text(encoding="utf-8"))["mode"] == "raw-websocket"


def test_read_settings_leaves_raw_websocket_unchanged(settings_path: Path) -> None:
    settings_path.write_text(json.dumps({"mode": "raw-websocket"}), encoding="utf-8")

    read = server._read_settings()

    assert read is not None
    assert read["mode"] == "raw-websocket"
    assert json.loads(settings_path.read_text(encoding="utf-8"))["mode"] == "raw-websocket"


def test_read_settings_returns_none_when_missing(settings_path: Path) -> None:
    assert server._read_settings() is None
