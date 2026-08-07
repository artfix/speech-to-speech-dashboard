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
from web_ui.settings_schema import build_argv


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


def test_read_settings_returns_default_profile_when_missing(settings_path: Path) -> None:
    read = server._read_settings()
    assert read is not None
    # Default profile (local llama.cpp + Qwen3-TTS) is used as the initial
    # dashboard state when the user has never saved settings.
    assert read.get("--mode") == "realtime"
    assert read.get("--llm-backend") == "responses-api"
    assert read.get("--tts") == "qwen3"
    assert "_enabled_flags" in read
    assert "--mode" in read["_enabled_flags"]


def test_build_argv_respects_enabled_flags() -> None:
    settings = {
        "--mode": "realtime",
        "--stt": "parakeet-tdt",
        "--llm-backend": "responses-api",
        "--tts": "qwen3",
        "--qwen3-tts-speaker": "Ono_Anna",
        "--qwen3-tts-device": "cuda",
        "_enabled_flags": ["--mode", "--stt", "--llm-backend", "--tts", "--qwen3-tts-speaker"],
    }
    argv = build_argv(settings)
    assert "--mode" in argv
    assert "--qwen3-tts-speaker" in argv
    # --qwen3-tts-device is present in settings but not in _enabled_flags, so it
    # must not be forwarded to the pipeline.
    assert "--qwen3-tts-device" not in argv


def test_build_argv_core_flags_always_forwarded() -> None:
    settings = {
        "--mode": "realtime",
        "--stt": "parakeet-tdt",
        "--llm-backend": "responses-api",
        "--tts": "qwen3",
        "_enabled_flags": [],
    }
    argv = build_argv(settings)
    assert "--mode" in argv
    assert "--stt" in argv
    assert "--llm-backend" in argv
    assert "--tts" in argv
