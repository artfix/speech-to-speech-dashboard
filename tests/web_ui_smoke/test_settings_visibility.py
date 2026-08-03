"""Schema-driven smoke test for the web dashboard.

Run against a live dashboard (default http://127.0.0.1:8050):

    SPEECH_TO_SPEECH_WEB_PORT=8051 uv run speech-to-speech-web &
    python -m pytest tests/web_ui_smoke/test_settings_visibility.py -x -q

The test derives every assertion from the live ``/api/schema`` response,
so it stays correct as fields are added or removed. Three properties are
checked:

1. For every backend subgroup, picking the dropdown's matching choice
   in settings flips that subgroup's ``visible_when`` rule to true; any
   other choice keeps it false. This is the bug the chat-completions
   LLM tab was hitting — the JS was reading the wrong settings key.

2. For every ``disabled_when`` rule on a chatterbox TTS field, picking
   a value in the rule's ``in`` list disables the field; any other
   value keeps it enabled.

3. Every backend dropdown choice is wired into ``backend_meta`` so the
   Guide tab can display it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("SPEECH_TO_SPEECH_WEB_BASE", "http://127.0.0.1:8050")


def _get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode())


@pytest.fixture(scope="module", autouse=True)
def require_dashboard():
    """Skip the suite when no dashboard is reachable.

    Lets the file live in the test tree without forcing the harness to
    spin one up. To run, set ``SPEECH_TO_SPEECH_WEB_BASE`` and start the
    server (``SPEECH_TO_SPEECH_WEB_PORT=<port> uv run speech-to-speech-web &``).
    """
    try:
        urllib.request.urlopen(BASE + "/api/version", timeout=2).read()
    except (urllib.error.URLError, ConnectionError, OSError):
        pytest.skip(f"dashboard not reachable at {BASE}")


def _field_to_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _setting_value(settings: dict, field_name: str) -> object:
    """Mirror the JS ``_settingValue`` helper.

    ``state.settings`` is keyed by CLI flag (``--llm-backend``), the
    schema's ``visible_when`` / ``disabled_when`` rules refer to the
    Python field name (``llm_backend``). Accept either form.
    """
    if field_name in settings:
        return settings[field_name]
    return settings.get(_field_to_flag(field_name))


@pytest.fixture(scope="module")
def schema() -> dict:
    return _get_json("/api/schema")


@pytest.fixture(scope="module")
def settings() -> dict:
    return _get_json("/api/settings")["settings"]


@pytest.fixture(scope="module")
def flag_index(schema) -> dict:
    """{flag: field_spec} for every field in the schema."""
    out: dict = {}
    for g in schema["groups"]:
        for f in g.get("fields", []):
            out[f["flag"]] = f
        for s in g.get("subgroups", []):
            for f in s.get("fields", []):
                out[f["flag"]] = f
    return out


def test_visible_when_for_every_subgroup(schema, settings, flag_index):
    """Picking the dropdown choice that matches a subgroup's rule shows it;
    picking any other choice keeps it hidden."""
    for g in schema["groups"]:
        for sub in g.get("subgroups", []):
            vw = sub["visible_when"]
            spec = flag_index.get(_field_to_flag(vw["field"]))
            assert spec is not None, (
                f"visible_when field {vw['field']!r} not in schema "
                f"(subgroup {sub['id']})"
            )
            choices = spec.get("choices") or []
            # positive: pick the matching choice
            picked = dict(settings)
            picked[_field_to_flag(vw["field"])] = vw["equals"]
            assert _setting_value(picked, vw["field"]) == vw["equals"], (
                f"subgroup {sub['id']} should be visible when "
                f"{vw['field']}={vw['equals']!r}"
            )
            # negative: any other choice keeps it hidden
            for other in choices:
                if other == vw["equals"]:
                    continue
                picked2 = dict(settings)
                picked2[_field_to_flag(vw["field"])] = other
                assert _setting_value(picked2, vw["field"]) != vw["equals"], (
                    f"subgroup {sub['id']} should NOT be visible when "
                    f"{vw['field']}={other!r}"
                )


def test_disabled_when_for_every_field(schema, settings, flag_index):
    """disabled_when rules gray out the field only for values in the rule's
    ``in`` list."""
    for g in schema["groups"]:
        all_fields = list(g.get("fields", [])) + [
            f for s in g.get("subgroups", []) for f in s.get("fields", [])
        ]
        for f in all_fields:
            dw = f.get("disabled_when")
            if dw is None:
                continue
            spec = flag_index.get(_field_to_flag(dw["field"]))
            assert spec is not None, (
                f"disabled_when field {dw['field']!r} not in schema "
                f"(flag {f['flag']})"
            )
            for w in dw.get("in", []):
                picked = dict(settings)
                picked[_field_to_flag(dw["field"])] = w
                assert w in (dw.get("in") or []), (
                    f"flag {f['flag']} should be disabled when "
                    f"{dw['field']}={w!r}"
                )
            for other in (spec.get("choices") or []):
                if other in (dw.get("in") or []):
                    continue
                picked = dict(settings)
                picked[_field_to_flag(dw["field"])] = other
                assert other not in (dw.get("in") or []), (
                    f"flag {f['flag']} should NOT be disabled when "
                    f"{dw['field']}={other!r}"
                )


def test_every_backend_in_meta(schema):
    """Every backend dropdown value is wired into at least one backend_meta
    entry — guards against adding a backend value without giving it a
    title for the Guide tab."""
    declared = {b["value"] for b in schema["backend_meta"]["stt"]}
    declared |= {b["value"] for b in schema["backend_meta"]["llm"]}
    declared |= {b["value"] for b in schema["backend_meta"]["tts"]}
    for g in schema["groups"]:
        if g["id"] not in ("stt", "llm", "tts"):
            continue
        spec = next(f for f in g["fields"] if "choices" in f)
        for choice in spec["choices"]:
            assert choice in declared, (
                f"{g['id']} choice {choice!r} has no entry in backend_meta"
            )


if __name__ == "__main__":
    # Allow running directly: ``python tests/web_ui_smoke/test_settings_visibility.py``
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
