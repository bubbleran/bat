"""Tests for the build-time telemetry redaction floor.

``resolve_hide_content`` ORs ``config.yaml``'s ``telemetry.hide_content``
with ``TELEMETRY_HIDE_CONTENT_FLOOR`` -- a module-level constant that only
exists (as ``True``) inside a binary built with
``bat build --hide-telemetry-content``. In a source checkout the import in
``bat.telemetry.build_policy`` fails and the floor is ``False``, so these
tests monkeypatch the resolved constant directly to simulate a "baked" build.
"""

from bat.telemetry import build_policy
from bat.telemetry.build_policy import resolve_hide_content


def test_no_floor_defaults_to_false():
    """Source checkouts (no build policy module) import nothing extra."""
    assert build_policy.TELEMETRY_HIDE_CONTENT_FLOOR is False


def test_no_floor_config_decides_alone():
    assert resolve_hide_content(False) is False
    assert resolve_hide_content(True) is True


def test_floor_forces_hide_content_even_when_config_says_no(monkeypatch):
    """A baked floor cannot be turned off by config.yaml."""
    monkeypatch.setattr(build_policy, "TELEMETRY_HIDE_CONTENT_FLOOR", True)
    assert resolve_hide_content(False) is True


def test_floor_and_config_both_true_stays_true(monkeypatch):
    monkeypatch.setattr(build_policy, "TELEMETRY_HIDE_CONTENT_FLOOR", True)
    assert resolve_hide_content(True) is True
