"""Tests for the build-time telemetry privacy floor.

``resolve_privacy`` takes the ``max`` of ``config.yaml``'s
``telemetry.privacy`` and ``TELEMETRY_PRIVACY_FLOOR`` -- a module-level
constant that only exists inside a binary built with
``bat build --telemetry-privacy LEVEL``. In a source checkout the import in
``bat.telemetry.build_policy`` fails and the floor is ``none``, so these
tests monkeypatch the resolved constant directly to simulate a "baked" build.
"""

from bat.telemetry import build_policy
from bat.telemetry.build_policy import resolve_privacy
from bat.telemetry.privacy import TelemetryPrivacy


def test_no_floor_defaults_to_none():
    """Source checkouts (no build policy module) import nothing extra."""
    assert build_policy.TELEMETRY_PRIVACY_FLOOR is TelemetryPrivacy.NONE


def test_no_floor_config_decides_alone():
    assert resolve_privacy("none") is TelemetryPrivacy.NONE
    assert resolve_privacy("content") is TelemetryPrivacy.CONTENT
    assert resolve_privacy("full") is TelemetryPrivacy.FULL


def test_floor_raises_level_when_config_asks_for_less(monkeypatch):
    """A baked floor cannot be lowered by config.yaml."""
    monkeypatch.setattr(
        build_policy, "TELEMETRY_PRIVACY_FLOOR", TelemetryPrivacy.NAMES
    )
    assert resolve_privacy("none") is TelemetryPrivacy.NAMES
    assert resolve_privacy("content") is TelemetryPrivacy.NAMES


def test_config_may_raise_above_the_floor(monkeypatch):
    """The floor is a minimum, not a ceiling."""
    monkeypatch.setattr(
        build_policy, "TELEMETRY_PRIVACY_FLOOR", TelemetryPrivacy.CONTENT
    )
    assert resolve_privacy("full") is TelemetryPrivacy.FULL


def test_floor_accepts_a_baked_level_name(monkeypatch):
    """The baked constant is written as a string literal by the CLI."""
    monkeypatch.setattr(build_policy, "TELEMETRY_PRIVACY_FLOOR", "names")
    assert resolve_privacy("none") is TelemetryPrivacy.NAMES
