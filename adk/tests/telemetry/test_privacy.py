"""Tests for the telemetry privacy floor.

``resolve_privacy`` takes the ``max`` of what ``config.yaml`` asked for and
the floor the agent set in its own source (``AgentApplication(...,
telemetry_privacy_floor=...)``). The floor is frozen into the packaged
binary, so a replaced ``config.yaml`` can raise the level but never lower it.
"""

import pytest

from bat.telemetry import TelemetryPrivacy, resolve_privacy


def test_default_floor_leaves_config_alone():
    """The default floor is ``none``, so config.yaml decides by itself."""
    assert resolve_privacy("none") is TelemetryPrivacy.NONE
    assert resolve_privacy("content") is TelemetryPrivacy.CONTENT
    assert resolve_privacy("full") is TelemetryPrivacy.FULL


def test_floor_raises_level_when_config_asks_for_less():
    """A floor set in the agent cannot be lowered by config.yaml."""
    floor = TelemetryPrivacy.NAMES
    assert resolve_privacy("none", floor) is TelemetryPrivacy.NAMES
    assert resolve_privacy("content", floor) is TelemetryPrivacy.NAMES


def test_config_may_raise_above_the_floor():
    """The floor is a minimum, not a ceiling."""
    floor = TelemetryPrivacy.CONTENT
    assert resolve_privacy("full", floor) is TelemetryPrivacy.FULL


def test_floor_accepts_a_level_name_or_ordinal():
    assert resolve_privacy("none", "names") is TelemetryPrivacy.NAMES
    assert resolve_privacy("none", 2) is TelemetryPrivacy.NAMES


def test_unknown_floor_falls_back_to_none():
    """A floor is annotated, so a typo is caught before it runs; at runtime
    it degrades rather than stopping the agent from starting."""
    assert resolve_privacy("none", "contnt") is TelemetryPrivacy.NONE
    assert resolve_privacy("full", "contnt") is TelemetryPrivacy.FULL


def test_unknown_configured_level_still_raises():
    """config.yaml is untyped input: a typo there must not turn redaction
    off quietly."""
    with pytest.raises(ValueError):
        resolve_privacy("contnt")
