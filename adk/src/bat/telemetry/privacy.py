"""Telemetry privacy levels: one ordered dial instead of several booleans.

Each level is a superset of the one below it, so the setting reads as "how
much of the agent's internals may leave the process" rather than as a set of
independent switches that can be combined into states nobody designed.

Being ordered is also what makes the privacy floor simple: an agent sets a
minimum level in its own source and the effective level is ``max(floor,
configured)`` -- see :func:`resolve_privacy`. With booleans this was an OR
per flag; with a ladder it is one comparison that cannot express a
contradiction.

Levels
------

``none`` (0)
    No redaction. Spans carry prompts, completions, tool definitions and
    node names in full. The default.

``content`` (1)
    Redact what the agent *says and knows*: prompts, messages, completions,
    invocation parameters, and every tool's description, parameter schema and
    call arguments. Tool names and node names still export, so traces stay
    fully navigable and the eval engine's tool-call metrics keep working.

``names`` (2)
    Also redact span names, so LangGraph node names stop leaking. Span kinds
    (``LLM``/``CHAIN``/``TOOL``) replace them, so the trace keeps its shape.
    The eval engine is still unaffected.

``full`` (3)
    Also redact tool names. Only token counts, span kinds, hierarchy and
    timing survive. NOTE: the eval engine reconstructs per-episode tool calls
    from span attributes and keys its metrics off the tool name, so at this
    level those metrics go empty. Use it when the tool inventory itself is
    considered proprietary.
"""

from enum import IntEnum
from typing import Literal

from ..logging import create_logger

logger = create_logger(__name__, "debug")
TelemetryPrivacyLevel = Literal["none", "content", "names", "full"]


class TelemetryPrivacy(IntEnum):
    """How much of the agent's internals may leave the process.

    Ordered and cumulative: each level redacts everything the level below it
    does. See the module docstring for what each one covers.
    """

    NONE = 0
    CONTENT = 1
    NAMES = 2
    FULL = 3

    @property
    def hides_content(self) -> bool:
        """Prompts, completions and tool descriptions/schemas/arguments."""
        return self >= TelemetryPrivacy.CONTENT

    @property
    def hides_span_names(self) -> bool:
        """Span (LangGraph node) names."""
        return self >= TelemetryPrivacy.NAMES

    @property
    def hides_tool_names(self) -> bool:
        """Tool names (costs the eval engine its tool-call metrics)."""
        return self >= TelemetryPrivacy.FULL


def parse_privacy(value: object) -> TelemetryPrivacy:
    """Coerce a ``config.yaml`` value into a :class:`TelemetryPrivacy`.

    Accepts the level name (case-insensitive, e.g. ``content``) or its
    ordinal (``1``). ``None`` means "not configured" and maps to
    :attr:`TelemetryPrivacy.NONE`.

    Raises:
        ValueError: If ``value`` is neither a known level name nor a valid
            ordinal. Failing loudly is deliberate: a typo'd privacy level
            silently falling back to ``none`` would export in the clear
            exactly when someone was trying to lock the agent down.
    """
    if value is None:
        return TelemetryPrivacy.NONE
    if isinstance(value, TelemetryPrivacy):
        return value
    if isinstance(value, bool):
        # `privacy: true` is meaningless on a ladder; refuse rather than
        # guess which level was meant.
        raise ValueError(
            "telemetry.privacy expects a level "
            f"({_level_names()}), not a boolean."
        )
    if isinstance(value, int):
        try:
            return TelemetryPrivacy(value)
        except ValueError:
            raise ValueError(
                f"Unknown telemetry privacy level {value!r}; expected one of "
                f"{_level_names()} (or 0-{max(TelemetryPrivacy).value})."
            ) from None
    if isinstance(value, str):
        key = value.strip().upper()
        if key in TelemetryPrivacy.__members__:
            return TelemetryPrivacy[key]
        raise ValueError(
            f"Unknown telemetry privacy level {value!r}; expected one of "
            f"{_level_names()}."
        )
    raise ValueError(
        f"Unknown telemetry privacy level {value!r}; expected one of "
        f"{_level_names()}."
    )


def _level_names() -> str:
    return ", ".join(level.name.lower() for level in TelemetryPrivacy)


def resolve_privacy(
    configured: object,
    floor: object = TelemetryPrivacy.NONE,
) -> TelemetryPrivacy:
    """Combine what ``config.yaml`` asked for with the agent's own floor. 
    Choose the more private of the two.

    Args:
        configured (object): The level ``config.yaml`` asked for, as a level
            name, an ordinal, or a :class:`TelemetryPrivacy`.
        floor (object): The minimum the agent allows. Defaults to
            :attr:`TelemetryPrivacy.NONE`, which leaves ``configured`` to
            decide alone.

    Raises:
        ValueError: If ``configured`` is not a known level.
    """
    try:
        floor_level = parse_privacy(floor)
    except ValueError:
        logger.warning(
            "Telemetry: unknown privacy floor %r; falling back to none. The "
            "agent will export whatever config.yaml asks for.",
            floor,
        )
        floor_level = TelemetryPrivacy.NONE
    return max(floor_level, parse_privacy(configured))
