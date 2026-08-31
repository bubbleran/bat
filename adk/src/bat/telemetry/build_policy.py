"""Build-time telemetry policy: a redaction floor `config.yaml` cannot lower.

``telemetry.privacy`` in ``config.yaml`` (see :mod:`bat.agent.config`) is
fully customer-editable at runtime -- a mounted ConfigMap, a replaced file, or
``CONFIG_PATH`` pointing elsewhere can all turn it back off. For an agent
*sold* as a packaged artifact, that makes it a default, not a guarantee.

``bat build --telemetry-privacy LEVEL`` closes that gap by writing a sibling
module named ``_telemetry_build_policy.py`` into the Docker build context
before PyInstaller freezes the agent (see ``cli/src/build/build.py``).
Because this module is imported below at *bat-adk's own* module load time --
a plain top-level import, kept optional via ``try``/``except`` -- PyInstaller's
static analysis traces it from the frozen entry point and compiles it
directly into the bytecode archive, alongside the rest of the dependency
graph. It is not a loose file sitting next to the executable the way
``config.yaml`` is: undoing it means decompiling the stripped binary, not
editing a mounted file.

In a source checkout (``uv run .``) or an un-flagged build, no such module
exists, the import fails, and the floor is ``none`` -- ``config.yaml`` is the
sole authority, exactly as before this existed.
"""

from .privacy import TelemetryPrivacy, parse_privacy

try:
    from _telemetry_build_policy import (  # type: ignore[import-not-found]
        TELEMETRY_PRIVACY_FLOOR,
    )
except ImportError:
    TELEMETRY_PRIVACY_FLOOR = TelemetryPrivacy.NONE


def resolve_privacy(configured: object) -> TelemetryPrivacy:
    """Combine the build-time floor with ``config.yaml``'s request.

    Monotonic ``max``: the floor can only make telemetry *more* private. Once
    a build bakes a level in, no runtime config can drop below it; absent a
    floor, ``configured`` alone decides.

    Using an ordered level rather than a set of booleans is what keeps this a
    single comparison -- there is no combination of flags that can express
    "more private in one respect, less in another".
    """
    return max(
        parse_privacy(TELEMETRY_PRIVACY_FLOOR),
        parse_privacy(configured),
    )
