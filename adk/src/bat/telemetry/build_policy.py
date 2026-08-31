"""Build-time telemetry policy: a redaction floor `config.yaml` cannot lower.

``telemetry.hide_content`` in ``config.yaml`` (see :mod:`bat.agent.config`) is
fully customer-editable at runtime -- a mounted ConfigMap, a replaced file, or
``CONFIG_PATH`` pointing elsewhere can all turn it back off. For an agent
*sold* as a packaged artifact, that makes it a default, not a guarantee.

``bat build --hide-telemetry-content`` closes that gap by writing a sibling
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
exists, the import fails, and the floor is ``False`` -- ``config.yaml`` is the
sole authority, exactly as before this existed.
"""

try:
    from _telemetry_build_policy import (  # type: ignore[import-not-found]
        TELEMETRY_HIDE_CONTENT_FLOOR,
    )
except ImportError:
    TELEMETRY_HIDE_CONTENT_FLOOR = False

try:
    from _telemetry_build_policy import (  # type: ignore[import-not-found]
        TELEMETRY_HIDE_SPAN_NAMES_FLOOR,
    )
except ImportError:
    # Older baked policy modules only carry the content floor.
    TELEMETRY_HIDE_SPAN_NAMES_FLOOR = False


def resolve_hide_content(configured: bool) -> bool:
    """Combine the build-time floor with ``config.yaml``'s request.

    Monotonic OR: the floor can only make telemetry *more* private. Once a
    build bakes the floor in, no runtime config can turn redaction back off;
    absent a floor, ``configured`` alone decides.
    """
    return TELEMETRY_HIDE_CONTENT_FLOOR or configured


def resolve_hide_span_names(configured: bool) -> bool:
    """Combine the build-time span-name floor with ``config.yaml``'s request.

    Same monotonic OR as :func:`resolve_hide_content`: a baked floor can only
    make telemetry more private, never less.
    """
    return TELEMETRY_HIDE_SPAN_NAMES_FLOOR or configured
