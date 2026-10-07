"""Fixtures shared by the scaffolding tests."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import pytest

# Top-level modules a generated project brings: every test generates them
# under the same names, so they are imported fresh and forgotten afterwards.
_GENERATED_MODULES = {"src"}


def _forget(top_level: set[str]) -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in top_level:
            del sys.modules[name]


@pytest.fixture
def floor_given_to_the_application(monkeypatch):
    """Start a generated entrypoint and return the privacy floor it passed.

    ``start(project_root)`` runs a standalone agent's ``__main__.py``;
    ``start(project_root, agent)`` calls a blueprint agent's ``run()``, the way
    the dispatcher does. AgentApplication is replaced by a recorder, so
    nothing binds a port or builds a model.
    """
    touched = set(_GENERATED_MODULES)
    recorded: list[dict] = []

    class _Recorder:
        def __init__(self, **kwargs) -> None:
            recorded.append(kwargs)

        def run(self) -> None:
            pass

    def start(project_root: Path, agent: str | None = None) -> str:
        monkeypatch.syspath_prepend(str(project_root))
        monkeypatch.setattr("bat.agent.AgentApplication", _Recorder)
        if agent is not None:
            touched.add(agent)
        _forget(touched)
        if agent is None:
            runpy.run_path(
                str(project_root / "__main__.py"), run_name="__main__"
            )
        else:
            importlib.import_module(agent).run()
        return recorded[-1]["telemetry_privacy_floor"]

    yield start
    _forget(touched)
