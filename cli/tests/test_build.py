from pathlib import Path

from typer.testing import CliRunner

from build.build import TELEMETRY_BUILD_POLICY_FILENAME
from cli import app

runner = CliRunner()


def _write_minimal_build_context(root: Path) -> None:
    (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")


def test_telemetry_privacy_writes_policy_file_during_build_and_cleans_up(
    tmp_path, monkeypatch
):
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME
    seen_during_build: dict[str, bool] = {}

    def fake_run(command, check, cwd):
        # The policy file must exist while `docker build` runs (it is the
        # build context PyInstaller freezes), and carry the baked level --
        # config.yaml must never be able to drop below it.
        seen_during_build["existed"] = policy_path.exists()
        seen_during_build["content"] = policy_path.read_text(encoding="utf-8")

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        [
            "build",
            "--context",
            str(tmp_path),
            "--telemetry-privacy",
            "content",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen_during_build["existed"] is True
    assert "TELEMETRY_PRIVACY_FLOOR = 'content'" in (
        seen_during_build["content"]
    )
    # Removed afterwards: it is a generated artifact, not part of the agent's
    # source tree or git history.
    assert not policy_path.exists()


def test_without_flag_no_policy_file_is_ever_written(tmp_path, monkeypatch):
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME

    def fake_run(command, check, cwd):
        assert not policy_path.exists()

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    result = runner.invoke(app, ["build", "--context", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert not policy_path.exists()


def test_reserved_filename_collision_aborts_before_docker_runs(
    tmp_path, monkeypatch
):
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME
    policy_path.write_text("# not ours\n", encoding="utf-8")

    called = False

    def fake_run(command, check, cwd):
        nonlocal called
        called = True

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        [
            "build",
            "--context",
            str(tmp_path),
            "--telemetry-privacy",
            "content",
        ],
    )

    assert result.exit_code != 0
    assert called is False
    # The pre-existing file must survive untouched.
    assert policy_path.read_text(encoding="utf-8") == "# not ours\n"


def test_policy_file_removed_even_when_docker_build_fails(
    tmp_path, monkeypatch
):
    import subprocess

    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME

    def fake_run(command, check, cwd):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        [
            "build",
            "--context",
            str(tmp_path),
            "--telemetry-privacy",
            "content",
        ],
    )

    assert result.exit_code != 0
    assert not policy_path.exists()



def _bake_policy(tmp_path, monkeypatch, level: str) -> str:
    """Run a build at ``level`` and return the policy file's contents.

    A helper rather than an inline closure so each invocation gets its own
    bindings (a closure over a loop variable is a B023 footgun).
    """
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME
    seen: dict[str, str] = {}

    def fake_run(command, check, cwd):
        seen["content"] = policy_path.read_text(encoding="utf-8")

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)
    result = runner.invoke(
        app,
        ["build", "--context", str(tmp_path), "--telemetry-privacy", level],
    )
    assert result.exit_code == 0, result.output
    # Generated artifact: never left behind in the agent's source tree.
    assert not policy_path.exists()
    return seen["content"]


def test_telemetry_privacy_levels_are_baked_verbatim(tmp_path, monkeypatch):
    """Each level reaches the frozen binary as written."""
    for level in ("content", "names", "full"):
        content = _bake_policy(tmp_path, monkeypatch, level)
        assert f"TELEMETRY_PRIVACY_FLOOR = {level!r}" in content


def test_telemetry_privacy_none_writes_no_policy_file(tmp_path, monkeypatch):
    """The default level must leave the build context untouched."""
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME
    seen: dict[str, bool] = {}

    def fake_run(command, check, cwd):
        seen["existed"] = policy_path.exists()

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)
    result = runner.invoke(
        app,
        ["build", "--context", str(tmp_path), "--telemetry-privacy", "none"],
    )
    assert result.exit_code == 0, result.output
    assert seen["existed"] is False


def test_unknown_telemetry_privacy_level_is_rejected(tmp_path, monkeypatch):
    """A typo must fail the build, not silently ship an unredacted agent."""
    _write_minimal_build_context(tmp_path)

    def fake_run(command, check, cwd):  # pragma: no cover - must not run
        raise AssertionError("docker build must not start")

    monkeypatch.setattr("build.build.subprocess.run", fake_run)
    result = runner.invoke(
        app,
        ["build", "--context", str(tmp_path), "--telemetry-privacy", "contnt"],
    )
    assert result.exit_code != 0
    assert not (tmp_path / TELEMETRY_BUILD_POLICY_FILENAME).exists()
