from pathlib import Path

from typer.testing import CliRunner

from build.build import TELEMETRY_BUILD_POLICY_FILENAME
from cli import app

runner = CliRunner()


def _write_minimal_build_context(root: Path) -> None:
    (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")


def test_hide_telemetry_content_writes_policy_file_during_build_and_cleans_up(
    tmp_path, monkeypatch
):
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME
    seen_during_build: dict[str, bool] = {}

    def fake_run(command, check, cwd):
        # The policy file must exist while `docker build` runs (it is the
        # build context PyInstaller freezes), and contain the floor set to
        # True -- config.yaml must never be able to override this back down.
        seen_during_build["existed"] = policy_path.exists()
        seen_during_build["content"] = policy_path.read_text(encoding="utf-8")

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        ["build", "--context", str(tmp_path), "--hide-telemetry-content"],
    )

    assert result.exit_code == 0, result.output
    assert seen_during_build["existed"] is True
    assert "TELEMETRY_HIDE_CONTENT_FLOOR = True" in seen_during_build["content"]
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
        ["build", "--context", str(tmp_path), "--hide-telemetry-content"],
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
        ["build", "--context", str(tmp_path), "--hide-telemetry-content"],
    )

    assert result.exit_code != 0
    assert not policy_path.exists()


def test_hide_telemetry_span_names_bakes_span_name_floor(
    tmp_path, monkeypatch
):
    """--hide-telemetry-span-names bakes its own floor, independently."""
    _write_minimal_build_context(tmp_path)
    policy_path = tmp_path / TELEMETRY_BUILD_POLICY_FILENAME
    seen_during_build: dict[str, str] = {}

    def fake_run(command, check, cwd):
        seen_during_build["content"] = policy_path.read_text(encoding="utf-8")

        class FakeResult:
            returncode = 0

        return FakeResult()

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    result = runner.invoke(
        app,
        ["build", "--context", str(tmp_path), "--hide-telemetry-span-names"],
    )

    assert result.exit_code == 0, result.output
    content = seen_during_build["content"]
    assert "TELEMETRY_HIDE_SPAN_NAMES_FLOOR = True" in content
    # Asking for one floor must not silently raise the other.
    assert "TELEMETRY_HIDE_CONTENT_FLOOR = False" in content
    assert not policy_path.exists()


def test_both_telemetry_floors_can_be_baked_together(tmp_path, monkeypatch):
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
        [
            "build",
            "--context",
            str(tmp_path),
            "--hide-telemetry-content",
            "--hide-telemetry-span-names",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "TELEMETRY_HIDE_CONTENT_FLOOR = True" in seen["content"]
    assert "TELEMETRY_HIDE_SPAN_NAMES_FLOOR = True" in seen["content"]
