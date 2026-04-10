from pathlib import Path
import os

from typer.testing import CliRunner

from bat_cli.cli import app


runner = CliRunner()


def test_create_new_agent_default_name() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "agent"])

        assert result.exit_code == 0

        root = Path("default")
        assert root.exists()

        assert (root / "src" / "llm_clients" / "example_client.py").exists()
        assert not (root / "src" / "llm_clients" / "courtesy_client.py").exists()
        assert not (root / "src" / "llm_clients" / "planner_client.py").exists()


def test_create_new_agent_custom_name() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "agent", "demo_agent"])

        assert result.exit_code == 0

        root = Path("demo_agent")
        assert root.exists()

        expected_files = [
            root / "__main__.py",
            root / "src" / "__init__.py",
            root / "src" / "graph.py",
            root / "src" / "llm_clients" / "__init__.py",
            root / "src" / "llm_clients" / "example_client.py",
            root / "tests" / "__init__.py",
            root / "agent.json",
            root / "agent.spec",
            root / "config.yaml",
            root / "Dockerfile",
            root / "Makefile",
            root / "pyproject.toml",
            root / "README.md",
            root / ".env",
            root / ".python-version",
            root / ".dockerignore",
        ]

        for file_path in expected_files:
            assert file_path.exists(), f"Missing scaffold file: {file_path}"

        pyproject_content = (root / "pyproject.toml").read_text(encoding="utf-8")
        assert "[project]" in pyproject_content
        assert 'name = "demo-agent"' in pyproject_content
        assert 'version = "1.0.0"' in pyproject_content
        assert 'description = "DEMO-AGENT Agent"' in pyproject_content
        assert 'readme = "README.md"' in pyproject_content
        assert 'requires-python = ">=3.13"' in pyproject_content
        assert '"bat-adk>=2026.3"' in pyproject_content

        agent_json_content = (root / "agent.json").read_text(encoding="utf-8")
        assert '"version": "2026_3"' in agent_json_content
        assert '"name": "demo_agent"' in agent_json_content
        assert '"description": ""' in agent_json_content
        assert '"skills": []' in agent_json_content

        dockerfile_content = (root / "Dockerfile").read_text(encoding="utf-8")
        assert "strip dist/demo-agent" in dockerfile_content
        assert "ENTRYPOINT [\"./demo-agent\"]" in dockerfile_content

        makefile_content = (root / "Makefile").read_text(encoding="utf-8")
        assert "REPO ?= orama/labs/demo-agent" in makefile_content
        assert "Building demo-agent Agent Docker image" in makefile_content

        agent_spec_content = (root / "agent.spec").read_text(encoding="utf-8")
        assert "name='demo-agent'" in agent_spec_content

        graph_content = (root / "src" / "graph.py").read_text(encoding="utf-8")
        assert "from .llm_clients.example_client import ExampleClient" in graph_content
        assert "class DemoAgentGraph(AgentGraph):" in graph_content
        assert "self.example_client = ExampleClient(" in graph_content
        assert "self.example_loop = ReActLoop(" in graph_content
        assert "DemoAgentGraph.NODE_1" not in graph_content


def test_create_new_agent_pyproject_name_from_camel_case() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "agent", "RNGAgent"])

        assert result.exit_code == 0
        pyproject_content = Path("RNGAgent", "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "rng-agent"' in pyproject_content


def test_create_new_agent_with_custom_clients() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "agent", "rng", "talk, discuss"])

        assert result.exit_code == 0
        root = Path("rng")

        talk_client = root / "src" / "llm_clients" / "talk_client.py"
        discuss_client = root / "src" / "llm_clients" / "discuss_client.py"
        example_client = root / "src" / "llm_clients" / "example_client.py"

        assert talk_client.exists()
        assert discuss_client.exists()
        assert not example_client.exists()

        talk_content = talk_client.read_text(encoding="utf-8")
        discuss_content = discuss_client.read_text(encoding="utf-8")

        assert "class TalkClient(ChatModelClient):" in talk_content
        assert "client_name=\"TalkClient\"" in talk_content
        assert "class DiscussClient(ChatModelClient):" in discuss_content
        assert "client_name=\"DiscussClient\"" in discuss_content

        graph_content = (root / "src" / "graph.py").read_text(encoding="utf-8")
        assert "from .llm_clients.talk_client import TalkClient" in graph_content
        assert "from .llm_clients.discuss_client import DiscussClient" in graph_content
        assert "self.talk_client = TalkClient(" in graph_content
        assert "self.discuss_client = DiscussClient(" in graph_content
        assert "self.talk_loop = ReActLoop(" in graph_content
        assert "self.discuss_loop = ReActLoop(" in graph_content
        assert "RngGraph.NODE_1" not in graph_content


def test_add_new_client_from_existing_agent_root() -> None:
    with runner.isolated_filesystem():
        init_result = runner.invoke(app, ["init", "agent", "api"]) 
        assert init_result.exit_code == 0

        start_dir = Path.cwd()
        os.chdir(Path("api"))
        try:
            result = runner.invoke(app, ["add", "client", "talk,discuss"])
        finally:
            os.chdir(start_dir)

        assert result.exit_code == 0

        root = Path("api")
        assert (root / "src" / "llm_clients" / "talk_client.py").exists()
        assert (root / "src" / "llm_clients" / "discuss_client.py").exists()


def test_add_new_client_requires_agent_root() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["add", "client", "talk"])

        assert result.exit_code == 1
        assert "src/llm_clients" in result.output


def test_build_command_runs_docker_build(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, check, cwd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr("bat_cli.build.build.subprocess.run", fake_run)

    with runner.isolated_filesystem():
        Path("agent").mkdir()
        result = runner.invoke(
            app,
            [
                "build",
                "--context",
                "agent",
                "--docker-registry",
                "hub.bubbleran.com",
                "--repo",
                "orama/labs/rng-agent",
                "--tag",
                "latest",
                "--version",
                "v1.2.3",
                "--no-cache",
            ],
        )

        assert result.exit_code == 0
        assert captured["cmd"] == [
            "docker",
            "build",
            "--no-cache",
            "--build-arg",
            "VERSION=v1.2.3",
            "--tag",
            "hub.bubbleran.com/orama/labs/rng-agent:latest",
            ".",
        ]
        assert captured["check"] is True
        assert captured["cwd"] == Path("agent").resolve()
        assert "Docker image built successfully: hub.bubbleran.com/orama/labs/rng-agent:latest" in result.output


def test_push_command_runs_docker_push(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, check, cwd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr("bat_cli.push.push.subprocess.run", fake_run)

    with runner.isolated_filesystem():
        Path("agent").mkdir()
        result = runner.invoke(
            app,
            [
                "push",
                "--context",
                "agent",
                "--docker-registry",
                "hub.bubbleran.com",
                "--repo",
                "orama/labs/rng-agent",
                "--tag",
                "latest",
            ],
        )

        assert result.exit_code == 0
        assert captured["cmd"] == [
            "docker",
            "push",
            "hub.bubbleran.com/orama/labs/rng-agent:latest",
        ]
        assert captured["check"] is True
        assert captured["cwd"] == Path("agent").resolve()
        assert "Docker image pushed successfully: hub.bubbleran.com/orama/labs/rng-agent:latest" in result.output
