import os
from pathlib import Path

from typer.testing import CliRunner

import create.agent as create_agent_module
from cli import app

runner = CliRunner()


def test_create_new_agent_requires_name(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent"])
def test_create_new_agent_requires_name(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent"])

    assert result.exit_code != 0
    assert result.exit_code != 0


def test_static_template_loader_ignores_bytecode_cache(
    monkeypatch, tmp_path
) -> None:
    templates_dir = tmp_path / "templates" / "agent"
    cache_dir = templates_dir / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "README.cpython-313.pyc").write_bytes(b"\xf3\x00\x00\x00")
    (templates_dir / "README.md").write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(create_agent_module, "TEMPLATES_DIR", templates_dir)
    monkeypatch.setattr(create_agent_module, "_DYNAMIC_TEMPLATE_FILES", set())

    assert create_agent_module._load_static_templates() == {
        "README.md": "hello\n"
    }


def test_create_new_agent_custom_name(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent", "demo_agent"])
def test_create_new_agent_custom_name(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent", "demo_agent"])

    assert result.exit_code == 0
    assert result.exit_code == 0

    root = Path("demo_agent")
    assert root.exists()
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
    for file_path in expected_files:
        assert file_path.exists(), f"Missing scaffold file: {file_path}"

    pyproject_content = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project]" in pyproject_content
    assert 'name = "demo"' in pyproject_content
    assert 'version = "1.0.0"' in pyproject_content
    assert 'description = "DEMO Agent"' in pyproject_content
    assert 'readme = "README.md"' in pyproject_content
    assert 'requires-python = ">=3.12"' in pyproject_content
    assert '"bat-adk>=2026.06"' in pyproject_content
    pyproject_content = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project]" in pyproject_content
    assert 'name = "demo"' in pyproject_content
    assert 'version = "1.0.0"' in pyproject_content
    assert 'description = "DEMO Agent"' in pyproject_content
    assert 'readme = "README.md"' in pyproject_content
    assert 'requires-python = ">=3.12"' in pyproject_content
    assert '"bat-adk>=2026.06"' in pyproject_content

    agent_json_content = (root / "agent.json").read_text(encoding="utf-8")
    assert '"version": "1.0.0"' in agent_json_content
    assert '"name": "demo_agent"' in agent_json_content
    assert '"description": ""' in agent_json_content
    assert '"skills": []' in agent_json_content
    assert '"defaultInputModes"' in agent_json_content
    assert '"defaultOutputModes"' in agent_json_content
    assert '"capabilities"' in agent_json_content
    agent_json_content = (root / "agent.json").read_text(encoding="utf-8")
    assert '"version": "1.0.0"' in agent_json_content
    assert '"name": "demo_agent"' in agent_json_content
    assert '"description": ""' in agent_json_content
    assert '"skills": []' in agent_json_content
    assert '"defaultInputModes"' in agent_json_content
    assert '"defaultOutputModes"' in agent_json_content
    assert '"capabilities"' in agent_json_content

    dockerfile_content = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "strip dist/demo" in dockerfile_content
    assert 'ENTRYPOINT ["./demo"]' in dockerfile_content
    dockerfile_content = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "strip dist/demo" in dockerfile_content
    assert 'ENTRYPOINT ["./demo"]' in dockerfile_content

    makefile_content = (root / "Makefile").read_text(encoding="utf-8")
    assert "REPO ?= YOUR_REPOSITORY/demo_agent" in makefile_content
    assert "Building demo_agent Docker image" in makefile_content
    makefile_content = (root / "Makefile").read_text(encoding="utf-8")
    assert "REPO ?= YOUR_REPOSITORY/demo_agent" in makefile_content
    assert "Building demo_agent Docker image" in makefile_content

    agent_spec_content = (root / "agent.spec").read_text(encoding="utf-8")
    assert "name='demo'" in agent_spec_content
    agent_spec_content = (root / "agent.spec").read_text(encoding="utf-8")
    assert "name='demo'" in agent_spec_content

    assert not (root / ".env.template").exists()

    # Endpoint/model now live in config.yaml; .env carries only the API key.
    config_content = (root / "config.yaml").read_text(encoding="utf-8")
    assert "port: 9900" in config_content
    assert "name: gpt-4o-mini" in config_content
    assert "provider: openai" in config_content
    assert "telemetry:" in config_content

    env_content = (root / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in env_content
    assert "MODEL=" not in env_content
    assert "PORT=" not in env_content

    graph_content = (root / "src" / "graph.py").read_text(encoding="utf-8")
    assert (
        "from .llm_clients.example_client import ExampleClient" in graph_content
    )
    assert "class DemoAgentGraph(AgentGraph):" in graph_content
    assert "self.example_client = ExampleClient(" in graph_content
    assert "DemoAgentGraph.NODE_1" not in graph_content
    graph_content = (root / "src" / "graph.py").read_text(encoding="utf-8")
    assert (
        "from .llm_clients.example_client import ExampleClient" in graph_content
    )
    assert "class DemoAgentGraph(AgentGraph):" in graph_content
    assert "self.example_client = ExampleClient(" in graph_content
    assert "DemoAgentGraph.NODE_1" not in graph_content


def test_create_new_agent_pyproject_name_from_camel_case(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent", "RNGAgent"])
def test_create_new_agent_pyproject_name_from_camel_case(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent", "RNGAgent"])

    assert result.exit_code == 0
    pyproject_content = Path("RNGAgent", "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'name = "rng"' in pyproject_content


def test_create_new_agent_rejects_empty_clients_option(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "agent", "rng", "--clients", " , "])

    assert result.exit_code != 0
    assert "Provide at least one client name" in result.output
    assert result.exit_code != 0
    assert "Provide at least one client name" in result.output


def test_create_new_agent_with_custom_clients(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "agent", "rng", "--clients", "talk, discuss"]
    )
def test_create_new_agent_with_custom_clients(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "agent", "rng", "--clients", "talk, discuss"]
    )

    assert result.exit_code == 0
    root = Path("rng")
    assert result.exit_code == 0
    root = Path("rng")

    talk_client = root / "src" / "llm_clients" / "talk_client.py"
    discuss_client = root / "src" / "llm_clients" / "discuss_client.py"
    example_client = root / "src" / "llm_clients" / "example_client.py"
    talk_client = root / "src" / "llm_clients" / "talk_client.py"
    discuss_client = root / "src" / "llm_clients" / "discuss_client.py"
    example_client = root / "src" / "llm_clients" / "example_client.py"

    assert talk_client.exists()
    assert discuss_client.exists()
    assert not example_client.exists()
    assert talk_client.exists()
    assert discuss_client.exists()
    assert not example_client.exists()

    talk_content = talk_client.read_text(encoding="utf-8")
    discuss_content = discuss_client.read_text(encoding="utf-8")
    talk_content = talk_client.read_text(encoding="utf-8")
    discuss_content = discuss_client.read_text(encoding="utf-8")

    assert "class TalkClient(ChatModelClient):" in talk_content
    assert 'client_name="TalkClient"' in talk_content
    assert "class DiscussClient(ChatModelClient):" in discuss_content
    assert 'client_name="DiscussClient"' in discuss_content
    assert "class TalkClient(ChatModelClient):" in talk_content
    assert 'client_name="TalkClient"' in talk_content
    assert "class DiscussClient(ChatModelClient):" in discuss_content
    assert 'client_name="DiscussClient"' in discuss_content

    graph_content = (root / "src" / "graph.py").read_text(encoding="utf-8")
    assert "from .llm_clients.talk_client import TalkClient" in graph_content
    assert (
        "from .llm_clients.discuss_client import DiscussClient" in graph_content
    )
    assert "self.talk_client = TalkClient(" in graph_content
    assert "self.discuss_client = DiscussClient(" in graph_content
    assert "RngGraph.NODE_1" not in graph_content
    graph_content = (root / "src" / "graph.py").read_text(encoding="utf-8")
    assert "from .llm_clients.talk_client import TalkClient" in graph_content
    assert (
        "from .llm_clients.discuss_client import DiscussClient" in graph_content
    )
    assert "self.talk_client = TalkClient(" in graph_content
    assert "self.discuss_client = DiscussClient(" in graph_content
    assert "RngGraph.NODE_1" not in graph_content


def test_add_new_client_from_existing_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0
def test_add_new_client_from_existing_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0

    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["add", "client", "talk,discuss"])
    finally:
        os.chdir(start_dir)
    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["add", "client", "talk,discuss"])
    finally:
        os.chdir(start_dir)

    assert result.exit_code == 0
    assert result.exit_code == 0

    root = Path("api")
    assert (root / "src" / "llm_clients" / "talk_client.py").exists()
    assert (root / "src" / "llm_clients" / "discuss_client.py").exists()
    root = Path("api")
    assert (root / "src" / "llm_clients" / "talk_client.py").exists()
    assert (root / "src" / "llm_clients" / "discuss_client.py").exists()


def test_create_new_agent_errors_for_non_empty_target_without_force(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("api")
    target.mkdir()
    (target / "keep.txt").write_text("do-not-touch", encoding="utf-8")
def test_create_new_agent_errors_for_non_empty_target_without_force(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("api")
    target.mkdir()
    (target / "keep.txt").write_text("do-not-touch", encoding="utf-8")

    result = runner.invoke(app, ["init", "agent", "api"])
    result = runner.invoke(app, ["init", "agent", "api"])

    assert result.exit_code == 1
    assert "already exists and is not empty" in result.output
    assert result.exit_code == 1
    assert "already exists and is not empty" in result.output


def test_create_new_agent_with_force_overwrites_existing_graph(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(app, ["init", "agent", "api", "--clients", "talk"])
    assert first.exit_code == 0
def test_create_new_agent_with_force_overwrites_existing_graph(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(app, ["init", "agent", "api", "--clients", "talk"])
    assert first.exit_code == 0

    graph_path = Path("api", "src", "graph.py")
    original_graph = graph_path.read_text(encoding="utf-8")
    graph_path.write_text("# stale", encoding="utf-8")
    graph_path = Path("api", "src", "graph.py")
    original_graph = graph_path.read_text(encoding="utf-8")
    graph_path.write_text("# stale", encoding="utf-8")

    second = runner.invoke(
        app, ["init", "agent", "api", "--clients", "plan", "--force"]
    )
    assert second.exit_code == 0
    second = runner.invoke(
        app, ["init", "agent", "api", "--clients", "plan", "--force"]
    )
    assert second.exit_code == 0

    updated_graph = graph_path.read_text(encoding="utf-8")
    assert updated_graph != "# stale"
    assert updated_graph != original_graph
    assert "from .llm_clients.plan_client import PlanClient" in updated_graph
    updated_graph = graph_path.read_text(encoding="utf-8")
    assert updated_graph != "# stale"
    assert updated_graph != original_graph
    assert "from .llm_clients.plan_client import PlanClient" in updated_graph


def test_create_new_agent_clients_are_normalized_and_deduplicated(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            "agent",
            "norm",
            "--clients",
            "Talk, talk , TALK_CLIENT, __, planner-agent",
        ],
    )
def test_create_new_agent_clients_are_normalized_and_deduplicated(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            "agent",
            "norm",
            "--clients",
            "Talk, talk , TALK_CLIENT, __, planner-agent",
        ],
    )

    assert result.exit_code == 0
    root = Path("norm", "src", "llm_clients")
    assert result.exit_code == 0
    root = Path("norm", "src", "llm_clients")

    assert (root / "talk_client.py").exists()
    assert not (root / "talk_client_client.py").exists()
    assert (root / "planner_agent_client.py").exists()
    assert (root / "talk_client.py").exists()
    assert not (root / "talk_client_client.py").exists()
    assert (root / "planner_agent_client.py").exists()


def test_create_new_agent_writes_custom_config_values(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            "agent",
            "envagent",
            "--port",
            "8088",
            "--model",
            "gpt-4.1-mini",
            "--model-provider",
            "openai",
        ],
    )

    assert result.exit_code == 0
    # The CLI options now populate config.yaml, not .env.
    config_file = Path("envagent", "config.yaml").read_text(encoding="utf-8")
    assert "port: 8088" in config_file
    assert "name: gpt-4.1-mini" in config_file
    assert "provider: openai" in config_file

    env_file = Path("envagent", ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in env_file
    assert "MODEL=" not in env_file
    assert not Path("envagent", ".env.template").exists()


def test_add_new_client_force_overwrites_existing_client_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(
        app, ["init", "agent", "api", "--clients", "talk"]
    )
    assert init_result.exit_code == 0
def test_add_new_client_force_overwrites_existing_client_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(
        app, ["init", "agent", "api", "--clients", "talk"]
    )
    assert init_result.exit_code == 0

    talk_client_path = Path("api", "src", "llm_clients", "talk_client.py")
    talk_client_path.write_text("# stale", encoding="utf-8")
    talk_client_path = Path("api", "src", "llm_clients", "talk_client.py")
    talk_client_path.write_text("# stale", encoding="utf-8")

    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["add", "client", "talk", "--force"])
    finally:
        os.chdir(start_dir)
    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["add", "client", "talk", "--force"])
    finally:
        os.chdir(start_dir)

    assert result.exit_code == 0
    assert "# stale" not in talk_client_path.read_text(encoding="utf-8")
    assert result.exit_code == 0
    assert "# stale" not in talk_client_path.read_text(encoding="utf-8")


def test_add_new_client_rejects_empty_client_input(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0
def test_add_new_client_rejects_empty_client_input(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0

    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["add", "client", " , "])
    finally:
        os.chdir(start_dir)
    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["add", "client", " , "])
    finally:
        os.chdir(start_dir)

    assert result.exit_code != 0
    assert "Provide at least one client name" in result.output
    assert result.exit_code != 0
    assert "Provide at least one client name" in result.output


def test_add_new_client_requires_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add", "client", "talk"])
def test_add_new_client_requires_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add", "client", "talk"])

    assert result.exit_code == 1
    assert "src/llm_clients" in result.output
    assert result.exit_code == 1
    assert "src/llm_clients" in result.output


def test_build_command_runs_docker_build(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, check, cwd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr("build.build.subprocess.run", fake_run)

    monkeypatch.chdir(tmp_path)
    Path("agent").mkdir()
    Path("agent", "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
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
            "--version",
            "v1.2.3",
            "--no-cache",
        ],
    )
    monkeypatch.chdir(tmp_path)
    Path("agent").mkdir()
    Path("agent", "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
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
        "hub.bubbleran.com/orama/labs/rng-agent:v1.2.3",
        ".",
    ]
    assert captured["check"] is True
    assert captured["cwd"] == Path("agent").resolve()
    assert (
        "Docker image built successfully: hub.bubbleran.com/orama/labs/rng-agent:v1.2.3"
        in result.output
    )
    assert result.exit_code == 0
    assert captured["cmd"] == [
        "docker",
        "build",
        "--no-cache",
        "--build-arg",
        "VERSION=v1.2.3",
        "--tag",
        "hub.bubbleran.com/orama/labs/rng-agent:v1.2.3",
        ".",
    ]
    assert captured["check"] is True
    assert captured["cwd"] == Path("agent").resolve()
    assert (
        "Docker image built successfully: hub.bubbleran.com/orama/labs/rng-agent:v1.2.3"
        in result.output
    )


def test_push_command_runs_docker_push(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, check, cwd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr("push.push.subprocess.run", fake_run)

    monkeypatch.chdir(tmp_path)
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
            "--version",
            "latest",
        ],
    )
    monkeypatch.chdir(tmp_path)
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
            "--version",
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
    assert (
        "Docker image pushed successfully: hub.bubbleran.com/orama/labs/rng-agent:latest"
        in result.output
    )
    assert result.exit_code == 0
    assert captured["cmd"] == [
        "docker",
        "push",
        "hub.bubbleran.com/orama/labs/rng-agent:latest",
    ]
    assert captured["check"] is True
    assert captured["cwd"] == Path("agent").resolve()
    assert (
        "Docker image pushed successfully: hub.bubbleran.com/orama/labs/rng-agent:latest"
        in result.output
    )


def test_build_command_errors_when_context_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["build", "--context", "missing"])
def test_build_command_errors_when_context_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["build", "--context", "missing"])

    assert result.exit_code == 1
    assert "Context directory not found" in result.output
    assert result.exit_code == 1
    assert "Context directory not found" in result.output


def test_push_command_errors_when_context_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["push", "--context", "missing"])
def test_push_command_errors_when_context_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["push", "--context", "missing"])

    assert result.exit_code == 1
    assert "Context directory not found" in result.output
    assert result.exit_code == 1
    assert "Context directory not found" in result.output


def test_set_env_writes_config_yaml_and_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0

    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(
            app,
            [
                "set",
                "env",
                "--port",
                "8080",
                "--model",
                "gpt-4.1-mini",
                "--model-provider",
                "openai",
                "--docker-registry",
                "hub.bubbleran.com",
                "--repo",
                "orama/labs/demo",
            ],
        )
    finally:
        os.chdir(start_dir)
    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(
            app,
            [
                "set",
                "env",
                "--port",
                "8080",
                "--model",
                "gpt-4.1-mini",
                "--model-provider",
                "openai",
                "--docker-registry",
                "hub.bubbleran.com",
                "--repo",
                "orama/labs/demo",
            ],
        )
    finally:
        os.chdir(start_dir)

    assert result.exit_code == 0

    # Endpoint/model land in config.yaml.
    import yaml

    config = yaml.safe_load(
        Path("api", "config.yaml").read_text(encoding="utf-8")
    )
    assert config["endpoint"]["port"] == 8080
    assert config["model"]["name"] == "gpt-4.1-mini"
    assert config["model"]["provider"] == "openai"

    # Docker build/push defaults stay in .env.
    env_content = Path("api", ".env").read_text(encoding="utf-8")
    assert "BAT_DOCKER_REGISTRY=hub.bubbleran.com" in env_content
    assert "BAT_DOCKER_REPO=orama/labs/demo" in env_content
    assert "MODEL=" not in env_content
    assert "PORT=" not in env_content


def test_set_env_requires_config_yaml_when_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0

    Path("api", "config.yaml").unlink()

    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["set", "env", "--port", "7777"])
    finally:
        os.chdir(start_dir)
    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["set", "env", "--port", "7777"])
    finally:
        os.chdir(start_dir)

    assert result.exit_code == 1
    assert "must contain config.yaml" in result.output


def test_set_env_requires_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["set", "env", "--port", "7777"])
def test_set_env_requires_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["set", "env", "--port", "7777"])

    assert result.exit_code == 1
    assert "must contain config.yaml" in result.output


def test_set_env_requires_at_least_one_option(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0
def test_set_env_requires_at_least_one_option(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_result = runner.invoke(app, ["init", "agent", "api"])
    assert init_result.exit_code == 0

    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["set", "env"])
    finally:
        os.chdir(start_dir)
    start_dir = Path.cwd()
    os.chdir(Path("api"))
    try:
        result = runner.invoke(app, ["set", "env"])
    finally:
        os.chdir(start_dir)

    assert result.exit_code == 1
    assert "Provide at least one option to set" in result.output
    assert result.exit_code == 1
    assert "Provide at least one option to set" in result.output
