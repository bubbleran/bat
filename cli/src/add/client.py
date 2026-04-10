from pathlib import Path
from create.agent import _write_llm_clients

def add_clients_to_existing_agent(
    agent_dir: Path,
    *,
    clients: list[str],
    force: bool = False,
) -> list[Path]:
    llm_clients_dir = agent_dir / "src" / "llm_clients"
    if not llm_clients_dir.is_dir():
        raise FileNotFoundError(
            f"Directory '{llm_clients_dir}' not found. Run this command from an agent root containing src/llm_clients."
        )

    return _write_llm_clients(llm_clients_dir, clients=clients, force=force)
