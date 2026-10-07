#Blueprint entrypoint: runs the agent named by the first argument.
import importlib
import os
import sys
from pathlib import Path


def _agents() -> list[str]:
    # An agent folder holds its card, here and next to the binary in the image.
    return sorted(card.parent.name for card in Path.cwd().glob("*/agent.json"))


def main() -> None:
    app = sys.argv[1] if len(sys.argv) > 1 else ""
    agents = _agents()
    if app not in agents:
        known = "|".join(agents) or "none yet -- run `bat add agent`"
        print(f"Usage: __BLUEPRINT_NAME__ <{known}>")
        sys.exit(1)
    sys.argv = sys.argv[1:]

    # The SDK reads ./config.yaml, which here is the blueprint root's. Point
    # it at the agent's own unless CONFIG_PATH is set (compose) or a config
    # is mounted at the root (the operator).
    if "CONFIG_PATH" not in os.environ and not Path("config.yaml").is_file():
        os.environ["CONFIG_PATH"] = f"{app}/config.yaml"

    importlib.import_module(app).run()


if __name__ == "__main__":
    main()
