from typing import Optional

from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore, TaskStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .config import PersistenceConfig


async def init_checkpointer(
    persistence: Optional[PersistenceConfig],
) -> Optional[BaseCheckpointSaver[str]]:
    """The checkpointer of config.yaml's `persistence`; None (no
    `persistence`) keeps no state between messages. Call it in the loop that
    serves the agent: the Postgres pool and the SQLite connection are bound
    to it."""
    saver: Optional[BaseCheckpointSaver[str]]
    if persistence is None:
        saver = None
    elif persistence.type == "memory":
        saver = InMemorySaver()
    elif persistence.type == "sqlite":
        # Imported here: they come with the `bat-adk[sqlite]` extra.
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        saver = AsyncSqliteSaver(aiosqlite.connect(persistence.url))
        await saver.setup()
    else:
        # Imported here: they come with the `bat-adk[postgres]` extra.
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        pool = AsyncConnectionPool(
            persistence.url,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            # replaces the connections a Postgres restart broke
            check=AsyncConnectionPool.check_connection,
            open=False,
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
    return saver


def _task_store(persistence: Optional[PersistenceConfig]) -> TaskStore:
    """Where A2A tasks are kept: with `postgres` or `sqlite`, in the
    checkpoints' database; otherwise in memory."""
    store: TaskStore
    if persistence is None or persistence.type == "memory":
        store = InMemoryTaskStore()
    else:
        # Imported here: they come with the postgres or sqlite extra.
        from sqlalchemy.engine import make_url
        from sqlalchemy.ext.asyncio import create_async_engine
        if persistence.type == "sqlite":
            url = make_url(f"sqlite+aiosqlite:///{persistence.url}")
        else:
            url = make_url(persistence.url).set(
                drivername="postgresql+psycopg"
            )
        store = DatabaseTaskStore(create_async_engine(url))
    return store
