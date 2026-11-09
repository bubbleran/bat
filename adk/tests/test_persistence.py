import asyncio

from a2a.server.tasks import DatabaseTaskStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from bat.config import PersistenceConfig
from bat.persistence import _task_store, init_checkpointer


def test_no_persistence_keeps_no_state():
    assert asyncio.run(init_checkpointer(None)) is None


def test_memory():
    saver = asyncio.run(init_checkpointer(PersistenceConfig(type="memory")))

    assert isinstance(saver, InMemorySaver)


def test_postgres_is_a_set_up_pool(monkeypatch):
    events = []

    async def open(pool):
        events.append(f"open {pool.conninfo}")

    async def setup(saver):
        events.append("setup")

    monkeypatch.setattr(AsyncConnectionPool, "open", open)
    monkeypatch.setattr(AsyncPostgresSaver, "setup", setup)

    persistence = PersistenceConfig(type="postgres", url="postgresql://db")
    saver = asyncio.run(init_checkpointer(persistence))

    assert isinstance(saver, AsyncPostgresSaver)
    assert isinstance(saver.conn, AsyncConnectionPool)
    assert events == ["open postgresql://db", "setup"]


def test_postgres_tasks_share_the_checkpointer_driver():
    persistence = PersistenceConfig(type="postgres", url="postgresql://db/a")
    store = _task_store(persistence)

    assert isinstance(store, DatabaseTaskStore)
    assert store.engine.url.drivername == "postgresql+psycopg"
