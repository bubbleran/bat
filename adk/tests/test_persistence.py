import asyncio

from a2a.server.context import ServerCallContext
from a2a.server.tasks import DatabaseTaskStore
from a2a.types import Task, TaskState, TaskStatus
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
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


def _confirm_graph(checkpointer):
    def confirm(state: MessagesState):
        return {"messages": [AIMessage(interrupt("Deploy?"))]}

    builder = StateGraph(MessagesState)
    builder.add_node(confirm)
    builder.add_edge(START, "confirm")
    return builder.compile(checkpointer=checkpointer)


def test_sqlite_resumes_every_paused_conversation_after_a_restart(tmp_path):
    persistence = PersistenceConfig(type="sqlite", url=str(tmp_path / "a.db"))

    async def send(thread, message):
        # A new checkpointer per message: the agent restarts in between.
        saver = await init_checkpointer(persistence)
        config = {"configurable": {"thread_id": thread}}
        try:
            return await _confirm_graph(saver).ainvoke(message, config)
        finally:
            await saver.conn.close()

    async def main():
        asked = [
            await send(thread, {"messages": [HumanMessage("deploy")]})
            for thread in ("a", "b")
        ]
        answered = [
            await send(thread, Command(resume=f"yes {thread}"))
            for thread in ("a", "b")
        ]
        return asked, answered

    asked, answered = asyncio.run(main())

    assert [r["__interrupt__"][0].value for r in asked] == ["Deploy?"] * 2
    assert [r["messages"][-1].text for r in answered] == ["yes a", "yes b"]


def test_sqlite_tasks_are_kept_in_the_checkpoints_file(tmp_path):
    persistence = PersistenceConfig(type="sqlite", url=str(tmp_path / "a.db"))
    paused = TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED)
    context = ServerCallContext()

    async def main():
        saver = await init_checkpointer(persistence)
        before, after = _task_store(persistence), _task_store(persistence)
        try:
            task = Task(id="t", context_id="c", status=paused)
            await before.save(task, context)
            return await after.get("t", context)
        finally:
            await before.engine.dispose()
            await after.engine.dispose()
            await saver.conn.close()

    task = asyncio.run(main())

    assert (task.context_id, task.status.state) == ("c", paused.state)
