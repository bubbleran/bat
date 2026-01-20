import time
from ..logging import create_logger
from .graph import AgentGraph
from .state import AgentTaskResult
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from typing import Dict
from typing_extensions import override, Any

_logger = create_logger(__name__, "debug")

class MinimalAgentExecutor(AgentExecutor):
    """Minimal Agent Executor.
    
    Minimal implementation of the AgentExecutor interface used by the `AgentApplication` class to execute agent tasks.
    """

    def __init__(
        self,
        agent_graph: AgentGraph
    ):
        self.agent_graph = agent_graph

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not self._request_ok(context):
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            config = {"configurable": {"thread_id": task.context_id}}
            ts = time.time()
            keep_streaming = True
            async for item in self.agent_graph.astream(query, config):
                if keep_streaming:
                    usage_metadata = self.agent_graph._get_usage_metadata(ts)
                    ts = time.time()
                    keep_streaming = await self._process_task_result(task, item, updater, {'usage': usage_metadata})
                else:
                    # TODO: add chunk status
                    _logger.warning("Artifact has been updated: ignoring additional streamed item.")
        except Exception as e:
            _logger.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    def _request_ok(self, context: RequestContext) -> bool:
        return True

    async def _process_task_result(
        self,
        task: Task,
        task_result: AgentTaskResult,
        updater: TaskUpdater,
        metadata: Dict[str, Any]
    ) -> bool:
        keep_streaming = True
        match task_result.task_status:
            case "working":
                message = new_agent_text_message(
                    task_result.content,
                    task.context_id,
                    task.id,
                )
                await updater.update_status(
                    TaskState.working,
                    message,
                    metadata=metadata,
                )
            case "input-required":
                message = new_agent_text_message(
                    task_result.content,
                    task.context_id,
                    task.id,
                )
                await updater.update_status(
                    TaskState.input_required,
                    message,
                    metadata=metadata,
                    final=True,
                )
                keep_streaming = False
            case "completed":
                await updater.add_artifact(
                    [Part(root=TextPart(text=task_result.content))],
                    metadata=metadata,
                )
                keep_streaming = False
            case "error":
                raise ServerError(error=InternalError(message=task_result.content))
            case _:
                _logger.warning(f"Unknown task status: {task_result.task_status}")
        return keep_streaming

    @override
    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
