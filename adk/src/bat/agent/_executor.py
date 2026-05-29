import time
from ..logging import create_logger
from .graph import AgentGraph
from .state import AgentTaskResult
from a2a.helpers import new_text_part, new_text_message, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils.errors import InvalidRequestError, InternalError
from typing import Dict
from typing_extensions import override, Any

logger = create_logger(__name__, "debug")

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
            raise InvalidRequestError(message="AgentExecutor could not validate the request.")

        query = context.get_user_input()
        task = context.current_task
        if not task:
            logger.debug("task not found, creating new task from user message")
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            config = {"configurable": {"thread_id": task.context_id}}
            ts = time.time()
            keep_streaming = True
            prev_item = None
            async for item in self.agent_graph.astream(query, config):
                if item != prev_item:
                    if keep_streaming:
                        metadata: Dict[str, Any] = {
                            'usage': self.agent_graph._get_usage_metadata(ts).model_dump(),
                            'trace': self.agent_graph._get_trace_metadata(ts).model_dump(),
                        }
                        ts = time.time()
                        keep_streaming = await self._process_task_result(task, item, updater, metadata)
                    else:
                        logger.warning("Artifact has been updated: ignoring additional streamed item.")
                prev_item = item
        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            raise InternalError(message="AgentExecutor encountered an error while streaming the response.") from e

    def _request_ok(self, context: RequestContext) -> bool:
        return True

    async def _process_task_result(
        self,
        task: Task,
        task_result: AgentTaskResult,
        updater: TaskUpdater,
        metadata: Dict[str, Any]
    ) -> bool:
        match task_result.task_status:
            case (
                TaskState.TASK_STATE_WORKING
                | TaskState.TASK_STATE_INPUT_REQUIRED
            ) as state:
                message = new_text_message(
                    text=task_result.content,
                    context_id=task.context_id,
                    task_id=task.id,
                )
                await updater.update_status(
                    state=state,
                    message=message,
                    metadata=metadata,
                )
            case TaskState.TASK_STATE_COMPLETED:
                await updater.add_artifact(
                    [new_text_part(task_result.content)],
                    metadata=metadata,
                )
                await updater.update_status(
                    state=TaskState.TASK_STATE_COMPLETED,
                )
            case TaskState.TASK_STATE_FAILED:
                raise InternalError(message=task_result.content)
            case _:
                logger.error(f"Unknown task status: {task_result.task_status}")
        return task_result.task_status == TaskState.TASK_STATE_WORKING

    @override
    async def cancel(
        self,
        request: RequestContext,
        event_queue: EventQueue,
    ) -> Task | None:
        raise UnsupportedOperationError(message="AgentExecutor does not support task cancel operation yet.")
