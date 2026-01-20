from a2a.client import ClientEvent
from a2a.types import (
    Message,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Self

AgentTaskStatus = Literal["working", "input-required", "completed", "error"]
"""AgentTaskStatus is a type alias for the status of an agent task.

The possible values are:
- `working`: The agent is currently processing the task.
- `input-required`: The agent requires additional input from the user to proceed.
- `completed`: The agent has successfully completed the task.
- `error`: An error occurred during the task execution.
"""

class AgentTaskResult(BaseModel):
    """Result of an agent invocation.

    Attributes
    ----------
        task_status (AgentTaskStatus): The status of the agent task.
        content (str): The content of the agent's response or message.
    
    Attributes meaning
    ----------
    | `task_status`  | `content`                                                            |
    |----------------|----------------------------------------------------------------------|
    | working        | Ongoing task description or progress update.                         |
    | input-required | Description of the required user input or context.                   |
    | completed      | Final response or result of the agent's processing.                  |
    | error          | Error message indicating what went wrong during the task execution.  |
    """

    task_status: AgentTaskStatus
    content: str

    @classmethod
    def from_send_message_stream(
        cls,
        item: ClientEvent | Message,
    ) -> Self:
        if isinstance(item, Message):
            first_part = item.parts[0].root
            if first_part.kind != "text":
                return cls(
                    task_status="error",
                    content=f"Unsupported message part kind in streaming response.",
                )
            return cls(
                task_status="completed",
                content=first_part.text,
            )
        _, event = item
        if isinstance(event, TaskArtifactUpdateEvent):
            first_part = event.artifact.parts[0].root
            if first_part.kind != "text":
                return cls(
                    task_status="error",
                    content=f"Unsupported artifact part kind in streaming response.",
                )
            return cls(
                task_status="completed",
                content=first_part.text,
            )
        elif isinstance(event, TaskStatusUpdateEvent):
            state = event.status.state
            full_message = event.status.message
            first_part = full_message.parts[0].root
            message = first_part.text if first_part.kind == "text" else ""
            match state:
                case TaskState.completed:
                    returned_task_status = "completed"
                case TaskState.input_required:
                    returned_task_status = "input-required"
                case TaskState.working:
                    returned_task_status = "working"
                case TaskState.failed:
                    returned_task_status = "error"
                case _:
                    returned_task_status = "error"
            return cls(
                task_status=returned_task_status,
                content=message,
            )
        else:
            return cls(
                task_status="error",
                content=f"Received unexpected None event in streaming response.",
            )

class AgentState(BaseModel, ABC):
    """Abstract Pydantic model from which agent's state classes should inherit.

    This class combines Pydantic's model validation with abstract state management
    requirements for agent operations. Subclasses should define concrete state models
    while implementing the required abstract methods.

    Attributes
    -------
        bat_extra (Dict[str, Any]): A dictionary for storing extra state information.
            The user should not modify this directly, as it is used internally by the SDK.
        bat_buffer (List): A list used as a buffer for intermediate state data.
            The user should not modify this directly, as it is used internally by the SDK.

    Methods
    -------
        from_query (**abstract**): Factory method to create an agent state from an initial query
        to_task_result (**abstract**): Convert current state to a `AgentTaskResult` object
        update_after_checkpoint_restore: Refresh state after checkpoint restoration
        is_waiting_for_human_input: Check if agent requires human input
    
    Example
    -------
    ```python
    from bat.agent import AgentState, AgentTaskResult
    from typing import List, Optional, Self
    from typing_extensions import override

    class MyAgentState(AgentState):
        user_inputs: List[str] = []
        assistant_outputs: List[str] = []
        question: str = ""
        answer: Optional[str] = None

        @classmethod
        def from_query(cls, query: str) -> Self:
            return cls(
                user_inputs=[query],
                question=query,
            )
        
        @override
        def update_after_checkpoint_restore(self, query: str) -> None:
            self.user_inputs.append(query)
            self.question = query
        
        @override
        def to_task_result(self) -> AgentTaskResult:
            if self.answer is None:
                return AgentTaskResult(
                    task_status="working",
                    content="Processing your request..."
                )
            return AgentTaskResult(
                task_status="completed",
                content=self.answer
            )
    ```
    """
    bat_extra: Dict[str, Any] = {}
    bat_buffer: List = []

    @classmethod
    @abstractmethod
    def from_query(
        cls,
        query: str
    ) -> Self:
        """Instantiate agent state from initial query.

        Factory method called by the execution framework to create a new state instance.
        Alternative to direct initialization, allowing state-specific construction logic.

        Args:
            query: Initial user query to bootstrap agent state

        Returns:
            Self: Fully initialized agent state instance
        """
        pass

    def update_after_checkpoint_restore(self, query: str) -> None:
        """Update state with new query after checkpoint restoration.

        Called by the SDK when restoring from a saved checkpoint. Allows the state
        to synchronize with new execution parameters before resuming the graph.

        Args:
            query: New query to execute with the restored state
        """
        return

    @abstractmethod
    def to_task_result(self) -> AgentTaskResult:
        """Convert current state to a task result object.

        Used to yield execution results during graph processing. This method defines
        how the agent's internal state translates to external-facing task results.

        Returns:
            AgentTaskResult: Task result representation of current state
        """
        pass

    def is_waiting_for_human_input(self) -> bool:
        """Check if agent is blocked waiting for human input.

        Default implementation returns `False`. Override in subclasses to implement
        human-in-the-loop pausing behavior.

        Returns:
            bool: True if agent requires human input to proceed, False otherwise
        """
        return False
