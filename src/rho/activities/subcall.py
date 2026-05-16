from __future__ import annotations

from temporalio import activity
from temporalio.client import Client

from ..constants import TASK_QUEUE
from ..models import SubcallActivityInput, ToolActivityOutput, WorkflowInput, WorkflowResult
from ..workflows import AgenticWorkflow


class SubcallActivities:
    def __init__(self, client: Client) -> None:
        self._client = client

    @activity.defn(name="ExecuteSubcallTool")
    async def execute_subcall_tool(self, input: SubcallActivityInput) -> ToolActivityOutput:
        max_depth = input.config.tools.max_subcall_depth
        child_depth = input.depth + 1
        if child_depth > max_depth:
            return ToolActivityOutput(
                call_id=input.call_id,
                content=f"Subtask delegation depth limit reached ({max_depth}).",
                success=False,
            )

        child_workflow_id = f"{input.parent_conversation_id}/subcall/{input.call_id}"
        child_handle = await self._client.start_workflow(
            AgenticWorkflow.run,
            WorkflowInput(
                conversation_id=child_workflow_id,
                user_message=input.prompt,
                config=input.config,
                depth=child_depth,
            ),
            id=child_workflow_id,
            task_queue=input.task_queue or TASK_QUEUE,
        )
        result = await child_handle.result()
        return ToolActivityOutput(
            call_id=input.call_id,
            content=result.final_message or "",
            success=bool(result.final_message),
        )
