from __future__ import annotations

from temporalio import workflow

from ..constants import (
    AGENT_STATUS_COMPLETED,
    AGENT_STATUS_ERRORED,
    AGENT_STATUS_RUNNING,
    QUERY_GET_AGENT_WORKFLOW_ID,
    SIGNAL_UPDATE_SESSION_STATUS,
)
from ..models import SessionWorkflowInput, UpdateSessionStatusRequest, WorkflowInput, WorkflowResult, default_session_configuration
from .agentic import AgenticWorkflow


@workflow.defn(name="SessionWorkflow")
class SessionWorkflow:
    def __init__(self) -> None:
        self._agent_workflow_id = ""

    @workflow.query(name=QUERY_GET_AGENT_WORKFLOW_ID)
    def get_agent_workflow_id(self) -> str:
        return self._agent_workflow_id

    @workflow.run
    async def run(self, input: SessionWorkflowInput) -> WorkflowResult:
        session_workflow_id = workflow.info().workflow_id
        self._agent_workflow_id = f"{session_workflow_id}/main"
        config = default_session_configuration()
        if input.overrides.cwd:
            config.cwd = input.overrides.cwd
        if input.overrides.model:
            config.model.model = input.overrides.model
        if input.overrides.provider:
            config.model.provider = input.overrides.provider
        if input.overrides.session_task_queue:
            config.session_task_queue = input.overrides.session_task_queue
        config.disable_suggestions = input.overrides.disable_suggestions
        config.memory_enabled = input.overrides.memory_enabled
        config.memory_db_path = input.overrides.memory_db_path
        config.apply_model_profile()

        child = await workflow.start_child_workflow(
            AgenticWorkflow.run,
            WorkflowInput(
                conversation_id=self._agent_workflow_id,
                user_message=input.user_message,
                config=config,
                crew_name=input.crew_name,
                crew_inputs=input.crew_inputs,
            ),
            id=self._agent_workflow_id,
            task_queue=workflow.info().task_queue,
        )
        await workflow.get_external_workflow_handle(input.harness_id).signal(
            SIGNAL_UPDATE_SESSION_STATUS,
            UpdateSessionStatusRequest(
                session_workflow_id=session_workflow_id,
                status=AGENT_STATUS_RUNNING,
            ),
        )
        try:
            result = await child
        except Exception:
            await workflow.get_external_workflow_handle(input.harness_id).signal(
                SIGNAL_UPDATE_SESSION_STATUS,
                UpdateSessionStatusRequest(
                    session_workflow_id=session_workflow_id,
                    status=AGENT_STATUS_ERRORED,
                ),
            )
            raise
        await workflow.get_external_workflow_handle(input.harness_id).signal(
            SIGNAL_UPDATE_SESSION_STATUS,
            UpdateSessionStatusRequest(
                session_workflow_id=session_workflow_id,
                status=AGENT_STATUS_COMPLETED,
            ),
        )
        return result
