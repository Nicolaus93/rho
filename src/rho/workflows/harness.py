from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from temporalio import workflow

from ..constants import (
    AGENT_STATUS_PENDING_INIT,
    QUERY_GET_SESSIONS,
    SIGNAL_UPDATE_SESSION_STATUS,
    UPDATE_SHUTDOWN,
    UPDATE_START_SESSION,
)
from ..models import (
    CLIOverrides,
    HarnessWorkflowInput,
    HarnessWorkflowState,
    SessionWorkflowInput,
    SessionEntry,
    ShutdownRequest,
    StartSessionRequest,
    StartSessionResponse,
    UpdateSessionStatusRequest,
)
from .session import SessionWorkflow


def _merge_cli_overrides(base: CLIOverrides, overlay: CLIOverrides | None) -> CLIOverrides:
    if overlay is None:
        return replace(base)
    result = replace(base)
    if overlay.cwd:
        result.cwd = overlay.cwd
    if overlay.codex_home:
        result.codex_home = overlay.codex_home
    if overlay.model:
        result.model = overlay.model
    if overlay.provider:
        result.provider = overlay.provider
    if overlay.session_task_queue:
        result.session_task_queue = overlay.session_task_queue
    if overlay.disable_suggestions:
        result.disable_suggestions = True
    if overlay.memory_enabled:
        result.memory_enabled = True
    if overlay.memory_db_path:
        result.memory_db_path = overlay.memory_db_path
    return result


@workflow.defn(name="HarnessWorkflow")
class HarnessWorkflow:
    def __init__(self) -> None:
        self._state = HarnessWorkflowState(harness_id="")
        self._version = 0
        self._shutdown_requested = False

    def _touch(self) -> None:
        self._version += 1

    def _snapshot(self) -> HarnessWorkflowState:
        return HarnessWorkflowState(
            harness_id=self._state.harness_id,
            overrides=replace(self._state.overrides),
            sessions=[replace(session) for session in self._state.sessions],
            session_counter=self._state.session_counter,
        )

    @workflow.query(name=QUERY_GET_SESSIONS)
    def get_sessions(self) -> list[SessionEntry]:
        return [replace(session) for session in self._state.sessions]

    @workflow.signal(name=SIGNAL_UPDATE_SESSION_STATUS)
    async def update_session_status(self, request: UpdateSessionStatusRequest) -> None:
        for index, session in enumerate(self._state.sessions):
            if session.session_workflow_id == request.session_workflow_id:
                updated = replace(session)
                if request.status:
                    updated.status = request.status
                if request.name:
                    updated.name = request.name
                self._state.sessions[index] = updated
                self._touch()
                return

    @workflow.update(name=UPDATE_START_SESSION)
    async def start_session(self, request: StartSessionRequest) -> StartSessionResponse:
        await workflow.wait_condition(lambda: bool(self._state.harness_id))
        self._state.session_counter += 1
        now = workflow.now()
        session_id = f"sess-{now.astimezone(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{self._state.session_counter}"
        session_workflow_id = f"{self._state.harness_id}/{session_id}"
        agent_workflow_id = f"{session_workflow_id}/main"
        overrides = _merge_cli_overrides(self._state.overrides, request.override_config)
        self._state.sessions.append(
            SessionEntry(
                session_id=session_id,
                session_workflow_id=session_workflow_id,
                workflow_id=agent_workflow_id,
                user_message="",
                model=overrides.model,
                status=AGENT_STATUS_PENDING_INIT,
                started_at=datetime.fromtimestamp(now.timestamp(), tz=timezone.utc),
                crew_type=request.crew_type,
            )
        )
        self._touch()
        await workflow.start_child_workflow(
            SessionWorkflow.run,
            SessionWorkflowInput(
                session_id=session_id,
                harness_id=self._state.harness_id,
                user_message="",
                overrides=overrides,
                crew_name=request.crew_name,
                crew_inputs=request.crew_inputs,
            ),
            id=session_workflow_id,
            task_queue=workflow.info().task_queue,
            parent_close_policy=workflow.ParentClosePolicy.ABANDON,
        )
        await workflow.wait_condition(
            lambda: any(s.session_id == session_id and s.status != AGENT_STATUS_PENDING_INIT for s in self._state.sessions)
        )
        return StartSessionResponse(session_id=session_id, session_workflow_id=agent_workflow_id)

    @workflow.update(name=UPDATE_SHUTDOWN)
    async def shutdown(self, _input: ShutdownRequest = ShutdownRequest()) -> None:
        self._shutdown_requested = True
        self._touch()

    @workflow.run
    async def run(self, input: HarnessWorkflowInput) -> None:
        if input.continued_state is not None:
            self._state = input.continued_state
        else:
            self._state = HarnessWorkflowState(harness_id=input.harness_id, overrides=input.overrides)
        idle_timeout = timedelta(seconds=input.idle_timeout_seconds)
        while True:
            version = self._version
            try:
                await workflow.wait_condition(
                    lambda: self._version != version or self._shutdown_requested,
                    timeout=idle_timeout,
                )
            except asyncio.TimeoutError:
                workflow.continue_as_new(
                    HarnessWorkflowInput(
                        harness_id=self._state.harness_id,
                        overrides=self._state.overrides,
                        continued_state=self._snapshot(),
                        idle_timeout_seconds=input.idle_timeout_seconds,
                    )
                )
            if self._shutdown_requested:
                return
