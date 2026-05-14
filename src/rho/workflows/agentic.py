from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from typing import cast

from temporalio import workflow
from temporalio.exceptions import ApplicationError

from ..constants import (
    ITEM_TYPE_ASSISTANT_MESSAGE,
    ITEM_TYPE_TURN_COMPLETE,
    ITEM_TYPE_TURN_STARTED,
    ITEM_TYPE_USER_MESSAGE,
    PHASE_LLM_CALLING,
    PHASE_WAITING_FOR_INPUT,
    QUERY_GET_CONVERSATION_ITEMS,
    QUERY_GET_TURN_STATUS,
    SIGNAL_AGENT_INPUT,
    SIGNAL_AGENT_SHUTDOWN,
    UPDATE_GET_STATE_UPDATE,
    UPDATE_INTERRUPT,
    UPDATE_SHUTDOWN,
    UPDATE_USER_INPUT,
)
from ..models import (
    AgenticWorkflowState,
    AgentInputSignal,
    ConversationItem,
    InterruptRequest,
    InterruptResponse,
    ModelConfig,
    ShutdownRequest,
    ShutdownResponse,
    StateUpdateRequest,
    StateUpdateResponse,
    TurnReplyActivityInput,
    TurnStatus,
    UserInput,
    WorkflowInput,
    WorkflowResult,
)


@workflow.defn(name="AgenticWorkflow")
class AgenticWorkflow:
    def __init__(self) -> None:
        self._conversation_id = ""
        self._config_context_window = 0
        self._history: list[ConversationItem] = []
        self._model_config = ModelConfig()
        self._pending_turns: list[tuple[str, str]] = []
        self._turn_counter = 0
        self._turns_in_run = 0
        self._current_turn_id = ""
        self._phase = PHASE_WAITING_FOR_INPUT
        self._total_tokens = 0
        self._total_cached_tokens = 0
        self._state_version = 0
        self._shutdown_requested = False
        self._draining = False
        self._interrupt_note = ""

    def _append_item(self, item: ConversationItem) -> None:
        item.seq = len(self._history) + 1
        self._history.append(item)
        self._state_version += 1

    def _next_turn_id(self) -> str:
        self._turn_counter += 1
        return f"turn-{self._turn_counter}"

    def _queue_turn(self, message: str) -> str:
        turn_id = self._next_turn_id()
        self._current_turn_id = turn_id
        self._append_item(ConversationItem(type=ITEM_TYPE_TURN_STARTED, turn_id=turn_id))
        self._append_item(ConversationItem(type=ITEM_TYPE_USER_MESSAGE, content=message, turn_id=turn_id))
        self._pending_turns.append((turn_id, message))
        self._phase = PHASE_LLM_CALLING
        self._state_version += 1
        return turn_id

    def _estimate_tokens(self) -> int:
        return sum(len(item.content.split()) for item in self._history if item.content)

    def _turn_count(self) -> int:
        return sum(1 for item in self._history if item.type == ITEM_TYPE_TURN_COMPLETE)

    def _status(self) -> TurnStatus:
        remaining = 100
        if self._config_context_window > 0:
            estimated = self._estimate_tokens()
            remaining = max(
                0,
                min(
                    100,
                    ((self._config_context_window - estimated) * 100) // self._config_context_window,
                ),
            )
        return TurnStatus(
            phase=self._phase,
            current_turn_id=self._current_turn_id,
            iteration_count=self._turns_in_run,
            total_tokens=self._total_tokens,
            total_cached_tokens=self._total_cached_tokens,
            turn_count=self._turn_count(),
            context_window_remaining_percent=remaining,
            context_window_total=self._config_context_window,
        )

    def _state_update(self, request: StateUpdateRequest) -> StateUpdateResponse:
        items = [replace(item) for item in self._history if item.seq > request.since_seq]
        return StateUpdateResponse(
            turn_id=self._current_turn_id,
            items=items,
            status=self._status(),
            compacted=False,
            completed=self._shutdown_requested,
        )

    def _snapshot(self) -> AgenticWorkflowState:
        return AgenticWorkflowState(
            conversation_id=self._conversation_id,
            history=[replace(item) for item in self._history],
            turn_counter=self._turn_counter,
            total_tokens=self._total_tokens,
            total_cached_tokens=self._total_cached_tokens,
        )

    @workflow.query(name=QUERY_GET_CONVERSATION_ITEMS)
    def get_conversation_items(self) -> list[ConversationItem]:
        return [replace(item) for item in self._history]

    @workflow.query(name=QUERY_GET_TURN_STATUS)
    def get_turn_status(self) -> TurnStatus:
        return self._status()

    @workflow.update(name=UPDATE_USER_INPUT)
    async def user_input(self, input: UserInput) -> StateUpdateResponse:
        if not input.content:
            raise ApplicationError("content must not be empty", type="InvalidRequest")
        if self._shutdown_requested:
            raise ApplicationError("session is shutting down", type="InvalidRequest")
        turn_id = self._queue_turn(input.content)
        return StateUpdateResponse(turn_id=turn_id, items=self.get_conversation_items(), status=self._status())

    @workflow.update(name=UPDATE_INTERRUPT)
    async def interrupt(self, _input: InterruptRequest = InterruptRequest()) -> InterruptResponse:
        if self._shutdown_requested:
            raise ApplicationError("session is shutting down", type="InvalidRequest")
        self._interrupt_note = "Interrupted."
        if self._current_turn_id:
            self._append_item(
                ConversationItem(
                    type=ITEM_TYPE_TURN_COMPLETE,
                    turn_id=self._current_turn_id,
                    content="interrupted",
                )
            )
        self._phase = PHASE_WAITING_FOR_INPUT
        self._state_version += 1
        return InterruptResponse(acknowledged=True)

    @workflow.update(name=UPDATE_SHUTDOWN)
    async def shutdown(self, _input: ShutdownRequest = ShutdownRequest()) -> ShutdownResponse:
        if self._shutdown_requested:
            raise ApplicationError("session is already shutting down", type="InvalidRequest")
        self._shutdown_requested = True
        self._phase = PHASE_WAITING_FOR_INPUT
        self._state_version += 1
        return ShutdownResponse(acknowledged=True)

    @workflow.update(name=UPDATE_GET_STATE_UPDATE)
    async def get_state_update(self, request: StateUpdateRequest) -> StateUpdateResponse:
        if self._has_immediately_available_state(request) or self._shutdown_requested or self._draining:
            return self._state_update(request)
        entry_version = self._state_version
        await workflow.wait_condition(
            lambda: self._state_version != entry_version or self._shutdown_requested or self._draining
        )
        return self._state_update(request)

    def _has_immediately_available_state(self, request: StateUpdateRequest) -> bool:
        has_new_items = any(item.seq > request.since_seq for item in self._history)
        return has_new_items or self._phase != request.since_phase

    @workflow.signal(name=SIGNAL_AGENT_INPUT)
    async def agent_input(self, signal: AgentInputSignal) -> None:
        if signal.interrupt:
            self._interrupt_note = "Interrupted."
        self._queue_turn(signal.content)

    @workflow.signal(name=SIGNAL_AGENT_SHUTDOWN)
    async def agent_shutdown(self) -> None:
        self._shutdown_requested = True
        self._phase = PHASE_WAITING_FOR_INPUT
        self._state_version += 1

    async def _process_turn(self, turn_id: str, message: str) -> None:
        self._phase = PHASE_LLM_CALLING
        self._state_version += 1
        interrupt_note = self._interrupt_note
        self._interrupt_note = ""
        try:
            reply = cast(
                str,
                await workflow.execute_activity(
                    "GenerateTurnReply",
                    TurnReplyActivityInput(
                        turn_id=turn_id,
                        message=message,
                        history=[replace(item) for item in self._history],
                        model_config=self._model_config,
                        interrupt_note=interrupt_note,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                ),
            )
        except Exception as error:
            self._append_item(
                ConversationItem(
                    type=ITEM_TYPE_ASSISTANT_MESSAGE,
                    content=f"Turn processing failed: {error}",
                    turn_id=turn_id,
                )
            )
            self._append_item(ConversationItem(type=ITEM_TYPE_TURN_COMPLETE, turn_id=turn_id, content="failed"))
            self._phase = PHASE_WAITING_FOR_INPUT
            self._state_version += 1
            self._turns_in_run += 1
            return
        self._append_item(ConversationItem(type=ITEM_TYPE_ASSISTANT_MESSAGE, content=reply, turn_id=turn_id))
        self._append_item(ConversationItem(type=ITEM_TYPE_TURN_COMPLETE, turn_id=turn_id))
        self._phase = PHASE_WAITING_FOR_INPUT
        self._state_version += 1
        self._turns_in_run += 1
        self._total_tokens += max(1, len(reply.split()))

    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowResult:
        self._conversation_id = input.conversation_id
        self._config_context_window = input.config.model.context_window
        self._model_config = input.config.model
        if input.continued_state is not None:
            self._history = [replace(item) for item in input.continued_state.history]
            self._turn_counter = input.continued_state.turn_counter
            self._total_tokens = input.continued_state.total_tokens
            self._total_cached_tokens = input.continued_state.total_cached_tokens
        else:
            if input.user_message:
                self._queue_turn(input.user_message)

        idle_timeout = timedelta(seconds=input.idle_timeout_seconds)
        while True:
            if self._pending_turns:
                turn_id, message = self._pending_turns.pop(0)
                self._current_turn_id = turn_id
                await self._process_turn(turn_id, message)
                if (
                    self._turns_in_run >= input.max_turns_per_run
                    and not self._pending_turns
                    and not self._shutdown_requested
                ):
                    self._draining = True
                    workflow.continue_as_new(
                        WorkflowInput(
                            conversation_id=self._conversation_id,
                            user_message="",
                            config=input.config,
                            depth=input.depth,
                            continued_state=self._snapshot(),
                            max_turns_per_run=input.max_turns_per_run,
                            idle_timeout_seconds=input.idle_timeout_seconds,
                            crew_name=input.crew_name,
                            crew_agent=input.crew_agent,
                            crew_inputs=input.crew_inputs,
                        )
                    )
            if self._shutdown_requested and not self._pending_turns:
                final_message = next(
                    (
                        item.content
                        for item in reversed(self._history)
                        if item.type == ITEM_TYPE_ASSISTANT_MESSAGE and item.content
                    ),
                    "",
                )
                return WorkflowResult(
                    conversation_id=self._conversation_id,
                    total_iterations=self._turns_in_run,
                    total_tokens=self._total_tokens,
                    total_cached_tokens=self._total_cached_tokens,
                    end_reason="shutdown",
                    final_message=final_message,
                )
            if self._pending_turns:
                continue
            try:
                await workflow.wait_condition(
                    lambda: bool(self._pending_turns) or self._shutdown_requested,
                    timeout=idle_timeout,
                )
            except asyncio.TimeoutError:
                self._draining = True
                workflow.continue_as_new(
                    WorkflowInput(
                        conversation_id=self._conversation_id,
                        user_message="",
                        config=input.config,
                        depth=input.depth,
                        continued_state=self._snapshot(),
                        max_turns_per_run=input.max_turns_per_run,
                        idle_timeout_seconds=input.idle_timeout_seconds,
                        crew_name=input.crew_name,
                        crew_agent=input.crew_agent,
                        crew_inputs=input.crew_inputs,
                    )
                )
