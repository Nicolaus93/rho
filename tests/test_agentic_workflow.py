from __future__ import annotations

import asyncio

from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from rho.activities import LLMActivities, LLMActivityInput, LLMActivityOutput
from rho.constants import (
    PHASE_WAITING_FOR_INPUT,
    UPDATE_GET_STATE_UPDATE,
    UPDATE_INTERRUPT,
    UPDATE_SHUTDOWN,
    UPDATE_USER_INPUT,
)
from rho.llm import (
    CompactRequest,
    CompactResponse,
    LLMRequest,
    LLMResponse,
    MultiProviderLLMClient,
)
from rho.models import (
    ConversationItem,
    InterruptRequest,
    InterruptResponse,
    ShutdownRequest,
    ShutdownResponse,
    StateUpdateRequest,
    StateUpdateResponse,
    TokenUsage,
    UserInput,
    WorkflowInput,
)
from rho.workflows import AgenticWorkflow


async def _wait_for_turn_complete(handle, *, since_seq: int, since_phase: str) -> StateUpdateResponse:
    latest = StateUpdateResponse(turn_id="", items=[], status=await handle.query(AgenticWorkflow.get_turn_status))
    current_seq = since_seq
    current_phase = since_phase
    while True:
        latest = await handle.execute_update(
            UPDATE_GET_STATE_UPDATE,
            StateUpdateRequest(since_seq=current_seq, since_phase=current_phase),
            result_type=StateUpdateResponse,
        )
        if any(item.type == "turn_complete" for item in latest.items):
            return latest
        if latest.items:
            current_seq = max(item.seq for item in latest.items)
        current_phase = latest.status.phase


class _FakeProvider:
    async def call(self, request: LLMRequest) -> LLMResponse:
        last_user_message = next(
            (item.content for item in reversed(request.history) if item.type == "user_message" and item.content),
            "",
        )
        return LLMResponse(
            items=[ConversationItem(type="assistant_message", content=f"LLM:{last_user_message}")],
            finish_reason="stop",
            token_usage=TokenUsage.from_counts(output_tokens=max(1, len(last_user_message.split()))),
        )

    async def compact(self, request: CompactRequest) -> CompactResponse:
        return CompactResponse(items=[], token_usage=TokenUsage())


def _make_llm_activities() -> LLMActivities:
    provider = _FakeProvider()
    return LLMActivities(client=MultiProviderLLMClient({"openai": provider, "anthropic": provider}))


async def test_agentic_workflow_handler_basics() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        llm_activities = _make_llm_activities()
        async with Worker(
            env.client,
            task_queue="test-agentic",
            workflows=[AgenticWorkflow],
            activities=[llm_activities.generate_turn_reply, llm_activities.execute_llm_call],
        ):
            handle = await env.client.start_workflow(
                AgenticWorkflow.run,
                WorkflowInput(conversation_id="agentic-1", user_message="hello"),
                id="agentic-1",
                task_queue="test-agentic",
            )
            await env.sleep(1)
            items = await handle.query(AgenticWorkflow.get_conversation_items)
            assert [item.type for item in items[:3]] == [
                "turn_started",
                "user_message",
                "assistant_message",
            ]
            status = await handle.query(AgenticWorkflow.get_turn_status)
            assert status.phase == PHASE_WAITING_FOR_INPUT

            update = await handle.execute_update(
                UPDATE_USER_INPUT,
                UserInput(content="follow up"),
                result_type=StateUpdateResponse,
            )
            assert update.turn_id == "turn-2"

            delta = await _wait_for_turn_complete(handle, since_seq=len(items), since_phase=status.phase)
            assert any(item.type == "assistant_message" and "follow up" in item.content for item in delta.items)

            interrupt = await handle.execute_update(
                UPDATE_INTERRUPT,
                InterruptRequest(),
                result_type=InterruptResponse,
            )
            assert interrupt.acknowledged is True
            shutdown = await handle.execute_update(
                UPDATE_SHUTDOWN,
                ShutdownRequest(reason="done"),
                result_type=ShutdownResponse,
            )
            assert shutdown.acknowledged is True

            result = await handle.result()
            assert result.end_reason == "shutdown"
            assert result.final_message


async def test_get_state_update_blocks_until_state_changes() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        llm_activities = _make_llm_activities()
        async with Worker(
            env.client,
            task_queue="test-agentic-blocking",
            workflows=[AgenticWorkflow],
            activities=[llm_activities.generate_turn_reply, llm_activities.execute_llm_call],
        ):
            handle = await env.client.start_workflow(
                AgenticWorkflow.run,
                WorkflowInput(conversation_id="agentic-blocking-1", user_message="hello"),
                id="agentic-blocking-1",
                task_queue="test-agentic-blocking",
            )
            await env.sleep(1)
            items = await handle.query(AgenticWorkflow.get_conversation_items)
            status = await handle.query(AgenticWorkflow.get_turn_status)

            pending_update = asyncio.create_task(
                handle.execute_update(
                    UPDATE_GET_STATE_UPDATE,
                    StateUpdateRequest(since_seq=len(items), since_phase=status.phase),
                    result_type=StateUpdateResponse,
                )
            )
            await asyncio.sleep(0)
            assert pending_update.done() is False

            await handle.execute_update(
                UPDATE_USER_INPUT,
                UserInput(content="later message"),
                result_type=StateUpdateResponse,
            )
            delta = await pending_update
            assert any(item.type == "user_message" and item.content == "later message" for item in delta.items)


async def test_turn_completion_waits_for_reply_activity() -> None:
    release_reply = asyncio.Event()

    @activity.defn(name="ExecuteLLMCall")
    async def delayed_llm_call(_input: LLMActivityInput) -> LLMActivityOutput:
        await release_reply.wait()
        return LLMActivityOutput(
            items=[ConversationItem(type="assistant_message", content="Handled delayed turn")],
            finish_reason="stop",
            token_usage=TokenUsage.from_counts(output_tokens=3),
        )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-agentic-delayed",
            workflows=[AgenticWorkflow],
            activities=[delayed_llm_call],
        ):
            handle = await env.client.start_workflow(
                AgenticWorkflow.run,
                WorkflowInput(conversation_id="agentic-delayed-1", user_message="hello"),
                id="agentic-delayed-1",
                task_queue="test-agentic-delayed",
            )
            await asyncio.sleep(0)

            status = await handle.query(AgenticWorkflow.get_turn_status)
            assert status.phase != PHASE_WAITING_FOR_INPUT

            pending_update = asyncio.create_task(_wait_for_turn_complete(handle, since_seq=0, since_phase=status.phase))
            await asyncio.sleep(0)
            assert pending_update.done() is False

            release_reply.set()
            delta = await pending_update
            assert any(item.type == "turn_complete" for item in delta.items)
