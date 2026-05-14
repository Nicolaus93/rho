from __future__ import annotations

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from rho.activities import (
    LLMActivities,
    SessionActivities,
    execute_tool,
)
from rho.constants import (
    UPDATE_SHUTDOWN,
    UPDATE_START_SESSION,
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
    HarnessWorkflowInput,
    ShutdownRequest,
    ShutdownResponse,
    StartSessionRequest,
    StartSessionResponse,
    StateUpdateResponse,
    TokenUsage,
    UserInput,
)
from rho.workflows import (
    AgenticWorkflow,
    HarnessWorkflow,
    SessionWorkflow,
)


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


async def test_harness_session_orchestration_basics() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        llm_activities = _make_llm_activities()
        async with Worker(
            env.client,
            task_queue="test-harness",
            workflows=[AgenticWorkflow, SessionWorkflow, HarnessWorkflow],
            activities=[llm_activities.generate_turn_reply, llm_activities.execute_llm_call],
        ):
            harness = await env.client.start_workflow(
                HarnessWorkflow.run,
                HarnessWorkflowInput(harness_id="harness-1"),
                id="harness-1",
                task_queue="test-harness",
            )

            response = await harness.execute_update(
                UPDATE_START_SESSION,
                StartSessionRequest(),
                result_type=StartSessionResponse,
            )
            assert response.session_id.startswith("sess-")
            assert response.session_workflow_id.endswith("/main")

            sessions = await harness.query(HarnessWorkflow.get_sessions)
            assert len(sessions) == 1
            assert sessions[0].session_workflow_id.startswith("harness-1/")
            assert sessions[0].workflow_id == response.session_workflow_id
            assert sessions[0].user_message == ""
            assert sessions[0].status in {"running", "completed"}

            agent = env.client.get_workflow_handle(response.session_workflow_id)
            await agent.execute_update(
                UPDATE_USER_INPUT,
                UserInput(content="start session"),
                result_type=StateUpdateResponse,
            )
            await env.sleep(1)
            history = await agent.query(AgenticWorkflow.get_conversation_items)
            assert any(item.type == "user_message" and item.content == "start session" for item in history)
            assert any(item.type == "assistant_message" for item in history)

            shutdown = await agent.execute_update(
                UPDATE_SHUTDOWN,
                ShutdownRequest(reason="complete test"),
                result_type=ShutdownResponse,
            )
            assert shutdown.acknowledged is True
            await agent.result()
            await env.sleep(1)

            sessions = await harness.query(HarnessWorkflow.get_sessions)
            assert sessions[0].status == "completed"
            await harness.terminate("test complete")


async def test_worker_accepts_registered_session_activity() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        llm_activities = _make_llm_activities()
        async with Worker(
            env.client,
            task_queue="test-worker-registration",
            workflows=[AgenticWorkflow, SessionWorkflow, HarnessWorkflow],
            activities=[
                execute_tool,
                SessionActivities(env.client).wait_for_session_ready,
                llm_activities.generate_turn_reply,
                llm_activities.execute_llm_call,
                llm_activities.execute_compact,
                llm_activities.generate_suggestions,
            ],
        ):
            pass
