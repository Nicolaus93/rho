from __future__ import annotations

import json

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_agent_harness.activities import LLMActivities
from temporal_agent_harness.constants import PHASE_LLM_CALLING, PHASE_WAITING_FOR_INPUT, TASK_QUEUE, UPDATE_SHUTDOWN, UPDATE_START_SESSION
from temporal_agent_harness.llm import CompactRequest, CompactResponse, LLMRequest, LLMResponse, MultiProviderLLMClient
from temporal_agent_harness.models import (
    ConversationItem,
    HarnessWorkflowInput,
    ShutdownRequest,
    ShutdownResponse,
    StartSessionRequest,
    StartSessionResponse,
    StateUpdateResponse,
    TokenUsage,
    TurnStatus,
    UserInput,
)
from temporal_agent_harness.runtime import rho
from temporal_agent_harness.workflows import AgenticWorkflow, HarnessWorkflow, SessionWorkflow


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


def test_rho_main_starts_session_for_current_directory(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeHarnessHandle:
        async def execute_update(self, update_name, request, *, result_type=None):
            captured["update_name"] = update_name
            captured["request"] = request
            captured["result_type"] = result_type
            return StartSessionResponse(session_id="sess-1", session_workflow_id="rho-test/main")

    class FakeAgentHandle:
        async def execute_update(self, update_name, request, *, result_type=None):
            captured["agent_update_name"] = update_name
            captured["agent_request"] = request
            captured["agent_result_type"] = result_type
            return StateUpdateResponse(
                turn_id="turn-1",
                items=[],
                status=TurnStatus(phase="llm_calling"),
            )

    class FakeClient:
        async def start_workflow(self, workflow, workflow_input, *, id, task_queue, **kwargs):
            captured["workflow"] = workflow
            captured["workflow_input"] = workflow_input
            captured["id"] = id
            captured["task_queue"] = task_queue
            captured["start_kwargs"] = kwargs
            return FakeHarnessHandle()

        def get_workflow_handle(self, workflow_id):
            captured["agent_workflow_id"] = workflow_id
            return FakeAgentHandle()

    async def fake_connect_client(host_port_override: str = "", namespace_override: str = ""):
        captured["temporal_host"] = host_port_override
        captured["namespace"] = namespace_override
        return FakeClient()

    monkeypatch.setattr(rho, "connect_client", fake_connect_client)
    monkeypatch.setattr(rho.os, "getcwd", lambda: str(tmp_path))

    rho.main(["inspect", "the", "repo", "--model", "gpt-4o-mini", "--provider", "openai"])

    workflow_input = captured["workflow_input"]
    request = captured["request"]
    assert isinstance(workflow_input, HarnessWorkflowInput)
    assert workflow_input.overrides.cwd == str(tmp_path)
    assert workflow_input.overrides.model == "gpt-4o-mini"
    assert workflow_input.overrides.provider == "openai"
    assert captured["task_queue"] == TASK_QUEUE
    assert captured["start_kwargs"] == {
        "id_reuse_policy": rho.WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        "id_conflict_policy": rho.WorkflowIDConflictPolicy.USE_EXISTING,
    }
    assert captured["update_name"] == UPDATE_START_SESSION
    assert isinstance(request, StartSessionRequest)
    assert request.user_message == ""
    assert request.override_config is not None
    assert request.override_config.cwd == str(tmp_path)
    assert captured["agent_workflow_id"] == "rho-test/main"
    assert captured["agent_request"] == UserInput(content="inspect the repo")

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "harness_id": captured["id"],
        "session_id": "sess-1",
        "workflow_id": "rho-test/main",
        "cwd": str(tmp_path),
    }


async def test_rho_cli_starts_harness_backed_session(monkeypatch, capsys, tmp_path) -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        llm_activities = _make_llm_activities()
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[AgenticWorkflow, SessionWorkflow, HarnessWorkflow],
            activities=[llm_activities.generate_turn_reply],
        ):
            async def fake_connect_client(host_port_override: str = "", namespace_override: str = ""):
                return env.client

            monkeypatch.setattr(rho, "connect_client", fake_connect_client)

            await rho._main([
                "List",
                "files",
                "in",
                "this",
                "directory",
                "--cwd",
                str(tmp_path),
                "--harness-id",
                "rho-test-harness",
            ])

            output = json.loads(capsys.readouterr().out)
            assert output["harness_id"] == "rho-test-harness"
            assert output["workflow_id"].endswith("/main")
            assert output["cwd"] == str(tmp_path)

            harness = env.client.get_workflow_handle("rho-test-harness")
            sessions = await harness.query(HarnessWorkflow.get_sessions)
            assert len(sessions) == 1
            assert sessions[0].workflow_id == output["workflow_id"]
            assert sessions[0].user_message == ""

            agent = env.client.get_workflow_handle(output["workflow_id"])
            history = await agent.query(AgenticWorkflow.get_conversation_items)
            assert any(item.type == "user_message" and item.content == "List files in this directory" for item in history)
            assert any(item.type == "assistant_message" for item in history)

            shutdown = await agent.execute_update(
                UPDATE_SHUTDOWN,
                ShutdownRequest(reason="test complete"),
                result_type=ShutdownResponse,
            )
            assert shutdown.acknowledged is True
            await agent.result()
            await harness.terminate("test complete")


async def test_rho_interactive_session_prints_replies_in_same_workflow(monkeypatch, capsys, tmp_path) -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        llm_activities = _make_llm_activities()
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[AgenticWorkflow, SessionWorkflow, HarnessWorkflow],
            activities=[llm_activities.generate_turn_reply],
        ):
            async def fake_connect_client(host_port_override: str = "", namespace_override: str = ""):
                return env.client

            prompts = iter(["follow up"])

            async def fake_read_terminal_line(prompt: str) -> str:
                try:
                    return next(prompts)
                except StopIteration as error:
                    raise EOFError() from error

            monkeypatch.setattr(rho, "connect_client", fake_connect_client)
            monkeypatch.setattr(rho, "_should_run_interactive", lambda: True)
            monkeypatch.setattr(rho, "_read_terminal_line", fake_read_terminal_line)

            await rho._main([
                "List",
                "files",
                "in",
                "this",
                "directory",
                "--cwd",
                str(tmp_path),
                "--harness-id",
                "rho-interactive",
            ])

            output = capsys.readouterr().out
            assert "LLM:List files in this directory" in output
            assert "LLM:follow up" in output

            harness = env.client.get_workflow_handle("rho-interactive")
            sessions = await harness.query(HarnessWorkflow.get_sessions)
            assert len(sessions) == 1

            agent = env.client.get_workflow_handle(sessions[0].workflow_id)
            history = await agent.query(AgenticWorkflow.get_conversation_items)
            assert any(item.type == "user_message" and item.content == "follow up" for item in history)

            shutdown = await agent.execute_update(
                UPDATE_SHUTDOWN,
                ShutdownRequest(reason="test complete"),
                result_type=ShutdownResponse,
            )
            assert shutdown.acknowledged is True
            await agent.result()
            await harness.terminate("test complete")
