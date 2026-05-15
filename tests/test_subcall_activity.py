from __future__ import annotations

from typing import cast

from temporalio.client import Client

from rho.activities.subcall import SubcallActivities
from rho.models import SessionConfiguration, SubcallActivityInput, WorkflowInput, WorkflowResult


class _FakeHandle:
    def __init__(self, result: WorkflowResult) -> None:
        self._result = result

    async def result(self, *, result_type=None):
        return self._result


class _FakeClient:
    def __init__(self, result: WorkflowResult) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result

    async def start_workflow(self, workflow_run, workflow_input, *, id: str, task_queue: str):
        self.calls.append(
            {
                "workflow_run": workflow_run,
                "workflow_input": workflow_input,
                "id": id,
                "task_queue": task_queue,
            }
        )
        return _FakeHandle(self._result)


async def test_subcall_activity_enforces_max_depth() -> None:
    config = SessionConfiguration()
    config.tools.max_subcall_depth = 1
    activities = SubcallActivities(client=cast(Client, _FakeClient(WorkflowResult("", 0, 0, 0))))

    result = await activities.execute_subcall_tool(
        SubcallActivityInput(
            call_id="call-1",
            prompt="delegate this",
            parent_conversation_id="parent-1",
            depth=1,
            config=config,
        )
    )

    assert result.success is False
    assert "depth limit" in result.content.lower()


async def test_subcall_activity_starts_child_workflow_and_returns_final_message() -> None:
    config = SessionConfiguration()
    config.tools.max_subcall_depth = 2
    fake_client = _FakeClient(
        WorkflowResult(
            conversation_id="parent-1/subcall/call-2",
            total_iterations=1,
            total_tokens=12,
            total_cached_tokens=0,
            end_reason="subcall_complete",
            final_message="child answer",
        )
    )
    activities = SubcallActivities(client=cast(Client, fake_client))

    result = await activities.execute_subcall_tool(
        SubcallActivityInput(
            call_id="call-2",
            prompt="Solve the delegated part",
            parent_conversation_id="parent-1",
            depth=0,
            config=config,
            task_queue="test-task-queue",
        )
    )

    assert result.success is True
    assert result.content == "child answer"
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["id"] == "parent-1/subcall/call-2"
    assert call["task_queue"] == "test-task-queue"
    workflow_input = cast(WorkflowInput, call["workflow_input"])
    assert workflow_input.conversation_id == "parent-1/subcall/call-2"
    assert workflow_input.user_message == "Solve the delegated part"
    assert workflow_input.depth == 1
