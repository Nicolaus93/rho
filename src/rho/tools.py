from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

from .models import (
    RETRY_DEFAULT,
    RETRY_NONE,
    ConversationItem,
    SessionConfiguration,
    SubcallActivityInput,
    ToolActivityInput,
    ToolActivityOutput,
    ToolRetryPolicy,
    ToolSpec,
)

DEFAULT_TOOL_TIMEOUT_MS = 120_000


def build_builtin_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec("shell", 10_000, RETRY_NONE),
        ToolSpec("shell_command", 10_000, RETRY_NONE),
        ToolSpec("read_file", 30_000, RETRY_DEFAULT),
        ToolSpec("write_file", 30_000, RETRY_NONE),
        ToolSpec("list_dir", 30_000, RETRY_DEFAULT),
        ToolSpec("grep_files", 30_000, RETRY_DEFAULT),
        ToolSpec("exec_command", DEFAULT_TOOL_TIMEOUT_MS, RETRY_NONE),
        ToolSpec("write_stdin", DEFAULT_TOOL_TIMEOUT_MS, RETRY_NONE),
        ToolSpec("delegate_subtask", DEFAULT_TOOL_TIMEOUT_MS, RETRY_NONE),
    ]


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def resolve_tool_timeout(spec_by_name: dict[str, ToolSpec], tool_name: str, args: dict[str, Any] | None) -> timedelta:
    if args:
        timeout_ms = _to_int(args.get("timeout_ms"))
        if timeout_ms and timeout_ms > 0:
            return timedelta(milliseconds=timeout_ms)
    spec = spec_by_name.get(tool_name)
    if spec and spec.default_timeout_ms > 0:
        return timedelta(milliseconds=spec.default_timeout_ms)
    return timedelta(milliseconds=DEFAULT_TOOL_TIMEOUT_MS)


def resolve_retry_policy(spec_by_name: dict[str, ToolSpec], tool_name: str) -> RetryPolicy:
    spec = spec_by_name.get(tool_name)
    policy = spec.retry_policy if spec else None
    if policy is not None:
        if policy.non_retryable:
            return RetryPolicy(maximum_attempts=1)
        max_attempts = policy.max_attempts or 3
        return RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=max_attempts,
        )
    return RetryPolicy(
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(minutes=1),
        maximum_attempts=3,
    )


def tool_activity_error_to_output(call_id: str, error: Exception) -> ToolActivityOutput:
    return ToolActivityOutput(call_id=call_id, content=str(error), success=False)


class ToolsExecutor:
    def __init__(
        self,
        specs: list[ToolSpec],
        cwd: str = "",
        task_queue: str = "",
        approval_mode: str = "",
        exec_policy_rules: str = "",
        conversation_id: str = "",
        depth: int = 0,
        session_config: SessionConfiguration | None = None,
    ) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._cwd = cwd
        self._task_queue = task_queue
        self._session_id = ""
        self._mcp_tool_lookup: dict[str, dict[str, str]] = {}
        self._approval_mode = approval_mode
        self._exec_policy_rules = exec_policy_rules
        self._conversation_id = conversation_id
        self._depth = depth
        self._session_config = session_config

    def with_mcp_context(self, session_id: str, lookup: dict[str, dict[str, str]]) -> "ToolsExecutor":
        self._session_id = session_id
        self._mcp_tool_lookup = lookup
        return self

    def with_policy_context(self, approval_mode: str = "", exec_policy_rules: str = "") -> "ToolsExecutor":
        self._approval_mode = approval_mode
        self._exec_policy_rules = exec_policy_rules
        return self

    async def execute_parallel(self, calls: list[ConversationItem]) -> list[ToolActivityOutput]:
        async def execute_one(call: ConversationItem) -> ToolActivityOutput:
            args = _decode_tool_args(call.arguments)
            activity_name = "ExecuteTool"
            activity_input: ToolActivityInput | SubcallActivityInput
            if call.name == "delegate_subtask":
                prompt = _build_subtask_prompt(args)
                activity_name = "ExecuteSubcallTool"
                activity_input = SubcallActivityInput(
                    call_id=call.call_id,
                    prompt=prompt,
                    parent_conversation_id=self._conversation_id,
                    depth=self._depth,
                    config=self._session_config or SessionConfiguration(),
                    task_queue=self._task_queue,
                )
            else:
                activity_input = ToolActivityInput(
                    call_id=call.call_id,
                    tool_name=call.name,
                    arguments=args,
                    cwd=self._cwd,
                    session_id=self._session_id,
                    mcp_tool_ref=self._mcp_tool_lookup.get(call.name),
                    approval_mode=self._approval_mode,
                    exec_policy_rules=self._exec_policy_rules,
                )
            kwargs: dict[str, Any] = {
                "start_to_close_timeout": resolve_tool_timeout(self._specs, call.name, args),
                "retry_policy": resolve_retry_policy(self._specs, call.name),
            }
            if call.name in {"exec_command", "write_stdin"}:
                kwargs["heartbeat_timeout"] = timedelta(seconds=15)
            if self._task_queue:
                kwargs["task_queue"] = self._task_queue
            try:
                return await workflow.execute_activity(activity_name, activity_input, result_type=ToolActivityOutput, **kwargs)
            except Exception as exc:  # pragma: no cover - exercised in runtime, not unit tests
                return tool_activity_error_to_output(call.call_id, exc)

        return await asyncio.gather(*(execute_one(call) for call in calls))


__all__ = [
    "DEFAULT_TOOL_TIMEOUT_MS",
    "RETRY_DEFAULT",
    "RETRY_NONE",
    "ToolRetryPolicy",
    "ToolSpec",
    "ToolsExecutor",
    "build_builtin_tool_specs",
    "resolve_retry_policy",
    "resolve_tool_timeout",
    "tool_activity_error_to_output",
]


def _decode_tool_args(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        import json

        value = json.loads(arguments)
        return value if isinstance(value, dict) else {"_raw": arguments}
    except Exception:
        return {"_raw": arguments}


def _build_subtask_prompt(arguments: dict[str, Any]) -> str:
    prompt = str(arguments.get("prompt") or "").strip()
    goal = str(arguments.get("goal") or "").strip()
    if prompt and goal:
        return f"{prompt}\n\nSuccess criteria:\n{goal}"
    if prompt:
        return prompt
    return goal
