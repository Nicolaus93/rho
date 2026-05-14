from datetime import timedelta

from temporal_agent_harness.tools import (
    RETRY_DEFAULT,
    RETRY_NONE,
    ToolRetryPolicy,
    ToolSpec,
    build_builtin_tool_specs,
    resolve_retry_policy,
    resolve_tool_timeout,
)


def test_resolve_retry_policy_non_retryable() -> None:
    policy = resolve_retry_policy(
        {"shell_command": ToolSpec("shell_command", retry_policy=RETRY_NONE)},
        "shell_command",
    )
    assert policy.maximum_attempts == 1


def test_resolve_retry_policy_retryable() -> None:
    policy = resolve_retry_policy({"read_file": ToolSpec("read_file", retry_policy=RETRY_DEFAULT)}, "read_file")
    assert policy.maximum_attempts == 3
    assert policy.initial_interval == timedelta(seconds=1)
    assert policy.backoff_coefficient == 2.0
    assert policy.maximum_interval == timedelta(minutes=1)


def test_resolve_retry_policy_custom_attempts() -> None:
    policy = resolve_retry_policy(
        {"custom": ToolSpec("custom", retry_policy=ToolRetryPolicy(max_attempts=5))},
        "custom",
    )
    assert policy.maximum_attempts == 5


def test_resolve_tool_timeout_prefers_llm_override() -> None:
    timeout = resolve_tool_timeout(
        {"read_file": ToolSpec("read_file", default_timeout_ms=30_000)},
        "read_file",
        {"timeout_ms": 1500},
    )
    assert timeout == timedelta(milliseconds=1500)


def test_builtin_tool_retry_shapes() -> None:
    spec_map = {spec.name: spec for spec in build_builtin_tool_specs()}
    for name in ["shell", "shell_command", "write_file", "exec_command", "write_stdin"]:
        assert resolve_retry_policy(spec_map, name).maximum_attempts == 1
    for name in ["read_file", "list_dir", "grep_files"]:
        assert resolve_retry_policy(spec_map, name).maximum_attempts == 3
