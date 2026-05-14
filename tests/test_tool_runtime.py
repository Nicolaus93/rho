from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rho.activities.tools import execute_tool
from rho.models import ToolActivityInput
from rho.tool_runtime import (
    Decision,
    ExecPolicyManager,
    Policy,
    build_default_tool_runtime,
    command_might_be_dangerous,
    is_known_safe_command,
    parse_shell_lc_plain_commands,
)


def test_parse_shell_lc_plain_commands_supports_simple_sequences() -> None:
    parsed = parse_shell_lc_plain_commands(["/bin/zsh", "-lc", "pwd && ls"])
    assert parsed == [["pwd"], ["ls"]]


def test_parse_shell_lc_plain_commands_rejects_pipes() -> None:
    assert parse_shell_lc_plain_commands(["/bin/zsh", "-lc", "pwd | wc -l"]) is None


def test_command_safety_helpers() -> None:
    assert is_known_safe_command(["rg", "needle", "."])
    assert is_known_safe_command(["git", "status"])
    assert is_known_safe_command(["find", ".", "-name", "*.py"])
    assert not is_known_safe_command(["sed", "-i", "s/a/b/", "file.txt"])
    assert command_might_be_dangerous(["rm", "-rf", "/tmp/x"])
    assert command_might_be_dangerous(["sudo", "rm", "-rf", "/tmp/x"])


def test_exec_policy_requires_approval_for_unknown_command_in_unless_trusted() -> None:
    evaluation = ExecPolicyManager(Policy(approval_mode="unless-trusted")).evaluate_command(["python3", "-V"])
    assert evaluation.decision == Decision.REQUIRE_APPROVAL
    assert evaluation.requires_approval is True


def test_exec_policy_allows_safe_and_denies_dangerous_commands() -> None:
    safe = ExecPolicyManager(Policy(approval_mode="unless-trusted")).evaluate_command(["git", "status"])
    dangerous = ExecPolicyManager(Policy(approval_mode="never")).evaluate_command(["rm", "-rf", "/tmp/x"])
    assert safe.decision == Decision.ALLOW
    assert dangerous.decision == Decision.DENY


def test_exec_policy_denies_when_rules_are_configured_but_unimplemented() -> None:
    evaluation = ExecPolicyManager(
        Policy(approval_mode="never", rules='prefix_rule(pattern=["git"])')
    ).evaluate_command(["git", "status"])
    assert evaluation.decision == Decision.DENY
    assert "not implemented" in evaluation.reason.lower()


async def test_tool_runtime_routes_filesystem_handlers(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    cwd = str(tmp_path)

    write_result = await runtime.execute(
        ToolActivityInput(
            call_id="write-1",
            tool_name="write_file",
            arguments={"path": "nested/example.txt", "content": "alpha\nbeta\n"},
            cwd=cwd,
        )
    )
    assert write_result.success is True
    assert (tmp_path / "nested" / "example.txt").read_text() == "alpha\nbeta\n"

    read_result = await runtime.execute(
        ToolActivityInput(
            call_id="read-1",
            tool_name="read_file",
            arguments={"file_path": "nested/example.txt", "offset": 2, "limit": 1},
            cwd=cwd,
        )
    )
    assert read_result.success is True
    assert read_result.content == "2: beta\n"

    list_result = await runtime.execute(
        ToolActivityInput(
            call_id="list-1",
            tool_name="list_dir",
            arguments={"dir_path": str(tmp_path), "depth": 2},
            cwd=cwd,
        )
    )
    assert list_result.success is True
    assert list_result.content.splitlines() == ["nested/", "nested/example.txt"]


async def test_list_dir_rejects_relative_dir_path(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    result = await runtime.execute(
        ToolActivityInput(
            call_id="list-relative-1",
            tool_name="list_dir",
            arguments={"dir_path": "relative"},
            cwd=str(tmp_path),
        )
    )
    assert result.success is False
    assert "absolute" in result.content.lower()


async def test_tool_runtime_grep_files_handles_missing_rg(tmp_path: Path, monkeypatch) -> None:
    runtime = build_default_tool_runtime()

    async def raise_missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", raise_missing)
    result = await runtime.execute(
        ToolActivityInput(
            call_id="grep-1",
            tool_name="grep_files",
            arguments={"pattern": "alpha", "path": str(tmp_path)},
            cwd=str(tmp_path),
        )
    )
    assert result.success is False
    assert "ripgrep" in result.content


async def test_tool_runtime_grep_files_content_mode(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    file_path = tmp_path / "notes.txt"
    file_path.write_text("alpha\nbeta\nalpha again\n", encoding="utf-8")

    result = await runtime.execute(
        ToolActivityInput(
            call_id="grep-2",
            tool_name="grep_files",
            arguments={
                "pattern": "alpha",
                "path": str(tmp_path),
                "output_mode": "content",
            },
            cwd=str(tmp_path),
        )
    )
    assert result.success is True
    assert "notes.txt:1:alpha" in result.content


async def test_shell_alias_routes_to_shell_command(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    result = await runtime.execute(
        ToolActivityInput(
            call_id="shell-echo-1",
            tool_name="shell",
            arguments={"command": "printf 'hello'"},
            cwd=str(tmp_path),
        )
    )
    assert result.success is True
    assert result.content == "hello"


async def test_exec_command_and_write_stdin_support_minimal_process_io(
    tmp_path: Path,
) -> None:
    runtime = build_default_tool_runtime()
    start = await runtime.execute(
        ToolActivityInput(
            call_id="exec-1",
            tool_name="exec_command",
            arguments={
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; print(sys.stdin.readline().strip())",
                ],
                "wait_for_exit": False,
            },
            cwd=str(tmp_path),
        )
    )
    assert start.success is True
    process_id = start.content.removeprefix("Started process ")

    finish = await runtime.execute(
        ToolActivityInput(
            call_id="stdin-1",
            tool_name="write_stdin",
            arguments={
                "process_id": process_id,
                "content": "hello\n",
                "close_stdin": True,
                "wait_for_exit": True,
            },
            cwd=str(tmp_path),
        )
    )
    assert finish.success is True
    assert finish.content == "hello"


async def test_shell_command_policy_blocks_dangerous_command(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    result = await runtime.execute(
        ToolActivityInput(
            call_id="shell-1",
            tool_name="shell_command",
            arguments={"command": "rm -rf /tmp/nope", "approval_mode": "never"},
            cwd=str(tmp_path),
        )
    )
    assert result.success is False
    assert "blocked" in result.content.lower()


async def test_shell_command_uses_authoritative_request_policy_fields(
    tmp_path: Path,
) -> None:
    runtime = build_default_tool_runtime()
    result = await runtime.execute(
        ToolActivityInput(
            call_id="shell-2",
            tool_name="shell_command",
            arguments={"command": "printf hello", "approval_mode": "never"},
            cwd=str(tmp_path),
            exec_policy_rules='prefix_rule(pattern=["printf"])',
        )
    )
    assert result.success is False
    assert "not implemented" in result.content.lower()


async def test_execute_tool_activity_uses_runtime(tmp_path: Path) -> None:
    result = await execute_tool(
        ToolActivityInput(
            call_id="activity-1",
            tool_name="write_file",
            arguments={"path": "activity.txt", "content": "hi"},
            cwd=str(tmp_path),
        )
    )
    assert result.success is True
    assert (tmp_path / "activity.txt").read_text() == "hi"


async def test_unsupported_tool_returns_error(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    result = await runtime.execute(
        ToolActivityInput(
            call_id="missing-1",
            tool_name="unknown_tool",
            arguments={},
            cwd=str(tmp_path),
        )
    )
    assert result.success is False
    assert "Unsupported tool" in result.content


async def test_apply_patch_is_not_exposed_by_python_runtime(tmp_path: Path) -> None:
    runtime = build_default_tool_runtime()
    result = await runtime.execute(
        ToolActivityInput(
            call_id="missing-apply",
            tool_name="apply_patch",
            arguments={},
            cwd=str(tmp_path),
        )
    )
    assert result.success is False
    assert "Unsupported tool" in result.content
