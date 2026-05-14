from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import uuid

from ..models import ToolActivityOutput
from .execpolicy import Decision, ExecPolicyManager, Policy
from .registry import RuntimeProcess, ToolContext


async def handle_exec_command(context: ToolContext) -> ToolActivityOutput:
    argv = _coerce_exec_argv(context.request.arguments)
    evaluation = _evaluate_command(context, argv)
    if evaluation is not None:
        return evaluation
    return await _run_process(context, argv)


async def handle_shell_command(context: ToolContext) -> ToolActivityOutput:
    command = _coerce_shell_command(context.request.arguments)
    shell_argv = [_shell_executable(), "-lc", command]
    evaluation = _evaluate_command(context, shell_argv)
    if evaluation is not None:
        return evaluation
    return await _run_process(context, shell_argv)


async def handle_write_stdin(context: ToolContext) -> ToolActivityOutput:
    process_id = str(context.request.arguments.get("process_id", ""))
    if not process_id:
        return context.error("process_id is required")
    runtime_process = context.process_store.get(process_id)
    if runtime_process is None:
        return context.error(f"Process not found: {process_id}")
    process = runtime_process.process
    if process.stdin is None:
        return context.error(f"Process has no writable stdin: {process_id}")
    data = str(context.request.arguments.get("content", context.request.arguments.get("input", "")))
    if data:
        process.stdin.write(data.encode("utf-8"))
        await process.stdin.drain()
    if context.request.arguments.get("close_stdin") or context.request.arguments.get("eof"):
        process.stdin.close()
    if context.request.arguments.get("wait_for_exit"):
        stdout, stderr = await _communicate_with_timeout(
            process, timeout_ms=context.request.arguments.get("timeout_ms")
        )
        context.process_store.pop(process_id, None)
        return _format_process_result(context, process, stdout, stderr)
    return ToolActivityOutput(
        call_id=context.request.call_id,
        content=f"Wrote stdin to process {process_id}",
        success=True,
    )


def _coerce_exec_argv(arguments: dict[str, object]) -> list[str]:
    raw_command = arguments.get("command", arguments.get("cmd", arguments.get("argv")))
    if raw_command is None:
        raw_command = arguments.get("_raw")
    if isinstance(raw_command, str):
        return shlex.split(raw_command)
    if isinstance(raw_command, (list, tuple)):
        return [str(part) for part in raw_command]
    raise ValueError("command is required")


def _coerce_shell_command(arguments: dict[str, object]) -> str:
    raw = arguments.get("command", arguments.get("cmd", arguments.get("_raw", "")))
    command = str(raw).strip()
    if not command:
        raise ValueError("command is required")
    return command


def _evaluate_command(context: ToolContext, command: list[str]) -> ToolActivityOutput | None:
    approval_mode = context.request.approval_mode or str(context.request.arguments.get("approval_mode", "") or "never")
    exec_policy_rules = context.request.exec_policy_rules or str(context.request.arguments.get("exec_policy", "") or "")
    policy = Policy(
        approval_mode=approval_mode,
        rules=exec_policy_rules,
    )
    evaluation = ExecPolicyManager(policy).evaluate_command(command)
    if evaluation.decision == Decision.DENY:
        return context.error(evaluation.reason or "Command denied by execution policy.")
    if evaluation.requires_approval:
        return context.error(evaluation.reason or "Command requires approval.")
    return None


async def _run_process(context: ToolContext, command: list[str]) -> ToolActivityOutput:
    stdin_mode = asyncio.subprocess.PIPE
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(context.cwd),
        stdin=stdin_mode,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if context.request.arguments.get("wait_for_exit") is False or context.request.arguments.get("background"):
        process_id = uuid.uuid4().hex
        context.process_store[process_id] = RuntimeProcess(
            process_id=process_id, process=process, command=tuple(command)
        )
        return ToolActivityOutput(
            call_id=context.request.call_id,
            content=f"Started process {process_id}",
            success=True,
        )
    stdin_data = _stdin_payload(context.request.arguments)
    stdout, stderr = await _communicate_with_timeout(
        process,
        stdin_data=stdin_data,
        timeout_ms=context.request.arguments.get("timeout_ms"),
    )
    return _format_process_result(context, process, stdout, stderr)


async def _communicate_with_timeout(
    process: asyncio.subprocess.Process,
    stdin_data: bytes | None = None,
    timeout_ms: object = None,
) -> tuple[bytes, bytes]:
    timeout_seconds = _timeout_seconds(timeout_ms)
    communicate = process.communicate(stdin_data)
    try:
        if timeout_seconds is None:
            return await communicate
        return await asyncio.wait_for(communicate, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("Command timed out") from None


def _format_process_result(
    context: ToolContext,
    process: asyncio.subprocess.Process,
    stdout: bytes,
    stderr: bytes,
) -> ToolActivityOutput:
    stdout_text = stdout.decode("utf-8", errors="replace").rstrip("\n")
    stderr_text = stderr.decode("utf-8", errors="replace").rstrip("\n")
    parts = [part for part in [stdout_text, stderr_text] if part]
    if not parts:
        parts.append(f"Command exited with code {process.returncode}")
    return ToolActivityOutput(
        call_id=context.request.call_id,
        content="\n".join(parts),
        success=process.returncode == 0,
    )


def _stdin_payload(arguments: dict[str, object]) -> bytes | None:
    value = arguments.get("input", arguments.get("stdin", arguments.get("content")))
    if value is None:
        return None
    return str(value).encode("utf-8")


def _timeout_seconds(timeout_ms: object) -> float | None:
    if isinstance(timeout_ms, bool) or timeout_ms is None:
        return None
    if isinstance(timeout_ms, (int, float)) and timeout_ms > 0:
        return float(timeout_ms) / 1000.0
    return None


def _shell_executable() -> str:
    shell = os.environ.get("SHELL")
    if shell:
        return shell
    for candidate in ("zsh", "bash", "sh"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "/bin/sh"
