from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ..models import ToolActivityInput, ToolActivityOutput

ToolHandler = Callable[["ToolContext"], Awaitable[ToolActivityOutput]]


@dataclass
class RuntimeProcess:
    process_id: str
    process: asyncio.subprocess.Process
    command: tuple[str, ...]


@dataclass
class ToolContext:
    request: ToolActivityInput
    cwd: Path
    process_store: dict[str, RuntimeProcess]
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))

    def resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.cwd / candidate).resolve()

    def error(self, message: str) -> ToolActivityOutput:
        return ToolActivityOutput(call_id=self.request.call_id, content=message, success=False)


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    @property
    def names(self) -> set[str]:
        return set(self._handlers)


class ToolRuntime:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry or ToolRegistry()
        self._process_store: dict[str, RuntimeProcess] = {}

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def execute(self, request: ToolActivityInput) -> ToolActivityOutput:
        handler = self._registry.get(request.tool_name)
        if handler is None:
            return ToolActivityOutput(call_id=request.call_id, content=f"Unsupported tool: {request.tool_name}", success=False)
        cwd = Path(request.cwd or os.getcwd()).expanduser()
        if not cwd.is_absolute():
            cwd = (Path(os.getcwd()) / cwd).resolve()
        else:
            cwd = cwd.resolve()
        try:
            return await handler(ToolContext(request=request, cwd=cwd, process_store=self._process_store))
        except Exception as exc:
            return ToolActivityOutput(call_id=request.call_id, content=str(exc), success=False)


def build_default_tool_runtime() -> ToolRuntime:
    from .commands import handle_exec_command, handle_shell_command, handle_write_stdin
    from .filesystem import handle_grep_files, handle_list_dir, handle_read_file, handle_write_file

    registry = ToolRegistry()
    registry.register("read_file", handle_read_file)
    registry.register("write_file", handle_write_file)
    registry.register("list_dir", handle_list_dir)
    registry.register("grep_files", handle_grep_files)
    registry.register("shell", handle_shell_command)
    registry.register("shell_command", handle_shell_command)
    registry.register("exec_command", handle_exec_command)
    registry.register("write_stdin", handle_write_stdin)
    return ToolRuntime(registry)
