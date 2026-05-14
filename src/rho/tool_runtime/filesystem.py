from __future__ import annotations

import asyncio
from pathlib import Path

from ..models import ToolActivityOutput
from .registry import ToolContext


async def handle_read_file(context: ToolContext) -> ToolActivityOutput:
    path = _resolve_input_path(context, "file_path", "path")
    if not path.is_file():
        return context.error(f"File not found: {path}")
    raw = await asyncio.to_thread(path.read_bytes)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    offset = max(_to_int(context.request.arguments.get("offset"), default=1), 1)
    limit = max(_to_int(context.request.arguments.get("limit"), default=2000), 0)
    selected = lines[offset - 1 :] if limit == 0 else lines[offset - 1 : offset - 1 + limit]
    if not selected and text and offset > len(lines):
        return ToolActivityOutput(call_id=context.request.call_id, content="", success=True)
    if not selected and not text:
        return ToolActivityOutput(call_id=context.request.call_id, content="", success=True)
    formatted = []
    for line_number, line in enumerate(selected, start=offset):
        formatted.append(f"{line_number}: {line}" if line.endswith("\n") else f"{line_number}: {line}\n")
    return ToolActivityOutput(call_id=context.request.call_id, content="".join(formatted), success=True)


async def handle_write_file(context: ToolContext) -> ToolActivityOutput:
    path = _resolve_input_path(context, "file_path", "path")
    content = str(context.request.arguments.get("content", ""))
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")
    return ToolActivityOutput(
        call_id=context.request.call_id,
        content=f"Wrote {len(content.encode('utf-8'))} bytes to {path}",
        success=True,
    )


async def handle_list_dir(context: ToolContext) -> ToolActivityOutput:
    dir_path = context.request.arguments.get("dir_path")
    if dir_path is not None and not Path(str(dir_path)).expanduser().is_absolute():
        return context.error("dir_path must be absolute")
    raw_path = dir_path if dir_path is not None else context.request.arguments.get("path", context.cwd)
    path = context.resolve_path(str(raw_path))
    if not path.exists() or not path.is_dir():
        return context.error(f"Directory not found: {path}")
    depth = max(_to_int(context.request.arguments.get("depth"), default=1), 1)
    entries = await asyncio.to_thread(_collect_entries, path, depth)
    offset = max(_to_int(context.request.arguments.get("offset"), default=0), 0)
    limit = max(_to_int(context.request.arguments.get("limit"), default=2000), 0)
    sliced = entries[offset:] if limit == 0 else entries[offset : offset + limit]
    return ToolActivityOutput(call_id=context.request.call_id, content="\n".join(sliced), success=True)


async def handle_grep_files(context: ToolContext) -> ToolActivityOutput:
    pattern = str(context.request.arguments.get("pattern", ""))
    if not pattern:
        return context.error("pattern is required")
    search_root = context.resolve_path(str(context.request.arguments.get("path", context.cwd)))
    output_mode = str(context.request.arguments.get("output_mode", "files_with_matches") or "files_with_matches")
    command = ["rg", "--color", "never", "--no-heading"]
    if output_mode == "content":
        command.append("--line-number")
    elif output_mode == "count":
        command.append("--count")
    else:
        command.append("--files-with-matches")
    include = context.request.arguments.get("include")
    if include:
        command.extend(["--glob", str(include)])
    command.extend([pattern, str(search_root)])
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(context.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return context.error("ripgrep (rg) is not installed")
    stdout, stderr = await process.communicate()
    if process.returncode not in {0, 1}:
        error = stderr.decode("utf-8", errors="replace").strip() or f"rg failed with exit code {process.returncode}"
        return context.error(error)
    output = stdout.decode("utf-8", errors="replace")
    head_limit = max(_to_int(context.request.arguments.get("head_limit"), default=0), 0)
    if head_limit > 0:
        output = "\n".join(output.splitlines()[:head_limit])
    return ToolActivityOutput(call_id=context.request.call_id, content=output.rstrip("\n"), success=True)


def _resolve_input_path(context: ToolContext, *keys: str) -> Path:
    for key in keys:
        value = context.request.arguments.get(key)
        if value:
            return context.resolve_path(str(value))
    raise ValueError(f"Missing required path argument; expected one of {', '.join(keys)}")


def _collect_entries(root: Path, depth: int) -> list[str]:
    entries: list[str] = []

    def visit(current: Path, current_depth: int) -> None:
        children = sorted(
            current.iterdir(),
            key=lambda child: (not child.is_dir(), child.name.lower(), child.name),
        )
        for child in children:
            relative = child.relative_to(root).as_posix()
            entries.append(relative + ("/" if child.is_dir() else ""))
            if child.is_dir() and current_depth < depth:
                visit(child, current_depth + 1)

    visit(root, 1)
    return entries


def _to_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default
