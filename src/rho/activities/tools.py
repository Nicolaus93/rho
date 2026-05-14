from __future__ import annotations

from temporalio import activity

from ..models import ToolActivityInput, ToolActivityOutput
from ..tool_runtime import build_default_tool_runtime

_RUNTIME = build_default_tool_runtime()


@activity.defn(name="ExecuteTool")
async def execute_tool(input: ToolActivityInput) -> ToolActivityOutput:
    return await _RUNTIME.execute(input)
