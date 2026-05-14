from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import activity
from temporalio.client import Client

from ..constants import QUERY_GET_AGENT_WORKFLOW_ID
from ..models import WaitForSessionReadyInput, WaitForSessionReadyOutput


class SessionActivities:
    def __init__(self, client: Client) -> None:
        self._client = client

    @activity.defn(name="WaitForSessionReady")
    async def wait_for_session_ready(self, input: WaitForSessionReadyInput) -> WaitForSessionReadyOutput:
        handle = self._client.get_workflow_handle(input.session_workflow_id)
        deadline = asyncio.get_running_loop().time() + (input.timeout_ms / 1000)
        while True:
            agent_workflow_id = await handle.query(QUERY_GET_AGENT_WORKFLOW_ID)
            if agent_workflow_id:
                return WaitForSessionReadyOutput(agent_workflow_id=agent_workflow_id)
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"session {input.session_workflow_id} did not become ready")
            await asyncio.sleep(
                max(
                    input.poll_interval_ms / 1000,
                    timedelta(milliseconds=50).total_seconds(),
                )
            )
