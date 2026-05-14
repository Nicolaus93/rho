from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow

from ..models import ConsolidationWorkflowInput, ConsolidationWorkflowResult, ConsolidationWorkflowState


@workflow.defn(name="ConsolidationWorkflow")
class ConsolidationWorkflow:
    def __init__(self) -> None:
        self._state = ConsolidationWorkflowState()
        self._shutdown = False

    @workflow.signal(name="add_items")
    async def add_items(self, items: list[str]) -> None:
        self._state.items.extend(items)

    @workflow.signal(name="shutdown")
    async def shutdown(self) -> None:
        self._shutdown = True

    @workflow.query(name="get_item_count")
    def get_item_count(self) -> int:
        return len(self._state.items)

    @workflow.run
    async def run(self, input: ConsolidationWorkflowInput) -> ConsolidationWorkflowResult:
        if input.continued_state is not None:
            self._state = input.continued_state
        else:
            self._state = ConsolidationWorkflowState(items=list(input.items))
        if len(self._state.items) >= input.max_items_per_run:
            workflow.continue_as_new(
                ConsolidationWorkflowInput(
                    continued_state=ConsolidationWorkflowState(
                        items=list(self._state.items),
                        continue_count=self._state.continue_count + 1,
                    ),
                    max_items_per_run=input.max_items_per_run,
                    idle_timeout_seconds=input.idle_timeout_seconds,
                )
            )
        try:
            await workflow.wait_condition(lambda: self._shutdown, timeout=timedelta(seconds=input.idle_timeout_seconds))
        except asyncio.TimeoutError:
            workflow.continue_as_new(
                ConsolidationWorkflowInput(
                    continued_state=ConsolidationWorkflowState(
                        items=list(self._state.items),
                        continue_count=self._state.continue_count + 1,
                    ),
                    max_items_per_run=input.max_items_per_run,
                    idle_timeout_seconds=input.idle_timeout_seconds,
                )
            )
        return ConsolidationWorkflowResult(
            item_count=len(self._state.items),
            continue_count=self._state.continue_count,
        )
