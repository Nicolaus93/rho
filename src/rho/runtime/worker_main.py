from __future__ import annotations

import asyncio
import logging

from temporalio.worker import Worker

from ..activities import LLMActivities, SessionActivities, execute_tool
from ..client import connect_client, load_client_config
from ..constants import TASK_QUEUE
from ..workflows import AgenticWorkflow, ConsolidationWorkflow, HarnessWorkflow, SessionWorkflow


LOGGER = logging.getLogger(__name__)


async def _run() -> None:
    connection_config = load_client_config()
    workflow_names = [workflow_type.__name__ for workflow_type in (AgenticWorkflow, SessionWorkflow, HarnessWorkflow, ConsolidationWorkflow)]
    client = await connect_client(config=connection_config)
    LOGGER.info(
        "Connected to Temporal host=%s namespace=%s tls=%s",
        connection_config.host_port,
        connection_config.namespace,
        connection_config.tls is not None,
    )
    session_activities = SessionActivities(client)
    llm_activities = LLMActivities()
    activity_names = [
        "generate_turn_reply",
        "execute_tool",
        "wait_for_session_ready",
        "execute_llm_call",
        "execute_compact",
        "generate_suggestions",
    ]
    LOGGER.info("Registering worker on task queue %s", TASK_QUEUE)
    LOGGER.info("Registered workflows: %s", ", ".join(workflow_names))
    LOGGER.info("Registered activities: %s", ", ".join(activity_names))
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgenticWorkflow, SessionWorkflow, HarnessWorkflow, ConsolidationWorkflow],
        activities=[
            llm_activities.generate_turn_reply,
            execute_tool,
            session_activities.wait_for_session_ready,
            llm_activities.execute_llm_call,
            llm_activities.execute_compact,
            llm_activities.generate_suggestions,
        ],
    )
    LOGGER.info("Starting Python worker run loop")
    try:
        await worker.run()
    except Exception:
        LOGGER.exception("Python worker exited with an error")
        raise
    finally:
        LOGGER.info("Python worker shutdown complete")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    LOGGER.info("Launching rho worker")
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
