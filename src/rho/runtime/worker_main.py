from __future__ import annotations

import argparse
import asyncio
import os

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.worker import Worker

from ..activities import LLMActivities, SessionActivities, execute_tool
from ..client import connect_client, load_client_config
from ..constants import TASK_QUEUE, UPDATE_START_SESSION
from ..models import CLIOverrides, HarnessWorkflowInput, StartSessionRequest, StartSessionResponse
from ..workflows import AgenticWorkflow, ConsolidationWorkflow, HarnessWorkflow, SessionWorkflow
from loguru import logger

from .rho import _default_harness_id, _should_run_interactive
from .tui import RhoApp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temporal-agent-harness-worker")
    parser.add_argument("--cwd", default="", help="Working directory for the session")
    parser.add_argument("--model", default="", help="Model override for the session")
    parser.add_argument("--provider", default="", help="Provider override for the session")
    parser.add_argument("--temporal-host", default="")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--harness-id", default="", help="Harness workflow id override")
    return parser


async def _start_session(client: Client, args: argparse.Namespace) -> str:
    cwd = os.path.abspath(args.cwd or os.getcwd())
    harness_id = args.harness_id or _default_harness_id(cwd)
    overrides = CLIOverrides(cwd=cwd, model=args.model, provider=args.provider)
    harness = await client.start_workflow(
        HarnessWorkflow.run,
        HarnessWorkflowInput(harness_id=harness_id, overrides=overrides),
        id=harness_id,
        task_queue=TASK_QUEUE,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )
    session = await harness.execute_update(
        UPDATE_START_SESSION,
        StartSessionRequest(override_config=overrides),
        result_type=StartSessionResponse,
    )
    return session.session_workflow_id


async def _run_worker(worker: Worker) -> None:
    try:
        await worker.run()
    except asyncio.CancelledError:
        logger.info("Python worker shutdown complete")
        raise
    except Exception:
        logger.exception("Python worker exited with an error")
        raise


async def _run(args: argparse.Namespace) -> None:
    connection_config = load_client_config(args.temporal_host, args.namespace)
    workflow_names = [workflow_type.__name__ for workflow_type in (AgenticWorkflow, SessionWorkflow, HarnessWorkflow, ConsolidationWorkflow)]
    client = await connect_client(config=connection_config)
    logger.info(
        "Connected to Temporal host={} namespace={} tls={}",
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
    logger.info("Registering worker on task queue {}", TASK_QUEUE)
    logger.info("Registered workflows: {}", ", ".join(workflow_names))
    logger.info("Registered activities: {}", ", ".join(activity_names))
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
    logger.info("Starting Python worker run loop")

    if not _should_run_interactive():
        try:
            await worker.run()
        except Exception:
            logger.exception("Python worker exited with an error")
            raise
        finally:
            logger.info("Python worker shutdown complete")
        return

    worker_task = asyncio.create_task(_run_worker(worker))
    try:
        workflow_id = await _start_session(client, args)
        await RhoApp(client, workflow_id).run_async()
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    logger.info("Launching rho worker")
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    main()
