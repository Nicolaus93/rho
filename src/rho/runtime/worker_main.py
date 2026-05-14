from __future__ import annotations

import argparse
import asyncio
import os

from loguru import logger
from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.worker import Worker

from ..activities import LLMActivities, SessionActivities, execute_tool
from ..client import connect_client, load_client_config
from ..constants import TASK_QUEUE, UPDATE_SHUTDOWN, UPDATE_START_SESSION
from ..models import (
    CLIOverrides,
    HarnessWorkflowInput,
    ShutdownRequest,
    StartSessionRequest,
    StartSessionResponse,
)
from ..workflows import (
    AgenticWorkflow,
    ConsolidationWorkflow,
    HarnessWorkflow,
    SessionWorkflow,
)
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


async def _start_session(client: Client, harness_id: str, args: argparse.Namespace) -> str:
    cwd = os.path.abspath(args.cwd or os.getcwd())
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


async def _shutdown_session(client: Client, agent_workflow_id: str, harness_id: str) -> None:
    session_workflow_id = agent_workflow_id.rsplit("/", 1)[0]
    try:
        handle = client.get_workflow_handle(agent_workflow_id)
        await handle.execute_update(UPDATE_SHUTDOWN, ShutdownRequest(reason="worker shutdown"))
        await asyncio.wait_for(handle.result(), timeout=2.0)
    except asyncio.TimeoutError:
        logger.debug("Agent workflow did not complete within timeout after shutdown signal")
    except Exception:
        logger.debug("Could not shut down agent workflow {}", agent_workflow_id)
    try:
        await asyncio.wait_for(client.get_workflow_handle(session_workflow_id).result(), timeout=2.0)
    except asyncio.TimeoutError:
        logger.debug("Session workflow did not complete within timeout")
    except Exception:
        logger.debug("Could not wait on session workflow {}", session_workflow_id)
    try:
        await client.get_workflow_handle(harness_id).execute_update(
            UPDATE_SHUTDOWN, ShutdownRequest(reason="worker shutdown")
        )
    except Exception:
        logger.debug("Could not shut down harness workflow {}", harness_id)


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
    workflow_names = [
        workflow_type.__name__
        for workflow_type in (
            AgenticWorkflow,
            SessionWorkflow,
            HarnessWorkflow,
            ConsolidationWorkflow,
        )
    ]
    cwd = os.path.abspath(args.cwd or os.getcwd())
    harness_id = args.harness_id or _default_harness_id(cwd)
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
        workflows=[
            AgenticWorkflow,
            SessionWorkflow,
            HarnessWorkflow,
            ConsolidationWorkflow,
        ],
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
    agent_workflow_id: str | None = None
    try:
        agent_workflow_id = await _start_session(client, harness_id, args)
        await RhoApp(client, agent_workflow_id).run_async()
    finally:
        if agent_workflow_id is not None:
            await _shutdown_session(client, agent_workflow_id, harness_id)
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
