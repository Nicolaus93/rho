from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from ..client import connect_client
from ..constants import TASK_QUEUE
from ..models import InterruptRequest, ShutdownRequest, UserInput, WorkflowInput, default_session_configuration
from ..workflows import AgenticWorkflow


async def _cmd_start(args: argparse.Namespace) -> None:
    client = await connect_client(args.temporal_host, args.namespace)
    workflow_id = f"py-agent-{uuid.uuid4().hex[:8]}"
    config = default_session_configuration()
    if args.model:
        config.model.model = args.model
    handle = await client.start_workflow(
        AgenticWorkflow.run,
        WorkflowInput(conversation_id=workflow_id, user_message=args.message, config=config),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    print(json.dumps({"workflow_id": workflow_id, "run_id": handle.result_run_id}))


async def _cmd_send(args: argparse.Namespace) -> None:
    client = await connect_client(args.temporal_host, args.namespace)
    handle = client.get_workflow_handle(args.workflow_id)
    result = await handle.execute_update(AgenticWorkflow.user_input, UserInput(content=args.message))
    print(json.dumps(result, default=lambda value: value.__dict__))


async def _cmd_history(args: argparse.Namespace) -> None:
    client = await connect_client(args.temporal_host, args.namespace)
    handle = client.get_workflow_handle(args.workflow_id)
    items = await handle.query(AgenticWorkflow.get_conversation_items)
    print(json.dumps(items, default=lambda value: value.__dict__, indent=2))


async def _cmd_interrupt(args: argparse.Namespace) -> None:
    client = await connect_client(args.temporal_host, args.namespace)
    handle = client.get_workflow_handle(args.workflow_id)
    result = await handle.execute_update(AgenticWorkflow.interrupt, InterruptRequest())
    print(json.dumps(result, default=lambda value: value.__dict__))


async def _cmd_end(args: argparse.Namespace) -> None:
    client = await connect_client(args.temporal_host, args.namespace)
    handle = client.get_workflow_handle(args.workflow_id)
    result = await handle.execute_update(AgenticWorkflow.shutdown, ShutdownRequest(reason=args.reason))
    print(json.dumps(result, default=lambda value: value.__dict__))


async def _main() -> None:
    parser = argparse.ArgumentParser(prog="temporal-agent-harness-client")
    parser.add_argument("--temporal-host", default="")
    parser.add_argument("--namespace", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--message", required=True)
    start.add_argument("--model", default="gpt-4o-mini")

    send = subparsers.add_parser("send")
    send.add_argument("--workflow-id", required=True)
    send.add_argument("--message", required=True)

    history = subparsers.add_parser("history")
    history.add_argument("--workflow-id", required=True)

    interrupt = subparsers.add_parser("interrupt")
    interrupt.add_argument("--workflow-id", required=True)

    end = subparsers.add_parser("end")
    end.add_argument("--workflow-id", required=True)
    end.add_argument("--reason", default="")

    args = parser.parse_args()
    if args.command == "start":
        await _cmd_start(args)
    elif args.command == "send":
        await _cmd_send(args)
    elif args.command == "history":
        await _cmd_history(args)
    elif args.command == "interrupt":
        await _cmd_interrupt(args)
    else:
        await _cmd_end(args)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover
    main()
