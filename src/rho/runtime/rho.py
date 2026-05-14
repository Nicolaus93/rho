from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Sequence

from temporalio.client import Client, WorkflowHandle
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

from ..client import connect_client
from ..constants import (
    ITEM_TYPE_ASSISTANT_MESSAGE,
    ITEM_TYPE_TURN_COMPLETE,
    TASK_QUEUE,
    UPDATE_GET_STATE_UPDATE,
    UPDATE_SHUTDOWN,
    UPDATE_START_SESSION,
    UPDATE_USER_INPUT,
)
from ..models import (
    CLIOverrides,
    HarnessWorkflowInput,
    ShutdownRequest,
    StartSessionRequest,
    StartSessionResponse,
    StateUpdateRequest,
    StateUpdateResponse,
    UserInput,
)
from ..workflows import AgenticWorkflow, HarnessWorkflow


def _default_harness_id(cwd: str) -> str:
    digest = hashlib.sha1(cwd.encode("utf-8")).hexdigest()[:8]
    return f"rho-{digest}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rho")
    parser.add_argument("message", nargs="*", help="Initial message for the new session")
    parser.add_argument("--cwd", default="", help="Working directory for the started session")
    parser.add_argument("--model", default="", help="Model override for the session")
    parser.add_argument("--provider", default="", help="Provider override for the session")
    parser.add_argument("--temporal-host", default="")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--harness-id", default="", help="Optional harness workflow id override")
    return parser


async def _start_session(args: argparse.Namespace) -> dict[str, str]:
    _, session = await _connect_and_start_session(args)
    return session


async def _connect_and_start_session(args: argparse.Namespace) -> tuple[Client, dict[str, str]]:
    cwd = os.path.abspath(args.cwd or os.getcwd())
    harness_id = args.harness_id or _default_harness_id(cwd)
    overrides = CLIOverrides(cwd=cwd, model=args.model, provider=args.provider)

    client = await connect_client(args.temporal_host, args.namespace)
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
    return client, {
        "harness_id": harness_id,
        "session_id": session.session_id,
        "workflow_id": session.session_workflow_id,
        "cwd": cwd,
    }


def _should_run_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


async def _read_terminal_line(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def _wait_for_turn_completion(
    handle: WorkflowHandle,
    *,
    since_seq: int,
    since_phase: str,
    target_turn_id: str | None,
) -> tuple[list[str], int, str]:
    assistant_messages: list[str] = []
    resolved_turn_id = target_turn_id
    while True:
        delta = await handle.execute_update(
            UPDATE_GET_STATE_UPDATE,
            StateUpdateRequest(since_seq=since_seq, since_phase=since_phase),
            result_type=StateUpdateResponse,
        )
        if resolved_turn_id is None and delta.turn_id:
            resolved_turn_id = delta.turn_id
        turn_completed = False
        for item in delta.items:
            since_seq = max(since_seq, item.seq)
            if resolved_turn_id and item.turn_id != resolved_turn_id:
                continue
            if item.type == ITEM_TYPE_ASSISTANT_MESSAGE and item.content:
                assistant_messages.append(item.content)
            if item.type == ITEM_TYPE_TURN_COMPLETE:
                turn_completed = True
        since_phase = delta.status.phase
        if resolved_turn_id and turn_completed:
            return assistant_messages, since_seq, since_phase
        if delta.completed:
            return assistant_messages, since_seq, since_phase


async def _run_interactive_session(client: Client, workflow_id: str) -> None:
    handle = client.get_workflow_handle(workflow_id)
    since_seq = 0
    since_phase = ""

    while True:
        try:
            user_message = (await _read_terminal_line("> ")).strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            await handle.execute_update(
                UPDATE_SHUTDOWN,
                ShutdownRequest(reason="rho interactive session ended"),
            )
            return

        update = await handle.execute_update(
            UPDATE_USER_INPUT,
            UserInput(content=user_message),
            result_type=StateUpdateResponse,
        )
        assistant_messages, since_seq, since_phase = await _wait_for_turn_completion(
            handle,
            since_seq=since_seq,
            since_phase=since_phase,
            target_turn_id=update.turn_id,
        )
        for message in assistant_messages:
            print(message)


async def _main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    initial_message = " ".join(args.message).strip()
    client, session = await _connect_and_start_session(args)
    if not _should_run_interactive():
        if initial_message:
            handle = client.get_workflow_handle(session["workflow_id"])
            await handle.execute_update(
                UPDATE_USER_INPUT,
                UserInput(content=initial_message),
                result_type=StateUpdateResponse,
            )
        print(json.dumps(session))
        return
    if initial_message:
        handle = client.get_workflow_handle(session["workflow_id"])
        update = await handle.execute_update(
            UPDATE_USER_INPUT,
            UserInput(content=initial_message),
            result_type=StateUpdateResponse,
        )
        assistant_messages, _, _ = await _wait_for_turn_completion(
            handle,
            since_seq=0,
            since_phase="",
            target_turn_id=update.turn_id,
        )
        for message in assistant_messages:
            print(message)
    await _run_interactive_session(client, session["workflow_id"])


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(_main(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
