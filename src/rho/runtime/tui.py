from __future__ import annotations

import asyncio
import re

from rich.markdown import Markdown
from rich.text import Text
from temporalio.client import Client
from textual import events
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Header, Input, RichLog, Static

from ..constants import (
    ITEM_TYPE_ASSISTANT_MESSAGE,
    ITEM_TYPE_FUNCTION_CALL,
    ITEM_TYPE_FUNCTION_CALL_OUTPUT,
    ITEM_TYPE_TURN_COMPLETE,
    PHASE_APPROVAL_PENDING,
    PHASE_COMPACTING,
    PHASE_ESCALATION_PENDING,
    PHASE_LLM_CALLING,
    PHASE_TOOL_EXECUTING,
    PHASE_WAITING_FOR_AGENTS,
    PHASE_WAITING_FOR_INPUT,
    UPDATE_GET_STATE_UPDATE,
    UPDATE_SHUTDOWN,
    UPDATE_USER_INPUT,
)
from ..models import (
    ShutdownRequest,
    StateUpdateRequest,
    StateUpdateResponse,
    TurnStatus,
    UserInput,
)

_PHASE_LABELS: dict[str, str] = {
    PHASE_WAITING_FOR_INPUT: "ready",
    PHASE_LLM_CALLING: "thinking…",
    PHASE_TOOL_EXECUTING: "running tool",
    PHASE_COMPACTING: "compacting…",
    PHASE_APPROVAL_PENDING: "⚠ approval needed",
    PHASE_ESCALATION_PENDING: "⚠ escalation pending",
    PHASE_WAITING_FOR_AGENTS: "waiting for agents…",
}


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def update_status(self, status: TurnStatus) -> None:
        phase = _PHASE_LABELS.get(status.phase, status.phase or "ready")
        tools = f"  [{', '.join(status.tools_in_flight)}]" if status.tools_in_flight else ""
        self.update(
            f"[bold]{phase}{tools}[/bold]"
            f"   tokens: {status.total_tokens:,}"
            f"   ctx: {status.context_window_remaining_percent}%"
            f"   turns: {status.turn_count}"
        )


class RhoApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #chat-log {
        height: 1fr;
        padding: 1 2;
        scrollbar-gutter: stable;
    }
    Input {
        border-top: solid $primary;
    }
    """

    def __init__(self, client: Client, workflow_id: str) -> None:
        super().__init__()
        self._client = client
        self._workflow_id = workflow_id
        self._since_seq = 0
        self._since_phase = ""
        self._message_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._log_entries: list[object] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat-log", markup=True, highlight=False, wrap=True)
        yield StatusBar("ready", id="status")
        yield Input(placeholder="Message…  (ctrl+c to quit)")

    async def on_mount(self) -> None:
        self.title = "rho"
        self.sub_title = self._workflow_id
        self.query_one(Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        event.input.clear()
        self._history_index = None
        self._history_draft = ""
        if not message:
            return
        if message.lower() in {"exit", "quit", "/exit", "/quit"}:
            await self.action_quit()
            return
        self._message_history.append(message)
        self._append_log_entry(Text.assemble(("You  ", "bold cyan"), message, "\n"))
        event.input.disabled = True
        self._send_message(message)  # type: ignore[unused-coroutine]

    async def on_key(self, event: events.Key) -> None:
        if event.key not in {"up", "down"}:
            return
        inp = self.query_one(Input)
        if self.focused is not inp or inp.disabled or not self._message_history:
            return
        if event.key == "up":
            self._show_previous_message(inp)
        else:
            self._show_next_message(inp)
        event.stop()

    def _show_previous_message(self, inp: Input) -> None:
        if self._history_index is None:
            self._history_draft = inp.value
            self._history_index = len(self._message_history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        self._set_input_value(inp, self._message_history[self._history_index])

    def _show_next_message(self, inp: Input) -> None:
        if self._history_index is None:
            return
        next_index = self._history_index + 1
        if next_index >= len(self._message_history):
            self._history_index = None
            self._set_input_value(inp, self._history_draft)
            return
        self._history_index = next_index
        self._set_input_value(inp, self._message_history[self._history_index])

    def _set_input_value(self, inp: Input, value: str) -> None:
        inp.value = value
        inp.cursor_position = len(value)

    def _append_log_entry(self, renderable: object) -> int:
        self._log_entries.append(renderable)
        self._render_log()
        return len(self._log_entries) - 1

    def _replace_log_entry(self, index: int, renderable: object) -> None:
        self._log_entries[index] = renderable
        self._render_log()

    def _render_log(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        for entry in self._log_entries:
            log.write(entry)

    async def _stream_assistant_reply(self, message: str) -> None:
        header_index = self._append_log_entry(Text.assemble(("Assistant  ", "bold green")))
        body_index = self._append_log_entry(Text(""))
        blank_index = self._append_log_entry("")
        del header_index, blank_index

        streamed = ""
        for token in self._iter_stream_tokens(message):
            streamed += token
            self._replace_log_entry(body_index, Text(streamed))
            if token.strip():
                await asyncio.sleep(0.035)

        self._replace_log_entry(body_index, Markdown(message))

    def _iter_stream_tokens(self, message: str) -> list[str]:
        return re.findall(r"\S+\s*|\s+", message)

    @work
    async def _send_message(self, message: str) -> None:
        handle = self._client.get_workflow_handle(self._workflow_id)
        status_bar = self.query_one(StatusBar)
        try:
            update = await handle.execute_update(
                UPDATE_USER_INPUT,
                UserInput(content=message),
                result_type=StateUpdateResponse,
            )
            turn_id = update.turn_id
            status_bar.update_status(update.status)
            assistant_buf: list[str] = []
            while True:
                delta = await handle.execute_update(
                    UPDATE_GET_STATE_UPDATE,
                    StateUpdateRequest(since_seq=self._since_seq, since_phase=self._since_phase),
                    result_type=StateUpdateResponse,
                )
                status_bar.update_status(delta.status)
                self._since_phase = delta.status.phase
                turn_completed = False
                for item in delta.items:
                    self._since_seq = max(self._since_seq, item.seq)
                    if turn_id and item.turn_id != turn_id:
                        continue
                    if item.type == ITEM_TYPE_ASSISTANT_MESSAGE and item.content:
                        assistant_buf.append(item.content)
                    elif item.type == ITEM_TYPE_FUNCTION_CALL:
                        self._append_log_entry(Text.from_markup(f"[dim]  ⚙  {item.name}[/dim]"))
                    elif item.type == ITEM_TYPE_FUNCTION_CALL_OUTPUT and item.output:
                        if item.output.success is False:
                            self._append_log_entry(Text.from_markup("[dim red]  ✗  error[/dim red]"))
                    elif item.type == ITEM_TYPE_TURN_COMPLETE:
                        turn_completed = True
                if turn_completed or delta.completed:
                    break
            if assistant_buf:
                combined = "\n\n".join(assistant_buf)
                await self._stream_assistant_reply(combined)
        except Exception as exc:
            self._append_log_entry(Text.from_markup(f"[bold red]Error:[/bold red] {exc}"))
        finally:
            inp = self.query_one(Input)
            inp.disabled = False
            inp.focus()

    async def action_quit(self) -> None:
        handle = self._client.get_workflow_handle(self._workflow_id)
        try:
            await handle.execute_update(
                UPDATE_SHUTDOWN,
                ShutdownRequest(reason="rho interactive session ended"),
            )
        except Exception:
            pass
        self.exit()
