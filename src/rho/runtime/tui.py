from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from temporalio.client import Client
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
        if not message:
            return
        if message.lower() in {"exit", "quit", "/exit", "/quit"}:
            await self.action_quit()
            return
        log = self.query_one(RichLog)
        log.write(Text.assemble(("You  ", "bold cyan"), message, "\n"))
        event.input.disabled = True
        self._send_message(message)  # type: ignore[unused-coroutine]

    @work
    async def _send_message(self, message: str) -> None:
        handle = self._client.get_workflow_handle(self._workflow_id)
        status_bar = self.query_one(StatusBar)
        log = self.query_one(RichLog)
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
                        log.write(Text.from_markup(f"[dim]  ⚙  {item.name}[/dim]"))
                    elif item.type == ITEM_TYPE_FUNCTION_CALL_OUTPUT and item.output:
                        if item.output.success is False:
                            log.write(Text.from_markup("[dim red]  ✗  error[/dim red]"))
                    elif item.type == ITEM_TYPE_TURN_COMPLETE:
                        turn_completed = True
                if turn_completed or delta.completed:
                    break
            if assistant_buf:
                combined = "\n\n".join(assistant_buf)
                log.write(Text.assemble(("Assistant  ", "bold green")))
                log.write(Markdown(combined))
                log.write("")
        except Exception as exc:
            log.write(Text.from_markup(f"[bold red]Error:[/bold red] {exc}"))
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
