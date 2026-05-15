from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ToolRetryPolicy:
    max_attempts: int = 0
    non_retryable: bool = False


RETRY_NONE = ToolRetryPolicy(non_retryable=True)
RETRY_DEFAULT = ToolRetryPolicy(max_attempts=3)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    default_timeout_ms: int = 0
    retry_policy: ToolRetryPolicy | None = None


REASONING_EFFORT_NONE = "none"
REASONING_EFFORT_MINIMAL = "minimal"
REASONING_EFFORT_LOW = "low"
REASONING_EFFORT_MEDIUM = "medium"
REASONING_EFFORT_HIGH = "high"
REASONING_EFFORT_XHIGH = "xhigh"

REASONING_SUMMARY_AUTO = "auto"
REASONING_SUMMARY_CONCISE = "concise"
REASONING_SUMMARY_DETAILED = "detailed"
REASONING_SUMMARY_NONE = "none"

WEB_SEARCH_DISABLED = "disabled"
WEB_SEARCH_CACHED = "cached"
WEB_SEARCH_LIVE = "live"

FINISH_REASON_STOP = "stop"
FINISH_REASON_TOOL_CALL = "tool_call"
FINISH_REASON_LENGTH = "length"
FINISH_REASON_ERROR = "error"


def parse_reasoning_effort(value: str) -> str | None:
    normalized = value.strip().lower()
    aliases = {
        REASONING_EFFORT_NONE: REASONING_EFFORT_NONE,
        REASONING_EFFORT_MINIMAL: REASONING_EFFORT_MINIMAL,
        REASONING_EFFORT_LOW: REASONING_EFFORT_LOW,
        REASONING_EFFORT_MEDIUM: REASONING_EFFORT_MEDIUM,
        "med": REASONING_EFFORT_MEDIUM,
        REASONING_EFFORT_HIGH: REASONING_EFFORT_HIGH,
        REASONING_EFFORT_XHIGH: REASONING_EFFORT_XHIGH,
        "x-high": REASONING_EFFORT_XHIGH,
        "extra-high": REASONING_EFFORT_XHIGH,
    }
    return aliases.get(normalized)


def parse_reasoning_summary(value: str) -> str | None:
    normalized = value.strip().lower()
    aliases = {
        REASONING_SUMMARY_AUTO: REASONING_SUMMARY_AUTO,
        REASONING_SUMMARY_CONCISE: REASONING_SUMMARY_CONCISE,
        REASONING_SUMMARY_DETAILED: REASONING_SUMMARY_DETAILED,
        REASONING_SUMMARY_NONE: REASONING_SUMMARY_NONE,
    }
    return aliases.get(normalized)


def detect_provider(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4", "codex-")):
        return "openai"
    if normalized.startswith("claude-"):
        return "anthropic"
    if normalized.startswith("gemini-"):
        return "google"
    return "openai"


@dataclass(frozen=True)
class ReasoningEffortPreset:
    effort: str
    description: str


@dataclass
class ToolOverrides:
    disable: list[str] = field(default_factory=list)


@dataclass
class ResolvedModelProfile:
    base_prompt: str = ""
    prompt_suffix: str = ""
    agents_file_names: list[str] = field(default_factory=list)
    tools: ToolOverrides | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    default_reasoning_effort: str | None = None
    supported_reasoning_efforts: list[ReasoningEffortPreset] = field(default_factory=list)


@dataclass
class ModelProfile:
    provider: str = ""
    model_pattern: str = ""
    base_prompt: str | None = None
    prompt_suffix: str = ""
    agents_file_names: list[str] | None = None
    tools: ToolOverrides | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    default_reasoning_effort: str | None = None
    supported_reasoning_efforts: list[ReasoningEffortPreset] | None = None


def merge_model_profiles(base: ModelProfile, overlay: ModelProfile) -> ModelProfile:
    merged = ModelProfile(
        provider=base.provider,
        model_pattern=base.model_pattern,
        base_prompt=base.base_prompt,
        prompt_suffix=base.prompt_suffix,
        agents_file_names=list(base.agents_file_names) if base.agents_file_names is not None else None,
        tools=ToolOverrides(disable=list(base.tools.disable)) if base.tools is not None else None,
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        context_window=base.context_window,
        default_reasoning_effort=base.default_reasoning_effort,
        supported_reasoning_efforts=(
            list(base.supported_reasoning_efforts) if base.supported_reasoning_efforts is not None else None
        ),
    )
    if overlay.base_prompt is not None:
        merged.base_prompt = overlay.base_prompt
    if overlay.prompt_suffix:
        merged.prompt_suffix = (
            f"{merged.prompt_suffix}\n\n{overlay.prompt_suffix}" if merged.prompt_suffix else overlay.prompt_suffix
        )
    if overlay.agents_file_names is not None:
        merged.agents_file_names = list(overlay.agents_file_names)
    if overlay.tools is not None:
        disable = list(merged.tools.disable) if merged.tools is not None else []
        disable.extend(overlay.tools.disable)
        merged.tools = ToolOverrides(disable=disable)
    if overlay.temperature is not None:
        merged.temperature = overlay.temperature
    if overlay.max_tokens is not None:
        merged.max_tokens = overlay.max_tokens
    if overlay.context_window is not None:
        merged.context_window = overlay.context_window
    if overlay.default_reasoning_effort is not None:
        merged.default_reasoning_effort = overlay.default_reasoning_effort
    if overlay.supported_reasoning_efforts is not None:
        merged.supported_reasoning_efforts = list(overlay.supported_reasoning_efforts)
    return merged


def _profile_matches(profile: ModelProfile, provider: str, model: str) -> bool:
    if not profile.provider and not profile.model_pattern:
        return True
    if profile.provider and profile.provider != provider:
        return False
    if not profile.model_pattern:
        return True
    try:
        return re.search(profile.model_pattern, model) is not None
    except re.error:
        return False


def _resolve_model_profile(profile: ModelProfile) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        base_prompt=profile.base_prompt or "",
        prompt_suffix=profile.prompt_suffix,
        agents_file_names=list(profile.agents_file_names or []),
        tools=ToolOverrides(disable=list(profile.tools.disable)) if profile.tools is not None else None,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        context_window=profile.context_window,
        default_reasoning_effort=profile.default_reasoning_effort,
        supported_reasoning_efforts=list(profile.supported_reasoning_efforts or []),
    )


def builtin_model_profiles() -> list[ModelProfile]:
    default_reasoning_effort = REASONING_EFFORT_MEDIUM
    return [
        ModelProfile(agents_file_names=["AGENTS.override.md", "AGENTS.md", "CLAUDE.md"]),
        ModelProfile(
            provider="anthropic",
            agents_file_names=["CLAUDE.md", "AGENTS.override.md", "AGENTS.md"],
            prompt_suffix=(
                "When using tools, prefer sequential calls when results depend on each other. "
                "Use parallel tool calls only for independent operations."
            ),
        ),
        ModelProfile(provider="openai", agents_file_names=["AGENTS.override.md", "AGENTS.md"]),
        ModelProfile(
            provider="openai",
            model_pattern=r"^(o1|o3|o4|codex)-",
            default_reasoning_effort=default_reasoning_effort,
            supported_reasoning_efforts=[
                ReasoningEffortPreset(REASONING_EFFORT_LOW, "Fastest responses, least reasoning"),
                ReasoningEffortPreset(REASONING_EFFORT_MEDIUM, "Balanced speed and reasoning (default)"),
                ReasoningEffortPreset(REASONING_EFFORT_HIGH, "More thorough reasoning"),
                ReasoningEffortPreset(REASONING_EFFORT_XHIGH, "Maximum reasoning effort"),
            ],
        ),
    ]


class ProfileRegistry:
    def __init__(self, profiles: list[ModelProfile] | None = None) -> None:
        self._profiles = profiles or builtin_model_profiles()

    def resolve(self, provider: str, model: str) -> ResolvedModelProfile:
        merged = ModelProfile()
        for profile in self._profiles:
            if _profile_matches(profile, provider, model):
                merged = merge_model_profiles(merged, profile)
        return _resolve_model_profile(merged)


@dataclass
class FunctionCallOutputPayload:
    content: str
    success: bool | None = None


@dataclass
class ConversationItem:
    type: str
    seq: int = 0
    content: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    output: FunctionCallOutputPayload | None = None
    web_search_action: str = ""
    web_search_status: str = ""
    web_search_url: str = ""
    turn_id: str = ""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(
        cls,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> "TokenUsage":
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
        )


@dataclass
class ModelConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 128000
    reasoning_effort: str = ""
    reasoning_summary: str = ""

    def normalize(self) -> "ModelConfig":
        self.provider = self.provider or detect_provider(self.model)
        if self.reasoning_effort:
            parsed_effort = parse_reasoning_effort(self.reasoning_effort)
            if parsed_effort is not None:
                self.reasoning_effort = parsed_effort
        if self.reasoning_summary:
            parsed_summary = parse_reasoning_summary(self.reasoning_summary)
            if parsed_summary is not None:
                self.reasoning_summary = parsed_summary
        return self

    def apply_profile_defaults(self, profile: ResolvedModelProfile | None = None) -> ResolvedModelProfile:
        self.normalize()
        resolved = profile or ProfileRegistry().resolve(self.provider, self.model)
        if resolved.temperature is not None:
            self.temperature = resolved.temperature
        if resolved.max_tokens is not None:
            self.max_tokens = resolved.max_tokens
        if resolved.context_window is not None:
            self.context_window = resolved.context_window
        if not self.reasoning_effort and resolved.default_reasoning_effort:
            self.reasoning_effort = resolved.default_reasoning_effort
        self.normalize()
        return resolved


@dataclass
class ToolsConfig:
    enabled_tools: list[str] = field(
        default_factory=lambda: [
            "shell",
            "shell_command",
            "read_file",
            "write_file",
            "list_dir",
            "grep_files",
            "request_user_input",
        ]
    )
    max_subcall_depth: int = 1

    def has_tool(self, name: str) -> bool:
        return name in self.enabled_tools

    def add_tools(self, *names: str) -> None:
        self.enabled_tools.extend(names)

    def remove_tools(self, *names: str) -> None:
        disabled = set(names)
        self.enabled_tools = [tool for tool in self.enabled_tools if tool not in disabled]


@dataclass
class Permissions:
    approval_mode: str = ""
    sandbox_mode: str = ""
    sandbox_writable_roots: list[str] = field(default_factory=list)
    sandbox_network_access: bool = False
    env_inherit: str = ""
    env_ignore_default_excludes: bool | None = None
    env_exclude: list[str] = field(default_factory=list)
    env_set: dict[str, str] = field(default_factory=dict)
    env_include_only: list[str] = field(default_factory=list)


@dataclass
class SessionConfiguration:
    base_instructions: str = ""
    developer_instructions: str = ""
    user_instructions: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    model_profile: ResolvedModelProfile = field(default_factory=ResolvedModelProfile)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    permissions: Permissions = field(default_factory=Permissions)
    cwd: str = ""
    codex_home: str = ""
    exec_policy_rules: str = ""
    auto_compact_token_limit: int = 0
    web_search_mode: str = ""
    disable_suggestions: bool = False
    session_source: str = "python"
    cli_project_docs: str = ""
    user_personal_instructions: str = ""
    session_task_queue: str = ""
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    personality: str = ""
    memory_enabled: bool = False
    memory_config: dict[str, Any] = field(default_factory=dict)
    memory_db_path: str = ""
    memory_root: str = ""
    disabled_skills: list[str] = field(default_factory=list)

    def apply_model_profile(self, registry: ProfileRegistry | None = None) -> ResolvedModelProfile:
        resolved = self.model.apply_profile_defaults(
            (registry or ProfileRegistry()).resolve(
                self.model.provider or detect_provider(self.model.model),
                self.model.model,
            )
        )
        self.model_profile = resolved
        return resolved


def default_session_configuration() -> SessionConfiguration:
    config = SessionConfiguration()
    config.apply_model_profile()
    return config


@dataclass
class PendingApproval:
    call_id: str
    tool_name: str
    arguments: str
    reason: str = ""


@dataclass
class EscalationRequest:
    call_id: str
    tool_name: str
    arguments: str
    output: str
    reason: str


@dataclass
class RequestUserInputQuestionOption:
    label: str
    description: str = ""


@dataclass
class RequestUserInputQuestion:
    id: str
    question: str
    header: str = ""
    is_other: bool = False
    options: list[RequestUserInputQuestionOption] = field(default_factory=list)


@dataclass
class PendingUserInputRequest:
    call_id: str
    questions: list[RequestUserInputQuestion] = field(default_factory=list)


@dataclass
class ChildAgentSummary:
    agent_id: str
    workflow_id: str
    role: str
    status: str


@dataclass
class TurnStatus:
    phase: str
    current_turn_id: str = ""
    tools_in_flight: list[str] = field(default_factory=list)
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    pending_escalations: list[EscalationRequest] = field(default_factory=list)
    pending_user_input_request: PendingUserInputRequest | None = None
    child_agents: list[ChildAgentSummary] = field(default_factory=list)
    iteration_count: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    turn_count: int = 0
    worker_version: str = ""
    suggestion: str = ""
    context_window_remaining_percent: int = 100
    context_window_total: int = 0


@dataclass
class UserInput:
    content: str


@dataclass
class StateUpdateRequest:
    since_seq: int = 0
    since_phase: str = ""


@dataclass
class StateUpdateResponse:
    turn_id: str
    items: list[ConversationItem]
    status: TurnStatus
    compacted: bool = False
    completed: bool = False


@dataclass
class TurnReplyActivityInput:
    turn_id: str
    message: str
    history: list[ConversationItem] = field(default_factory=list)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    tool_specs: list[ToolSpec] = field(default_factory=list)
    interrupt_note: str = ""


@dataclass
class TurnReplyActivityOutput:
    items: list[ConversationItem] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class InterruptRequest:
    pass


@dataclass
class InterruptResponse:
    acknowledged: bool


@dataclass
class ShutdownRequest:
    reason: str = ""


@dataclass
class ShutdownResponse:
    acknowledged: bool


@dataclass
class CLIOverrides:
    cwd: str = ""
    codex_home: str = ""
    model: str = ""
    provider: str = ""
    permissions: Permissions = field(default_factory=Permissions)
    session_task_queue: str = ""
    disable_suggestions: bool = False
    memory_enabled: bool = False
    memory_db_path: str = ""


@dataclass
class AgenticWorkflowState:
    conversation_id: str
    history: list[ConversationItem]
    turn_counter: int
    total_tokens: int
    total_cached_tokens: int


@dataclass
class WorkflowInput:
    conversation_id: str
    user_message: str = ""
    config: SessionConfiguration = field(default_factory=default_session_configuration)
    depth: int = 0
    continued_state: AgenticWorkflowState | None = None
    max_turns_per_run: int = 100
    idle_timeout_seconds: int = 24 * 60 * 60
    crew_name: str = ""
    crew_agent: str = ""
    crew_inputs: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    conversation_id: str
    total_iterations: int
    total_tokens: int
    total_cached_tokens: int
    tool_calls_executed: list[str] = field(default_factory=list)
    end_reason: str = ""
    final_message: str = ""


@dataclass
class SessionWorkflowInput:
    session_id: str
    harness_id: str
    user_message: str = ""
    overrides: CLIOverrides = field(default_factory=CLIOverrides)
    crew_name: str = ""
    crew_inputs: dict[str, str] = field(default_factory=dict)


@dataclass
class UpdateSessionStatusRequest:
    session_workflow_id: str
    status: str = ""
    name: str = ""


@dataclass
class StartSessionRequest:
    user_message: str = ""
    override_config: CLIOverrides | None = None
    crew_name: str = ""
    crew_inputs: dict[str, str] = field(default_factory=dict)
    crew_type: str = ""


@dataclass
class StartSessionResponse:
    session_id: str
    session_workflow_id: str


@dataclass
class SessionEntry:
    session_id: str
    session_workflow_id: str
    workflow_id: str
    user_message: str
    name: str = ""
    model: str = ""
    status: str = "pending_init"
    started_at: datetime | None = None
    crew_type: str = ""


@dataclass
class HarnessWorkflowState:
    harness_id: str
    overrides: CLIOverrides = field(default_factory=CLIOverrides)
    sessions: list[SessionEntry] = field(default_factory=list)
    session_counter: int = 0


@dataclass
class HarnessWorkflowInput:
    harness_id: str
    overrides: CLIOverrides = field(default_factory=CLIOverrides)
    continued_state: HarnessWorkflowState | None = None
    idle_timeout_seconds: int = 24 * 60 * 60


@dataclass
class AgentInputSignal:
    content: str
    interrupt: bool = False


@dataclass
class ConsolidationWorkflowState:
    items: list[str] = field(default_factory=list)
    continue_count: int = 0


@dataclass
class ConsolidationWorkflowInput:
    items: list[str] = field(default_factory=list)
    continued_state: ConsolidationWorkflowState | None = None
    max_items_per_run: int = 50
    idle_timeout_seconds: int = 24 * 60 * 60


@dataclass
class ConsolidationWorkflowResult:
    item_count: int
    continue_count: int


@dataclass
class ToolActivityInput:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    cwd: str = ""
    session_id: str = ""
    mcp_tool_ref: dict[str, str] | None = None
    approval_mode: str = ""
    exec_policy_rules: str = ""


@dataclass
class ToolActivityOutput:
    call_id: str
    content: str
    success: bool | None = None


@dataclass
class SubcallActivityInput:
    call_id: str
    prompt: str
    parent_conversation_id: str
    depth: int
    config: SessionConfiguration = field(default_factory=default_session_configuration)
    task_queue: str = ""


@dataclass
class WaitForSessionReadyInput:
    session_workflow_id: str
    timeout_ms: int = 30_000
    poll_interval_ms: int = 250


@dataclass
class WaitForSessionReadyOutput:
    agent_workflow_id: str
