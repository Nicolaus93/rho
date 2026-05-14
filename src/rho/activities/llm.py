from __future__ import annotations

from dataclasses import dataclass, field

from temporalio import activity

from ..llm import (
    CompactRequest,
    LLMRequest,
    MultiProviderLLMClient,
    create_default_llm_client,
)
from ..models import ConversationItem, ModelConfig, TokenUsage, TurnReplyActivityInput
from ..tools import ToolSpec


class LLMActivityError(RuntimeError):
    pass


@dataclass
class LLMActivityInput:
    history: list[ConversationItem]
    model_config: ModelConfig
    tool_specs: list[ToolSpec] = field(default_factory=list)
    base_instructions: str = ""
    developer_instructions: str = ""
    user_instructions: str = ""
    previous_response_id: str = ""


@dataclass
class LLMActivityOutput:
    items: list[ConversationItem]
    finish_reason: str
    token_usage: TokenUsage
    response_id: str = ""


@dataclass
class CompactActivityInput:
    model: str
    input: list[ConversationItem]
    instructions: str = ""


@dataclass
class CompactActivityOutput:
    items: list[ConversationItem]
    token_usage: TokenUsage


@dataclass
class SuggestionInput:
    user_message: str
    assistant_message: str
    model_config: ModelConfig
    tool_summaries: list[str] = field(default_factory=list)


@dataclass
class SuggestionOutput:
    suggestion: str = ""


class LLMActivities:
    def __init__(self, client: MultiProviderLLMClient | None = None) -> None:
        self._client = client or create_default_llm_client()

    @activity.defn(name="GenerateTurnReply")
    async def generate_turn_reply(self, input: TurnReplyActivityInput) -> str:
        response = await self._client.call(
            LLMRequest(
                history=input.history,
                model_config=input.model_config,
            )
        )
        reply = next(
            (
                item.content.strip()
                for item in response.items
                if item.type == "assistant_message" and item.content.strip()
            ),
            "",
        )
        if not reply:
            raise LLMActivityError("LLM did not return an assistant message")
        if input.interrupt_note:
            reply = f"{input.interrupt_note} {reply}".strip()
        return reply

    @activity.defn(name="ExecuteLLMCall")
    async def execute_llm_call(self, input: LLMActivityInput) -> LLMActivityOutput:
        response = await self._client.call(
            LLMRequest(
                history=input.history,
                model_config=input.model_config,
                tool_specs=input.tool_specs,
                base_instructions=input.base_instructions,
                developer_instructions=input.developer_instructions,
                user_instructions=input.user_instructions,
                previous_response_id=input.previous_response_id,
            )
        )
        return LLMActivityOutput(
            items=response.items,
            finish_reason=response.finish_reason,
            token_usage=response.token_usage,
            response_id=response.response_id,
        )

    @activity.defn(name="ExecuteCompact")
    async def execute_compact(self, input: CompactActivityInput) -> CompactActivityOutput:
        response = await self._client.compact(
            CompactRequest(model=input.model, input=input.input, instructions=input.instructions)
        )
        return CompactActivityOutput(items=response.items, token_usage=response.token_usage)

    @activity.defn(name="GenerateSuggestions")
    async def generate_suggestions(self, input: SuggestionInput) -> SuggestionOutput:
        try:
            response = await self._client.call(
                LLMRequest(
                    history=[
                        ConversationItem(
                            type="user_message",
                            content="\n".join(
                                part
                                for part in [
                                    input.user_message,
                                    input.assistant_message,
                                    *input.tool_summaries,
                                ]
                                if part
                            ),
                        )
                    ],
                    model_config=input.model_config,
                    base_instructions="Return a single concise follow-up suggestion.",
                )
            )
        except Exception:
            return SuggestionOutput()
        for item in response.items:
            if item.type == "assistant_message" and item.content:
                return SuggestionOutput(suggestion=item.content.strip())
        return SuggestionOutput()
