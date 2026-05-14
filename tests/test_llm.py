from temporal_agent_harness.activities import LLMActivities, LLMActivityInput
from temporal_agent_harness.llm import (
    CompactResponse,
    AnthropicClient,
    CompactRequest,
    LLMResponse,
    LLMRequest,
    MultiProviderLLMClient,
    OpenAIClient,
    detect_provider_from_model,
)
from temporal_agent_harness.models import ConversationItem, ModelConfig, TokenUsage, TurnReplyActivityInput


class FakeProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.called_with: list[str] = []

    async def call(self, request: LLMRequest):
        self.called_with.append(request.model_config.model)
        return LLMResponse(
            items=[ConversationItem(type="assistant_message", content=f"{self.name}:{request.model_config.model}")],
            finish_reason="stop",
            token_usage=TokenUsage.from_counts(input_tokens=2, output_tokens=1),
            response_id=f"{self.name}-resp",
        )

    async def compact(self, request: CompactRequest):
        self.called_with.append(request.model)
        return CompactResponse(
            items=[ConversationItem(type="assistant_message", content=f"compact:{self.name}")],
            token_usage=TokenUsage.from_counts(input_tokens=1, output_tokens=1),
        )


def test_detect_provider_from_model() -> None:
    assert detect_provider_from_model("claude-3-7-sonnet") == "anthropic"
    assert detect_provider_from_model("gpt-4o-mini") == "openai"


async def test_multi_provider_dispatch_routes_by_provider_and_model() -> None:
    openai = FakeProvider("openai")
    anthropic = FakeProvider("anthropic")
    client = MultiProviderLLMClient({"openai": openai, "anthropic": anthropic})

    response = await client.call(
        LLMRequest(history=[], model_config=ModelConfig(provider="anthropic", model="claude-3-7-sonnet"))
    )
    compact = await client.compact(CompactRequest(model="gpt-4o-mini", input=[]))

    assert response.items[0].content == "anthropic:claude-3-7-sonnet"
    assert compact.items[0].content == "compact:openai"
    assert anthropic.called_with == ["claude-3-7-sonnet"]
    assert openai.called_with == ["gpt-4o-mini"]


async def test_openai_adapter_uses_transport_stub() -> None:
    def transport(path: str, payload: dict[str, object]) -> dict[str, object]:
        assert path == "/responses"
        assert payload["model"] == "gpt-4o-mini"
        return {
            "id": "resp-1",
            "output_text": "hello",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "finish_reason": "stop",
        }

    client = OpenAIClient(transport=transport)
    result = await client.call(LLMRequest(history=[], model_config=ModelConfig(model="gpt-4o-mini")))

    assert result.response_id == "resp-1"
    assert result.items[0].content == "hello"
    assert result.token_usage.total_tokens == 5


async def test_openai_chat_completions_mode_uses_standard_request_and_response_shape() -> None:
    def transport(path: str, payload: dict[str, object]) -> dict[str, object]:
        assert path == "/chat/completions"
        assert payload["model"] == "gpt-4o-mini"
        assert payload["temperature"] == 0.25
        assert payload["max_tokens"] == 128
        assert "provider" not in payload
        assert "instructions" not in payload
        messages = payload["messages"]
        assert messages == [
            {
                "role": "system",
                "content": "Base instructions:\nBe helpful.\n\nDeveloper instructions:\nStay concise.\n\nUser instructions:\nReply in one sentence.",
            },
            {"role": "user", "content": "hi"},
        ]
        return {
            "id": "chat-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello from chat"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    client = OpenAIClient(transport=transport, api_mode="chat_completions")
    result = await client.call(
        LLMRequest(
            history=[ConversationItem(type="user_message", content="hi")],
            model_config=ModelConfig(model="gpt-4o-mini", temperature=0.25, max_tokens=128),
            base_instructions="Be helpful.",
            developer_instructions="Stay concise.",
            user_instructions="Reply in one sentence.",
        )
    )

    assert result.response_id == "chat-1"
    assert result.items[0].content == "hello from chat"
    assert result.finish_reason == "stop"
    assert result.token_usage.total_tokens == 5


async def test_openai_auto_mode_falls_back_to_chat_completions_for_missing_responses_endpoint() -> None:
    calls: list[str] = []

    def transport(path: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(path)
        if path == "/responses":
            raise RuntimeError("openai request failed: 404 Not Found")
        assert path == "/chat/completions"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        return {
            "id": "chat-2",
            "choices": [{"message": {"role": "assistant", "content": "fallback ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    client = OpenAIClient(transport=transport, api_mode="auto")
    result = await client.call(
        LLMRequest(history=[ConversationItem(type="user_message", content="hi")], model_config=ModelConfig(model="gpt-4o-mini"))
    )

    assert calls == ["/responses", "/chat/completions"]
    assert result.items[0].content == "fallback ok"
    assert result.response_id == "chat-2"
    assert result.token_usage.total_tokens == 6


async def test_openai_auto_mode_does_not_fallback_for_responses_specific_request_fields() -> None:
    calls: list[str] = []

    def transport(path: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(path)
        raise RuntimeError("openai request failed: 404 Not Found")

    client = OpenAIClient(transport=transport, api_mode="auto")

    try:
        await client.call(
            LLMRequest(
                history=[],
                model_config=ModelConfig(model="gpt-4o-mini"),
                previous_response_id="resp-prev",
            )
        )
    except RuntimeError as exc:
        assert "404" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == ["/responses"]


async def test_openai_auto_mode_does_not_fallback_for_non_endpoint_errors() -> None:
    calls: list[str] = []

    def transport(path: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(path)
        raise RuntimeError("openai request failed: 401 Unauthorized")

    client = OpenAIClient(transport=transport, api_mode="auto")

    try:
        await client.call(LLMRequest(history=[], model_config=ModelConfig(model="gpt-4o-mini")))
    except RuntimeError as exc:
        assert "401" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == ["/responses"]


async def test_anthropic_adapter_uses_transport_stub() -> None:
    def transport(path: str, payload: dict[str, object]) -> dict[str, object]:
        assert path == "/messages"
        assert payload["model"] == "claude-3-7-sonnet"
        return {
            "id": "msg-1",
            "content": [{"type": "text", "text": "world"}],
            "usage": {"input_tokens": 4, "output_tokens": 3},
            "stop_reason": "end_turn",
        }

    client = AnthropicClient(transport=transport)
    result = await client.call(
        LLMRequest(history=[], model_config=ModelConfig(provider="anthropic", model="claude-3-7-sonnet"))
    )

    assert result.response_id == "msg-1"
    assert result.items[0].content == "world"
    assert result.finish_reason == "end_turn"


async def test_llm_activity_behavior_with_fake_client() -> None:
    class FakeClient(MultiProviderLLMClient):
        async def call(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                items=[ConversationItem(type="assistant_message", content="activity ok")],
                finish_reason="stop",
                token_usage=TokenUsage.from_counts(input_tokens=10, output_tokens=4),
                response_id="fake",
            )

        async def compact(self, request: CompactRequest) -> CompactResponse:
            return CompactResponse(
                items=[ConversationItem(type="assistant_message", content="summary")],
                token_usage=TokenUsage.from_counts(input_tokens=5, output_tokens=2),
            )

    activities = LLMActivities(client=FakeClient({}))
    result = await activities.execute_llm_call(
        LLMActivityInput(
            history=[ConversationItem(type="user_message", content="hi")],
            model_config=ModelConfig(model="gpt-4o-mini"),
        )
    )

    assert result.items[0].content == "activity ok"
    assert result.token_usage.total_tokens == 14


async def test_generate_turn_reply_activity_returns_assistant_message() -> None:
    class FakeClient(MultiProviderLLMClient):
        async def call(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                items=[ConversationItem(type="assistant_message", content="llm reply")],
                finish_reason="stop",
                token_usage=TokenUsage.from_counts(output_tokens=2),
            )

    activities = LLMActivities(client=FakeClient({}))
    result = await activities.generate_turn_reply(TurnReplyActivityInput(turn_id="turn-7", message="hello there"))

    assert result == "llm reply"
