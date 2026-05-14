from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol
from urllib import error, request as urllib_request

from .models import (
    FINISH_REASON_STOP,
    ConversationItem,
    ModelConfig,
    TokenUsage,
    detect_provider,
)
from .tools import ToolSpec


LOGGER = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    history: list[ConversationItem]
    model_config: ModelConfig
    tool_specs: list[ToolSpec] = field(default_factory=list)
    base_instructions: str = ""
    developer_instructions: str = ""
    user_instructions: str = ""
    previous_response_id: str = ""
    web_search_mode: str = ""


@dataclass
class LLMResponse:
    items: list[ConversationItem]
    finish_reason: str = FINISH_REASON_STOP
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str = ""


@dataclass
class CompactRequest:
    model: str
    input: list[ConversationItem]
    instructions: str = ""


@dataclass
class CompactResponse:
    items: list[ConversationItem]
    token_usage: TokenUsage = field(default_factory=TokenUsage)


class LLMClient(Protocol):
    async def call(self, request: LLMRequest) -> LLMResponse: ...

    async def compact(self, request: CompactRequest) -> CompactResponse: ...


def detect_provider_from_model(model: str) -> str:
    return detect_provider(model)


def _history_to_messages(history: list[ConversationItem]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        if item.type == "assistant_message":
            messages.append({"role": "assistant", "content": item.content})
        elif item.type == "user_message":
            messages.append({"role": "user", "content": item.content})
        elif item.type == "function_call_output" and item.output is not None:
            messages.append({"role": "tool", "content": item.output.content})
    return messages


def _estimate_usage_from_text(*parts: str) -> TokenUsage:
    total_chars = sum(len(part) for part in parts)
    estimated = max(1, total_chars // 4) if total_chars else 0
    return TokenUsage.from_counts(input_tokens=estimated, output_tokens=max(1, estimated // 3) if estimated else 0)


class BaseHTTPProviderClient:
    provider_name = ""
    api_key_env = ""
    default_base_url = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Any | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv(self.api_key_env, "")
        env_base_url = os.getenv(f"{self.provider_name.upper()}_BASE_URL", "")
        self._base_url = (base_url or env_base_url or self.default_base_url).rstrip("/")
        self._transport = transport
        LOGGER.info(
            "Initialized %s client base_url=%s api_key_present=%s transport_stub=%s",
            self.provider_name,
            self._base_url,
            bool(self._api_key),
            self._transport is not None,
        )

    async def call(self, request: LLMRequest) -> LLMResponse:
        config = request.model_config.normalize()
        LOGGER.info(
            "Dispatching %s call url=%s%s model=%s provider=%s history_items=%d tool_specs=%d",
            self.provider_name,
            self._base_url,
            self._call_path(),
            config.model,
            config.provider or self.provider_name,
            len(request.history),
            len(request.tool_specs),
        )
        payload = self._build_call_payload(request)
        response = await self._request_json(self._call_path(), payload)
        return self._parse_call_response(request, response)

    async def compact(self, request: CompactRequest) -> CompactResponse:
        LOGGER.info(
            "Dispatching %s compact url=%s%s model=%s input_items=%d",
            self.provider_name,
            self._base_url,
            self._compact_path(),
            request.model,
            len(request.input),
        )
        payload = self._build_compact_payload(request)
        response = await self._request_json(self._compact_path(), payload)
        return self._parse_compact_response(request, response)

    def _call_path(self) -> str:
        return "/responses"

    def _compact_path(self) -> str:
        return "/responses/compact"

    async def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            result = self._transport(path, payload)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, dict):
                raise TypeError(f"{self.provider_name} transport must return a dict")
            return result
        LOGGER.info(
            "Sending %s HTTP request url=%s%s payload_keys=%s",
            self.provider_name,
            self._base_url,
            path,
            sorted(payload.keys()),
        )
        return await asyncio.to_thread(self._perform_http_request, path, payload)

    def _perform_http_request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib_request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:  # pragma: no cover - network path
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} request failed: {exc.code} {details}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"{self.provider_name} request failed: {exc.reason}") from exc
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise RuntimeError(f"{self.provider_name} returned non-object JSON")
        return data

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_call_payload(self, request: LLMRequest) -> dict[str, Any]:
        config = request.model_config.normalize()
        return {
            "model": config.model,
            "provider": config.provider or self.provider_name,
            "messages": _history_to_messages(request.history),
            "instructions": {
                "base": request.base_instructions,
                "developer": request.developer_instructions,
                "user": request.user_instructions,
            },
            "tool_specs": [asdict(spec) for spec in request.tool_specs],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reasoning_effort": config.reasoning_effort,
            "reasoning_summary": config.reasoning_summary,
            "previous_response_id": request.previous_response_id,
            "web_search_mode": request.web_search_mode,
        }

    def _build_compact_payload(self, request: CompactRequest) -> dict[str, Any]:
        return {
            "model": request.model,
            "input": _history_to_messages(request.input),
            "instructions": request.instructions,
        }

    def _parse_call_response(self, request: LLMRequest, payload: dict[str, Any]) -> LLMResponse:
        text = str(payload.get("output_text") or payload.get("content") or "")
        items = [ConversationItem(type="assistant_message", content=text)] if text else []
        usage = self._parse_usage(payload.get("usage"), fallback_text=text)
        return LLMResponse(
            items=items,
            finish_reason=str(payload.get("finish_reason") or FINISH_REASON_STOP),
            token_usage=usage,
            response_id=str(payload.get("id") or payload.get("response_id") or ""),
        )

    def _parse_compact_response(self, request: CompactRequest, payload: dict[str, Any]) -> CompactResponse:
        if isinstance(payload.get("items"), list):
            items = [ConversationItem(**item) if isinstance(item, dict) else ConversationItem(type="assistant_message", content=str(item)) for item in payload["items"]]
        else:
            text = str(payload.get("summary") or payload.get("content") or "")
            items = [ConversationItem(type="assistant_message", content=text)] if text else []
        usage = self._parse_usage(payload.get("usage"), fallback_text=" ".join(item.content for item in items))
        return CompactResponse(items=items, token_usage=usage)

    def _parse_usage(self, usage: Any, *, fallback_text: str) -> TokenUsage:
        if isinstance(usage, dict):
            return TokenUsage.from_counts(
                input_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
                output_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
                cached_input_tokens=int(usage.get("cached_input_tokens", 0) or 0),
                reasoning_tokens=int(usage.get("reasoning_tokens", 0) or 0),
            )
        return _estimate_usage_from_text(fallback_text)


class OpenAIClient(BaseHTTPProviderClient):
    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Any | None = None,
        api_mode: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, transport=transport)
        self._api_mode = self._resolve_api_mode(api_mode)

    async def call(self, request: LLMRequest) -> LLMResponse:
        mode = self._api_mode
        config = request.model_config.normalize()
        LOGGER.info(
            "OpenAI call mode=%s base_url=%s model=%s previous_response_id=%s web_search_mode=%s",
            mode,
            self._base_url,
            config.model,
            bool(request.previous_response_id),
            bool(request.web_search_mode),
        )
        if mode == "chat_completions":
            return await self._call_chat_completions(request)
        if mode == "auto":
            try:
                return await super().call(request)
            except RuntimeError as exc:
                if self._can_fallback_to_chat_completions(request) and self._should_retry_with_chat_completions(exc):
                    LOGGER.info(
                        "OpenAI auto mode falling back from /responses to /chat/completions base_url=%s model=%s",
                        self._base_url,
                        config.model,
                    )
                    return await self._call_chat_completions(request)
                raise
        return await super().call(request)

    def _headers(self) -> dict[str, str]:
        return self._headers_for_path(self._call_path())

    async def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            result = self._transport(path, payload)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, dict):
                raise TypeError(f"{self.provider_name} transport must return a dict")
            return result
        LOGGER.info(
            "Sending %s HTTP request url=%s%s api_mode=%s payload_keys=%s",
            self.provider_name,
            self._base_url,
            path,
            self._api_mode,
            sorted(payload.keys()),
        )
        return await asyncio.to_thread(self._perform_http_request_for_path, path, payload)

    def _perform_http_request_for_path(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib_request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers=self._headers_for_path(path),
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:  # pragma: no cover - network path
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} request failed: {exc.code} {details}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"{self.provider_name} request failed: {exc.reason}") from exc
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise RuntimeError(f"{self.provider_name} returned non-object JSON")
        return data

    def _parse_call_response(self, request: LLMRequest, payload: dict[str, Any]) -> LLMResponse:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message") if isinstance(first_choice, dict) else None
            text = ""
            if isinstance(message, dict):
                text = self._chat_content_to_text(message.get("content"))
            if not text:
                text = self._chat_content_to_text(first_choice.get("text"))
            items = [ConversationItem(type="assistant_message", content=text)] if text else []
            usage = self._parse_usage(payload.get("usage"), fallback_text=text)
            return LLMResponse(
                items=items,
                finish_reason=str(first_choice.get("finish_reason") or payload.get("finish_reason") or FINISH_REASON_STOP),
                token_usage=usage,
                response_id=str(payload.get("id") or payload.get("response_id") or ""),
            )
        return super()._parse_call_response(request, payload)

    def _headers_for_path(self, path: str) -> dict[str, str]:
        headers = super()._headers()
        if path.startswith("/responses"):
            headers["OpenAI-Beta"] = "responses-api"
        return headers

    def _call_path(self) -> str:
        if self._api_mode == "chat_completions":
            return "/chat/completions"
        return super()._call_path()

    def _build_call_payload(self, request: LLMRequest) -> dict[str, Any]:
        if self._api_mode == "chat_completions":
            return self._build_chat_completions_payload(request)
        return super()._build_call_payload(request)

    def _resolve_api_mode(self, api_mode: str | None) -> str:
        raw_mode = api_mode or os.getenv("OPENAI_API_MODE", "")
        if not raw_mode:
            use_chat = os.getenv("OPENAI_USE_CHAT_COMPLETIONS", "")
            if use_chat.strip().lower() in {"1", "true", "yes", "on"}:
                raw_mode = "chat_completions"
        normalized = raw_mode.strip().lower().replace("-", "_")
        if normalized in {"", "responses"}:
            return "responses"
        if normalized in {"chat", "chat_completions", "chatcompletions"}:
            return "chat_completions"
        if normalized == "auto":
            return "auto"
        raise ValueError(f"unsupported OpenAI API mode: {raw_mode}")

    async def _call_chat_completions(self, request: LLMRequest) -> LLMResponse:
        self._ensure_chat_completions_compatible(request)
        payload = self._build_chat_completions_payload(request)
        LOGGER.info(
            "Calling OpenAI chat completions url=%s/chat/completions model=%s message_count=%d",
            self._base_url,
            payload.get("model", ""),
            len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0,
        )
        response = await self._request_json("/chat/completions", payload)
        return self._parse_call_response(request, response)

    def _build_chat_completions_payload(self, request: LLMRequest) -> dict[str, Any]:
        self._ensure_chat_completions_compatible(request)
        config = request.model_config.normalize()
        messages = self._chat_completions_messages(request)
        return {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

    def _chat_completions_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        messages = _history_to_messages(request.history)
        system_message = self._system_message_from_request(request)
        if system_message:
            return [{"role": "system", "content": system_message}, *messages]
        return messages

    def _system_message_from_request(self, request: LLMRequest) -> str:
        parts = [
            ("Base instructions", request.base_instructions),
            ("Developer instructions", request.developer_instructions),
            ("User instructions", request.user_instructions),
        ]
        chunks = [f"{label}:\n{text.strip()}" for label, text in parts if text.strip()]
        return "\n\n".join(chunks)

    def _ensure_chat_completions_compatible(self, request: LLMRequest) -> None:
        if request.tool_specs:
            raise ValueError("OpenAI chat completions mode does not support tool_specs in this client yet")
        if request.previous_response_id:
            raise ValueError("OpenAI chat completions mode does not support previous_response_id")
        if request.web_search_mode:
            raise ValueError("OpenAI chat completions mode does not support web_search_mode")

    def _can_fallback_to_chat_completions(self, request: LLMRequest) -> bool:
        return not request.tool_specs and not request.previous_response_id and not request.web_search_mode

    def _should_retry_with_chat_completions(self, exc: RuntimeError) -> bool:
        details = str(exc).lower()
        endpoint_markers = (
            " 404 ",
            " 405 ",
            "not found",
            "unsupported",
            "unknown url",
            "unknown path",
            "no route",
            "does not exist",
        )
        return any(marker in details for marker in endpoint_markers)

    def _chat_content_to_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return ""


class AnthropicClient(BaseHTTPProviderClient):
    provider_name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _call_path(self) -> str:
        return "/messages"

    def _compact_path(self) -> str:
        return "/messages/compact"

    def _parse_call_response(self, request: LLMRequest, payload: dict[str, Any]) -> LLMResponse:
        content = payload.get("content")
        text = ""
        if isinstance(content, list):
            text = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        elif isinstance(content, str):
            text = content
        elif payload.get("output_text"):
            text = str(payload["output_text"])
        usage = self._parse_usage(payload.get("usage"), fallback_text=text)
        return LLMResponse(
            items=[ConversationItem(type="assistant_message", content=text)] if text else [],
            finish_reason=str(payload.get("stop_reason") or payload.get("finish_reason") or FINISH_REASON_STOP),
            token_usage=usage,
            response_id=str(payload.get("id") or payload.get("response_id") or ""),
        )


class MultiProviderLLMClient:
    def __init__(self, providers: Mapping[str, LLMClient] | None = None) -> None:
        self._providers = dict(providers) if providers is not None else {
            "openai": OpenAIClient(),
            "anthropic": AnthropicClient(),
        }

    def provider_for(self, provider: str) -> LLMClient:
        if provider not in self._providers:
            raise ValueError(f"unsupported LLM provider: {provider}")
        return self._providers[provider]

    async def call(self, request: LLMRequest) -> LLMResponse:
        provider = request.model_config.provider or detect_provider_from_model(request.model_config.model)
        return await self.provider_for(provider).call(request)

    async def compact(self, request: CompactRequest) -> CompactResponse:
        provider = detect_provider_from_model(request.model)
        try:
            return await self.provider_for(provider).compact(request)
        except Exception:
            if provider != "anthropic" and "anthropic" in self._providers:
                return await self._providers["anthropic"].compact(request)
            raise


def create_default_llm_client() -> MultiProviderLLMClient:
    return MultiProviderLLMClient()
