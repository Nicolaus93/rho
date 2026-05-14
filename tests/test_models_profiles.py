from temporal_agent_harness.models import (
    ModelConfig,
    ModelProfile,
    ProfileRegistry,
    ReasoningEffortPreset,
    ToolOverrides,
    builtin_model_profiles,
    detect_provider,
    merge_model_profiles,
    parse_reasoning_effort,
    parse_reasoning_summary,
)


def test_parse_reasoning_variants() -> None:
    assert parse_reasoning_effort("med") == "medium"
    assert parse_reasoning_effort("x-high") == "xhigh"
    assert parse_reasoning_effort("invalid") is None
    assert parse_reasoning_summary("concise") == "concise"
    assert parse_reasoning_summary("invalid") is None


def test_profile_registry_resolves_provider_model_and_suffixes() -> None:
    registry = ProfileRegistry(
        [
            ModelProfile(
                prompt_suffix="base",
                agents_file_names=["BASE.md"],
                tools=ToolOverrides(disable=["a"]),
            ),
            ModelProfile(
                provider="anthropic",
                prompt_suffix="provider",
                agents_file_names=["CLAUDE.md"],
            ),
            ModelProfile(
                provider="anthropic",
                model_pattern=r"^claude-3",
                prompt_suffix="model",
                tools=ToolOverrides(disable=["b"]),
                supported_reasoning_efforts=[ReasoningEffortPreset("high", "High")],
            ),
        ]
    )

    profile = registry.resolve("anthropic", "claude-3-7-sonnet")

    assert profile.prompt_suffix == "base\n\nprovider\n\nmodel"
    assert profile.agents_file_names == ["CLAUDE.md"]
    assert profile.tools is not None
    assert profile.tools.disable == ["a", "b"]
    assert profile.supported_reasoning_efforts[0].effort == "high"


def test_builtin_openai_reasoning_profile_applies_defaults() -> None:
    config = ModelConfig(model="o3-mini", provider="")
    resolved = config.apply_profile_defaults(ProfileRegistry(builtin_model_profiles()).resolve("openai", "o3-mini"))

    assert config.provider == "openai"
    assert config.reasoning_effort == "medium"
    assert resolved.agents_file_names == ["AGENTS.override.md", "AGENTS.md"]
    assert [preset.effort for preset in resolved.supported_reasoning_efforts] == [
        "low",
        "medium",
        "high",
        "xhigh",
    ]


def test_detect_provider_matches_model_prefixes() -> None:
    assert detect_provider("gpt-4o") == "openai"
    assert detect_provider("claude-sonnet-4-5") == "anthropic"
    assert detect_provider("gemini-2.5-pro") == "google"


def test_merge_model_profiles_replaces_and_unions_fields() -> None:
    base = ModelProfile(
        agents_file_names=["A.md"],
        prompt_suffix="one",
        tools=ToolOverrides(disable=["x"]),
    )
    overlay = ModelProfile(
        agents_file_names=["B.md"],
        prompt_suffix="two",
        tools=ToolOverrides(disable=["y"]),
    )

    merged = merge_model_profiles(base, overlay)

    assert merged.agents_file_names == ["B.md"]
    assert merged.prompt_suffix == "one\n\ntwo"
    assert merged.tools is not None
    assert merged.tools.disable == ["x", "y"]
