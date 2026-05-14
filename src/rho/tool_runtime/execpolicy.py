from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Sequence

from .command_safety import command_might_be_dangerous, is_known_safe_command, parse_shell_lc_plain_commands


class Decision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class Evaluation:
    decision: Decision
    reason: str = ""
    parsed_subcommands: list[list[str]] = field(default_factory=list)

    @property
    def requires_approval(self) -> bool:
        return self.decision == Decision.REQUIRE_APPROVAL


@dataclass(frozen=True)
class Policy:
    approval_mode: str = ""
    rules: str = ""


class ExecPolicyManager:
    def __init__(self, policy: Policy | None = None) -> None:
        self._policy = policy or Policy()

    def evaluate_command(self, command: Sequence[str] | str, approval_mode: str = "") -> Evaluation:
        parsed_subcommands = parse_shell_lc_plain_commands(command)
        command_groups = parsed_subcommands or [_normalize_command(command)]
        effective_mode = (approval_mode or self._policy.approval_mode or "never").strip().lower()

        if any(command_might_be_dangerous(parts) for parts in command_groups):
            if effective_mode == "never":
                return Evaluation(Decision.DENY, "Command blocked by safety heuristics.", parsed_subcommands or [])
            return Evaluation(Decision.REQUIRE_APPROVAL, "Command requires approval due to safety heuristics.", parsed_subcommands or [])

        if self._policy.rules.strip():
            return Evaluation(
                Decision.DENY,
                "Custom exec policy rules are configured but not implemented in the Python runtime yet.",
                parsed_subcommands or [],
            )

        if effective_mode in {"", "never", "on-failure"}:
            return Evaluation(Decision.ALLOW, parsed_subcommands=(parsed_subcommands or []))

        if all(is_known_safe_command(parts) for parts in command_groups):
            return Evaluation(Decision.ALLOW, parsed_subcommands=(parsed_subcommands or []))

        return Evaluation(
            Decision.REQUIRE_APPROVAL,
            "Command is not in the trusted safe-command allowlist.",
            parsed_subcommands or [],
        )


def _normalize_command(command: Sequence[str] | str) -> list[str]:
    if isinstance(command, str):
        return [command]
    return [str(part) for part in command]
