from __future__ import annotations

import os
import shlex
from collections.abc import Sequence

_SHELL_PROGRAMS = {"bash", "sh", "zsh", "/bin/bash", "/bin/sh", "/bin/zsh"}
_SHELL_META_TOKENS = ("|", "||", "<", ">", "$(", "`", "\n", "\r")
_SAFE_COMMANDS = {
    "basename",
    "cat",
    "cut",
    "date",
    "dirname",
    "du",
    "echo",
    "env",
    "file",
    "head",
    "id",
    "ls",
    "nl",
    "pwd",
    "printf",
    "readlink",
    "realpath",
    "sort",
    "stat",
    "tail",
    "tree",
    "uname",
    "uniq",
    "wc",
    "which",
    "whoami",
}
_SAFE_GIT_SUBCOMMANDS = {
    "branch",
    "diff",
    "fetch",
    "grep",
    "log",
    "ls-files",
    "remote",
    "rev-parse",
    "show",
    "status",
}


def _normalize_argv(argv: Sequence[str] | str) -> list[str]:
    if isinstance(argv, str):
        return shlex.split(argv)
    return [str(part) for part in argv]


def parse_shell_lc_plain_commands(argv: Sequence[str] | str) -> list[list[str]] | None:
    parts = _normalize_argv(argv)
    shell_program = parts[0] if parts else ""
    if len(parts) < 3 or (
        shell_program not in _SHELL_PROGRAMS and os.path.basename(shell_program) not in _SHELL_PROGRAMS
    ):
        return None
    if parts[1] not in {"-c", "-lc"}:
        return None
    if len(parts) != 3:
        return None
    script = parts[2]
    if any(token in script for token in _SHELL_META_TOKENS):
        return None
    normalized = script.replace("&&", ";")
    commands: list[list[str]] = []
    for chunk in normalized.split(";"):
        candidate = chunk.strip()
        if not candidate:
            continue
        try:
            parsed = shlex.split(candidate)
        except ValueError:
            return None
        if not parsed:
            return None
        commands.append(parsed)
    return commands or None


def is_known_safe_command(argv: Sequence[str] | str) -> bool:
    parts = _normalize_argv(argv)
    if not parts:
        return False
    command = parts[0]
    if command in _SAFE_COMMANDS:
        return True
    if command == "base64":
        return True
    if command == "rg":
        return True
    if command == "find":
        return not any(arg in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for arg in parts[1:])
    if command == "sed":
        return not any(arg == "-i" or arg.startswith("--in-place") for arg in parts[1:])
    if command == "git":
        return len(parts) > 1 and parts[1] in _SAFE_GIT_SUBCOMMANDS
    return False


def command_might_be_dangerous(argv: Sequence[str] | str) -> bool:
    parts = _normalize_argv(argv)
    if not parts:
        return False
    if parts[0] == "sudo":
        return True if len(parts) == 1 else command_might_be_dangerous(parts[1:])
    if parts[0] == "rm":
        for arg in parts[1:]:
            if arg.startswith("-") and "f" in arg and "r" in arg:
                return True
            if arg in {"-f", "-rf", "-fr"}:
                return True
    return False
