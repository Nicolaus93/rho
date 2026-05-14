# Python runtime

This repo includes a Python implementation under `src/rho`.

## Current status

The Python side currently provides:

- Worker + workflow entry point (`rho`)
- Low-level workflow CLI (`rho-client`)
- Core workflow/activity scaffolding
- Phase-1 tool runtime for file and command tools

It does **not** yet provide full feature parity with the Go implementation.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Temporal CLI

## Install

From the repo root:

```bash
uv sync --extra dev
```

## Start Temporal locally

In one terminal:

```bash
temporal server start-dev
```

By default, the Python client connects to:

- host: `localhost:7233`
- namespace: `default`

You can override those with:

- `TEMPORAL_HOST_URL`
- `TEMPORAL_NAMESPACE`

TLS env vars are also supported:

- `TEMPORAL_TLS_CERT`
- `TEMPORAL_TLS_KEY`
- `TEMPORAL_TLS_CA`
- `TEMPORAL_TLS_SERVER_ROOT_CA_CERT`
- `TEMPORAL_TLS_SERVER_NAME`

## Start rho

A single command starts the worker, launches the harness workflow, and opens the interactive TUI:

```bash
uv run rho
```

Optional flags:

```bash
uv run rho --provider openai --model gpt-4o "Hello"
uv run rho --cwd /path/to/project
uv run rho --harness-id my-harness
```

On exit (`ctrl+c` or closing the TUI), `rho` sends a shutdown signal to all open workflows so they complete cleanly.

## Notes about models and API keys

The Python worker includes LLM activity implementations for OpenAI and Anthropic and looks for:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

### Using a standard OpenAI-compatible `/chat/completions` server

By default, the OpenAI client uses the newer `/responses` endpoint.
If your provider exposes the more common `POST /chat/completions` API instead, set:

```bash
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_MODE="chat_completions"
```

Then start rho normally:

```bash
uv run rho --provider openai --model your-model-name
```

To avoid stale shell state, you can also pass the values inline:

```bash
OPENAI_BASE_URL="http://localhost:1234/v1" \
OPENAI_API_MODE="chat_completions" \
uv run rho --provider openai --model your-model-name
```

For a public Modal vLLM deployment:

```bash
OPENAI_BASE_URL="https://your-workspace--your-app-serve.modal.run/v1" \
OPENAI_API_MODE="chat_completions" \
uv run rho --provider openai --model your-model-name
```

Notes:

- `OPENAI_BASE_URL` must include the `/v1` prefix
- if you do not pass `--model`, the harness defaults to `gpt-4o-mini`
- many local or self-hosted servers do not require `OPENAI_API_KEY`; if yours does, set it before starting
- you can also set `OPENAI_API_MODE="auto"` to try `/responses` first and fall back to `/chat/completions` for simple requests; `chat_completions` is the safest setting for local providers

## rho-client

`rho-client` is a low-level CLI that talks directly to a running `AgenticWorkflow` over the Temporal API. It is useful when you want to script interactions with a workflow, inspect conversation state, or control a workflow that was started by a separate `rho` process (e.g. in a CI pipeline or a headless environment where the TUI is not available).

### Why it exists

`rho` bundles the worker and the TUI into one process. `rho-client` does neither — it only acts as a Temporal client. This means you can use it to interact with any `AgenticWorkflow` that a worker elsewhere is keeping alive, without starting your own worker.

### Commands

**Start a new workflow** (requires a running worker):

```bash
uv run rho-client start --message "List files in the current directory"
```

Prints JSON with the workflow ID:

```json
{"workflow_id": "py-agent-1234abcd", "run_id": "..."}
```

Save the `workflow_id` and pass it to the commands below.

**Send a follow-up message:**

```bash
uv run rho-client send --workflow-id py-agent-1234abcd --message "Read README.md"
```

**Read conversation history:**

```bash
uv run rho-client history --workflow-id py-agent-1234abcd
```

**Interrupt the current turn:**

```bash
uv run rho-client interrupt --workflow-id py-agent-1234abcd
```

**Shut down the workflow:**

```bash
uv run rho-client end --workflow-id py-agent-1234abcd --reason "done"
```

### Connection flags

All `rho-client` subcommands accept `--temporal-host` and `--namespace` to override the defaults:

```bash
uv run rho-client --temporal-host my-cluster:7233 --namespace prod start --message "Hello"
```

## Run tests

```bash
uv run pytest tests -q
uv run python -m compileall src tests
```

## Implemented Python tool runtime

Currently exposed in Python:

- `read_file`
- `write_file`
- `list_dir`
- `grep_files`
- `exec_command`
- `write_stdin`
- `shell`
- `shell_command`

Not yet implemented in Python:

- `apply_patch`

## Troubleshooting

If the client cannot connect, confirm the Temporal dev server is running and that `TEMPORAL_HOST_URL` / `TEMPORAL_NAMESPACE` match your server.

If the worker starts but command tools fail, make sure common local tools are installed. `grep_files` uses `rg` (ripgrep).
