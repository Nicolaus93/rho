# Python runtime

This repo now includes a phase-1 Python port under `src/rho`.

## Current status

The Python side currently provides:

- Temporal worker entry point
- Temporal client CLI for starting and interacting with workflows
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

## Start the Python worker

In a second terminal:

```bash
uv run temporal-agent-harness-worker
```

You can also run it as a module:

```bash
uv run python -m rho.runtime.worker_main
```

## Start a workflow

In a third terminal:

```bash
uv run rho --provider openai --model llm "Hello"
```

If you want to use the lower-level workflow client directly, the compatibility command still works:

```bash
uv run temporal-agent-harness-client start --message "List files in the current directory"
```

That prints JSON like:

```json
{"workflow_id":"py-agent-1234abcd","run_id":"..."}
```

Save the `workflow_id` and use it in the commands below.

## Interact with a running workflow

Send another user message:

```bash
uv run temporal-agent-harness-client send --workflow-id py-agent-1234abcd --message "Read README.md"
```

Read conversation history:

```bash
uv run temporal-agent-harness-client history --workflow-id py-agent-1234abcd
```

Interrupt the workflow:

```bash
uv run temporal-agent-harness-client interrupt --workflow-id py-agent-1234abcd
```

Request shutdown:

```bash
uv run temporal-agent-harness-client end --workflow-id py-agent-1234abcd --reason "done"
```

## Notes about models and API keys

The Python worker includes LLM activity implementations for OpenAI and Anthropic and looks for:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

But the current phase-1 workflow path is still limited and is **not** yet full Go-style agent parity. So for basic local workflow bring-up, you may not need an API key depending on what path you exercise.

### Using a standard OpenAI-compatible `/chat/completions` server

By default, the OpenAI client in the worker uses the newer `/responses` endpoint.
If your local or remote provider exposes the more common OpenAI-compatible `POST /chat/completions` API instead, set:

```bash
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_MODE="chat_completions"
```

If the server does not require authentication, you can leave `OPENAI_API_KEY` unset.
If it does require auth, also set:

```bash
export OPENAI_API_KEY="your-token-here"
```

Then start the worker and use `rho` normally:

```bash
uv run temporal-agent-harness-worker
uv run rho "List files in the current directory"
```

The worker reads these values at startup time.
If you change `OPENAI_BASE_URL`, `OPENAI_API_MODE`, or `OPENAI_API_KEY`, restart the worker so it picks up the new configuration.

To avoid stale shell state, you can also start the worker with the values inline:

```bash
OPENAI_BASE_URL="http://localhost:1234/v1" \
OPENAI_API_MODE="chat_completions" \
uv run temporal-agent-harness-worker
```

For a public Modal vLLM deployment, the working shape looks like this:

```bash
OPENAI_BASE_URL="https://your-workspace--your-app-serve.modal.run/v1" \
OPENAI_API_MODE="chat_completions" \
uv run temporal-agent-harness-worker

uv run rho --provider openai --model llm "Hello"
```

Notes:

- `OPENAI_BASE_URL` must include the `/v1` prefix
- if you do not pass `--model`, the harness may still default to `gpt-4o-mini`
- many local or self-hosted OpenAI-compatible servers do not require `OPENAI_API_KEY`; if yours does, set it before starting the worker

You can also use:

```bash
export OPENAI_API_MODE="auto"
```

In `auto` mode, the client tries `/responses` first and only falls back to `/chat/completions` for simple text requests when the responses endpoint is clearly unsupported.
If you want predictable behavior with local providers, `chat_completions` is the safest setting.

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
