# coding-agent-cli

A permission-gated coding agent built as a complete example project on top of
`ferragent`. It can inspect an existing or new workspace, edit files, use a
constrained Git interface, install dependencies, and iteratively build, test,
and run code.

## Run

From the `ferragent` repository root:

```bash
cargo run --manifest-path examples/coding-agent-cli/Cargo.toml -- \
  --workspace /path/to/project
```

The preferred provider configuration is `~/.code-agent-cli.config`. It selects
one provider and stores only the fields needed by that provider. Create it with
one of these shapes:

```ini
# OpenAI
provider=openai
api_key=sk-...
model=gpt-5-mini
```

```ini
# Anthropic
provider=anthropic
api_key=sk-ant-...
model=claude-sonnet-4-6
```

```ini
# Local or any OpenAI-compatible server
provider=compatible
base_url=http://127.0.0.1:8080/v1
model=LiquidAI/LFM2.5-230M-GGUF:Q8_0
# api_key is optional for servers that do not require one
```

Protect API keys on Unix-like systems:

```bash
chmod 600 ~/.code-agent-cli.config
```

The file must be a regular file rather than a symlink. Blank lines, lines
starting with `#`, and single- or double-quoted values are supported. When the
file is absent, the CLI falls back to interactive provider selection using the
existing `OPENAI_*`, `ANTHROPIC_*`, or `OPENAI_COMPATIBLE_*` environment
variables (including values loaded from `.env`).

## Safety model

- Workspace access is explicitly approved at startup. `.env`, credential
  files, and `.git` internals are never exposed through file tools.
- Every write, Git operation, command, and dependency installation requires a
  direct terminal confirmation. Model output cannot grant approval.
- Commands use an executable plus argument array; model-provided shell strings
  are never evaluated.
- Commands run with a five-minute timeout, capped output, scrubbed environment,
  and no inherited API keys.
- The provider config is read only by the CLI. It is excluded from workspace
  listing/reading and denied to sandboxed child processes, even if the home
  directory itself is selected as the workspace.
- On macOS, `sandbox-exec` confines writes to the workspace and disables
  network by default. On Linux, `bwrap` provides the equivalent boundary.
- Dependency installation requires a separate approval and enables network
  only for that sandboxed command.
- If no supported sandbox exists or it cannot start, the CLI explains why and
  asks again before running that exact command locally. It never silently
  falls back.
- Git is restricted to `status`, `diff`, `log`, `init`, `add`, and `commit`;
  destructive and remote operations are not exposed.

Local models must support OpenAI-style chat completions and tool calling. Very
small models may serve the API successfully but still be unable to plan and
emit reliable tool calls; use a capable coding/tool-use model for real work.
