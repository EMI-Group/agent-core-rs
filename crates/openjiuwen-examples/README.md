# openjiuwen examples

Twelve runnable programs that tour the Rust SDK. They mirror the Python reference tree
`examples/` (reduced: the research drops, device/training pipelines and vendor-backed
stores are not ported), and every one of them runs **offline** out of the box.

```bash
# list them
cargo metadata --no-deps --format-version 1 -p openjiuwen-examples | grep example

# run one
cargo run -p openjiuwen-examples --example 01_hello_agent

# the A2A example is the only one behind a feature flag
cargo run -p openjiuwen-examples --example 11_a2a_server_client --features a2a
```

## The set

| File | What it teaches | Python counterpart | Offline |
| --- | --- | --- | --- |
| `01_hello_agent.rs` | build a model + `ReActAgent`, one `invoke`, print the answer | `examples/react_agent/build_react_agent.ipynb` (no plain hello-world script exists) | `ScriptedModel` |
| `02_react_with_tools.rs` | two function tools (`#[tool]` macro + `FunctionTool`), one tool round | `examples/rl_calculator/tools.py` | scripted tool call |
| `03_sessions_and_resume.rs` | `AgentSession` + checkpointer, interrupt and resume | `examples/client_tool/client_tool_demo.py` | in-memory |
| `04_workflow_pipeline.rs` | workflow with a conditional edge and `${node.field}` references | `examples/intelli_router/intelliRouter_demo.py` | in-process |
| `05_context_and_memory.rs` | context windowing/compression, long-term memory write + recall | `examples/context_evolver/quickstart.py`, `examples/graph_memory/showcase_graph_memory.py` | in-memory |
| `06_retrieval.rs` | chunk → embed → store → search | `examples/retrieval/showcase_text_embedding.py` | `HashEmbedding` |
| `07_mcp_tool.rs` | an MCP-exposed tool mounted on a ReAct agent | `examples/mcp/stdio/client_as_tool.py` | local `McpClient` |
| `08_deep_agent.rs` | `DeepAgent` in a temp workspace doing a file-tool round | `examples/harness/deepagent_with_worktree.py` | `ScriptedModel` |
| `09_harness_provider.rs` | `create_harness("native", …)`: start → send → events → stop | — | loopback `MockLlmServer` |
| `10_agent_team.rs` | `TeamAgentSpec` → build → invoke | `examples/multi_agent/team_hybrid.py` | loopback `MockLlmServer` |
| `11_a2a_server_client.rs` | A2A server + client over loopback (`--features a2a`) | `examples/a2a/*` | loopback |
| `12_self_improvement.rs` | auto-harness orchestration and an evolving trainer over fakes | `examples/agent_evolving/react_agent_evolving.py` | fake runner/target |

## Offline by default, live on request

Nothing here needs a network. Examples that talk to a model use
`openjiuwen-test-support`:

- `ScriptedModel` — an in-process `Model` with an ordered response script (the default for
  01–03, 08, 12).
- `MockLlmServer` — a real loopback OpenAI-compatible HTTP server, used where the wire path
  matters (09, 10: the harness provider and the team runtime build their own clients from a
  base URL).

To run an example against a real provider instead, set the live switch and credentials:

```bash
export OPENJIUWEN_LIVE=1
export MODEL_PROVIDER=openai        # or anthropic
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...        # or API_KEY / OPENJIUWEN_API_KEY
export OPENAI_BASE_URL=...          # optional, for OpenAI-compatible vendors
cargo run -p openjiuwen-examples --example 01_hello_agent
```

Live mode is opt-in and defensive: with `OPENJIUWEN_LIVE` set but no model name or
credential, the example prints what is missing and continues offline instead of failing.

## Reading order

`01` → `03` cover a single agent and its session; `04` → `06` cover the composition
crates (workflow, context/memory, retrieval); `07` → `10` climb to the harness tier
(MCP, `DeepAgent`, the provider protocol, agent teams); `11` and `12` are the optional
protocol and self-improvement surfaces.

Each example prints its narrative to stdout and its diagnostics to stderr
(`OPENJIUWEN_LOG_LEVEL=debug` for detail). No example prints a credential, and each
one's stdout is deterministic so runs can be diffed.

## Notes and known workarounds

- **07** runs against a local `McpClient`; set `OPENJIUWEN_MCP_SERVER_CMD="<program> [args]"`
  to point it at a real stdio MCP server instead. A server that fails to answer is
  reported as a note (bounded by a timeout) and the example still exits 0.
- **10** wraps its model resolver so the resolved model name reaches the member's
  config. That is a workaround for a defect in `openjiuwen-agent-teams`
  (`NativeMemberFactory` does not propagate the name); once it is fixed in that
  crate the wrapper here can go.
- **11** logs the ephemeral loopback port to stderr, so its stdout stays stable
  across runs.
