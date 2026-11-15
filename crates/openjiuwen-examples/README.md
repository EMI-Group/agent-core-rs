# openjiuwen examples

Twenty-six runnable programs that tour the Rust SDK. They mirror the Python reference tree
`examples/` (reduced: the research drops, device/training pipelines and vendor-backed
stores are not ported), and every one of them runs **offline** out of the box.

```bash
# list them (one file per `[[example]]` target)
ls crates/openjiuwen-examples/examples/*.rs

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
| `03_sessions_and_resume.rs` | `AgentSession` + checkpointer, interrupt and resume | `examples/client_tool/client_tool_demo.py` | temp file (`FileCheckpointer`) |
| `04_workflow_pipeline.rs` | workflow with a conditional edge and `${node.field}` references | `examples/intelli_router/intelliRouter_demo.py` | in-process |
| `05_context_and_memory.rs` | context windowing/compression, long-term memory write + recall | `examples/context_evolver/quickstart.py`, `examples/graph_memory/showcase_graph_memory.py` | in-memory |
| `06_retrieval.rs` | chunk → embed → store → search | `examples/retrieval/showcase_text_embedding.py` | `HashEmbedding` |
| `07_mcp_tool.rs` | an MCP-exposed tool mounted on a ReAct agent | `examples/mcp/stdio/client_as_tool.py` | local `McpClient` |
| `08_deep_agent.rs` | `DeepAgent` in a temp workspace doing a file-tool round | `examples/harness/deepagent_with_worktree.py` | `ScriptedModel` |
| `09_harness_provider.rs` | `create_harness("native", …)`: start → send → events → stop | — | loopback `MockLlmServer` |
| `10_agent_team.rs` | `TeamAgentSpec` → build → invoke | `examples/multi_agent/team_hybrid.py` | loopback `MockLlmServer` |
| `11_a2a_server_client.rs` | A2A server + client over loopback (`--features a2a`) | `examples/a2a/*` | loopback |
| `12_self_improvement.rs` | auto-harness orchestration and an evolving trainer over fakes | `examples/agent_evolving/react_agent_evolving.py` | fake runner/target |
| `13_context_processors.rs` | the processor pipeline stage by stage: accounting, truncation, tool-result offload, summarization, reload | `examples/context_evolver/quickstart.py` (engine-level only) | scripted summary |
| `14_workflow_components.rs` | sub-workflow, HTTP request, and a questioner suspension/resume in one workflow | `examples/workflow_agent/multi_workflow_agent_demo/multi_workflow_agent_demo.py` (questioner only) | loopback `TcpListener` |
| `15_retrieval_quality.rs` | BM25 vs dense ranking, query rewriting, DOCX/XLSX/PDF parsing | `examples/retrieval/showcase_query_rewriter.py` (rewrite only) | `Bm25Index` + `HashEmbedding` |
| `16_memory_lite.rs` | engine-backed `CodingMemory` and `LocalCodingMemory` (`MEMORY.md` + FTS5) | — | temp workspace |
| `17_team_persistence.rs` | SQLite team store plus the organisation layer | — | temp SQLite file |
| `18_team_attach.rs` | `AttachServer`/`AttachClient` over a Unix socket | — | Unix socket |
| `19_symphony_skills.rs` | SKILL.md discovery, fingerprinted inventory, lexical search, registry/graph | — | temp files |
| `20_prompt_tuning.rs` | `dev_tools::tune`: cases, scoring, optimiser, trainer | `examples/agent_evolving/react_agent_evolving.py` | `ScriptedPromptRunner` |
| `21_native_tokenizers.rs` | a generated `tokenizer.json` and tiktoken `.model`, the selector fallback ladder, the artefact cache | — | generated artefacts |
| `22_context_telemetry.rs` | `context.usage` snapshots over a compressed transcript plus `context.compression_state` records | — | `ApproxTokenCounter` + scripted summary |
| `23_memory_dreaming.rs` | `DreamOrchestrator` ticks, a deterministic consolidator, backoff and the SQLite job queue | `memory/dreaming/` (no example) | temp SQLite + injected clock |
| `24_runner_spawn_registry.rs` | `SpawnerRegistry` routing, an in-process and an external-command backend, health probes, `active_spawns` | — | in-process echo + self re-exec |
| `25_harness_moa_persistent.rs` | MoA consultation/ranking/aggregation and a persistent-agent journal replay + resume plan | — | `ScriptedModel` + file journal |
| `26_online_training_service.rs` | the online RL state machine (task → capture → ledger → train → status) over the in-memory store | — | `MemoryOnlineStore` + local executor |

## Offline by default, live on request

Nothing here needs a network. Examples that talk to a model use
`openjiuwen-test-support`:

- `ScriptedModel` — an in-process `Model` with an ordered response script (the default for
  01–03, 07, 08, 13, 22, 25).
- `MockLlmServer` — a real loopback OpenAI-compatible HTTP server, used where the wire path
  matters (09, 10: the harness provider and the team runtime build their own clients from a
  base URL).

Examples with no model seam use in-memory or generated data instead: `HashEmbedding` and
`Bm25Index` (06, 15), a scripted harness runner and fake target (12), `ScriptedPromptRunner`
+ `DeterministicEvaluator` (20), generated tokenizer artefacts (21), a temp SQLite dream
queue (23), pipes and an in-process handler (24), a file journal (25), and in-memory
stores, temp files, a Unix socket, or a temp SQLite database (14, 16–19, 26).

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
Every example still exits 0 with a note when its optional live prerequisite is absent.

## Reading order

`01` → `03` cover a single agent and its session; `04` → `06` cover the composition
crates (workflow, context/memory, retrieval); `07` → `10` climb to the harness tier
(MCP, `DeepAgent`, the provider protocol, agent teams); `13` → `16` go deeper into the
kernel seams (context processors, workflow components, retrieval quality, lite memory);
`17` and `18` cover durable team state and cross-process attach; `19` and `20` cover
symphony skill discovery and prompt tuning; `21` → `26` tour the newest seams (native
tokenizers, context telemetry, memory dreaming, the spawn registry, MoA plus the persistent
agent, and the online training service); `11` and `12` remain the optional protocol and
self-improvement surfaces.

Each example prints its narrative to stdout and its diagnostics to stderr
(`OPENJIUWEN_LOG_LEVEL=debug` for detail). No example prints a credential. The
narrative output is deterministic, but identifiers such as generated UUIDs, temp
paths, interrupt ids, and ephemeral ports vary between runs, so a diff shows those
lines.

## Testing

```bash
cargo test -p openjiuwen-examples --offline
```

runs the support-library unit tests (15 of them: run-mode detection, subscriber
installation, banner formatting) and the offline smoke suite in
`tests/examples_offline.rs`, whose single test executes every example binary the test
profile builds in a clean environment — live variables removed, a temp working directory,
a per-example timeout, and captured output — and asserts exit code 0 with non-empty stdout.
That is 25 examples with default features, and 26 under `--features a2a`:
`11_a2a_server_client` joins the run only then, because with the feature off its example
target is not built.

The three SQLite-backed examples (`16` memory FTS index, `17` team store, `23` dream queue)
need no setup: the backend is compiled in through the crate's dev-dependencies, so the
suite stays offline.

## Notes and known workarounds

- **07** runs against a local `McpClient`; set `OPENJIUWEN_MCP_SERVER_CMD="<program> [args]"`
  to point it at a real stdio MCP server instead. A server that fails to answer is
  reported as a note (bounded by a timeout) and the example still exits 0.
- **10** injects `NativeMemberFactory` with an `EnvModelResolver` pointed at the loopback
  server; the factory copies the resolved model's name and provider onto the member's
  `DeepAgentConfig`, so no wrapper model is needed.
- **11** logs the ephemeral loopback port to stderr and prints only the host on stdout,
  so the varying port number never appears in the narrative.
- **15** writes its own DOCX/XLSX/ZIP and PDF fixtures because the retrieval crate ships
  parsers but no archive writer; the auto loader replaces `metadata.title` with the file
  stem.
- **17** records message stamps verbatim and leaves a dependent task's status alone when a
  dependency edge is added — the task board owns that transition.
- **19** sends `SkillFs` diagnostics (including the absolute skill path) to stderr; the
  narrative prints only relative paths, names, scores, and digests.
- **21** generates both tokenizer artefacts (BPE JSON and a base64 rank table) in a temp
  dir, so no download is attempted; cache misses are reported as `None` plus a compact
  `last_error()` code, not as an error, and the selector's `source`/`fallback_reason` pair
  is the observable fallback ladder.
- **22** reads `context.usage` snapshots from the engine (`PreparedContext.usage`,
  `usage_snapshot`) and processor records from a `RecordingSink`; without an analyzer the
  engine answers a typed `config` error rather than a silent zero.
- **23** ticks with an injected `DateTime` (no sleeping), and the dreaming pass only moves
  `dream_jobs` through its state machine — nothing is merged back into memory records.
- **24**'s external backend re-execs the example binary as an NDJSON child; child mode must
  not print a banner, and both children are shut down explicitly so `active_spawns` empties.
- **25** keeps MoA and the persistent agent standalone (no `DeepAgentConfig.moa` field);
  the journal is a directory, so dropping the handle and reopening is the crash simulation.
