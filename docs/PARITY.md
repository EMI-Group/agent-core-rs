# Parity Matrix

Reference: openJiuwen Python SDK v0.1.20 (`/home/bill/Source/agent-core`).
Status legend: **port** = targeted for full behavioral parity; **subset** = core behavior ported, peripheral paths deferred; **stub** = API-shaped placeholder that returns `Error::Unsupported`; **deferred** = not implemented in this rewrite.

## Tier 0 — kernel (port)

| Python area | Rust crate | Status | Notes |
|---|---|---|---|
| `core.common` (logging, exceptions, Cards/params, HTTP, task manager, security utils) | `openjiuwen-common` | port | `tracing` logging facade; thiserror hierarchy; serde Cards. |
| `core.foundation.llm` | `openjiuwen-foundation` | port | OpenAI-compatible + Anthropic clients; streaming, tool calls, usage, retries, timeouts; provider registry. |
| `core.foundation.prompt` | `openjiuwen-foundation` | port | `{{slot}}` templates, message/content parts, JSON conversion. |
| `core.foundation.tools` | `openjiuwen-foundation` | port | `Tool` trait, `ToolCard`, `FunctionTool`, REST tool, `#[tool]` macro. |
| `core.foundation.tools` MCP | `openjiuwen-foundation` | subset | stdio and streamable-HTTP/SSE client via `rmcp`; reconnect/expiry; OpenAPI/Playwright transports deferred. |
| `core.foundation.storage` | `openjiuwen-foundation` | subset | kv/db/vector/graph/object traits; in-memory + local filesystem + optional SQLite. Redis/Milvus/Chroma/Elasticsearch/pgvector/S3 deferred. |
| `core.session` | `openjiuwen-session` | port | Session hierarchy, state scopes, streaming, interactions, checkpoint/resume, tracer hooks. |
| `core.graph` | `openjiuwen-graph` | port | Pregel/BSP execution, channels, conditional edges, streaming. |
| `core.workflow` | `openjiuwen-workflow` | port | Builder API, `${node.field}` resolution, Start/End, LLM/tool/branch/loop/retrieval components. |
| `core.single_agent` | `openjiuwen-agent` | port | `BaseAgent`, `ReActAgent`, abilities, rails, interrupts, streaming. |
| `core.application` | `openjiuwen-agent` | port | `LLMAgent`, `WorkflowAgent`. |
| `core.runner` | `openjiuwen-runner` | port | `Runner`, `ResourceMgr`, callbacks, in-process MQ; Pulsar/ZeroMQ/subprocess spawn subset. |
| `core.context_engine` | `openjiuwen-context` | subset | Token accounting, window management, compression/summarisation/offload; exact Python tokenizer parity approximate (`tiktoken-rs`). |
| `core.memory` | `openjiuwen-memory` | subset | Long-term memory, store/search/recall; entity/graph memory basic; Lite coding memory port. |
| `core.retrieval` | `openjiuwen-retrieval` | subset | Knowledge base, chunking, embeddings trait, in-memory vector store, reranker trait; external vector DBs deferred. |
| `core.sys_operation` | `openjiuwen-sysop` | subset | Local fs/shell/code ops with allowlists, timeouts, output limits; remote sandbox HTTP backend stubbed behind a feature. |
| `core.security` | `openjiuwen-security` | subset | Guardrail pipeline (critical-only blocking); vendor/cloud moderation adapters deferred. |
| `core.multi_agent` | `openjiuwen-multiagent` | subset | `BaseTeam`, `TeamCard`, `TeamConfig`, in-process message bus, core strategies; distributed transports deferred. |

## Tier 1 — harness and teams (subset → port)

| Python area | Rust crate | Status | Notes |
|---|---|---|---|
| `harness` DeepAgent runtime (task loop, rails, workspace, permissions, skills, subagents, prompts, observability) | `openjiuwen-harness` | subset | Core coding-agent loop, tool set, workspace, permission engine, skills, plan/explore/research/verification subagents, interrupts/HITL. Browser/CUA/mobile subagents deferred. |
| `harness.cli` | `openjiuwen-cli` | subset | `run` and interactive REPL; session resume and permission prompts. |
| `harness_protocol` | `openjiuwen-harness-protocol` | port | Provider-neutral SPI, events/results/streams. |
| `harness_providers` | `openjiuwen-harness-providers` | subset | `native` provider; `claudecode`, `codex`, `dsh` deferred (vendor SDK/CLI integrations). |
| `agent_teams` | `openjiuwen-agent-teams` | subset | Leader/teammate runtime, worktrees, HITL, skill CLI/MCP server; process/ZeroMQ/PG distributed mode deferred. |
| `extensions.a2a` | `openjiuwen-extensions` | subset | A2A client/server adapter with agent-card exchange; optional feature. |
| `extensions.tracer_otel` | `openjiuwen-extensions` | subset | OTel tracing behind `otel` feature. |
| `extensions.checkpointer/store/message_queue/sys_operation/context_evolver` | `openjiuwen-extensions` | subset | In-process and file/SQLite variants; Redis/Pulsar/S3 variants deferred. |
| `dev_tools` (agent builder, prompt builder, skill creator/evaluator, tune) | `openjiuwen-dev-tools` | subset | Prompt builder and skill creator ported; optimizer/tuning pipelines stub. |

## Tier 2 — explicitly deferred

These are stubbed with API-shaped modules and `Error::Unsupported`, or omitted with a documented reason. They are reported as known gaps, not silent omissions.

| Area | Reason |
|---|---|
| `agent_evolving` (trainer/evaluator/optimizer, online RL service) | Depends on Python ML stack (`transformers`, `torch`, distributed RL, `agentdescent` internals) with no maintained Rust equivalents; no drop-in replacement. |
| `rsi` / `auto_harness` self-improvement pipelines | Depends on `agent_evolving`, PUCT program optimisation and vendor harness SDKs. |
| `symphony` capability-asset engine | Large domain engine, near-leaf; depends on retrieval/vector backends. |
| Sandbox remote AIO container (`PreDeployLauncher`) | External service protocol; local sys-op backend + trait seam provided. |
| CUA desktop, mobile GUI, browser Playwright subagents | Platform automation and external workers; not part of the agent runtime core. |
| Claude Code / Codex / DSH harness providers | Vendor SDK/CLI transports; protocol seam provided. |
| External storage/MQ backends: Redis, Milvus, Chroma, Elasticsearch, Pulsar, ZeroMQ, S3, PostgreSQL | Adapter work; traits and in-memory/local implementations provided. |
| Document parsing: PDF/DOCX/XLSX, `trafilatura` article extraction | Format coverage differs; basic text/HTML/Markdown handled, rich extraction deferred. |
| Full Python unit/system test suite (~1600 unit files) | Not ported wholesale; representative behavior tests ported per module. |
| `plugins/medical_kb`, `deploy/` manifests, docs site | Outside the SDK package. |
| Prompt tuning / agent-descent optimisation internals | Research code tied to Python ML ecosystem. |

## Intentional deviations

- Python runtime introspection, decorators, and pydantic validation are replaced with typed structs, traits, `schemars` schemas, and the `#[tool]` attribute macro.
- Process-global `Runner` remains available for parity; additionally a local `Runner` can be constructed for isolation and tests.
- Streaming is represented as `BoxStream<Result<T>>` instead of asyncio queues; cancellation uses `CancellationToken`.
- Checkpoint formats preserve JSON field names, but Python pickle-specific artifacts are not supported.
- Token counting uses `tiktoken-rs` with a heuristic fallback; counts may differ from Python for non-OpenAI models.
- Errors are a single typed enum instead of Python's exception class hierarchy; error kinds preserve the original category names.
