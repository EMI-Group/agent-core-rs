# Workspace crates

## Intent

Every crate in the Rust rewrite of openJiuwen.
Each crate maps to a Python package (or focused kernel module) and owns its own `CONTEXT.md`.
Dependencies point downward through the layers defined in `../docs/ARCHITECTURE.md`.

## API Surface

The public API of each crate is documented in its own `CONTEXT.md`.
The stable seams between crates are defined in `../docs/ARCHITECTURE.md` §2.

## Constraints

- A crate may depend only on crates in lower layers (see table); no cycles.
- `openjiuwen-core` and `openjiuwen` are umbrella crates; concrete crates must not depend on them except the harness layer and above.
- Shared types belong in `openjiuwen-common` or `openjiuwen-foundation`, never duplicated across crates.
- Crates are `publish = false` until the API stabilizes.
- Test-only crates are never dependencies of library crates.

## Routing Table

### Layer 0 — independent foundations

- `./openjiuwen-common/` → Card/Config schema, typed errors, logging, env/config, HTTP helpers, task manager, utilities
- `./openjiuwen-macros/` → proc macros (`#[tool]`, schema derivation helpers)
- `./openjiuwen-harness-protocol/` → provider-neutral harness SPI (traits, events, results, streams)

### Layer 1 — primitives

- `./openjiuwen-foundation/` → LLM clients and `Model`, prompts/messages, tools + MCP, storage backends
- `./openjiuwen-session/` → sessions, state scopes, streaming, interactions, checkpointers, tracer
- `./openjiuwen-graph/` → Pregel/BSP graph execution engine
- `./openjiuwen-retrieval/` → documents, chunking, embeddings, vector stores, rerankers, knowledge base
- `./openjiuwen-sysop/` → local/sandbox filesystem, shell, and code operations

### Layer 2 — composition

- `./openjiuwen-workflow/` → workflow builder, components, `${node.field}` data flow
- `./openjiuwen-context/` → context windows, token accounting, compression/offload
- `./openjiuwen-memory/` → long-term and coding memory
- `./openjiuwen-security/` → guardrails and content-safety pipeline

### Layer 3 — agents

- `./openjiuwen-agent/` → `BaseAgent`, `ReActAgent`, abilities, rails, interrupts, `LLMAgent`, `WorkflowAgent`
- `./openjiuwen-multiagent/` → `BaseTeam`, team runtime, message bus, strategies

### Layer 4 — orchestration

- `./openjiuwen-runner/` → `Runner`, `ResourceMgr`, callbacks, MQ abstraction
- `./openjiuwen-core/` → kernel umbrella re-exporting `common`, `foundation`, `session`, `graph`, `workflow`, `context_engine`, `memory`, `retrieval`, `sys_operation`, `security`, `single_agent`, `application`, `multi_agent`, `runner`

### Layer 5 — product frameworks

- `./openjiuwen-harness/` → DeepAgent task loop, rails, workspace, permissions, skills, subagents, prompts, CLI logic
- `./openjiuwen-extensions/` → A2A, OTel tracer, checkpointer/store/MQ/sysop adapters (feature-gated)
- `./openjiuwen-dev-tools/` → agent builder, prompt builder, skill creator/evaluator, tuning helpers
- `./openjiuwen-symphony/` → capability-asset engine (stub, deferred — see `../docs/PARITY.md`)

### Layer 6 — providers and self-improvement

- `./openjiuwen-harness-providers/` → harness provider registry; `native` implemented, vendor providers deferred
- `./openjiuwen-rsi/` → recursive self-improvement (stub, deferred)
- `./openjiuwen-agent-evolving/` → evaluation/optimisation/RL (stub, deferred)
- `./openjiuwen-auto-harness/` → re-export shim over `rsi::auto_harness`
- `./openjiuwen-agent-teams/` → leader/teammate runtime, worktrees, HITL, skill CLI/MCP server

### Layer 7 — top-level

- `./openjiuwen/` → top-level umbrella re-exporting all public module trees
- `./openjiuwen-cli/` → binary `openjiuwen`

### Development crates

- `./openjiuwen-test-support/` → mock LLM server, fakes, fixtures
- `./openjiuwen-e2e/` → cross-crate integration scenarios
- `./openjiuwen-examples/` → runnable examples mirroring the Python `examples/` tree
