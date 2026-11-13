# abilities/

## Intent

Per-agent tool registry, tool execution, and the concrete ability kinds agents mount.
Ports Python `core/single_agent/ability_manager.py` minus the `Runner.resource_mgr` resolution: the manager owns `Arc<dyn Tool>` instances directly.
Python's workflow, sub-agent, and MCP ability paths are ported as ordinary `openjiuwen_foundation::Tool` implementations mounted through `AbilityManager::add_tool`.

## API Surface

### `mod.rs` — registry

- `AbilityManager` (`new(owner_id)`, `add_tool`, `remove_tool`, `get`, `cards`, `tool_info`, `revision`, `reorder`, `len`, `is_empty`, `execute`).
- `AddAbilityResult { name, added, reason }` (`added_tool | refreshed_tool | duplicate_tool`).
- `ToolExecutionOptions` and `ToolCallResult { call, output, text, success, error }` with `ToolCallResult::message()`.

### `arguments.rs` — argument repair

- `parse_tool_arguments(&str) -> Result<Value>`: empty input is `{}`; truncated JSON is repaired (bracket/quote balancing); unrepairable input is `Error::Tool`.

### `execute.rs` — execution engine

- Execution policy: model order, contiguous `parallel_safe` batches run concurrently (`futures::future::join_all`, capped by `MAX_PARALLEL_BATCH`), non-parallel-safe tools act as barriers.
- Per-call `tokio::time::timeout` from `ToolExecutionOptions::timeout`; cancellation via `CancellationToken`.
- Failures, timeouts, malformed arguments, and unknown tools become error results, never `Err`.

### `sub_agent.rs` — `SubAgentAbility`

- `SubAgentAbility::new(Arc<dyn crate::Agent>)`: card name/description/id come from the child `AgentCard`; input schema is `{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`.
- Builders: `with_name`, `with_description`, `with_session(Arc<dyn Session>)`; accessors `agent()`, `session()`; `boxed() -> Arc<dyn Tool>`; manual `Debug`.
- `invoke` accepts an object with a string `query` or a bare JSON string; anything else is `Error::Validation`.
- Calls the child with `json!({"query": q})`; an `answer` result returns `Value::String(output.output)`; `error`/`interrupt` results become `Error::Tool` naming the child and result type.
- The child session is fixed at construction because `ToolContext` only carries a `session_id`, not an `Arc<dyn Session>`; without `with_session` the child runs session-less.

### `workflow.rs` — `WorkflowAbility`

- `WorkflowAbility::new(Arc<openjiuwen_workflow::Workflow>)`: card name is `WorkflowCard::name` when non-empty else `id`, description from the card, card id is the workflow id; input schema is `{"type":"object","properties":{"query":{"type":"string"}},"additionalProperties":true}`.
- Builders `with_name`, `with_description`; accessor `workflow()`; `boxed()`; manual `Debug`.
- `invoke` merges the input object into the workflow payload: a string `query` is forwarded as `query` and every other key is merged verbatim; a non-object is `Error::Validation`, and an object with neither `query` nor extras is rejected.
- Each call gets a fresh `WorkflowSession` (`session_id = ToolContext::session_id` or the invocation id, `workflow_id = workflow card id`).
- `Completed` returns `WorkflowOutput::result` (the model-visible text is the default `render_for_llm` compact JSON); `InputRequired` becomes `Error::Tool` stating interactive workflow resume from a tool call is out of scope; `Error` and run failures become `Error::Tool`.

### `mcp.rs` — MCP abilities

- `McpToolAbility::new(ToolCard, Arc<dyn McpClient>, tool_name)` wraps one remote tool by delegating to `openjiuwen_foundation::McpTool`; `with_options(McpToolOptions)`, accessors `remote_name()`, `server_name()`, `mcp_tool()`, `boxed()`; manual `Debug`.
- `McpToolAbility::invoke` maps `Validation`/`Cancelled`/`Timeout` through and every other client/server failure to `Error::Tool` naming the card.
- `McpAbilityOptions { namespace: Option<String>, allowlist: Option<McpToolAllowlist>, prefix: bool }` with `new`, `with_namespace`, `with_allowlist`, `with_prefix`, `without_prefix`; default `prefix = true`.
- `load_mcp_abilities(client, options) -> Result<Vec<Arc<dyn Tool>>>` lists via `load_mcp_tools`, filters, renames, and returns mountable tools; `list_tools` errors propagate unchanged.
- Naming: prefixed names are `<namespace>__<tool>` (double underscore), with `namespace` defaulting to `client.server_name()`; `prefix = false` keeps the remote name.
- The card id stays foundation's `<server_id>.<server_name>.<tool>` so registry identity/refresh semantics are unaffected; metadata always carries `mcp`, `server_name`, `server_id`, `remote_name`, plus `namespace` when prefixed.
- Allowlist key: `McpToolAllowlist::allows(server, remote_name)` where `server` is `client.server_id()` and falls back to `client.server_name()` when the id is empty.
- Documented deviation: Python names MCP tools `mcp_<server>_<tool>`; the double-underscore form avoids ambiguity when names contain underscores.

## Constraints

- No rails here: the ReAct loop fires `BeforeToolCall`/`AfterToolCall` around `execute`.
- `tool_info()` order equals registration order (the model-visible tool order), `cards()` returns clones.
- `revision` increments only on successful mutations; `reorder` keeps unknown names out.
- Registry is keyed by tool name; a duplicate name with the same `ToolCard.id` is a refresh, a different id is rejected.
- Rendering for the LLM uses `Tool::render_for_llm`.
- The ability kinds are plain `Tool` impls: they never touch `AbilityManager` internals, so `add_tool`/`remove_tool`/`execute` stay the only registry paths.
- Sub-agent and workflow abilities hold their `Arc<dyn Agent>` / `Arc<Workflow>` directly; there is no global resource lookup.
- Workflow interrupt/resume from a tool call is out of scope: an `InputRequired` result is surfaced as `Error::Tool`, not as a paused tool call.

## Test Strategy

- Inline unit tests: registration/refresh/duplicate/removal, revision bumping, reorder, argument repair cases, parallel vs barrier ordering, timeout and unknown-tool error results.
- `sub_agent.rs` (10): card inheritance, builders/Debug, answer forwarding and recorded `{"query": …}`, bare-string query, missing/non-string/non-object input, child `error`/`interrupt` → `Error::Tool`, `boxed`.
- `workflow.rs` (12): card derivation/fallbacks, builders/Debug, completed `{"answer": "hi"}` via a real start→end workflow, extra-key merge, empty/null query rules, broken workflow → `Error::Tool`, `boxed`.
- `mcp.rs` (14): local `FakeClient` (foundation's `FakeMcpClient` is test-only) covering projection preservation, namespace/prefix rules, allowlist filtering and server-name fallback, invocation forwarding + `render_for_llm`, client error → `Error::Tool`, cancellation passthrough, card metadata, Debug, and option builders.

## Routing Table

- `./mod.rs` → registry types
- `./arguments.rs` → argument parsing/repair
- `./execute.rs` → execution engine
- `./sub_agent.rs` → `SubAgentAbility` (child agent as a tool)
- `./workflow.rs` → `WorkflowAbility` (workflow as a tool)
- `./mcp.rs` → `McpToolAbility`, `McpAbilityOptions`, `load_mcp_abilities` (MCP server tools)
