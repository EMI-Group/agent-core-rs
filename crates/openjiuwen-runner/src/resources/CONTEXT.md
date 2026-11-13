# resources/ — four registries and five sub-managers

## Intent

The runner's catalogue of the resources a host publishes: four run-able registries (tools, agents, workflows, agent teams) plus five sub-managers ported from Python `openjiuwen/core/runner/resources_manager/` (v0.1.20): tags, models, prompts, system operations, and MCP servers.
Everything here is reached through `ResourceMgr`, which the owning `Runner` holds as an `Arc`; the sub-managers are reachable both through typed accessors (`mgr.tags()`, `mgr.models()`, …) and through module paths (`openjiuwen_runner::resources::tag::TagMgr`, …).
The four registries are the run-able kinds (`Runner::run_agent` / `run_workflow` / `run_agent_team` resolve ids here); the five sub-managers add the catalogue Python keeps in the same class: tag indexes, model factories, prompt templates, sys-operation instances, and MCP server lifecycle.
Every manager is interior-mutable behind `parking_lot` locks and takes `&self`, so a host can hold one `Arc<ResourceMgr>` and mutate it from anywhere.

## API Surface

### `mod.rs` — `ResourceMgr`
```rust
pub struct ResourceMgr { /* 4 registries + tags, models, prompts, sysops, mcp_servers */ }
impl ResourceMgr {
    pub fn new() -> Arc<Self>;
    // tools / agents / workflows / teams: register_*[_with], get, *_by_name, *_card,
    // remove_*, plural cards, *_ids, *_revision, clear_*, *_count (as before)
    pub fn tags(&self) -> &TagMgr;                        // #[must_use]
    pub fn models(&self) -> &ModelMgr;
    pub fn prompts(&self) -> &PromptMgr;
    pub fn sysops(&self) -> &SysOperationMgr;
    pub fn mcp_servers(&self) -> &McpServerMgr;
    // tag facade (declared in tag.rs as `impl ResourceMgr`)
    pub fn tag(&self, kind, id, tags: impl IntoIterator<Item = Tag>) -> Result<Vec<Tag>>;
    pub fn update_tags(&self, kind, id, tags, strategy: TagUpdateStrategy) -> Result<Vec<Tag>>;
    pub fn tags_of(&self, kind, id) -> Vec<Tag>;
    pub fn resources_by_tags(&self, kind, tags: &[Tag], strategy: TagMatchStrategy) -> Result<Vec<String>>;
    pub fn untag(&self, kind, id) -> Vec<Tag>;
    pub fn remove_tag(&self, tag: &Tag) -> Vec<String>;
    pub fn tag_revision(&self) -> u64;
    pub fn is_empty(&self) -> bool;                       // four run-able registries only
    pub fn is_pristine(&self) -> bool;                    // + all five sub-managers, incl. the tag index
    pub fn release(&self);                                // resets all nine
}
pub fn workflow_key(id: &str, version: &str) -> String;    // "{id}_{version}"
```
Flat re-exports make the whole surface reachable as `openjiuwen_runner::resources::<Name>`: `Provider`, `RegisterPolicy`, `Registration`, `Registry`, `ResourceCard`, `Tag`, `TagMgr`, `ResourceKind`, `TagMatchStrategy`, `TagUpdateStrategy`, `GLOBAL_TAG`, `ModelCard`, `ModelMgr`, `ModelProvider`, `PromptCard`, `PromptMgr`, `SysOperationMgr`, `SysOperationProvider`, `DefaultMcpConnector`, `McpConnector`, `McpServerInfo`, `McpServerMgr`, `McpServerState`.
Each module is also a `pub mod`, so `openjiuwen_runner::resources::tag::TagMgr` and friends resolve.
Foundation's `McpClient`, `McpServerConfig`, and `McpTool` are deliberately **not** re-exported here; hosts import them from `openjiuwen_foundation::tool::mcp`.
`ResourceMgr` has a manual `Debug`: registry ids + revisions, `(count, revision)` per sub-manager, MCP server ids + states, tag count + revision — never an `McpServerConfig`, so `auth_headers`/`auth_query_params` cannot leak into a log.

### `registry.rs` — the generic machinery
`ResourceCard` (impls for `ToolCard`, `AgentCard`, `WorkflowCard`, `TeamCard`, `SysOperationCard`), `Provider<T> = Arc<dyn Fn() -> T + Send + Sync>`, `RegisterPolicy { Fail, Keep, Replace }`, `Registration { id, name, replaced }`, `Registry<C, T>` with `new`/`with_kind`/`register`/`get`/`card`/`id_by_name`/`get_by_name`/`remove`/`contains`/`cards`/`ids`/`len`/`is_empty`/`clear`/`revision`/`pub(crate) reset`.

### `tag.rs` — `TagMgr` and the `ResourceMgr` tag facade
```rust
pub const GLOBAL_TAG: &str = "__global__";
#[serde(transparent)] pub struct Tag(String);
impl Tag { pub fn new(v: impl Into<String>) -> Self; pub fn global() -> Self; pub fn is_global(&self) -> bool; pub fn as_str(&self) -> &str; /* Display, Deref<str>, From<&str>, From<String> */ }
#[serde(rename_all = "snake_case")] pub enum ResourceKind { Tool, Agent, Workflow, Team, McpServer }
impl ResourceKind { pub const ALL: [Self; 5]; pub fn as_str(&self) -> &'static str; }  // "mcp_server" for McpServer
#[serde(rename_all = "snake_case")] pub enum TagMatchStrategy { Any, All }
#[serde(rename_all = "snake_case")] pub enum TagUpdateStrategy { Merge, Replace }
impl TagMgr {
    pub fn new() -> Self;
    pub fn tag(&self, kind, id, tags) -> Result<Vec<Tag>>;
    pub fn update_tags(&self, kind, id, tags, strategy) -> Result<Vec<Tag>>;
    pub fn tags_of(&self, kind, id) -> Vec<Tag>;
    pub fn has_resource_tag(&self, kind, id, tag: &Tag) -> bool;
    pub fn has_tag(&self, tag: &Tag) -> bool;
    pub fn resources_with_tag(&self, kind, tag: &Tag) -> Vec<String>;
    pub fn resources_by_tags(&self, kind, tags: &[Tag], strategy) -> Result<Vec<String>>;  // ids sorted
    pub fn remove_tags(&self, kind, id, tags: &[Tag], skip_if_missing: bool) -> Result<Vec<Tag>>;
    pub fn untag(&self, kind, id) -> Vec<Tag>;
    pub fn remove_tag(&self, tag: &Tag) -> Vec<String>;
    pub fn remove_resource(&self, kind, id) -> Vec<Tag>;
    pub fn list_tags(&self) -> Vec<Tag>;        // non-empty tags, sorted
    pub fn resources(&self, kind) -> Vec<String>;
    pub fn len(&self) -> usize;                 // distinct live tags
    pub fn is_empty(&self) -> bool;
    pub fn revision(&self) -> u64;
    pub fn clear(&self) -> usize;
    pub(crate) fn reset(&self) -> usize;
}
```

### `model.rs` — `ModelMgr`
```rust
pub type ModelProvider = Provider<Arc<dyn Model>>;   // not foundation's `llm::registry::ModelProvider` trait
pub struct ModelCard { pub id: String, pub name: String }   // `name` defaults to the id; an empty name is skipped on the wire
impl ModelMgr {
    pub fn new() -> Self;
    pub fn register_model(&self, id: impl Into<String>, model: Arc<dyn Model>) -> Result<()>;
    pub fn register_model_with(&self, id, provider: ModelProvider, policy: RegisterPolicy) -> Result<Registration>;
    pub fn register_factory(&self, card: ModelCard, provider: ModelProvider, policy) -> Result<Registration>;
    pub fn model(&self, id: &str) -> Option<Arc<dyn Model>>;   // provider runs per lookup
    pub fn model_card(&self, id: &str) -> Option<ModelCard>;
    pub fn remove_model(&self, id: &str) -> Option<ModelCard>;
    pub fn model_ids(&self) -> Vec<String>;
    pub fn models(&self) -> Vec<ModelCard>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn revision(&self) -> u64;  pub fn clear(&self) -> usize;
    pub(crate) fn reset(&self) -> usize;
}
```

### `prompt.rs` — `PromptMgr`
```rust
pub struct PromptCard { pub id: String }     // ResourceCard: name() == id
impl PromptMgr {
    pub fn new() -> Self;
    pub fn add_prompt(&self, id: impl Into<String>, template: PromptTemplate) -> Result<()>;   // overwrites (RegisterPolicy::Replace)
    pub fn add_prompts(&self, prompts: impl IntoIterator<Item = (String, PromptTemplate)>) -> Result<()>;
    pub fn prompt(&self, id: &str) -> Option<PromptTemplate>;
    pub fn remove_prompt(&self, id: &str) -> Option<PromptTemplate>;
    pub fn prompt_ids(&self) -> Vec<String>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn revision(&self) -> u64;  pub fn clear(&self) -> usize;
    pub(crate) fn reset(&self) -> usize;
}
```

### `sysop.rs` — `SysOperationMgr`
```rust
pub type SysOperationProvider = Provider<Arc<SysOperation>>;
impl SysOperationMgr {
    pub fn new() -> Self;
    pub fn register(&self, operation: SysOperation) -> Result<()>;          // keys on `operation.card().id`
    pub fn register_with(&self, operation: SysOperation, policy: RegisterPolicy) -> Result<Registration>;
    pub fn get(&self, id: &str) -> Option<Arc<SysOperation>>;
    pub fn card(&self, id: &str) -> Option<SysOperationCard>;
    pub fn remove(&self, id: &str) -> Option<SysOperationCard>;            // releases its isolation-key template
    pub fn ids(&self) -> Vec<String>;
    pub fn cards(&self) -> Vec<SysOperationCard>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn revision(&self) -> u64;  pub fn clear(&self) -> usize;
    pub(crate) fn reset(&self) -> usize;
    pub fn tool_specs(&self, id: &str) -> Result<Vec<OperationToolSpec>>;   // 13 specs for a local op
}
```

### `mcp_server.rs` — `McpServerMgr` (async lifecycle)
```rust
#[serde(rename_all = "snake_case", tag = "state")]
pub enum McpServerState { Registered, Connected, Failed { reason: String }, Disconnected }
pub struct McpServerInfo { pub server_id: String, pub server_name: String, pub state: McpServerState }
pub trait McpConnector: Send + Sync { fn build(&self, config: &McpServerConfig) -> Result<Arc<dyn McpClient>>; }
pub struct DefaultMcpConnector;                                  // calls foundation's `connect_mcp_server`
impl McpServerMgr {
    pub fn new() -> Self;
    pub fn with_connector(connector: Arc<dyn McpConnector>) -> Self;
    pub fn register(&self, config: McpServerConfig) -> Result<()>;                       // never connects
    pub fn register_with(&self, config: McpServerConfig, policy: RegisterPolicy) -> Result<Registration>;
    pub fn config(&self, id: &str) -> Option<McpServerConfig>;
    pub fn state(&self, id: &str) -> Option<McpServerState>;
    pub fn client(&self, id: &str) -> Option<Arc<dyn McpClient>>;    // Some only while connected
    pub async fn connect(&self, id: &str) -> Result<()>;
    pub async fn load_tools(&self, id: &str) -> Result<Vec<McpTool>>;                     // connects if needed
    pub async fn register_tools(&self, mgr: &ResourceMgr, id: &str) -> Result<Vec<String>>;
    pub async fn disconnect(&self, id: &str) -> Result<()>;
    pub async fn remove(&self, id: &str) -> Result<McpServerConfig>;
    pub fn list(&self) -> Vec<McpServerInfo>;   // registration order
    pub fn ids(&self) -> Vec<String>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn revision(&self) -> u64;  pub fn clear(&self) -> usize;
    pub(crate) fn reset(&self) -> usize;
}
```

## Constraints

- Duplicate registration under `RegisterPolicy::Fail` is `Error::Validation("<kind> '<id>' is already registered")` and leaves the entry untouched; `Keep` reports `replaced: false`; `Replace` swaps card and provider and reports `replaced: true`.
- Ids come from the card, never from the caller: an empty or whitespace-only id is `Error::Validation` (`crate::error::invalid_id(kind)`), and nothing is stored.
- Lookups never panic and never invoke a provider for a missing id; `get*` returns `Option` and a missing id becomes the runner's `Error::NotFound` at the calling seam.
- A provider is taken out of the registry guard and invoked **after** the guard is dropped, so a provider may touch the registry without deadlocking.
- No `parking_lot` guard is ever held across `.await`; the only awaits in this directory are the MCP lifecycle methods, which copy what they need out of the guard first.
- Tag messages: empty id → `invalid_id("resource")`, empty/whitespace tag value → `Validation("tag cannot be empty or whitespace only")`, empty tag list → `Validation("at least one tag is required")`, a missing tag with `skip_if_missing = false` → `Validation("<kind> '<id>' does not have tag '<tag>'")`, and an untagged unknown resource on the remove path → `Error::NotFound`.
- Tags are one flat namespace and `ResourceKind` is only the index partition: the same `Tag` value can label an agent and a tool, and `list_tags`/`has_tag`/`remove_tag` span every kind.
- `GLOBAL_TAG` (`"__global__"`) is exclusive: tagging a resource with it drops that resource's other tags, and adding further tags to an already-global resource adds nothing.
- Tags are garbage-collected: a tag whose last carrier loses it disappears from `list_tags`, `has_tag`, and `len`.
- `ResourceMgr`'s tag facade rejects an id that is not registered for that `ResourceKind` up front (`Error::NotFound`); reads (`tags_of`, `untag`, `remove_tag`) stay infallible so removal races registry cleanup without erroring.
- `ResourceMgr::tag` / `update_tags` validate against the owning registry (or, for `McpServer`, `McpServerMgr::config`); `resources_by_tags` has no id parameter, so it validates only tag values.
- Removing a tool/agent/workflow/team drops that resource's tags, and clearing a kind drops the tags of every id it removed; a failed removal (missing id) leaves the tag index untouched.
- `is_empty()` keeps the historical meaning (the four run-able registries are empty) so existing callers and `../tests/resources.rs` are unaffected; `is_pristine()` is the strict form (all nine managers empty, including the tag index).
- `release()` resets the four registries and all five sub-managers, zeroing every revision counter.
- Every manager keeps its own revision counter, so a host caching one projection is invalidated only by changes to that projection.
- MCP: `register` never connects (state `Registered`), and a `build` or `McpClient::connect` failure returns the error **and** records `Failed { reason }` with `client(id) == None` — success is never reported without a live client.
- MCP: state and client live behind one guard that is never held across `.await`; a per-server `tokio::sync::Mutex` (from a `parking_lot`-guarded map) serializes `connect`/`disconnect`/`remove` per server id, and the guard is acquired *before* the state re-check so two racing `connect` calls open exactly one session.
- MCP: dropping a `connect` future leaves no stale state — the state is written only once the awaited result is known.
- MCP: `disconnect` on a `Registered`/`Failed`/`Disconnected` server is `Ok(())`; a failing client `disconnect` still drops the client and marks `Disconnected`; `remove` logs the failure with `tracing::warn!` and unregisters regardless (Python parity).
- MCP: `register_tools` loads the remote tools and publishes each into the tool registry with `RegisterPolicy::Keep`, returning the tool card ids (the ids come from `McpTool`'s card, not from this manager).
- No secrets in logs or `Debug`: never print `McpServerConfig::auth_headers`/`auth_query_params`.
- Sys operations: two *different* ids may not share a non-empty `isolation_key_template` (`Error::Validation` naming the template and the owning id); re-registering the same id keeps its own template, and removal releases it.
- Sys operations: `tool_specs` exposes the specs; projecting them into `openjiuwen_foundation::tool::Tool` instances is a caller concern, not this manager's.
- Prompts overwrite silently (`RegisterPolicy::Replace`) because Python's `add_prompt` assigns into a dict; `add_prompts` stops at the first invalid id and keeps the entries added before it.
- Files stay ≲1000 lines; `tag/tests.rs` and `mcp_server/tests.rs` are legitimately long (one comprehensive suite per manager, kept out of the module file).
- Tests are offline and deterministic: no network, no sleeps used for ordering, loopback only, bounded awaits.

## Design Decisions

- **Nine managers on one `ResourceMgr`.** Python keeps tags, models, prompts, sys-ops, and MCP servers inside the same `ResourceMgr` class; the Rust port keeps one owner object so a host has a single lifecycle (`release`) and one `Arc` to share.
- **Accessors returning `&Manager`, not `Arc`.** Every sub-manager is interior-mutable, so a shared reference suffices and a manager cannot be dropped from under a borrow.
- **Sub-managers keep their own error nouns.** Tool/agent/workflow/team errors read `"<kind> '<id>' …"`, models `"model"`, prompts `"prompt"`, sys operations `"sys_operation"`, MCP servers `"mcp server"`.
- **The tag facade lives in `tag.rs`.** The `impl ResourceMgr` block for tags sits with `TagMgr` and validates through the public card lookups, so `mod.rs` stays near the size guideline and no registry internals leak.
- **`Tag` is a newtype with `#[serde(transparent)]`.** Python's `Tag = str`; the newtype serializes as a bare JSON string while giving one place for the emptiness check and the GLOBAL constant.
- **`ResourceKind` partitions the tag index; it is not part of the tag value.** Two resources of different kinds can share a tag, and `GLOBAL_TAG` behaves identically in every partition.
- **Models and prompts get card types (`ModelCard`, `PromptCard`) even though Python keys them on bare strings.** `Registry` keys on a card, so the card carries the identity and the revision/policy semantics come for free; `PromptCard::name()` returns the id and `ModelCard` names itself after the id unless told otherwise.
- **MCP lifecycle is a small state machine guarded per server.** `Registered → Connected → Disconnected`, with `Failed { reason }` as the recorded error state, and a per-server async mutex instead of one global lock so unrelated servers connect in parallel.
- **`McpConnector` exists because foundation's fake client is `#[cfg(test)]`-only.** The seam lets this crate test the whole lifecycle (success, build failure, connect failure, racing connects) with a local fake, while `DefaultMcpConnector` ships the real `connect_mcp_server` path.
- **Sys-op isolation-key ownership is tracked in a manager-level `Mutex`.** Check-and-register happen in one critical section (index lock → registry lock ordering) so two concurrent registrations cannot both claim a template.
- **`is_empty` and `is_pristine` are both kept.** `is_empty` means "no run-able resources", which is what `../tests/resources.rs` relies on; `is_pristine` is the stronger "nothing registered at all".
- **Tag deletion is part of resource removal, not a caller chore.** `remove_*`/`clear_*` call `TagMgr::remove_resource`, so a host cannot forget to untag a removed resource.

## Deferred Python Behaviour

- Python's `TagMgr.display()` log dump is not ported: `Debug` prints per-kind counts, the live tag count, and the revision, which is the diagnostic content a host needs without an unbounded dump.
- Python's `Tag.ALL` (`"*"`), `Tag.ACTIVE`, and `Tag.INACTIVE` constants carry no behaviour in the Python manager either; they are omitted rather than added as inert `const`s.
- Tagging is limited to the five `ResourceKind` variants (tools, agents, workflows, teams, MCP servers); models, prompts, and sys operations are addressed by id through their own managers.
- MCP servers keep no `last_update_time`/`expiry_time` and no cached tool-id list: Python's `McpServerResource` tracks both for periodic refresh, while here `load_tools`/`register_tools` re-read the server on demand and the tool registry holds the result.
- MCP lookup and tag filtering by `server_name` (Python's `remove_mcp_server`/`get_mcp_tool_ids` accept a name) is not ported: this manager keys on `server_id`, and `McpServerInfo` carries the name for display.
- Prompt/template loading from a filesystem or config directory is not ported (Python loads `.txt`/`.yaml` prompt packs); hosts call `PromptMgr::add_prompt`/`add_prompts` with a `PromptTemplate`.
- Model tracing decoration (Python's `decorate_model_with_trace` wraps a `get_model` result in a traced proxy) is not ported: the runner does not own tracing decoration, and hosts wrap a model before registering it.
- Python's `get_model(model_id, session)` session parameter is not ported: model lookup is session-independent here.
- Sys-operation tool binding (Python's `SysOperationToolAdapter`) is not ported into this directory; `tool_specs` exposes the 13 specs so a caller can build `Tool`s.

## Routing Table

- `./mod.rs` → `ResourceMgr`: four registries, five accessors, tag facade wiring, cleanup hooks, `is_empty`/`is_pristine`/`release`, manual `Debug`, flat re-exports
- `./registry.rs` → `Registry`, `Provider`, `RegisterPolicy`, `Registration`, `ResourceCard` (five impls)
- `./tag.rs` → `Tag`, `ResourceKind`, `TagMatchStrategy`, `TagUpdateStrategy`, `GLOBAL_TAG`, `TagMgr`, and the `impl ResourceMgr` tag facade
- `./model.rs` → `ModelCard`, `ModelProvider`, `ModelMgr`
- `./prompt.rs` → `PromptCard`, `PromptMgr`
- `./sysop.rs` → `SysOperationProvider`, `SysOperationMgr` (isolation-key ownership)
- `./mcp_server.rs` → `McpServerState`, `McpServerInfo`, `McpConnector`, `DefaultMcpConnector`, `McpServerMgr`
- `./tag/tests.rs` → tag manager and tag-facade suite (included from `tag.rs` via `#[path]`)
- `./mcp_server/tests.rs` → MCP lifecycle suite with a local fake client/connector
- `../tests/resources.rs` → registry integration suite (sibling — read-only, escalate writes to the crate root)
- `../tests/api_surface.rs` → crate-level API surface assertions (sibling — read-only)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)] mod tests`, `#[allow(clippy::unwrap_used, clippy::expect_used)]`), with `tag/tests.rs` and `mcp_server/tests.rs` included via `#[path]` to keep the module files small.
- Coverage: registry insert/lookup/policy/revision/debug; tag add/update/ANY/ALL/GLOBAL-exclusivity/garbage-collection/validation/NotFound; model and prompt register/get/remove/overwrite/validation/revision with a counting factory proving the provider runs per lookup; sys-op register/get/remove/tool-specs/isolation-key conflicts; MCP register/list/connect/failure/load-tools/disconnect/remove/racing connects/revision.
- Wiring tests in `mod.rs`: tag cleanup on `remove_*` and `clear_*`, `release()` resetting all nine managers, `is_pristine` vs `is_empty`, and the manual `Debug` output.
- Everything is offline and deterministic: `ScriptedModel` for model registrations, `McpServerConfig::stdio("srv", "cmd")` for servers (registration never connects), `SysOperation::new(SysOperationCard::new(id))` for local system operations, and a `tokio::sync`-based fake for the MCP lifecycle — no sleeps used for ordering, no network.
- Gates: `cargo fmt -p openjiuwen-runner`, `cargo test -p openjiuwen-runner --offline`, `cargo clippy -p openjiuwen-runner --all-targets --offline -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc -p openjiuwen-runner --no-deps`.

## Notes for Agents

- Use `ResourceMgr::new()` in tests; never a process global.
- Reach the sub-managers through the accessors (`mgr.tags()`, `mgr.models()`, …) or the module paths; the flat re-exports exist for hosts that prefer one `use`.
- Prefer the `ResourceMgr` tag facade when the resource lives in a registry: it rejects an unknown id instead of creating an orphan tag entry.
- `TagMgr` itself never checks registries — an id it has not seen is simply untagged, which is what makes `tags_of` infallible.
- `McpServerMgr` never connects during registration; call `connect` (or `load_tools`/`register_tools`, which connect implicitly) and keep the per-server lock discipline if you extend the lifecycle.
- `ModelMgr::model` and `PromptMgr::prompt` run the provider per lookup; register a factory when the caller needs fresh per-run state, or `register_model` to hand back the same `Arc`.
- `is_empty()` is not "nothing registered"; use `is_pristine()` when you mean that.
- `registry.rs` is the only place the five `ResourceCard` impls live; a new card-bearing manager belongs there, not in the manager module.
