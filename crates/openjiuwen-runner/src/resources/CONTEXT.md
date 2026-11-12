# resources/ — the four typed registries

## Intent

The runner's registry of the resources a host publishes: tools, agents, workflows, and agent teams.
Ports Python `openjiuwen/core/runner/resources_manager/resource_manager.py` (2127 lines) and its typed managers, reduced to the four public resource kinds and the semantics hosts actually depend on: ids, duplicate policy, card-vs-name lookup, and a revision counter for cache invalidation.
Deliberately **not** ported (see the crate-level `## Deferred Python Behavior`): tags (`TagMgr`), MCP servers/tools, sys-operations, models, and prompts.

## API Surface

```rust
// resources/registry.rs
pub trait ResourceCard: Clone + Send + Sync + 'static {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
}
impl ResourceCard for ToolCard / AgentCard / WorkflowCard / TeamCard

pub type Provider<T> = Arc<dyn Fn() -> T + Send + Sync>;
pub type ToolProvider = Provider<Arc<dyn Tool>>;
pub type AgentProvider = Provider<Arc<dyn Agent>>;
pub type WorkflowProvider = Provider<Arc<Workflow>>;
pub type TeamProvider = Provider<Arc<dyn Team>>;

pub enum RegisterPolicy { Fail, Keep, Replace }   // Python default / skip_if_exists=True / refresh=True
pub struct Registration { pub id: String, pub name: String, pub replaced: bool }

pub struct Registry<C: ResourceCard, T> { .. }     // the generic machinery
impl<C: ResourceCard, T> Registry<C, T> {
    pub fn new() -> Self;
    pub fn register(&self, card: C, provider: Provider<T>, policy: RegisterPolicy) -> Result<Registration>;
    pub fn get(&self, id: &str) -> Option<T>;              // provider() invoked per call
    pub fn card(&self, id: &str) -> Option<C>;
    pub fn id_by_name(&self, name: &str) -> Option<String>;
    pub fn get_by_name(&self, name: &str) -> Option<T>;
    pub fn remove(&self, id: &str) -> Option<C>;
    pub fn contains(&self, id: &str) -> bool;
    pub fn cards(&self) -> Vec<C>;                          // registration order
    pub fn ids(&self) -> Vec<String>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn clear(&self) -> usize;
    pub fn revision(&self) -> u64;                          // +1 on every mutation
}

// resources/mod.rs
pub struct ResourceMgr { /* four registries */ }
impl ResourceMgr {
    pub fn new() -> Arc<Self>;
    // tools
    pub fn register_tool(&self, tool: Arc<dyn Tool>) -> Result<()>;
    pub fn register_tool_with(&self, tool: Arc<dyn Tool>, policy: RegisterPolicy) -> Result<Registration>;
    pub fn tool(&self, id: &str) -> Option<Arc<dyn Tool>>;
    pub fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn tool_card(&self, id: &str) -> Option<ToolCard>;
    pub fn remove_tool(&self, id: &str) -> Option<ToolCard>;
    pub fn tools(&self) -> Vec<ToolCard>;
    pub fn tool_ids(&self) -> Vec<String>;
    pub fn tool_revision(&self) -> u64;
    pub fn clear_tools(&self) -> usize;
    // agents / workflows / teams: the same shape with register_agent(card, provider) etc.
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> Result<()>;
    pub fn register_agent_with(&self, card: AgentCard, provider: AgentProvider, policy: RegisterPolicy) -> Result<Registration>;
    pub fn agent(&self, id: &str) -> Option<Arc<dyn Agent>>;
    pub fn agent_by_name(&self, name: &str) -> Option<Arc<dyn Agent>>;
    pub fn agent_card(&self, id: &str) -> Option<AgentCard>;
    pub fn remove_agent(&self, id: &str) -> Option<AgentCard>;
    pub fn agents(&self) -> Vec<AgentCard>;
    pub fn agent_ids(&self) -> Vec<String>;
    pub fn agent_revision(&self) -> u64;
    pub fn clear_agents(&self) -> usize;
    // workflows: register_workflow(card, provider) / workflow(id) / workflow_card / remove_workflow / workflows / workflow_ids / workflow_revision / clear_workflows
    // teams:     register_team(card, provider)     / team(id)     / team_card     / remove_team     / teams     / team_ids     / team_revision     / clear_teams
    pub fn tool_count(&self) -> usize;  pub fn agent_count(&self) -> usize;
    pub fn workflow_count(&self) -> usize;  pub fn team_count(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn release(&self);                                  // clear all four (Python ResourceMgr.release)
}
pub fn workflow_key(id: &str, version: &str) -> String;     // "{id}_{version}" (Python generate_workflow_key)
```
`ResourceMgr` is cheap to clone? No: it is shared as `Arc<ResourceMgr>` (the runner owns it). Interior mutability is `parking_lot::RwLock<Inner>` on the registry, guarded only for synchronous sections.

## Constraints

- Duplicate registration under `RegisterPolicy::Fail` is `Error::Validation("resource '<id>' is already registered")`; the existing entry is untouched (Python keeps the first and returns an error).
- `Keep` returns `Ok(Registration { replaced: false })` and leaves the original; `Replace` swaps the provider and card and returns `replaced: true` (a fresh id is `replaced: false`).
- Ids come from the card (`card.id()`), never from the provider or caller; an empty or whitespace-only id is `Error::Validation`.
- Lookups never invoke a provider for a missing id and never panic; `get*` returns `Option`, resolution failures are the runner's `Error::NotFound`.
- `by_name` is a scan over registered cards (names are not unique); the first registration wins. Callers that need uniqueness use ids.
- `release()` clears every registry (Python parity: `Runner::stop` calls it) and resets revision counters to zero.
- No provider call happens while the registry lock is held: take the provider out under the lock, drop the guard, then call it.
- Tests: unit tests beside the module plus `../tests/resources.rs`; no global state, each test builds its own `ResourceMgr::new()`.

## Routing Table

- `./src/resources/registry.rs` → `Registry`, `Provider`, `RegisterPolicy`, `Registration`, `ResourceCard`
- `./src/resources/mod.rs` → `ResourceMgr`, the four typed facades, `workflow_key`, re-exports
- `../tests/resources.rs` → registry behavior suite (sibling — read-only, escalate writes to the crate root)
