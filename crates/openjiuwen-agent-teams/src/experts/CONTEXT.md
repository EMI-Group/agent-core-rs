# experts — expert catalog, AgentGroup packages, template → member assembly

## Intent

The expert domain of the team runtime: how a host's installed expert packages are located, how an AgentGroup package is parsed, and how an AgentTemplate becomes a team member.
Python parity source: `openjiuwen/agent_teams/experts/` (`catalog.py`, `agent_group.py`, `template_member.py`, `bound_group.py`).
Three concerns stay separate: **locating** (host knowledge, `catalog.rs`), **package parsing** (`agent_group.rs`), **assembly into a member** (`template.rs`).
Leaf node: no subdirectories.

## API Surface

```rust
// catalog.rs
pub trait ExpertCatalog: Send + Sync {                      // #[async_trait]
    async fn list_expert_packages(&self) -> Result<Vec<PackageLocation>>;
    async fn locate_expert(&self, name: &str) -> Result<Option<PackageLocation>>;
    async fn list_group_packages(&self) -> Result<Vec<PackageLocation>>;
    async fn locate_group(&self, name: &str) -> Result<Option<PackageLocation>>;
}
pub struct ExpertRef { pub name: String, pub group: Option<String> }
impl ExpertRef { pub fn new(name) -> Self; pub fn with_group(self, group) -> Self;
                 pub fn label(&self) -> String;            // "group/name" | "name"
                 pub fn to_value(&self) -> Value; }        // {"name", "group"}
pub struct PackageLocation { pub name: String, pub path: PathBuf, pub source: String }  // path ↔ string
pub fn set_process_expert_catalog(catalog: Arc<dyn ExpertCatalog>);
pub fn get_process_expert_catalog() -> Option<Arc<dyn ExpertCatalog>>;
pub fn clear_process_expert_catalog();

// agent_group.rs
pub const LEADER_MEMBER: &str = "leader";
pub enum LocalizedText { Plain(String), Localized(BTreeMap<String, String>) }  // untagged serde
impl LocalizedText { pub fn resolve(&self, language: &str) -> &str;
                     pub fn resolve_optional(&self, language: Option<&str>) -> &str;
                     pub fn is_empty(&self) -> bool; }
pub struct AgentGroupCard { pub name: String, pub description: LocalizedText,
    pub display_name: Option<LocalizedText>, pub category: Option<String>,
    pub tags: Vec<String>, pub version: Option<String>, pub instruction: Option<String> }
pub struct AgentTemplateCard { pub name: String, pub description: LocalizedText,
    pub display_name: Option<LocalizedText>, pub display_description: Option<LocalizedText>,
    pub category: Option<String>, pub tags: Vec<String>, pub version: Option<String>,
    pub avatar: Option<String>, pub runtime: String /* "native"|"external" */,
    pub runtime_provider: Option<String>, pub runtime_config: Option<Value>,
    pub skills: Vec<String>, pub connectors: Vec<String> }
pub struct AgentGroupMemberCard { pub name: String, pub group: String, pub card: AgentTemplateCard }
pub struct LoadedAgentGroupPackage { pub package_dir: PathBuf, pub manifest: AgentGroupCard,
                                     pub templates: BTreeMap<String, AgentTemplateCard> }
impl LoadedAgentGroupPackage {
    pub fn instruction(&self) -> &str; pub fn capabilities(&self) -> &[String];
    pub fn template(&self, name: &str) -> Option<&AgentTemplateCard>;
    pub fn member_cards(&self) -> Vec<AgentGroupMemberCard>;
    pub fn load_member_persona(&self, member: &str) -> Result<String>;
}
pub fn load_agent_group(dir: impl AsRef<Path>) -> Result<LoadedAgentGroupPackage>;

// template.rs
pub struct BoundAgentGroup { pub name: String, pub members: Vec<BoundGroupMember>,
                             pub skipped: Vec<SkippedMember> }
impl BoundAgentGroup { pub fn new(name) -> Self;
                       pub fn member(&self, name: &str) -> Option<&BoundGroupMember>;
                       pub fn member_names(&self) -> Vec<&str>;
                       pub fn is_complete(&self) -> bool;          // no non-leader skip
                       pub fn skipped_names(&self) -> Vec<&str>;
                       pub fn skipped_reason(&self, name: &str) -> Option<&str>; }
pub struct BoundGroupMember { pub name: String, pub display_name: String, pub desc: String,
                              pub persona: String, pub external_cli: bool }
pub struct SkippedMember { pub name: String, pub reason: String, pub kind: SkipReason }
pub enum SkipReason { Leader, EmptyPersona, PersonaUnreadable }  // snake_case wire names
impl SkipReason { pub fn as_str(self) -> &'static str; }         // leader|empty_persona|persona_unreadable
pub fn bind_group(package: &LoadedAgentGroupPackage) -> BoundAgentGroup;
pub fn compose_template_member(base: &DeepAgentSpec, template: &AgentTemplateCard,
                               persona: &str, language: Option<&str>) -> DeepAgentSpec;
pub fn render_persona(persona: &str, language: &str, workspace: &str) -> String;
```

`ExpertCatalog` is location-only: it answers which packages exist and where (source `builtin|local|hub`) and never opens a manifest.
The catalog is process-global because leaders are assembled while the spec is built, before any runtime-manager hook can reach them, and because `TeamAgentSpec` must stay serializable.
This static and the process-wide team registry are the crate's two documented singletons; tests must call `clear_process_expert_catalog()`.

## AgentGroup package format

```text
<group>/
  manifest.json               package_type="agent_group", name, agents, instruction, capabilities|tags, skills
  skills/<skill>/SKILL.md     shared skills (declared and auto-discovered)
  agents/<member>/
    manifest.json             package_type="agent_template", persona.dir, skills, connectors, runtime, ...
    AGENT.md                  leader only, optional
    <persona.dir>/**/*.md     persona sections, concatenated in resolved-path order
```

`manifest.json` rules enforced by `load_agent_group`:
- `package_type` must be `"agent_group"`; `name` must be a string equal to the package directory name.
- `agents` must be a non-empty string list; every name matches `^[a-z][a-z0-9_-]*$` (hand-written check, no regex), is unique, and `"leader"` is present.
- `instruction` is optional (trimmed; empty becomes `None`), `skills` is an optional list of names or `{"dir": …}` mappings, and `version`/`category` accept a string or localized object.
- `capabilities` (fallback `tags`) becomes `AgentGroupCard::tags`: entries are strings or `{id|zh|en}` objects, deduplicated in order.
- `description`/`display_name` are kept as `LocalizedText`; `display_description` and `avatar` are accepted by Python's card but are not retained by the Rust card (display-only, see Deferred).

Member manifests:
- `package_type` must be `"agent_template"`; `persona.dir` is required, package-relative, must exist, and must contain at least one `.md` file (recursive; sorted by resolved path).
- Persona text is those files joined with a blank line, front matter preserved; no prompt-section or placeholder rendering happens at load time.
- `name` defaults to the directory (member) name; the member's identity is always the directory name.
- `AGENT.md`: optional for `leader` (validated as a file inside its directory); any `AGENT.md` in a non-leader directory is a validation error.
- `skills` entries (name strings or `{dir|name}` objects) are reduced to basenames and merged with the group's shared skills, declared first then discovered, deduplicated.
- `connectors` entries are strings or `{name|id}` objects reduced to names.
- `runtime` absent/`null`/`"native"` is native; an object with a non-empty `provider_name` (or the string `"external"`) marks an external member and records `runtime_provider` plus `runtime_config` (`runtime.config`). A leader may not declare an external runtime.
- `tools|mcps|rails|model|subagents|metadata` are accepted and not interpreted here: there is no `AgentTemplateSpec` in the Rust port; the member runtime reads what it needs from the package.

Shared skills:
- Declared names resolve to `<group>/skills/<name>/SKILL.md`; `{"dir": …}` mappings resolve relative to the package.
- Every declared skill must exist inside the package and contain `SKILL.md`; a duplicate resolved path is a validation error.
- Children of `<group>/skills/` that contain `SKILL.md` are auto-discovered in file-name order after the declared skills.
- The loader never reads skill content; only `SKILL.md`'s existence is checked.

`LocalizedText::resolve` aliases `zh` (and `zh-CN`/`zh_CN`) to `cn`, treats a `zh` map key as a `cn` synonym, and falls back to `cn` when the requested language is absent.

## Design Decisions

`compose_template_member` is the single template → member implementation. Python stores the capability snapshot in `DeepAgentSpec.agent_template_spec`; the Rust `DeepAgentSpec` has no such field, so the contract is **idempotence**: composing the same base, template and persona twice yields the same spec.
- `system_prompt` = rendered persona + one blank line + any existing prompt; an existing prompt that already starts with the rendered persona is left untouched, which makes composition idempotent.
- `skills` = base skill basenames ∪ template skill basenames, deduplicated and sorted (`BTreeSet`); an empty union preserves the base value (`None` stays `None`).
- `language` = `language` argument, else `base.language`, else `crate::constants::DEFAULT_LANGUAGE`.
- every other `DeepAgentSpec` field is preserved.
`render_persona` substitutes `{{language}}`, `{{workspace}}` (the base workspace root, `""` when absent) and `{{persona}}` in one non-recursive pass; `{{persona}}` expands once to the unrendered text.
`bind_group` returns the spawnable (non-leader) members, ordered by member name; it re-reads each persona from the package and skips members whose persona is empty or no longer readable, so one broken member does not hide the rest.
Every skip is observable, never silent: `bind_group` logs it with the package and member identity (`debug!` for the leader, `warn!` for a persona failure) and appends a `SkippedMember { name, reason, kind }` to `BoundAgentGroup::skipped`.
The leader is the one expected filter (`SkipReason::Leader`, so `is_complete()` stays `true` for a healthy package); `EmptyPersona` and `PersonaUnreadable` mark a package the caller's input no longer fully backs, and `reason` always names the member and the failing condition.
`skipped` keeps `bind_group` infallible and its signature unchanged: a broken member degrades to a recorded skip instead of a typed error, and the caller decides whether to abort.
`agent_group.rs`'s skill discovery logs the one remaining silent filter at `debug!`: a directory under `<group>/skills/` without `SKILL.md` is not mounted (Python's behaviour), while non-Markdown persona files are excluded by the package format itself.
A member with `runtime = "external"` is flagged `external_cli = true`; its launch configuration stays on the template card.

## Constraints

- Only `src/experts/**` is edited; `src/lib.rs`, `Cargo.toml`, `src/schema/` and sibling modules belong to other nodes.
- `openjiuwen_common::error::{Error, Result}` only; malformed JSON is a `Error::validation` naming the offending path, never a panic or `unwrap`.
- No `unsafe`; `tracing` only; no `println!`/`eprintln!`.
- Path safety: components reject `""`, `.`, `..`, `/`, `\`, absolute paths and `X:`/`\\` prefixes; every resolved file/directory must stay inside its package (symlink-aware `canonicalize` + `starts_with`).
- Loader is synchronous `std::fs`; async callers may call it directly or hop to a blocking thread. There is no `aload_*` twin (see Deferred).
- Files stay well under 1000 lines; test-only fixtures live in `test_support.rs`, compiled only under `cfg(test)`.

## Deferred

- Async loader wrappers (`aload_agent_group_package`, `read_agent_group_card`) — callers own threading; the loader is sync and cheap.
- Card-only browsing (`load_agent_group_card`): the Rust loader always validates personas, so there is no cheap display-only path yet.
- `AgentTemplateSpec` materialization, prompt sections, skill mounting, `agent_group_instruction` injection, leader `AGENT.md` mounting: the member runtime owns these; the package loader exposes cards and persona text only.
- `display_description` / `avatar` on the group card, and localized tag objects as structured values (kept as resolved `String`s).
- Hub install/builtin package discovery: `ExpertCatalog` implementations are host-owned.
- YAML manifests, remote workspaces and the expert-package launch of a vendor CLI agent (`ExternalCliAgentSpec`): the package loader exposes cards and persona text only, and vendor CLI harness providers are deferred crate-wide.

## Routing Table

- `./catalog.rs` → `ExpertCatalog`, `ExpertRef`, `PackageLocation`, process-wide catalog accessors
- `./agent_group.rs` → `LocalizedText`, group/template/member cards, `LoadedAgentGroupPackage`, `load_agent_group`, package validation and path safety
- `./template.rs` → `compose_template_member`, `render_persona`, `bind_group`, `BoundAgentGroup`, `BoundGroupMember`
- `./test_support.rs` → test-only package fixtures shared by the unit tests
- `./mod.rs` → module docs and flat re-exports

## Test Strategy

Unit tests live in `#[cfg(test)] mod tests` beside each definition (25 tests), using `tempfile::tempdir()` and the shared fixtures in `test_support.rs`:
`agent_group.rs` exceeds the 1000-line baseline by design — the package format and its fixture-driven tests are one cohesive unit, and the file carries a top-of-file note saying so.
- happy path: leader + two members, localized group/member text, instruction trimming, roster order independence, persona loading;
- name/directory mismatch, empty roster, missing leader, invalid/kebab/snake roster names, wrong `package_type`, missing `persona.dir`;
- non-leader `AGENT.md` rejection, optional leader `AGENT.md`;
- persona concatenation order across nested and non-Markdown files;
- `LocalizedText` string/object deserialization, `zh` alias, `cn` fallback, `zh` map key;
- shared-skill declaration + auto-discovery merge and missing-`SKILL.md` failure;
- external runtime recording and leader rejection;
- `bind_group` leader exclusion, display/desc resolution, external flag, persona content, the leader `SkippedMember` (`kind`/`as_str`/`reason` naming the member), a member skipped for an empty persona and one skipped for an unreadable persona (`is_complete()` false, `skipped_reason` populated), and `SkippedMember`/`BoundAgentGroup` serde round-trip plus default-fill for a payload written before `skipped` existed;
- `compose_template_member` language precedence, placeholder rendering, skill union/dedup/sort, prompt prepending, idempotence, field preservation;
- catalog set/get/clear, fake catalog listing/locating, `ExpertRef::label`/`to_value`, `PackageLocation` path-as-string round-trip.
Definition of done: `cargo test -p openjiuwen-agent-teams --lib`, `cargo test -p openjiuwen-agent-teams --no-default-features --lib`, `cargo fmt -p openjiuwen-agent-teams -- --check`, and clippy clean for `src/experts/` (the empty sibling placeholder modules currently trip `missing_docs`, which belongs to the crate root node).

## Notes for Agents

- Python's `group_capabilities` and its card `tags` are merged into `AgentGroupCard::tags` here; use `capabilities()` for the Python `LoadedAgentGroupPackage.capabilities` behaviour.
- Python requires a leader `AGENT.md`; the Rust loader makes it optional (packages authored without leader rules stay loadable).
- Python preserves skill order and mounts prompt sections; the Rust port sorts the composed skill list and leaves persona text flat.
- `BoundAgentGroup::skipped` and its `SkippedMember`/`SkipReason` types are reachable as `openjiuwen_agent_teams::experts::…`; the crate root (`src/lib.rs`) still lists `experts` exports item by item and does not re-export them (that file belongs to another node).
- Binding never reports a broken member as an error: `bind_group` stays infallible, and a caller that needs strictness fails on `!bound.is_complete()` (or on any `SkipReason` other than `Leader`).
- `LoadedAgentGroupPackage::templates` is a `BTreeMap`, so member iteration is alphabetical, not roster order; `bind_group` inherits that order for both `members` and `skipped`.
- Localized maps in published Python packages often use `zh`, not `cn`; `resolve` accepts both.
- `ExpertCatalog::locate_*` returns `Ok(None)` for a missing package; Python raises `ValueError`. Callers that need an error map it to `Error::not_found`.
- Never hold the process-catalog read lock across `.await`: `get_process_expert_catalog` clones the `Arc` and releases it before returning.
