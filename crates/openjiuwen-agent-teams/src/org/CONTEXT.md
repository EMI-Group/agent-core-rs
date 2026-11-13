# src/org — organisation layer, swarmflow and fork/compact

## Intent

The cross-team organisation layer of `openjiuwen-agent-teams`, plus the two context mechanisms a team needs to grow: swarmflow lanes and fork/compact.
It sits *above* [`crate::runtime`]: member spawning, permissions, pause/resume and dispatch stay inside one team's runtime; this layer owns who reports to whom, whether a delegation is legal, where a cross-team task lives, how leaders exchange mail, how a plan fans out, and how a member's context is forked or compacted.
Scope is in-process and deterministic: no LLM call, no network, no wall-clock ordering, no process-global state.
The durable half (task pool, mailbox) writes through the existing [`crate::storage::TeamStore`] seam under a reserved namespace instead of introducing a second database.
Python parity sources: `openjiuwen/agent_teams/organization/` (`schema.py`, `task_pool.py`, `message_service.py`, `manager.py`, `events.py`), `fork.py`, `fork_compact.py` and `workflow/` (`swarmflow_infra.py`, `schema.py`, `runner.py`, `tool_swarmflow.py`).

## API Surface

```rust
// mod.rs — the namespace and addressing primitives every submodule shares
pub const MAX_ORG_ID_LEN: usize = 96;
pub const ORG_ID_ALLOWED_CHARS: &str = "A-Za-z0-9_.-";
pub const ORG_NAMESPACE_PREFIX: &str = "org::";
pub const ORG_SYSTEM_SENDER: &str = "__organization__";
pub const ORG_MESSAGE_PROTOCOL: &str = "org";
pub const MEMBER_QUALIFIER: &str = "::";
pub fn is_valid_org_id(id: &str) -> bool;
pub fn validate_org_id(id: &str) -> Result<()>;
pub fn org_namespace(organization_id: &str) -> String;          // "org::<id>"
pub fn is_org_namespace(team_name: &str) -> bool;
pub fn org_namespace_id(team_name: &str) -> Option<&str>;
pub fn qualify_member(team: &str, member: &str) -> String;      // "<team>::<member>"
pub fn split_qualified(value: &str) -> Option<(&str, &str)>;

// roles.rs — the chart
pub enum OrgRole { Member, Leader, Owner, Summary }             // default Member; as_str/can_lead/can_summarize
pub struct OrgTeam { team_name, display_name, desc, role, leader: Option<String>, reports_to: Option<String>, metadata }
pub struct OrgMember { member_id, team_name, display_name, desc, role, reports_to, metadata }   // node_id()
pub struct OrganizationSpec { organization_id, display_name, desc, owner, teams, members, summary_team, metadata }
impl OrganizationSpec { pub fn new(id) -> Self; pub fn validate(&self) -> Result<()>; team(..); member(team, member); }
pub enum OrgNodeRef<'a> { Team(&'a OrgTeam), Member(&'a OrgMember) }
pub struct OrgChart;            // Clone, Send + Sync; built once from a validated spec
impl OrgChart { new, spec, organization_id, node, team, member, teams, members, team_members, role_of,
                parent_of, direct_reports, descendants, ancestors, depth_of, is_ancestor, leader_of,
                team_of_node, to_value }

// delegation.rs — the rule engine
pub struct RolePair { from: OrgRole, to: OrgRole }
pub struct DelegationRules { max_delegation_depth, require_reporting_line, allowed_pairs,
                             team_task_budget, default_task_budget }     // new + builders + allows_pair + budget_for
pub struct DelegationRequest { from_team, from_member, to_team, to_member, depth, open_tasks_in_target }
pub enum DelegationRefusal { UnknownDelegator, UnknownTargetTeam, UnknownTargetMember, SelfDelegation,
                             ReportingLineRequired, DepthExceeded{..}, RoleNotAllowed{..}, BudgetExhausted{..} }
pub enum DelegationDecision { Allowed { depth, cross_team }, Refused { refusal } }
pub struct DelegationRouter;    // Arc<OrgChart> + DelegationRules
impl DelegationRouter { new, chart, rules, decide, can_delegate, check }   // check -> Err(Error::Validation(message))

// task_pool.rs (+ task_pool_types.rs) — the cross-team task pool
pub struct OrgTaskPool;         // Arc<dyn TeamStore> + Arc<DelegationRouter> + namespace
impl OrgTaskPool { new(store, router), with_event_sink(OrgTaskEventSink), organization_id, namespace, store, router,
                   create, delegate, claim, start, complete, fail, review, vote, settle_review,
                   get, list, children, open_task_count }
// task_pool_types.rs: OrgTaskFailure, OrgTaskSidecar, OrgTaskSpec, OrgTask, OrgTaskFilter,
//                     OrgTaskOpResult, OrgTaskEvent, OrgTaskMutation, OrgTaskEventSink

// mailbox.rs — the leader mailbox
pub enum OrgMessageKind { Leader, System, Broadcast }
pub struct OrgMessage { message_id, organization_id, from_team, to_team: Option<String>, kind, subject, body, timestamp, read }
pub struct OrgMailbox;          // Arc<dyn TeamStore> + organization_id + namespace
impl OrgMailbox { new(store, organization_id), organization_id, namespace, store,
                  send, broadcast, notify, announce, inbox, acknowledge, acknowledge_all, watermark, unread_count }

// swarmflow.rs — plan → bounded fan-out → fan-in
pub struct LaneSpec { lane_id, title, payload, max_attempts, backoff_ms, metadata }
pub struct SwarmflowPlan { plan_id, lanes, max_concurrency, continue_on_failure, max_backoff_ms, metadata }
pub enum LaneStatus { Completed, Failed, Skipped }
pub struct LaneOutcome { lane_id, status, attempts, output, error }
pub struct SwarmflowOutcome { plan_id, lanes, completed, failed, skipped, total_attempts, peak_concurrency }
pub trait SwarmLane { fn name(&self) -> &str; async fn run(&self, lane: &LaneSpec, attempt: u32) -> Result<Value>; }
pub struct LaneClosure;  pub struct SwarmflowRunner;
pub async fn run_swarmflow<L: SwarmLane + ?Sized>(plan, lane: Arc<L>) -> Result<SwarmflowOutcome>;

// fork_compact.rs — fork a member's context, compact a transcript
pub struct ForkTaskOverride { title, content, parent_task_id, metadata }
pub struct ForkToolPolicy { denied_tools, allowed_tools, additive_tools }        // apply(&[ToolSpec]) -> Vec<ToolSpec>
pub struct ForkSource { team_name, member_name, session_id, member: TeamMemberSpec, spec: DeepAgentSpec, lineage }
pub struct ForkedMemberSpec { member, spec, parent_team, parent_member, session_id, workspace, cwd,
                              project_root, task, lineage, fork_depth }
pub fn fork_member_context(source: &ForkSource, child: TeamMemberSpec, task: Option<ForkTaskOverride>,
                           tool_policy: &ForkToolPolicy) -> Result<ForkedMemberSpec>;
pub enum CompactRole { System, User, Assistant, Tool }
pub struct TranscriptMessage { role, content, tool_name, tool_call_id }          // system/user/assistant/tool/byte_len
pub struct CompactPolicy { keep_system, keep_last_messages, max_tool_result_bytes, truncated_suffix }
pub struct CompactedTranscript { messages, summary, dropped_messages, truncated_tool_results,
                                 original_bytes, compacted_bytes }              // is_lossless/text
pub fn compact_transcript(messages: &[TranscriptMessage], policy: &CompactPolicy) -> CompactedTranscript;
```

`mod.rs` re-exports every submodule flat, so the whole layer reads as `crate::org::OrgChart`, `crate::org::OrgTaskPool`, `crate::org::SwarmflowPlan`, `crate::org::fork_member_context`, …
`task_pool::task_pool_types` stays reachable as a submodule as well as through the glob.

## Invariants

### Addressing and identity

- A team node id is its `team_name`; a member node id is `qualify_member(team, member)` = `"<team>::<member>"`, so the two namespaces cannot collide.
- `reports_to` accepts either form: a team reports to a team or a member, a member reports to a member or a team, and every link must name a node that exists (`Error::validation` otherwise).
- `OrganizationSpec::validate` enforces: the org-id grammar (`1..=96` of `[A-Za-z0-9_.-]`), no duplicate team names, no duplicate `(team, member)` pairs, no `::` in a team name, non-empty member ids, a declared team for every member, no self-report, no cycle in the union reporting graph (`"reporting cycle: a -> b -> a"`, path rotated to start at its smallest id), each team's resolved leader (`leader`, else `team_leader`) present in that team, `owner` an existing member node, and `summary_team` an existing team with `OrgRole::Summary`.
- Hierarchy reads are sorted: `teams()` by `team_name`, `members()` by `(team_name, member_id)`, `direct_reports`/`descendants` lexicographically, `ancestors` nearest-first. `depth_of` is root = 0 and an unknown node is `Error::NotFound`.

### Delegation (`DelegationRouter::decide`)

First refusal wins, in this order (the module docs carry the same table):

| # | Check | Refusal |
| --- | --- | --- |
| 1 | delegator node exists in the chart | `UnknownDelegator` |
| 2 | target team exists | `UnknownTargetTeam` |
| 3 | target member, when named, is a member of the target team (`None` = that team's leader) | `UnknownTargetMember` |
| 4 | resolved target is not the delegator | `SelfDelegation` |
| 5 | `depth` (0 reads as 1) `<= max_delegation_depth`, unless the cap is 0 | `DepthExceeded` |
| 6 | non-empty `allowed_pairs` contains `(from_role, to_role)` | `RoleNotAllowed` |
| 7 | when `require_reporting_line`: same team, or target team descends from the delegator's team, or the delegator is the `Owner`, or the delegator node is an ancestor of the target node | `ReportingLineRequired` |
| 8 | `open_tasks_in_target < budget_for(to_team)` (falling back to `default_task_budget`) | `BudgetExhausted` |

`check` maps a refusal to `Error::validation(refusal.message())`, so a pool caller can use one call; `decide` stays total and never errors.

### Task pool storage mapping (`OrgTaskPool`)

- Rows live in the existing store under `org::<organization_id>` — one `TaskRecord` per org task, never a second database.
- `assignee`, `reviewers` and `created_by` are team-qualified; `team_name` is the namespace; `title`/`content`/`status`/`summary`/`parent_task_id`/`review_round`/`max_review_rounds`/`updated_at` keep their board meaning.
- `note` carries the JSON `OrgTaskSidecar` (`owner_team`, `target_team`, `depth`, `failure`, `metadata`) because the pool is the only writer of these rows; human-facing history goes into `change_notes` instead.
- Creating a child adds `add_dependency(namespace, parent, child)`, so a parent is `blocked` until its children are terminal, and a terminal write unblocks every direct dependent whose dependencies are all terminal (`blocked -> pending`).
- Statuses: `pending` → `in_progress` on claim/start → `in_review` when reviewers exist (with `review_round += 1`) → `completed`; `planning` resolves through `in_progress` because `TASK_TRANSITIONS` has no direct edge.
- `fail` becomes `cancelled` (there is no failed status) with the `OrgTaskFailure` persisted in the sidecar and `failed: <code> — <message>` appended to `change_notes`.
- Every status change is a `TeamStore::transition_task` compare-and-set; a `false` result means another writer won and is reported as `OrgTaskOpResult::failed(reason)`, never as `Err`. Only the CAS winner writes `assignee`, so two concurrent claims cannot both succeed.
- Review: `review` is the autonomous path (one verdict settles), `vote` + `settle_review` are the scheduled path over `crate::taskboard`'s `ReviewTally`/`tally_with`; a tie never passes.
- `delegate` builds a `DelegationRequest` whose `open_tasks_in_target` is `open_task_count(to_team)`, so the rule engine's budget is enforced in the pool and not only by the caller.
- `OrgTaskEvent`s are pure data returned in `OrgTaskMutation`/delivered to the optional `OrgTaskEventSink`; the pool never publishes to the bus itself.

### Mailbox storage mapping (`OrgMailbox`)

- Message rows live in the same namespace but the message table: `from_member_name` is the sending team (or `__organization__`), `to_member_name` is `Some(team)` for a direct row and `None` for a broadcast, `protocol` is `"org"`, and `content` is the JSON `{kind, subject, body}` payload.
- Reads ignore rows whose protocol is not `org` and skip corrupt payloads instead of panicking; order is the store's `(timestamp, message_id)`.
- Receipts are exactly the store's read state: `mark_read` for direct rows and the per-team watermark for broadcasts; `acknowledge` returns the number of read-state changes and ignores ids that do not belong to the team, `acknowledge_all` clears both.
- `notify`/`announce` are ordinary rows from `ORG_SYSTEM_SENDER` with `OrgMessageKind::System` and the default subject `organization`.
- Self-messaging is refused; team names are validated as store keys and path segments (non-empty, no `::`, no `org::` prefix, `<= 96` chars); subject and body are bounded by `crate::constants::MAX_MESSAGE_CONTENT_CHARS`.

### Swarmflow

- A plan is validated before any lane starts: non-empty `plan_id`, at least one lane, unique non-blank `lane_id`s, `max_concurrency >= 1`, `max_attempts >= 1`.
- At most `max_concurrency` lanes run at once, each as its own future; `peak_concurrency` is measured with an in-flight counter, not estimated.
- A lane is retried up to `max_attempts`; the wait before attempt *n+1* is `backoff_ms << (n-1)` capped at `max_backoff_ms`.
- With `continue_on_failure = false` the runner stops launching after the first exhausted failure and marks never-launched lanes `Skipped`; already-running lanes finish and keep their real status.
- `SwarmflowOutcome::lanes` is always in plan order regardless of completion order; `is_success` requires no `Failed` and no `Skipped`.
- The runner makes no LLM call and owns no state: real work arrives through `SwarmLane`.

### Fork and compact

- `fork_member_context` is pure: the child name must match `^[a-z][a-z0-9_-]*$`, must not be reserved and must differ from the parent; `session_id`, `workspace`, `cwd` and `project_root` are inherited so the child shares the parent's session identity and workspace layout.
- The child's `DeepAgentSpec` starts from the parent's with the tool policy applied, `system_prompt` cleared and `card` cleared, so the parent's role identity never leaks; the child's identity comes from its `TeamMemberSpec`.
- `lineage` is the parent chain with the parent appended and `fork_depth == lineage.len()`.
- `compact_transcript` keeps system messages at the front (unless `keep_system` is false), keeps the last `keep_last_messages` non-system messages verbatim, replaces everything older with exactly one deterministic summary line (`"[compacted] <n> earlier messages dropped (<u> user, <a> assistant, <t> tool)"`), and truncates only `Tool` content beyond `max_tool_result_bytes` on a UTF-8 char boundary with `truncated_suffix`.
- Compaction never grows the transcript (`compacted_bytes <= original_bytes`), is stable when applied twice, and `is_lossless()` means nothing was dropped and nothing truncated.

## Constraints

- Only `src/org/**` is edited; `src/runtime/`, `src/schema/`, `src/taskboard/`, `src/storage/`, `src/lib.rs`, `Cargo.toml`, `tests/`, the crate-root `CONTEXT.md` and `docs/` are read-only here.
- `openjiuwen_common::error::{Error, Result}` only: `validation` for user input, `not_found` for unknown tasks/nodes, `config` for a missing build dependency; no `unwrap`/`expect` outside tests.
- Async-first tokio with `#[async_trait]`; the router and chart are `Arc`-shared and immutable, so no lock is needed; the pools hold no interior mutable state (the mailbox's timestamp guard is the only one, and it is a single short `parking_lot`/atomic section, never held across an `.await`).
- `tracing` only (a `debug!` on task-pool mutations), no `println!`/`eprintln!`; no `unsafe`; no undocumented global state, and none in this module at all.
- Every public item documented; fallible methods carry `# Errors`; pure accessors are `#[must_use]`; every data type is `serde` with `rename_all = "snake_case"`, `default` and unknown-field tolerance; event enums are internally tagged (`type`).
- Files ≤ 1000 lines; `task_pool.rs` is the sanctioned long file (1140 lines: one policy object whose lifecycle rules only read as a whole) with a top-of-file note, and its value objects live in `task_pool_types.rs`.
- `--no-default-features` must build: this module has no feature gates and no `mcp`/`sqlite` reference.

## Deferred (documented, not implemented)

- Cross-process organisation: a shared store plus an external transport, attach/reattach, leader-turn queues, resume-from-DB binding and the `(db, organization, session)` manager pool of Python `organization/pool.py`.
- Expert-team launcher, the reusable Summary Team (and the LLM "laundering" closeout rail), aggregation modes (`HIERARCHICAL` vs `SUMMARY_TEAM`, `aggregation_json`), root/child aggregation preconditions and the summary-execution rows.
- The unclaimed-task state machine and its deadline scanner (`unclaimed.py`), repairs as sibling children, `retry_limit`/`retry_count`, and durable system-notification replay.
- The organisation workspace manager and write policy (`teams/<id>/`, `shared/`, `summary/`, git init/commit lock, conflict strategy) plus `workspace_rail.py` enforcement.
- `TransportAPI` negotiation/handshake and the organisation event bus beyond the in-process `OrgTaskEvent`/`OrgMessage` values.
- The LLM-facing organisation tool wrappers (`tools.py`: control, task-pool, summary and inbox tools) and their prompt packs.
- LLM-generated compaction: Python asks the model to summarise; this port emits the deterministic template summary line instead.
- Swarmflow's external host/bridge (`external_swarmflow_host.py`), worktree lanes and in-runner review gate (`tool_swarmflow.py`).
- Org-level task lease/expiry: `TaskRecord` carries no lease column, so staleness stays the board's `updated_at` policy.

## Routing Table

- `./mod.rs` → namespace constants, org-id validation, qualified-member helpers, flat re-exports (`#![warn(missing_docs)]` surface)
- `./roles.rs` → `OrgRole`, `OrgTeam`, `OrgMember`, `OrganizationSpec`, `OrgNodeRef`, `OrgChart` and every hierarchy query
- `./delegation.rs` → `RolePair`, `DelegationRules`, `DelegationRequest`, `DelegationRefusal`, `DelegationDecision`, `DelegationRouter`
- `./task_pool.rs` → `OrgTaskPool` and its lifecycle operations, sidecar codec, reason strings, unblocking
- `./task_pool_types.rs` → the pool's value objects (`OrgTaskSpec`, `OrgTask`, `OrgTaskFilter`, `OrgTaskOpResult`, `OrgTaskEvent`, `OrgTaskMutation`, `OrgTaskFailure`, `OrgTaskSidecar`, `OrgTaskEventSink`)
- `./mailbox.rs` → `OrgMailbox`, `OrgMessage`, `OrgMessageKind`, mailbox constants
- `./swarmflow.rs` → `LaneSpec`, `SwarmflowPlan`, `SwarmLane`, `LaneClosure`, `SwarmflowRunner`, `SwarmflowOutcome`
- `./fork_compact.rs` → `ForkSource`/`ForkToolPolicy`/`ForkedMemberSpec`/`fork_member_context` and `TranscriptMessage`/`CompactPolicy`/`CompactedTranscript`/`compact_transcript`
- `./roles_tests.rs`, `./delegation_tests.rs`, `./task_pool_tests.rs`, `./mailbox_tests.rs`, `./swarmflow_tests.rs`, `./fork_compact_tests.rs` → the crate-internal suites, included with `#[cfg(test)] #[path = "…"] mod …;`

## Test Strategy

133 unit tests live beside the modules and run offline against `InMemoryTeamStore` plus small in-memory fixtures; no `#[ignore]`, no network, no sleeps beyond short bounded waits.

- `roles_tests.rs` (18): org-id grammar and message, duplicate/malformed team names, duplicate members, undeclared team, unknown/self `reports_to`, team and member reporting cycles with exact path text, leader membership and the `team_leader` default, owner checks, `summary_team` role check, sorted listings, a full 5-deep hierarchy walk, unknown-node `NotFound`, serde and `to_value` round-trips.
- `delegation_tests.rs` (18): the allow/refuse matrix (intra-team, descendant team, owner bypass, upward, lateral, self, unknown delegator/team/member, depth cap incl. `depth == 0` and unlimited, role matrix, budget boundary and per-team override), `check()`'s validation error, `code()` strings, rules/request serde.
- `task_pool_tests.rs` (41): create root/child and parent parking, claim CAS with a second-claimer refusal, start, complete with and without reviewers, review pass/fail, scheduled vote/settle (undecided, majority, tie), fail → `cancelled` + sidecar + unblocking, delegation allowed/refused (line, budget, unknown delegator), `open_task_count`, list filters/order, `children`, namespace isolation from a real team's `TaskBoard`, and the type serde.
- `mailbox_tests.rs` (15): stamped send and round-trip, receiver-only visibility, self-send and name validation, oversize subject/body, broadcast visibility and watermark advance/non-regression, direct ack idempotence, `acknowledge_all`, system `notify`/`announce`, ordering, namespace isolation, foreign-protocol and corrupt rows skipped, serde, `preview`.
- `swarmflow_tests.rs` (12): measured peak concurrency ≤ limit and reaching it, retry-then-success, retries exhausted, continue-on-failure vs fail-fast skipping, plan-order results under out-of-order completion, payload/attempt delivery, validation errors, outcome lookups, serde, `LaneClosure` as `Arc<dyn SwarmLane>`.
- `fork_compact_tests.rs` (25): inheritance of session/workspace/cwd/project_root/model, identity clearing, tool deny/allow/additive ordering, invalid/reserved/self fork, two-level lineage, serde shape, exact summary lines, system front-keeping and `keep_system = false`, `keeping_last(0)`, multibyte truncation on a char boundary, tool-only truncation, losslessness, monotonicity, idempotence, empty transcript.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline org::`, `cargo test -p openjiuwen-agent-teams --offline --no-default-features org::`, `cargo clippy -p openjiuwen-agent-teams --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-agent-teams --no-deps --offline`.

## Runtime integration points (next step, outside this node)

1. Spec adoption: `TeamAgentSpec` is closed to this node, so an organisation is carried as JSON — suggested `TeamAgentSpec.metadata["organization"] = {"spec": OrganizationSpec, "rules": DelegationRules}` — and the runtime parses it with `serde_json::from_value`, then builds `Arc<OrgChart>` and `Arc<DelegationRouter>`.
2. `TeamBackend` (`src/runtime/backend.rs`) is the natural owner of the organisation handles (`Arc<OrgChart>`, `Arc<DelegationRouter>`, `Arc<OrgTaskPool>`, `Arc<OrgMailbox>`), mirroring Python's `_bind_team`; `TeamAgent::backend()` already hands that facade to tools, CLI and MCP.
3. Task-pool events: construct the pool with `OrgTaskPool::with_event_sink`, wrap each `OrgTaskEvent` into `EventMessage { event_type: "org_task", payload, sender_id: team }` and publish it on `TeamTopic::Task` through `TeamBackend::post_event`.
4. Mailbox drain: the leader loop in `src/runtime/leader.rs` should read `OrgMailbox::inbox(team)` per tick, fold each `OrgMessage` into the existing batched leader `send` (respecting `steer_batch_size`), and call `acknowledge` only after a successful leader turn — Python's ack gate.
5. Fork: `TeamMemberSpawner` (`src/runtime/spawner.rs`) should call `fork_member_context` when a spawn directive carries a parent and a task override, feeding `ForkedMemberSpec` into `MemberHarnessContext` and turning the override into a `TaskGraphSpec` with `parent_task_id`.
6. Compact: the harness adapter (`src/harness/`) converts the DeepAgent context into `TranscriptMessage`s and calls `compact_transcript` at Python's split point instead of the model-generated summary.
7. Swarmflow: implement `SwarmLane` in `src/runtime/` over `TeamAgent::invoke` (leader-side fan-out) or `MemberRuntime::send` (member-side), building the plan from `TaskGraphSpec` children.
8. Crate root (`src/lib.rs`, parent node): add `pub use org::{...}` for the types listed under API Surface so the layer mirrors the other flat re-exports; `openjiuwen-agent-teams/src/org` already re-exports its six submodules flat.

## Notes for Agents

- The organisation layer owns no member lifecycle: a "leader turn", a spawn, a pause or a workspace bind belongs to `src/runtime/`, and this layer only decides *whether* and *where* work goes.
- Two dormant-but-real deviations from Python worth remembering: `fail` is `cancelled` plus a sidecar (there is no failed status), and the mailbox stamps its own timestamps (`max(now, previous + 1)`, shared across clones) because `TeamStore::insert_message` writes the caller's stamp verbatim and has no single-message getter; a future storage change that stamps `0` and adds `message(team, id)` keeps this code working.
- Parent parking in the task pool writes `blocked` directly for `planning`/`in_progress` parents, the same repair `TaskBoard::create_subtask` makes, because `TASK_TRANSITIONS` has no edge back to `blocked`.
- `OrgTaskPool::list` fetches un-limited when the `team` or bare-`assignee` filters apply, filters in memory, then truncates to `limit`.
- The pool and the mailbox share one namespace and therefore one store instance; they touch different tables, so a task id and a message id can never collide and a real team's `TaskBoard` rows stay invisible to both.
- `swarmflow` deliberately does not depend on `crate::runtime`; keeping the lane seam injectable is what makes the runner testable with fakes and keeps the fan-out policy independent of how a member is spawned.
- `fork_compact` is pure and reusable outside the team runtime: it has no store, no clock and no session dependency, so the harness can call the compaction step on any transcript.
- `task_pool.rs` is the sanctioned long file; `task_pool_tests.rs` (1368 lines) and `fork_compact_tests.rs` (809 lines) are comprehensive suites rather than policy code.
