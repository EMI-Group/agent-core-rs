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

// compact_template.rs — Python's split/direction compaction with a total byte budget
pub const DEFAULT_TEMPLATE_MAX_FACTS: usize = 4;
pub const DEFAULT_EXCERPT_CHARS: usize = 48;
pub const DEFAULT_BUDGET_PRIORITY: [BudgetTier; 3] = [Summary, KeptSide, ToolResults];
pub enum CompactDirection { Before, After }        // default Before; as_str/parse/compresses
pub enum CompactionSide { Head, Tail }             // default Head; as_str
pub enum BudgetTier { Summary, KeptSide, ToolResults }   // as_str
pub enum SummaryOmission { Regressive, Budget }    // as_str
pub enum SummaryFact { Paths, Decisions, Errors, Feedback }   // as_str
pub struct SummaryRetention { facts, max_facts, max_excerpt_chars }   // new/with_facts/without/with_max_facts/with_excerpt_chars/retains
pub struct CompactTemplate { split_at, direction, keep_system, total_budget_bytes: Option<usize>,
                             budget_priority, max_tool_result_bytes, truncated_suffix,
                             extend_through_tool_results, retention }   // new + builders + validate
pub struct TemplateCompaction { messages, summary, summary_omitted, direction, compressed_side,
                                split_at, requested_split_at, compressed_messages,
                                stripped_system_messages, kept_messages, squeezed_messages,
                                truncated_tool_results, original_bytes, compacted_bytes }  // is_lossless/text
pub fn compact_with_template(messages: &[TranscriptMessage], template: &CompactTemplate)
    -> Result<TemplateCompaction>;
```

`mod.rs` re-exports every submodule flat, so the whole layer reads as `crate::org::OrgChart`, `crate::org::OrgTaskPool`, `crate::org::SwarmflowPlan`, `crate::org::fork_member_context`, …
`task_pool::task_pool_types` stays reachable as a submodule as well as through the glob.
`fork_compact` re-exports every `compact_template` item, so `crate::org::fork_compact::CompactTemplate` and `crate::org::CompactTemplate` are the same type.

## Invariants

### Addressing and identity

- A team node id is its `team_name`; a member node id is `qualify_member(team, member)` = `"<team>::<member>"`, so the two namespaces cannot collide.
- `reports_to` accepts either form: a team reports to a team or a member, a member reports to a member or a team, and every link must name a node that exists (`Error::validation` otherwise).
- `OrganizationSpec::validate` enforces: the org-id grammar (`1..=96` of `[A-Za-z0-9_.-]`), no duplicate team names, no duplicate `(team, member)` pairs, no `::` in a team name, non-empty member ids, a declared team for every member, no self-report, no cycle in the union reporting graph (`"reporting cycle: a -> b -> a"`, path rotated to start at its smallest id), each team's resolved leader (`leader`, else `team_leader`) present in that team, `owner` an existing member node, and `summary_team` an existing team with `OrgRole::Summary`.
- Exact validation message templates (all `Error::validation`, rendered with a `validation: ` prefix): `invalid organization id '{id}': expected 1..=96 characters of [A-Za-z0-9_.-]`, `team name must not be empty`, `team name '{team}' must not contain '::'`, `duplicate team name '{team}'`, `team '{team}' has a member with an empty id`, `member '{team}::{member}' belongs to unknown team '{team}'`, `duplicate member '{team}::{member}'`, `'{node}' cannot report to itself`, `'{node}' reports to unknown node '{parent}'`, `team '{team}' leader '{leader}' is not a member of the team`, `owner '{owner}' is not an organisation member`, `summary_team '{team}' is not a team of the organisation`, `summary_team '{team}' must have role 'summary'`.
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

### Template-driven compaction (`compact_with_template`)

- `CompactDirection::Before` compresses `stripped[..split_at]` and keeps the tail verbatim; `After` keeps `stripped[..head_end]` and compresses the rest, where `head_end` starts at `split_at` and advances through trailing `Tool` messages when `extend_through_tool_results` (Python's dangling-tool-call guard).
- Every `System` message is stripped *before* the split is computed and counted in `stripped_system_messages`; it is re-emitted at the front only when `keep_system` (default false, matching Python, which never puts them back).
- The split is validated against the stripped transcript (`1 <= split_at < non-system messages`) and a bad split, an empty `budget_priority` or a repeated tier is a typed `Error::validation`; Python silently skips instead.
- The summary line is `"<SUMMARY_PREFIX><side>@<split>/<total>: <n> message(s) compressed (<role counts>)"` followed by `paths`, `decisions`, `errors` and `feedback` segments in that fixed order; an empty or disabled category costs no byte.
- Facts are extracted deterministically from the compressed side: path-like tokens (first-seen order, deduped), lines containing an error marker, lines containing a decision marker, and every non-empty line of a `User` message; `max_facts == 0` reports counts, `max_excerpt_chars` cuts an excerpt on a char boundary with `…`.
- Python's retention priorities are the template contract: `SummaryRetention::facts` selects the categories and the default enables all four (paths, decisions, errors, feedback).
- Budget tiers are reduced lowest-priority first until the result fits `min(budget, original_bytes)`; within a tier the messages farthest from the split go first, so the side nearest the split survives longest.
- Default `budget_priority` is `[Summary, KeptSide, ToolResults]`: the summary is protected first, then the verbatim non-tool messages, then their tool results; a template that lists `Summary` last makes the summary the first thing the budget removes.
- A summary that would cost more than the messages it replaces is omitted with `SummaryOmission::Regressive`; a summary the budget removed is `SummaryOmission::Budget`; `squeezed_messages` counts kept-side and system messages only.
- Kept system messages are the budget's last resort, so `compacted_bytes <= min(budget, original_bytes)` always holds; `compacted_bytes` is the summed `byte_len` of `messages`, and compaction is monotone by construction.

## Constraints

- Only `src/org/**` is edited; `src/runtime/`, `src/schema/`, `src/taskboard/`, `src/storage/`, `src/lib.rs`, `Cargo.toml`, `tests/`, the crate-root `CONTEXT.md` and `docs/` are read-only here.
- `openjiuwen_common::error::{Error, Result}` only: `validation` for user input, `not_found` for unknown tasks/nodes, `config` for a missing build dependency; no `unwrap`/`expect` outside tests.
- Async-first tokio with `#[async_trait]`; the router and chart are `Arc`-shared and immutable, so no lock is needed; the pools hold no interior mutable state (the mailbox's timestamp guard is the only one, and it is a single short `parking_lot`/atomic section, never held across an `.await`).
- `tracing` only (a `debug!` on task-pool mutations), no `println!`/`eprintln!`; no `unsafe`; no undocumented global state, and none in this module at all.
- Every public item documented; fallible methods carry `# Errors`; pure accessors are `#[must_use]`; every data type is `serde` with `rename_all = "snake_case"`, `default` and unknown-field tolerance; event enums are internally tagged (`type`).
- Files ≤ 1000 lines; `task_pool.rs` (1140 lines) and `compact_template.rs` (1056 lines) are the sanctioned long files, each carrying a top-of-file note, because a lifecycle policy and a split/budget/retention pipeline only read as a whole; `task_pool`'s value objects live in `task_pool_types.rs`.
- `--no-default-features` must build: this module has no feature gates and no `mcp`/`sqlite` reference.

## Deferred (documented, not implemented)

- Cross-process organisation: a shared store plus an external transport, attach/reattach, leader-turn queues, resume-from-DB binding and the `(db, organization, session)` manager pool of Python `organization/pool.py`.
- Expert-team launcher, the reusable Summary Team (and the LLM "laundering" closeout rail), aggregation modes (`HIERARCHICAL` vs `SUMMARY_TEAM`, `aggregation_json`), root/child aggregation preconditions and the summary-execution rows.
- The unclaimed-task state machine and its deadline scanner (`unclaimed.py`), repairs as sibling children, `retry_limit`/`retry_count`, and durable system-notification replay.
- The organisation workspace manager and write policy (`teams/<id>/`, `shared/`, `summary/`, git init/commit lock, conflict strategy) plus `workspace_rail.py` enforcement.
- `TransportAPI` negotiation/handshake and the organisation event bus beyond the in-process `OrgTaskEvent`/`OrgMessage` values.
- The LLM-facing organisation tool wrappers (`tools.py`: control, task-pool, summary and inbox tools) and their prompt packs.
- LLM-generated compaction: Python's `fork_compact.compact_context` asks the model for the summary and writes it back into the live context; this port's `fork_compact::compact_transcript` and `compact_template::compact_with_template` produce a deterministic template summary instead and have no runtime caller.
- The missing runtime integration: neither compaction entry point is adopted by `src/harness/` or `src/runtime/`, the fork path carries a blueprint (`ForkedMemberSpec`) so nothing in this crate captures a member transcript, Python's `ForkContext.messages` → `create_new_context_engine(messages=…)` hand-off has no counterpart, and `SwarmflowRunner` has no runtime lane.
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
- `./fork_compact.rs` → `ForkSource`/`ForkToolPolicy`/`ForkedMemberSpec`/`fork_member_context` and `TranscriptMessage`/`CompactPolicy`/`CompactedTranscript`/`compact_transcript`, plus the re-export of every `compact_template` item
- `./compact_template.rs` → `CompactTemplate`, `CompactDirection`, `CompactionSide`, `BudgetTier`, `SummaryRetention`/`SummaryFact`, `SummaryOmission`, `TemplateCompaction`, `compact_with_template`, the internal `BudgetState` and the deterministic fact extractor
- `./roles_tests.rs`, `./delegation_tests.rs`, `./task_pool_tests.rs`, `./mailbox_tests.rs`, `./swarmflow_tests.rs`, `./fork_compact_tests.rs`, `./compact_template_tests.rs` → the crate-internal suites, included with `#[cfg(test)] #[path = "…"] mod …;`

## Test Strategy

153 unit tests live beside the modules and run offline against `InMemoryTeamStore` plus small in-memory fixtures; no `#[ignore]`, no network, no sleeps beyond short bounded waits.

- `roles_tests.rs` (18): org-id grammar and message, duplicate/malformed team names, duplicate members, undeclared team, unknown/self `reports_to`, team and member reporting cycles with exact path text, leader membership and the `team_leader` default, owner checks, `summary_team` role check, sorted listings, a full 5-deep hierarchy walk, unknown-node `NotFound`, serde and `to_value` round-trips.
- `delegation_tests.rs` (18): the allow/refuse matrix (intra-team, descendant team, owner bypass, upward, lateral, self, unknown delegator/team/member, depth cap incl. `depth == 0` and unlimited, role matrix, budget boundary and per-team override), `check()`'s validation error, `code()` strings, rules/request serde.
- `task_pool_tests.rs` (41): create root/child and parent parking, claim CAS with a second-claimer refusal, start, complete with and without reviewers, review pass/fail, scheduled vote/settle (undecided, majority, tie), fail → `cancelled` + sidecar + unblocking, delegation allowed/refused (line, budget, unknown delegator), `open_task_count`, list filters/order, `children`, namespace isolation from a real team's `TaskBoard`, and the type serde.
- `mailbox_tests.rs` (15): stamped send and round-trip, receiver-only visibility, self-send and name validation, oversize subject/body, broadcast visibility and watermark advance/non-regression, direct ack idempotence, `acknowledge_all`, system `notify`/`announce`, ordering, namespace isolation, foreign-protocol and corrupt rows skipped, serde, `preview`.
- `swarmflow_tests.rs` (12): measured peak concurrency ≤ limit and reaching it, retry-then-success, retries exhausted, continue-on-failure vs fail-fast skipping, plan-order results under out-of-order completion, payload/attempt delivery, validation errors, outcome lookups, serde, `LaneClosure` as `Arc<dyn SwarmLane>`.
- `fork_compact_tests.rs` (25): inheritance of session/workspace/cwd/project_root/model, identity clearing, tool deny/allow/additive ordering, invalid/reserved/self fork, two-level lineage, serde shape, exact summary lines, system front-keeping and `keep_system = false`, `keeping_last(0)`, multibyte truncation on a char boundary, tool-only truncation, losslessness, monotonicity, idempotence, empty transcript.
- `compact_template_tests.rs` (20): both directions and the head-widening tool extension, an empty compressed side, system stripping before the split, `keep_system`, split at 0 / at the end / out of range as typed errors, direction parsing and side/tier/reason names, the exact summary line with `head@3/6`-style side and split, excerpt caps and counts-only mode, retention selection, the regressive-summary guard, budget-first summary omission, tool-results-before-kept-messages squeezing under the default priority, nearest-the-split survival in both directions, a custom priority order inverting the squeeze, `budget_priority` validation, tool truncation, the never-grow/never-exceed-budget matrix, and the template/result serde round-trip.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --offline org::`, `cargo test -p openjiuwen-agent-teams --offline --no-default-features org::`, `cargo clippy -p openjiuwen-agent-teams --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-agent-teams --no-deps --offline`.

## Runtime integration (owned by the sibling nodes)

The runtime adopts this layer through one optional metadata block: `TeamAgentSpec.metadata["organization"] = {"spec": OrganizationSpec, "rules": DelegationRules}` is parsed in `src/runtime/spec_meta.rs` and installed by `build_with` as an `OrgBinding` on `TeamBackend`.
`OrgBinding` owns the `Arc<OrgChart>`, `Arc<DelegationRouter>`, `Arc<OrgTaskPool>` and `Arc<OrgMailbox>`, and `TeamAgent::org()` hands it to callers.
`OrgTaskPool::with_event_sink` receives `runtime::org_task_sink`, which publishes each `OrgTaskEvent` as an `"org_task"` payload on `TeamTopic::Task`.
The leader loop in `src/runtime/leader.rs` reads `OrgBinding::unread(team)` (over `OrgMailbox::inbox`) per tick, folds each message into the batched leader `send`, and acknowledges only after a successful flush (Python's ack gate).
`TeamMemberSpawner::forked_context` calls `fork_member_context` for a forked member and feeds the `ForkedMemberSpec` into the spawn path.
`src/lib.rs` re-exports this module flat (`pub use org::*;`).

## Notes for Agents

- The organisation layer owns no member lifecycle: a "leader turn", a spawn, a pause or a workspace bind belongs to `src/runtime/`, and this layer only decides *whether* and *where* work goes.
- Two dormant-but-real deviations from Python worth remembering: `fail` is `cancelled` plus a sidecar (there is no failed status), and the mailbox stamps its own timestamps (`max(now, previous + 1)`, shared across clones) because `TeamStore::insert_message` writes the caller's stamp verbatim and has no single-message getter; a future storage change that stamps `0` and adds `message(team, id)` keeps this code working.
- Parent parking in the task pool writes `blocked` directly for `planning`/`in_progress` parents, the same repair `TaskBoard::create_subtask` makes, because `TASK_TRANSITIONS` has no edge back to `blocked`.
- `OrgTaskPool::list` fetches un-limited when the `team` or bare-`assignee` filters apply, filters in memory, then truncates to `limit`.
- The pool and the mailbox share one namespace and therefore one store instance; they touch different tables, so a task id and a message id can never collide and a real team's `TaskBoard` rows stay invisible to both.
- `swarmflow` deliberately does not depend on `crate::runtime`; keeping the lane seam injectable is what makes the runner testable with fakes and keeps the fan-out policy independent of how a member is spawned.
- `fork_compact` is pure and reusable outside the team runtime: it has no store, no clock and no session dependency, so the harness can call the compaction step on any transcript.
- `task_pool.rs` and `compact_template.rs` are the sanctioned long files; `task_pool_tests.rs` (1368 lines), `compact_template_tests.rs` (691 lines) and `fork_compact_tests.rs` (809 lines) are comprehensive suites rather than policy code.
- `compact_transcript` and `compact_with_template` are both uncalled: the former keeps its "last N non-system messages" heuristic while only the template entry point models Python's `split_at` + `direction` contract, because this crate never captures a member transcript to hand to either; a new fork path should adopt `compact_with_template` rather than grow the other heuristic.
- The template path never re-emits stripped system messages by default and never truncates the compressed side, so a summary can only disappear through the monotonicity guard or the budget — both are reported on `TemplateCompaction` (`summary_omitted`).
