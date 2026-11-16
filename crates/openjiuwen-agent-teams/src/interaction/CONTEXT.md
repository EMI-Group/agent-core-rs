# interaction

## Intent

The external interaction layer of `openjiuwen-agent-teams`: everything between an external caller and the running team.
Python parity source: `openjiuwen/agent_teams/interaction/` plus `runtime/gate.py` and `rails/tool_approval_rail.py` (v0.1.20).
It owns the typed payload union, the single `#`/`$`/`@` routing grammar, the per-run admission gate, the human-agent inbox and the tool-approval bridge.
Parsing happens exactly once, here; the runtime dispatch switch consumes typed values and never re-parses a body.

## API Surface

```rust
// payload.rs — every channel's input shape and the uniform outcome
pub enum InteractPayload { GodView(GodViewMessage), Operator(OperatorMessage),
                           HumanAgent(HumanAgentMessage), HumanAgentToolCall(HumanAgentToolCall),
                           GroupChat(GroupChatMessage), ExternalTeam(ExternalTeamEvent) }
impl InteractPayload { pub const fn kind(&self) -> &'static str; pub const fn requires_hitt(&self) -> bool; }

pub struct GodViewMessage   { pub body: String, pub sender: Option<String> }
pub struct OperatorMessage  { pub target: Option<String>, pub body: String, pub sender: Option<String> }
pub struct HumanAgentMessage{ pub human_name: String, pub target: Option<String>, pub body: String, pub sender: Option<String> }
pub struct HumanAgentToolCall { pub human_name: String, pub tool_name: String, pub tool_args: Value }
pub struct GroupChatMessage { pub body: String, pub sender: Option<String>, pub mentions: Vec<String> }
pub struct ExternalTeamEvent{ pub event: Value }

pub struct DeliverResult { pub ok: bool, pub message_id: Option<String>, pub reason: Option<String>,
                           pub output: Value, pub data: Value }
impl DeliverResult {
    pub fn ok(message_id: Option<String>) -> Self;
    pub fn failed(reason: impl Into<String>) -> Self;
    pub fn not_active() -> Self;                  // reason "not_active"
    pub fn gate_closed() -> Self;                 // reason "gate_closed"
    pub fn unsupported(reason: impl Into<String>) -> Self;  // reason "unsupported:<reason>"
}

// router.rs — free-form text to a roster-resolved routing decision
pub enum Route { GodView { body: String }, HumanAgent { name: String, body: String },
                 Member { targets: Vec<String>, body: String }, Broadcast { body: String } }
pub fn parse_interact_str(text: &str, roster: &[String]) -> Route;
pub fn resolve_targets(mentions: &[String], roster: &[String]) -> (Vec<String>, Vec<String>);

// gate.rs — admission control for one run cycle (synchronous, lock-guarded)
pub struct InteractGate;
impl InteractGate {
    pub fn new() -> Self;
    pub fn admit(&self, payload: InteractPayload) -> Result<()>;
    pub fn try_admit(&self) -> Result<Option<InteractPayload>>;
    pub fn drain(&self) -> Vec<InteractPayload>;
    pub fn close(&self);
    pub fn is_closed(&self) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn dispatch(&self, payload: InteractPayload, active: bool) -> DeliverResult;
}

// approval.rs — interrupted tool call → harness interaction → bus event
pub struct ConfirmPayload { pub approved: bool, pub feedback: String, pub auto_confirm: bool }
pub struct ToolApprovalRequest { pub team_name, member_name, tool_call_id, tool_name: String, pub tool_args: Value }
impl ToolApprovalRequest { pub fn to_interaction(&self) -> InteractionRequest; }
pub fn parse_confirm(question_id: &str, answer: &Value) -> Result<ConfirmPayload>;
pub fn approval_result_event(request: &ToolApprovalRequest, payload: &ConfirmPayload) -> EventPayload;

// inbox.rs — FIFO for already-parsed human-agent input
pub struct HumanAgentInbox;
impl HumanAgentInbox {
    pub fn new() -> Self;                       // starts disabled
    pub async fn push(&self, msg: HumanAgentMessage) -> Result<String>;   // id "msg_<uuid hex>"
    pub async fn drain(&self) -> Vec<(String, HumanAgentMessage)>;
    pub fn send(&self, body: impl Into<String>, to: Option<&str>, sender: Option<&str>) -> HumanAgentMessage;
    pub fn is_enabled(&self) -> bool;
    pub fn set_enabled(&self, enabled: bool);
    pub fn require_enabled(&self) -> Result<()>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
pub const HUMAN_AGENT_NOT_ENABLED: &str = "human agent is not enabled for this team";
```

## Routing grammar

`parse_interact_str` is pure syntax plus roster membership; it never touches a session or the bus.

| Input | Route | Notes |
| --- | --- | --- |
| `# body` | `GodView { body }` | `#` must be followed by a space; `#hashtag` is prose |
| `$name body` | `HumanAgent { name, body }` | drives that avatar's loop; `$name` must be on the roster |
| `@member body` | `Member { targets: [member], body }` | uses `constants::parse_mention` |
| `@a @b body` | `Member { targets: [a, b], body }` | one shared body, mention order, deduplicated |
| `@all body`, `@* body` | `Broadcast { body }` | broadcast supersedes any co-listed recipient |
| bare text | `Broadcast { body }` | no channel prefix |
| `@ghost body` (all mentions unknown) | `Broadcast { body: <original text> }` | mention folded back as prose |
| `@ghost @dev-1 body` | `Member { targets: [dev-1], body }` | unknown names dropped from `targets` |
| `$ghost body` (unknown channel) | `Broadcast { body: <original text> }` | nobody to drive, so unrouted |
| `""` / whitespace | `Broadcast { body }` | deterministic empty result (Python returns no payloads) |

`route.kind` wire tags: `god_view`, `human_agent`, `member`, `broadcast`.
Broadcast tokens are recognized before roster resolution; `resolve_targets` is pure membership and reports unknown names in order, deduplicated.

## Gate semantics

- `dispatch` checks in one atomic lock section: closed → `DeliverResult::gate_closed()`; `!active` → `DeliverResult::not_active()`; otherwise enqueue and return `DeliverResult::ok(None)`.
- `gate_closed` wins over `not_active` when both apply.
- Once `close()` is called, `admit` and `try_admit` fail with `Error::Validation("interact gate is closed")`; `dispatch` reports `gate_closed`.
- `drain` is allowed after close and returns the payloads queued before it, oldest first.
- The gate mints no message id: the run loop mints one when it delivers the queued payload.

## Constraints

- Only files under `src/interaction/`; `src/lib.rs`, `Cargo.toml`, `src/schema/` and sibling modules are read-only.
- Errors: `openjiuwen_common::error::{Error, Result}`; no panics on user input, no `unwrap`/`expect` outside tests.
- `tracing` only for diagnostics; no `println!`/`eprintln!`.
- Every public data type is `serde` with `snake_case`, tolerant of unknown fields; collections default to empty.
- The approval bridge uses `openjiuwen_session::{InteractionRequest, InteractionKind}` and `crate::schema::{EventPayload, ToolApprovalResultEvent}`; `HarnessState`/schema types are never redefined here.
- No `unsafe`; gate and inbox are `Send + Sync` and lock only for short, non-`await` sections.
- Files ≤1000 lines; unit tests live beside each module.

## Routing Table

- `./payload.rs` → `InteractPayload` union, per-channel message structs, `DeliverResult`
- `./router.rs` → `Route`, `parse_interact_str`, `resolve_targets`
- `./gate.rs` → `InteractGate` admission queue and run/interact decisions
- `./approval.rs` → `ToolApprovalRequest`, `ConfirmPayload`, `parse_confirm`, `approval_result_event`
- `./inbox.rs` → `HumanAgentInbox` FIFO, enablement check, message builder
- `./mod.rs` → flat re-exports of the whole seam

## Test Strategy

- Unit tests in `#[cfg(test)] mod tests` beside each module (`#[allow(clippy::unwrap_used, clippy::expect_used)]`).
- `payload.rs`: serde round-trip and `kind` tag for every variant, unknown-field tolerance, `requires_hitt` per channel, `DeliverResult` constructor reason tokens.
- `router.rs`: every grammar row above, multi-mention fan-out, unknown mention drop/fold, unknown `$` channel, empty body, `resolve_targets` order/dedup, route JSON round-trip.
- `gate.rs`: FIFO admit/try_admit, drain, `not_active`, `gate_closed` after close (including after close+drain), close idempotence.
- `approval.rs`: `to_interaction` kind/id/question, `parse_confirm` happy path/defaults/malformed, `approval_result_event` shape and `tool_approval_result` event type.
- `inbox.rs`: FIFO push/drain, `msg_` id format, enabled/disabled `require_enabled`, builder field resolution.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings`.

## Notes for Agents

- `parse_interact_str` returns one `Route`, not Python's payload list: the `Route` normalizes the audience, so a `$name` channel with a named recipient (`$alice@dev-1 hi`) collapses to `Member` and loses the human-agent sender. The runtime that needs the `$` identity must keep the channel separately or construct `HumanAgentMessage` directly.
- Python's inline mid-body segmentation (`@a body1 @b body2`) is not modeled; each parsed mention list fans out one shared body.
- Python defaults bare text to the `#` channel (god view); this port routes bare text to `Broadcast` per the crate contract, and `#` is the only explicit god-view marker.
- Python's gate counts in-flight payloads via `AdmissionTicket` and awaits `close_and_drain`; this port queues admitted payloads and exposes synchronous decisions. The wire-visible tokens (`not_active`, `gate_closed`) are unchanged.
- Python mints `intr_<uuid>` per interaction; `ToolApprovalRequest::to_interaction` derives `intr_team_tool_approval_<tool_call_id>` so re-rendering the same interrupted call is idempotent.
- Python's `ToolApprovalResultEvent` carries `feedback` and `auto_confirm`; this crate's schema event does not, so `approval_result_event` carries only team/member/tool-call/verdict (schema is out of this node's scope).
- `HumanAgentInbox::new()` starts disabled (`enable_hitt` is opt-in); the runtime must call `set_enabled(true)` before `require_enabled()` succeeds. `push` itself does not re-check enablement.
- `ExternalTeamEvent` is data-only: the runtime answers it with `DeliverResult::unsupported("external_team")` because distributed transports are deferred.
