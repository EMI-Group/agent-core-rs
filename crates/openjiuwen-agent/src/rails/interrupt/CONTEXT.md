# rails/interrupt/

## Intent

Interrupt (human-in-the-loop) rails: pause a tool call for approval or user input, and consume the answer on resume.
Ports the kernel-side behavior of Python `harness/rails/interrupt/` (`BaseInterruptRail`, `ConfirmInterruptRail`) and the `ask_user` kind, expressed on this crate's `AgentRail` seam.

## API Surface

- `ToolFilter { All, Names(BTreeSet<String>), Any(Arc<dyn Fn(&ToolCall) -> bool + Send + Sync>) }` with `matches(&ToolCall)`.
- `BaseInterruptRail` — `new(kind, owner, message)`, `with_tools(ToolFilter)`, `with_filter(ToolFilter)`, `with_payload_schema(Value)`, `with_auto_confirm_key(&str)` plus accessors; its `AgentRail::before_tool_call` implements the full approve/reject/ask/pause flow.
- `ConfirmInterruptRail` — kind `confirm`, owner `openjiuwen.agent.rails.interrupt.ConfirmInterruptRail`, message `"Please approve or reject?"`, payload schema `{approved, feedback, auto_confirm, persist_allow}` with `required: ["approved"]`, `auto_confirm_key = tool name`; `::new()`, `::for_tools(names)`.
- `AskUserRail` — kind `ask_user`, owner `openjiuwen.agent.rails.interrupt.AskUserRail`, default message `"Please provide your input."`; `::new()`, `::for_tools(names)`.
- An `InterruptRequest` serializes with Python wire names `interrupt_kind`/`interrupt_owner` (the Rust fields stay `kind`/`owner`, and deserialization accepts either), so an `__interaction__` frame payload is `{id, value}` whose `value` is the Python-shaped request object.
- `AgentRail::name()` is `"interrupt"` (`BaseInterruptRail`), `"confirm_interrupt"` (`ConfirmInterruptRail`), `"ask_user"` (`AskUserRail`); mount with `ReActAgent::register_rail(Arc::new(rail))` and remove with `unregister_rail(<that name>)`.

## Semantics of `before_tool_call`

- No `ctx.tool_call`, or a call the `ToolFilter` rejects, is left untouched.
- Answers are read from `ctx.extra["_resume_interrupt_answers"]`, an object keyed by `ToolCall.id`; an absent or empty id never matches.
- `parse_approval(answer) == Some(true)` approves the call by returning without recording anything, so the tool runs.
- `parse_approval(answer) == Some(false)` calls `ctx.skip_tool_with(reject_feedback(answer))`, defaulting to `"User feedback: rejected\n action"` and using the answer's `feedback`/`reason` otherwise.
- For `InterruptKind::AskUser` a non-approval answer is the user's text: `ctx.skip_tool_with(answer_text(answer))` becomes the tool result and the tool never runs.
- For any other kind a non-approval answer is malformed: the rail records a fresh request (new `interrupt_id`) instead of approving.
- Auto-confirm reads `ctx.extra["_interrupt_auto_confirm"]`: a bool approves every matching call, an object approves only the entries with name → `true`; it only applies when no answer was seeded.
- The loop seeds that key from the session's `__interrupt_auto_confirm__` on every invoke, before the first `BeforeToolCall`, so a host can pre-approve tools for a fresh turn as well as a resumed one.
- Otherwise the rail records `InterruptRequest::for_tool_call(kind, owner, message, call)`, applies the configured `payload_schema` and `auto_confirm_key` (falling back to the tool name), and calls `ctx.request_interrupt`.
- `answer_text` keeps a string as-is, prefers `text`/`value`/`answer`/`feedback`/`input` string fields of an object, stringifies anything else, and never returns empty text (`"User input: (empty)"`).
- `AskUserRail` derives the message from the call's `question` argument (parsed with `parse_tool_arguments`), falling back to `"Please provide your input."` and then `"Please provide input to continue."`; `ConfirmInterruptRail` always uses its configured message.

## Constraints

- An unanswered call only records an `InterruptRequest` on the context; the loop owns state persistence and the interrupt output.
- The rail records requests only; it never mutates the agent, mints `InterruptAnswer` values, or persists state.
- Approve → no context mutation; reject/ask-user → `ctx.skip_tool_with(..)`; malformed → one fresh request (never silently approves).
- No harness dependencies; owner strings are fixed per rail as listed above.
- Extra keys are private consts (`_resume_interrupt_answers`, `_interrupt_auto_confirm`) shared with the loop.

## Test Strategy

- Unit tests beside the module cover the canonical confirm request (tool-call id, tool name, kind, owner, schema with `required`, auto-confirm key), approve via bool/object/`"yes"`, reject with default and custom feedback, ask-user question extraction and answer-as-result, auto-confirm global and per-tool, filter rejection, a missing tool call, and malformed-answer re-ask with distinct interrupt ids.
- `../../tests/interrupt.rs` drives the same flows end to end through `ReActAgent`: the pause frame shape and persisted state, approve/reject resumes (with and without feedback), a session-seeded auto-confirm that skips the pause, ask-user answer-as-tool-result, and two pending calls pausing once then resuming individually.

## Routing Table

- `./mod.rs` → filters and the three rails
