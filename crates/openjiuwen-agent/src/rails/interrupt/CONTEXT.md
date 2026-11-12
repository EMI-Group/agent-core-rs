# rails/interrupt/

## Intent

Interrupt (human-in-the-loop) rails: pause a tool call for approval or user input, and consume the answer on resume.
Ports the kernel-side behavior of Python `harness/rails/interrupt/` (`BaseInterruptRail`, `ConfirmInterruptRail`) and the `ask_user` kind, expressed on this crate's `AgentRail` seam.

## API Surface

- `ToolFilter { All, Names(BTreeSet<String>), Any(Arc<dyn Fn(&ToolCall) -> bool + Send + Sync>) }` with `matches(&ToolCall)`.
- `BaseInterruptRail` — `new(kind, owner, message)`, `with_tools(ToolFilter)`, `with_payload_schema(Value)`, `with_auto_confirm_key(&str)`; implements `AgentRail::before_tool_call`: matching calls consult `ctx.extra` seeded answers (`_resume_interrupt_answers`) and auto-confirm (`_interrupt_auto_confirm`), then approve / skip-tool (reject feedback or ask-user text) / `ctx.request_interrupt(InterruptRequest::for_tool_call(..))`.
- `ConfirmInterruptRail` — kind `confirm`, message `"Please approve or reject?"`, payload schema `{approved, feedback, auto_confirm, persist_allow}`, `auto_confirm_key = tool name`; `::new()`, `::for_tools(names)`.
- `AskUserRail` — kind `ask_user`; the question comes from the tool call `question` argument (fallback to the configured message); an answer becomes the tool result text without invoking the tool.

## Constraints

- An unanswered call only records an `InterruptRequest` on the context; the loop owns state persistence and the interrupt output.
- Answers are matched by tool-call id first, then by interrupt id (Python `match_answers` precedence).
- Approve → `Ok(())` so the tool runs; reject → `ctx.skip_tool_with(feedback)` with the Python default `"User feedback: rejected\n action"`.
- A malformed answer re-asks with a fresh `interrupt_id` (never silently approves).
- No harness dependencies; owner strings default to the rail name (`openjiuwen.agent.rails.interrupt.<Rail>`).

## Test Strategy

`../../../tests/interrupt.rs`: confirm pause produces `result_type = interrupt` with one `__interaction__` frame; approve resumes and runs the tool; reject never runs it and returns feedback as the tool message; `AskUserRail` answer becomes the tool content; auto-confirm bypasses the pause; malformed answer re-asks.

## Routing Table

- `./mod.rs` → filters and the three rails
