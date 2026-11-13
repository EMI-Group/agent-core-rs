# callbacks/

## Intent
Per-agent callback framework: lifecycle events, typed payloads, and a priority-ordered handler registry.
Ports `openjiuwen/core/single_agent/agent_callback_manager.py` without Python's process-global `Runner.callback_framework`; the registry is an ordinary value owned by the agent host.

## API Surface
- `event.rs` — `AgentCallbackEvent` (`invoke_start`, `step_start`, `step_end`, `tool_call_start`, `tool_call_end`, `invoke_end`) with `as_str()` and `ALL`.
- `payload.rs` — `InvokeStartPayload`, `StepStartPayload`, `StepEndPayload`, `ToolCallPayload`, `ToolCallEndPayload`, `InvokeEndPayload` (all `Default`, `snake_case`, unknown-field tolerant) and the `CallbackPayload` union with `event()`, `agent_id()`, `session_id()`, `to_value()`.
- `framework.rs` — `AgentCallback` trait (`name`, `priority` default 50, six no-op async hooks) and `CallbackFramework` (`register`, `unregister`, `clear`, `has_hooks`, `handlers`, `invoke`) plus `DEFAULT_CALLBACK_PRIORITY`.
- `mod.rs` re-exports every public item; `src/lib.rs` only declares `pub mod callbacks;`.
- `CallbackPayload::to_value()` serializes the inner payload so the wire object is the Python-shaped `{agent_id, session_id, ...}` record.

## Constraints
- `CallbackFramework::invoke` dispatches only the hook matching `payload.event()`, in stable descending priority, and stops at the first `Err`.
- Registering the same name for the same event replaces the handler; separate events keep separate name spaces.
- The manual `Debug` impl reports per-event handler counts because `dyn AgentCallback` is not `Debug`.
- No global mutable state; payloads are plain data and the framework is `Send + Sync` through `Arc` handlers.
- Tests live inline beside each module and are offline.
