# openjiuwen-agent controller

## Intent

Rust port of the Python package `openjiuwen/core/controller/`: routing events to task executors (controller agents, an LLM router, or workflows) through intent recognition and a task scheduler.
The port lands in phases; this node currently holds **Phase A — the data model and configuration**.
Later phases add `executors` (task-executor seam), `intent` (intent recognition), `base` (event queue, scheduler, controller loop), and `legacy` (string-routed controller), and will declare those modules in `mod.rs`.

## API Surface

- `config::ControllerConfig` — behaviour knobs (intent/task budgets, history, event queue, concurrency, timeouts, intent list) with Python field names and defaults.
- `config::DefaultResponse` — optional fallback answer text.
- `schema::DataFrame` (`TextDataFrame`, `FileDataFrame`, `JsonDataFrame`) — payload frames; wire shape `{"type":"text", …}`.
- `schema::Event` / `EventType` — one flat record for Python's `InputEvent`, `TaskInteractionEvent`, `TaskCompletionEvent`, `TaskFailedEvent`, and `FollowUpEvent`.
- `schema::Intent` / `IntentType` — recognized intent plus Python's per-kind validators.
- `schema::Task` / `TaskStatus` / `TaskResult` — scheduled task, lifecycle status (including `input-required`), and executor result.
- `schema::ControllerOutputPayload` / `ControllerOutputChunk` / `ControllerOutput` / `OutputKind` — streaming and batch controller output, plus the `processing` and `all_tasks_processed` sentinels.
- `schema::frames_text` — joins the text and JSON frames of a slice.

## Constraints

- Every public type is serde-serializable with Python wire names (`mimeType`, `input-required`, `controller_output`, `*_task`) and tolerates unknown fields.
- Serde defaults match `Default` for every type, so a partial JSON object deserializes.
- Validators mirror Python: `FileDataFrame` needs exactly one content source, `Intent` enforces its per-kind fields and confidence range, `Task` enforces ids/priority/parent and status-specific fields.
- Errors are `openjiuwen_common::error::{Error, Result}`; no panics, no unwrap/expect outside tests.
- `ControllerOutputChunk` bridges to `openjiuwen_session::stream::StreamItem` under the `controller_output` frame type.
- This node owns only its own files; `crate::controller` is declared from `src/lib.rs`, which this node does not edit.

## Routing Table

- `./config.rs` → `ControllerConfig` and `DefaultResponse`
- `./schema/` → controller data model (`schema/CONTEXT.md`)
- `./schema/dataframe.rs` → payload frames and `frames_text`
- `./schema/event.rs` → events and `EventType`
- `./schema/intent.rs` → intents and `IntentType`
- `./schema/task.rs` → tasks, statuses, and results
- `./schema/output.rs` → streaming/batch controller output
