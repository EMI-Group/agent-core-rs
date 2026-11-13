# openjiuwen-agent controller schema

## Intent

Payload, event, intent, task, and output types shared by every controller phase.
Ports `openjiuwen/core/controller/schema/`; Python's pydantic models become serde data types with the same wire names.

## API Surface

- `dataframe` — `DataFrame` (tagged enum), `TextDataFrame`, `FileDataFrame`, `JsonDataFrame`, `frames_text`.
- `event` — `Event` (Python's event subclasses flattened into one record), `EventType`.
- `intent` — `Intent`, `IntentType`, and the per-kind validation rules.
- `task` — `Task`, `TaskStatus`, `TaskResult`.
- `output` — `OutputKind`, `ControllerOutputPayload`, `ControllerOutputChunk`, `ControllerOutput`, plus `CONTROLLER_OUTPUT_TYPE`, `TASK_PROCESSING`, `ALL_TASKS_PROCESSED`.
- `mod.rs` re-exports the whole surface for consumers.

## Constraints

- Wire names follow Python exactly (`mimeType`, `input-required`, `controller_output`); enums serialize in `snake_case`.
- Unknown JSON fields are tolerated; every type has `Default` matching its serde field defaults.
- `Task::created_seq` is a Rust-only ordering aid and is never serialized.
- Constructors mint fresh ids; `Default`/deserialization keep ids deterministic.
- Files stay around 600 lines, with inline `#[cfg(test)] mod tests` asserting the wire shapes.

## Routing Table

- `./dataframe.rs` → payload frames
- `./event.rs` → events
- `./intent.rs` → intents
- `./task.rs` → tasks and results
- `./output.rs` → controller output chunks and batch output
