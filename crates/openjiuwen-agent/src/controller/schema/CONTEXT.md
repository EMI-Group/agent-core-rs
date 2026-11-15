# openjiuwen-agent controller schema

## Intent

Payload, event, intent, task, and output types shared by every controller phase.
Ports `openjiuwen/core/controller/schema/`; Python's pydantic models become serde data types with the same wire names, plus the Rust-only fields the pipeline needs.

## API Surface

- `mod.rs` re-exports `DataFrame`, `FileDataFrame`, `JsonDataFrame`, `TextDataFrame`, `frames_text`, `Event`, `EventType`, `Intent`, `IntentType`, `Task`, `TaskStatus`, `TaskResult`, `OutputKind`, `ControllerOutputPayload`, `ControllerOutputChunk`, `ControllerOutput`, `CONTROLLER_OUTPUT_TYPE`, `TASK_PROCESSING`, `ALL_TASKS_PROCESSED`.
- `dataframe` — `DataFrame` is an internally tagged enum (`tag = "type"`; `text`/`file`/`json`) with `text`, `json`, `kind`, `as_text`, `validate`; `TextDataFrame { text }`, `FileDataFrame { name, mimeType, bytes, uri }` (`from_uri`, `from_bytes`, and a `validate` that requires exactly one of `bytes`/`uri`), `JsonDataFrame { data: BTreeMap<String, Value> }` (`from_value` wraps non-objects under `"value"`); `frames_text` joins text/JSON contributions with `"\n"` and ignores file frames.
- `event` — `EventType` (`input`, `task_interaction`, `task_completion`, `task_failed`, `follow_up`, default `input`); `Event` fields `event_type`, `event_id`, `metadata`, `input_data`, `task_result`, `interaction`, `error_message`, `task_id`; constructors `new`, `user_input`, `from_frames`, `from_user_inputs`, `task_interaction`, `task_completion`, `task_failed`, `follow_up`; `text()`, `frames()`.
- `intent` — `IntentType` (`create_task`, `pause_task`, `resume_task`, `continue_task`, `supplement_task`, `cancel_task`, `modify_task`, `switch_task`, `unknown_task`) with `as_str`/`is_task_mutating`; `Intent` fields `intent_type`, `event`, `target_task_id`, `target_task_description`, `depend_task_id`, `supplementary_info`, `modification_details`, `confidence`, `metadata`, `clarification_prompt`, plus the Rust extensions `name` and `task_type`; constructors `new`, `create_task`, `default_for`, `unknown`; accessors `label`, `description`, `parameters`; `validate`.
- `task` — `TaskStatus` (`submitted`, `working`, `paused`, `input-required`, `completed`, `canceled`, `failed`, `waiting`, `unknown`, default `unknown`) with `is_terminal`/`is_active`; `Task` fields `session_id`, `task_id`, `task_type`, `description`, `priority`, `inputs`, `outputs`, `status`, `parent_task_id`, `context_id`, `input_required_fields`, `error_message`, `metadata`, `extensions`, `created_seq`; `new`/`submitted`, builders, `validate`, `input_text`, `is_terminal`, `is_active`; `TaskResult` (`status`, `output`, `error`, `metadata`, `data`, `reasoning`) with `completed`, `text`, `failed`, `input_required`, `interrupted`, `display`, `with_reasoning`.
- `output` — `OutputKind` (`task_completion`, `task_interaction`, `task_failed`, `processing`, `all_tasks_processed`, `reasoning`, where `Reasoning` marks a task's reasoning text); `ControllerOutputPayload` (`type`, `data`, `metadata`) with `new`, `processing`, `all_tasks_processed`, `text`; `ControllerOutputChunk` (`index`, `type`, `payload`, `last_chunk`) with `new`, `last`, `to_value`, `to_stream_item`, `from_stream_item`; `ControllerOutput` (`output`, `task_id`, `status`, `data`, `frames`) with `new`, `empty`, `is_interrupt`, `to_value`, `from_value`, builders `with_task_id`, `with_data`, `with_frames`.
- Frame-kind constants — `CONTROLLER_OUTPUT_TYPE = "controller_output"`, `TASK_PROCESSING = "processing"`, `ALL_TASKS_PROCESSED = "all_tasks_processed"`.

## Constraints

- Structs use `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields; enums use `#[serde(rename_all = "snake_case")]` and reject unknown variant strings.
- Wire renames follow Python: `FileDataFrame.mime_type` → `mimeType`, `TaskStatus::InputRequired` → `input-required`, and the payload/chunk `kind` field serializes as `type`.
- `Option` fields skip `None`, `Vec`/`BTreeMap` fields skip empty, `TaskResult.output` skips `null`, and `ControllerOutputChunk.last_chunk` is emitted only when `true`.
- `Task::created_seq` is `#[serde(default, skip)]`: never serialized, always restored as `0`.
- `Default` exists everywhere except `DataFrame` and `IntentType`, which have no `Default` implementation; `Intent`, `Task`, and `ControllerOutputChunk` implement it manually with the same values as their serde defaults.
- Only `Event` mints ids: every constructor creates a UUID v4, and deserializing an event without `event_id` mints one too.
- Files stay under ~470 lines, each with an inline `#[cfg(test)] mod tests` asserting the wire shapes.

## Notes for Agents

- `Event` flattens Python's five event subclasses into one record and stores `task_id: Option<String>` where Python carries the `Task` object.
- `Event::from_user_inputs` accepts a string, a string array (one text frame per element), `{"query": ...}` (string or JSON payload), and any other object as one JSON frame, and rejects `null`/scalars with `Error::validation`.
- `Event::text()` concatenates `input_data` + `task_result` + `interaction` regardless of kind, while `frames()` selects the list by kind and returns an empty slice for `TaskFailed`.
- `Intent::validate` is explicit, not a deserialization hook: create/switch need `target_task_description`, continue needs `depend_task_id`, supplement/modify need a target plus their payload, pause/resume/cancel need a target, unknown needs a prompt, and confidence must be `0.0..=1.0`.
- `Intent::default_for` labels the intent `"default"` and takes its description from the event text; `Task::new` starts `Submitted` while `Task::default()` starts `Unknown` with priority 1.
- `ControllerOutput::empty()` is `Completed` while `Default` is `Unknown`; `is_interrupt()` means status `InputRequired`.
- `ControllerOutputChunk::to_stream_item` treats a missing payload as the `all_tasks_processed` sentinel, and `from_stream_item` always returns `last_chunk = false`, so the flag does not survive a stream round-trip.
- The sentinels travel as `OutputKind` values, never as `TaskStatus`.
- `OutputKind::Reasoning` serializes as `"reasoning"` and is distinct from the `processing`/`all_tasks_processed` sentinels.
- `TaskResult::reasoning` is additive and empty by default, so a result without reasoning stays Python-shaped on the wire.
- These types are shared as-is by the task manager, scheduler loop, queue, group seam, and reasoner, which define their own types in their own modules.

## Test Strategy

- 39 inline tests: `dataframe` 6, `event` 8, `intent` 8, `task` 8, `output` 8, `mod.rs` 1, all offline and asserting wire shapes, defaults, validation, and the sentinel constants.

## Routing Table

- `./mod.rs` → module declarations and the re-export surface
- `./dataframe.rs` → payload frames
- `./event.rs` → events
- `./intent.rs` → intents
- `./task.rs` → tasks and results
- `./output.rs` → controller output chunks, batch output, sentinels
