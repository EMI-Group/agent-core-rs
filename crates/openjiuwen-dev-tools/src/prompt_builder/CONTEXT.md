# prompt_builder

## Intent

Ports `openjiuwen.dev_tools.prompt_builder`: standardize a prompt, rewrite it from user feedback, and repair it from evaluated counterexamples.
Three builders sit on two bundled template banks (`zh-CN` default, `en-US` alternative) whose texts are part of the observable contract.
Deterministic transforms (flattening, splicing, rendering, parsing) are ordinary public functions; a `Model` only supplies text.
The builders take `Arc<dyn Model>` instead of Python's `(ModelRequestConfig, ModelClientConfig)` pair, so configuration stays with the caller.

## API Surface

- `Language` — `ZhCn` (default) / `EnUs`; serde tags `"zh-CN"` / `"en-US"`, `Display`, `as_str`, and a strict `FromStr` (canonical tags case-insensitively plus `zh`/`zh_cn`/`zhcn`/`en`/`en_us`/`enus`).
- `templates::PromptBank` — the eleven template texts of one language; `bank(Language)`, `system_template(&str)`, `user_template(&str)`.
- `MetaTemplateBuilder` — `new(Arc<dyn Model>)`, `build(&str, &MetaBuildOptions)`, `stream_build(impl Into<String>, MetaBuildOptions)`, `register_meta_template`, `get_meta_template`, `pop_meta_template`, `registered_names`.
- `TemplateType` — `General` (default) / `Plan` / `Other`, serde snake_case; `MetaBuildOptions { tools, template_type, custom_template_name, language }`.
- `FeedbackPromptBuilder` — `new`, `build(&str, &FeedbackOptions)`, `stream_build(String, FeedbackOptions)`.
- `FeedbackMode` — `General` (default) / `Insert` / `Select`; `FeedbackOptions { feedback, mode, start_pos, end_pos, language }`.
- `BadCasePromptBuilder` — `new`, `build(&str, &[EvaluatedCase], Language)`, `stream_build(String, Vec<EvaluatedCase>, Language)`.
- `MAX_CASES_LIMIT = 10`, `INSERT_STR = "[用户要插入的位置]"`, `JSON_STRING_MAX_LENGTH = 10000`.
- `utils` — `get_string_prompt`, `insert_marker`, `select_range`, `render_bad_cases`, `FeedbackIntent`, `parse_feedback_intent`, `parse_bad_case_analysis`.
- `IntoMetaTemplate` — conversion bound of `register_meta_template` (implemented for `&str`, `String`, `PromptTemplate`).

## Prompt Banks

`templates/zh_cn.rs` and `templates/en_us.rs` hold the eleven Python constants verbatim (UPPER_SNAKE names, `pub const … : &str`) plus `BANK`.
Texts are byte-identical to the Python `"""…"""` literals, including the leading and trailing newline; placeholders are `{{name}}`.
`templates/mod.rs` maps them onto `PromptBank` fields `meta_general_system`, `meta_general_user`, `meta_plan_system`, `meta_plan_user`, `feedback_intent`, `feedback_general`, `feedback_select`, `feedback_insert`, `bad_case_analyze`, `bad_case_optimize`, `format_bad_case`.

Placeholder keys per text:
- `meta_general_user` — `{{instruction}}`; `meta_plan_user` — `{{instruction}}`, `{{tools}}`; the two system texts have no placeholders.
- `feedback_intent` — `{{original_prompt}}`, `{{feedbacks}}`; `feedback_general` and `feedback_insert` — `{{original_prompt}}`, `{{suggestion}}`.
- `feedback_select` — `{{original_prompt}}`, `{{pending_optimized_prompt}}`, `{{suggestion}}`.
- `bad_case_analyze` — `{{original_prompt}}`, `{{bad_cases}}`; `bad_case_optimize` — `{{original_prompt}}`, `{{feedback}}`.
- `format_bad_case` — `{{question}}`, `{{label}}`, `{{answer}}`, `{{reason}}`, and it is the one text identical in both banks.

## Behavior

`MetaTemplateBuilder::build` makes exactly one call with `ModelRequestConfig::new(model.model_name())`.
`General` sends the general system text plus the general user text rendered with `{{instruction}}`; tools are never interpolated there.
`Plan` sends the plan system text plus the plan user text rendered with `{{instruction}}` and `{{tools}}` (compact JSON of the `ToolInfo` array, or the literal `None` when the list is empty).
`Other` renders the template registered under `custom_template_name` with `{{instruction}}` (and `{{tools}}`, which templates without that slot ignore).

`FeedbackPromptBuilder::build` makes exactly two calls for every mode.
Call 1 renders `feedback_intent` with `{{original_prompt}}` (always the untagged prompt) and `{{feedbacks}}`; `parse_feedback_intent` supplies the phrasing, and a missing/false intent or a blank `optimized_feedback` falls back to the raw feedback.
Call 2 renders `feedback_general` (`{{original_prompt}}` = prompt), `feedback_insert` (`{{original_prompt}}` = prompt with `INSERT_STR` spliced at `start_pos`) or `feedback_select` (`{{original_prompt}}` = prompt, `{{pending_optimized_prompt}}` = the selected fragment).

`BadCasePromptBuilder::build` makes exactly two calls.
Call 1 renders `bad_case_analyze` with `{{original_prompt}}` and `{{bad_cases}}` from `render_bad_cases`; `parse_bad_case_analysis` returns the `<intent>` flag and the last `<summary>` (or the trimmed whole answer) as `{{feedback}}` for call 2, which renders `bad_case_optimize`.

`stream_build` on all three runs the same assembly on a `'static` owned `async_stream`, then forwards only non-empty text deltas as `Ok(String)`; usage/done/empty deltas are skipped, model errors are yielded once, and an assembly error is the single item.
Feedback and bad-case streaming perform the normalization/analysis stage as a non-streamed `invoke`, so the stream always shows only the final answer.

## Constraints

- Only this directory may be edited here; `lib.rs`, `Cargo.toml`, `src/util.rs`, `src/tune/`, sibling modules, and `docs/` are out of scope.
- Errors: `openjiuwen_common::error::{Error, Result}`; `Error::validation` for user input, `Error::llm` for unusable model output; no panics outside tests.
- Positions are character indices (`start_pos`/`end_pos`), never byte offsets.
- Public data types are serde with snake_case and unknown-field tolerance; `PromptBank` is `Copy` and not serde.
- No `unsafe`, no `println!`/`eprintln!`, no `#[allow]` blankets; rustdoc on every public item; clippy `all` + `pedantic` clean under `-D warnings`.
- Files ≤1000 lines; the largest is `feedback.rs` at 833 lines.

## Design Decisions

- Validation replaces Python's warn-and-degrade: blank prompts/feedback, missing or out-of-range positions, missing/unregistered custom templates, empty case lists, and more than `MAX_CASES_LIMIT` cases all fail before any model call.
- An answer with no choice or blank text is `Error::Llm`; Python returned `None` or an empty string.
- `register_meta_template` keys are the caller-supplied name verbatim; Python's hidden `META_TEMPLATE_` prefix is not reproduced.
- `register_meta_template` takes `impl IntoMetaTemplate` because `openjiuwen_foundation::prompt::PromptTemplate` has no `From<&str>` (that impl lives on `TemplateContent`); the trait covers the three Python-accepted shapes.
- `render_bad_cases` renders the `inputs`/`label` maps as compact truncated JSON (Python used `str(dict)`), so output is deterministic and BTreeMap-ordered; `reason` is never truncated.
- `get_string_prompt` flattens a message list by joining each message's text with `\n` and rejects non-text content parts — the error path Python's `item.values()` fallback represented.
- `parse_feedback_intent` normalizes the `intent` flag from a JSON bool or string into the string `"true"`/`"false"`, keeps `optimized_feedback` trimmed, and accepts `optimization_directions` as a list or a bare string.

## Deferred

No part of the module is stubbed; the Python warn-and-degrade branches, `None` returns, and the `META_TEMPLATE_` name prefix are intentionally not reproduced.

## Test Strategy

Unit tests live in each file (`#[cfg(test)]`), async ones under `#[tokio::test]`, all offline and deterministic.
`ScriptedModel` scripts the model: `last_call()`/`calls()` assert the exact rendered messages and the call modes (two `Invoke`s for `build`, `Invoke` then `Stream` for `stream_build`), `ScriptedTurn::fail` covers error propagation.
Template tests prove every text is non-empty, keeps its placeholders, and loses every `{{…}}` after formatting, and that the two banks differ only where the Python sources differ.
`templates` provenance is checked by extracting the Python `"""…"""` literals and the Rust constants and comparing them byte for byte (22/22 identical).

## Routing Table

- `./templates/` → the verbatim `zh-CN`/`en-US` banks, `PromptBank`, `bank`, `system_template`, `user_template` (`templates/mod.rs`, `templates/zh_cn.rs`, `templates/en_us.rs`)
- `./language.rs` → `Language` selector, serde tags, `Display`/`FromStr`
- `./utils.rs` → prompt flattening, char-indexed insert/select, bad-case rendering, model-output parsing
- `./meta.rs` → `MetaTemplateBuilder`, `TemplateType`, `MetaBuildOptions`, custom-template registry
- `./feedback.rs` → `FeedbackPromptBuilder`, `FeedbackMode`, `FeedbackOptions`
- `./bad_case.rs` → `BadCasePromptBuilder`, `MAX_CASES_LIMIT`
- `./mod.rs` → module docs, shared constants, re-exports
- `../tune/` → `EvaluatedCase`/`Case` consumed by bad-case rendering and the bad-case builder (sibling — read-only, escalate writes to parent)
- `../util.rs` → `extract_json_fence`, `extract_tag`, `extract_bool_tag`, `truncate_json` (sibling — read-only, escalate writes to parent)
