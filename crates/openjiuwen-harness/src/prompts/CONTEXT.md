# prompts

## Intent

Assemble the DeepAgent system prompt from ordered bilingual sections: identity, safety, tools, skills, memory, workspace, project context, task loop, subagent tools, completion signal, mode instructions, todo plan and environment.
Also owns the model-facing tool description table, the prompt report (size/token accounting) and the interpolation sanitizers.
Ports `openjiuwen/harness/prompts/` (builder, report, sanitize, `sections/`, `tools/`) plus the section half of `openjiuwen/core/single_agent/prompts/builder.py`.
Everything here is pure data assembly: no IO, no async, no filesystem access.

## API Surface

`builder.rs`:
- `pub use crate::schema::PromptMode` (re-export; defined in `crate::schema::mode`).
- `DEFAULT_LANGUAGE: &str = "cn"`, `SUPPORTED_LANGUAGES: [&str; 2] = ["cn", "en"]`, `resolve_language(Option<&str>) -> String`.
- `SectionName` — associated string constants: `IDENTITY`, `SAFETY`, `TOOLS`, `SKILLS`, `MEMORY`, `WORKSPACE`, `CONTEXT`, `TASK_LOOP`, `SUBAGENT_TOOLS`, `COMPLETION_SIGNAL`, `MODE_INSTRUCTIONS`, `TODO`, `ENVIRONMENT`.
- `PromptCategory { SystemPrompt, Tools, Skills, Memory }` with `as_str`, `parse`, `display_name(language)`, `ORDER` (the fixed report order).
- `PromptSection { name: String, content: BTreeMap<String, String>, priority: i32, category: Option<PromptCategory>, carrier: String }`:
  `new(name, content, priority)`, `bilingual(name, cn, en, priority)`, `with_category`, `with_carrier`, `render(language) -> String`, `char_count(language) -> usize`, `is_empty(language) -> bool`.
  `render` fallback is exactly `content[language]` → `content["cn"]` → first value → `""`.
- `HarnessPromptBuilder`:
  `new(language, mode)`, `default()` (= cn + `PromptMode::Full`), `language()`, `mode()`, `with_mode(mode)`,
  `add_section(PromptSection) -> Result<()>` (name-keyed upsert; replacing an existing section with a different `priority` is `Error::validation`),
  `remove_section(name) -> Option<PromptSection>`, `get_section(name) -> Option<&PromptSection>`, `has_section(name) -> bool`, `sections() -> Vec<&PromptSection>` (insertion order),
  `build() -> String` (mode-filtered, ascending priority, stable insertion-order ties, `"\n\n"` join, blank renders dropped),
  `build_report() -> PromptReport`.
- `MINIMAL_SECTIONS: &[&str] = &[IDENTITY, SAFETY, SKILLS, TOOLS, MEMORY]` — the subset rendered in `PromptMode::Minimal`.
- `PromptMode::None` renders the identity section text only, raw (no join, no filtering, `""` when absent).

`sections.rs` — factories return a `PromptSection`; all content is inline and bilingual (cn + en):
- `identity()`, `safety()`, `tools_section(&[openjiuwen_foundation::ToolCard])`, `skills_section(&[crate::skills::Skill])`,
  `memory_section()`, `workspace_section(&crate::workspace::Workspace)`, `context_section(&[(String, String)])` (file name, loaded content; blank content skipped),
  `task_loop_section()`, `subagent_section()`, `completion_signal()`, `completion_signal_with(promise)`,
  `mode_instructions(AgentMode)`, `todo_section(&[crate::schema::TodoItem])`,
  `environment_section(cwd: &str, model: &str, date: &str, git_branch: Option<&str>)`.
- `DEFAULT_COMPLETION_PROMISE: &str = "TASK_COMPLETE"` and `promise_tag(promise) -> String` producing `<promise>{promise}</promise>`.
- A section whose data is empty (no tools, no skills, no todos) still carries its usage guidance; `mode_instructions(AgentMode::Normal)` returns an empty section so `build` drops it.

`tool_descriptions.rs`:
- `ToolDescription { name: &'static str, description_cn: &'static str, description_en: &'static str }`, `TOOL_DESCRIPTIONS: &[ToolDescription]`.
- `tool_names() -> &'static [&'static str]`; `description(name, language) -> Option<&'static str>`; `validate_all() -> Result<()>`.
- `build_tool_card(name, tool_id, language, input_params: serde_json::Value) -> Result<ToolCard>` — description from the table, JSON schema supplied by the caller.
- Exactly these 23 names: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `list_files`, `grep`, `todo_create`, `todo_list`, `todo_get`, `todo_modify`, `ask_user`, `task_tool`, `subagent_spawn`, `subagent_wait`, `subagent_list`, `subagent_send_input`, `subagent_close`, `subagent_resume`, `memory_search`, `memory_get`, `write_memory`, `read_memory`.

`report.rs`:
- `SectionInfo { name, priority, char_count, category }`, `CategoryInfo { category, display_name, sections, char_count, estimated_tokens }`,
  `PromptReport { total_chars, estimated_tokens, section_count, sections, mode, language, categories }`.
- `PromptReport::from_sections(mode, language, &[&PromptSection])`, `chars_per_token(language) -> f64`, `summary() -> String` (`[PromptReport] mode=… lang=… sections=… chars=… est_tokens≈…`).
- Categories are emitted in the fixed order `system_prompt`, `tools`, `skills`, `memory`; sections with `category: None` count as `SystemPrompt`; empty groups are omitted.
- Token heuristic: `chars / 2.5` for `"cn"`, `chars / 4.0` otherwise, truncated to `usize`.

`sanitize.rs`:
- `DEFAULT_MAX_USER_CONTENT_LEN: usize = 2000`.
- `sanitize_path(&str) -> String` and `sanitize_user_content(&str) -> String` / `sanitize_user_content_with_limit(&str, max_len) -> String`.
- Strips `<`, `>`, `{`, `}`, `[`, `]`, `` ` ``, `$`, any run of three or more dots, and the literal two-character sequences `\n` / `\r`; then truncates to `max_len` characters with no ellipsis.

## Section catalog (name → priority → category → factory)

- `identity` → 10 → `SystemPrompt` → `identity()` (both languages in one section, no heading).
- `safety` → 20 → `SystemPrompt` → `safety()`.
- `tools` → 30 → `Tools` → `tools_section(&[ToolCard])`.
- `skills` → 40 → `Skills` → `skills_section(&[Skill])`.
- `memory` → 50 → `Memory` → `memory_section()`.
- `workspace` → 70 → `Memory` → `workspace_section(&Workspace)`.
- `task_loop` → 80 → `SystemPrompt` → `task_loop_section()`.
- `subagent_tools` → 81 → `SystemPrompt` → `subagent_section()`.
- `completion_signal` → 83 → `SystemPrompt` → `completion_signal()`.
- `mode_instructions` → 84 → `SystemPrompt` → `mode_instructions(AgentMode)`.
- `environment` → 90 → `SystemPrompt` → `environment_section(...)`.
- `todo` → 92 → `SystemPrompt` → `todo_section(&[TodoItem])`.
- `context` → 100 → `Memory` → `context_section(&[(String, String)])`.

## Constraints

- No IO, no async, no `std::env` reads, no filesystem; everything is data in and data out.
- Must not depend on `crate::tools`; tool cards and todo items arrive as values. May depend on `crate::schema`, `crate::workspace`, `crate::skills` and `openjiuwen_foundation::ToolCard`.
- Errors: `openjiuwen_common::error::{Error, Result}` only; no panics, no `unwrap`/`expect` outside tests.
- Section content is a `BTreeMap` keyed by language; injection slots are formatted in Rust, never `str.format`, so literal braces must survive untouched.
- `add_section` is the only mutation path that can fail; it never overwrites a section's priority silently.
- `build` is pure and idempotent; no caching, no interior mutability.
- Logging: `tracing` only (debug for add/remove), no stdout.
- Files ≤1000 lines; public items documented; no `unsafe`; workspace clippy pedantic is enforced with `-D warnings`.

## Design Decisions

- Section names, priorities and category order are settled in the catalog above; the harness `agent`/`rails` modules build their prompt by calling these factories and adding the result to the builder.
- One `tools` section carries both the tool usage rules and the model-facing tool list (Python has two producers on the same name/priority, one of which silently replaced the other).
- The completion signal is the Python `<promise>…</promise>` tag with `DEFAULT_COMPLETION_PROMISE` (`TASK_COMPLETE`) as the default token; Python requires a caller-supplied promise, the Rust default makes the marker usable without configuration.
- Plan mode instructions are a condensed five-phase workflow (understand, design, review, write plan, exit) rather than Python's full manual; only `AgentMode::Plan` emits text.
- `PromptCategory` is an enum instead of Python's free-form string; categories only drive report grouping.
- `environment` is bilingual here, although Python's CLI builder emits English only.

## Routing Table

- `./builder.rs` → `HarnessPromptBuilder`, `PromptSection`, `PromptCategory`, `SectionName`, `PromptMode` re-export, language resolution
- `./sections.rs` → every section factory and its bilingual content, completion promise marker
- `./tool_descriptions.rs` → bilingual description table for the 23 implemented tools, `build_tool_card`
- `./report.rs` → `PromptReport`, category grouping, token heuristic
- `./sanitize.rs` → `sanitize_path`, `sanitize_user_content`
- `./mod.rs` → module declarations and flat re-exports

## Test Strategy

- Builder: ascending priority order, stable insertion-order ties, blank sections dropped, upsert by name, priority change rejected with `ErrorKind::Validation`, `remove_section`, mode filtering (`Full`/`Minimal`/`None`), language fallback chain including unknown language.
- Sections: exact headings and working-directory sentence, empty tools/skills omit content but keep guidance, plan mode non-empty and normal empty, completion tag text, todo rendering uses `status_icon` + id + content, environment lists cwd/model/date/branch with `N/A` fallback.
- Tool descriptions: `tool_names()` has exactly 23 unique names, `description` returns cn and en for all, unknown name → `None`, `validate_all()` passes and `build_tool_card` sets name/description/schema.
- Report: fixed category order, `None` category folded into `system_prompt`, cn vs en token divisor, empty-group omission, summary format.
- Sanitize: each stripped character class, 3+ dot runs, literal `\n`/`\r`, char-boundary truncation on multi-byte input, no ellipsis.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --no-default-features`, `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`.
