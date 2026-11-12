# prompts

## Intent

Assemble the DeepAgent system prompt from ordered, bilingual sections: identity, safety, tools, skills, memory, workspace/environment, task loop, todo/plan formatting and mode instructions; plus the packaged tool descriptions/cards and prompt reporting/sanitization.
Ports `openjiuwen/harness/prompts/` and the section half of `openjiuwen/core/single_agent/prompts/`.

## API Surface

- `PromptMode { Full, Minimal, None }` (`full`/`minimal`/`none`); `None` renders identity only.
- `SectionName` — string constants: `IDENTITY`, `SAFETY`, `TOOLS`, `SKILLS`, `MEMORY`, `WORKSPACE`, `TASK_TOOL`, `SUBAGENT_TOOLS`, `COMPLETION_SIGNAL`, `MODE_INSTRUCTIONS`, `TODO`, `CONTEXT`, `ENVIRONMENT`.
- `PromptSection { name, content: BTreeMap<String, String>, priority: i32, category: Option<String>, carrier: String }` with `render(language)` (language → `cn` → first value → `""`).
- `HarnessPromptBuilder::new(language, mode)`, `add_section`, `remove_section`, `sections`, `build() -> String` (ascending priority, two-newline join, stable ties, empty dropped), `build_report() -> PromptReport`.
- `sections` factories: `identity`, `safety`, `tools_section(&[ToolCard])`, `skills_section(&[Skill])`, `memory_section`, `workspace_section(&Workspace)`, `task_loop_section`, `subagent_section`, `completion_signal`, `mode_instructions(AgentMode)`, `todo_section(&[TodoItem])`, `environment_section(cwd, model, date, git_branch)`.
- `tool_descriptions.rs`: bilingual description + JSON-schema tables for every implemented builtin tool with Python names, `build_tool_card(name, tool_id, language) -> Result<ToolCard>`, `register_tool_description`, `validate_all()`, `tool_names()`.
- `report.rs`: `PromptReport { system_prompt, tools, skills, memory }` with `categories()` fixed order and `token_estimate` heuristic (chars/2.5 cn, chars/4.0 en).
- `sanitize.rs`: `sanitize_path`, `sanitize_user_content(content, max_len=2000)`.

## Constraints

- Sections are data: no IO in `build`; content for context files is injected by the caller (agent) after reading the workspace through sysop.
- Tool descriptions are OpenAI function-calling JSON (name/description/parameters), never markdown.
- Language fallback order is exactly `language → cn → first`; default language `"cn"`.
- `add_section` upserts by name; changing the priority of an existing name is a `ValueError` equivalent (`Error::validation`).
- No dependency on `crate::tools` (takes `ToolCard`s / prebuilt tables); may depend on `crate::schema`, `crate::workspace`, `crate::skills`.
- Packaged YAML/Markdown prompt resources use `include_str!` and are parsed once.

## Routing Table

- `./builder.rs` → `HarnessPromptBuilder`, `PromptSection`, `PromptMode`, `SectionName`
- `./sections.rs` → section factories (identity, safety, tools, skills, memory, workspace, task loop, mode, todo, environment)
- `./tool_descriptions.rs` → bilingual tool descriptions and JSON schemas
- `./report.rs` → `PromptReport`
- `./sanitize.rs` → path/content sanitization
- `./resources/` → packaged markdown/YAML prompt fragments

## Test Strategy

- Unit tests: ascending priority order and stable ties, empty sections dropped, duplicate-name upsert, priority change rejected, language fallback, `PromptMode::None` identity-only, report category order/token heuristic, sanitize strips control characters and truncates.
- Parity references: `tests/unit_tests/harness/prompts/test_priority_registry.py`, `test_report.py`, `test_sanitize.py`, `test_tool_metadata.py`, `test_tools_section.py`.
