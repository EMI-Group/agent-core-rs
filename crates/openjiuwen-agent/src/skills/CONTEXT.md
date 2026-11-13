# openjiuwen-agent skills

## Intent

Markdown-front-matter skill library the agent discovers on disk and advertises to the model.
Ports `openjiuwen/core/single_agent/skills/` (`Skill`, `SkillManager`, `SkillUtil`) onto this crate's rail seam.
A skill is metadata plus the body of a `SKILL.md` document; the crate never executes skill content — the agent reads it with `read_file`.
Remote skill download (`RemoteSkillUtil`, GitHub) is not ported.

## API Surface

- `Skill` — `name`, `description`, `version`, `keywords`, `path`, `body`, `metadata: BTreeMap<String, Value>`; serde snake_case with `default`.
- `Skill::new/with_body/with_keywords/with_metadata/with_path`, `matches(&str)` (case-insensitive name/description/keyword substrings), `render()` (the `# name` Markdown block).
- `Skill::from_markdown` / `Skill::to_markdown` — YAML front matter (`name`/`description` required, unknown keys into `metadata`) plus body, or a first `# ` heading when there is no front matter; `to_markdown` round-trips and never writes `path`.
- `SkillFrontMatter` — the serialized front-matter shape; `keywords` defaulted, `metadata` flattened.
- `SkillManager` — `new/with_skills/add/remove/get/len/is_empty/list/search`, `async load_dir/load_file/discover` over `tokio::fs`; the registry is a `BTreeMap` so `list`/`search` are name-sorted, and a duplicate name replaces the earlier skill with a `tracing::warn!`.
- `SkillUtil` — namespace type: `render_catalog`, `render_prompt(&[Skill], max_chars)`, `select(&[Skill], query)`.
- `SkillRail` — `AgentRail` named `skill` (priority `DEFAULT_RAIL_PRIORITY`, configurable) inserting the rendered prompt as the first system message on `BeforeModelCall`.
- `SKILL_PROMPT_METADATA_KEY` (`_openjiuwen_skill_prompt`), `DEFAULT_SKILL_PROMPT_BUDGET`, `SKILLS_HEADING`, `TRUNCATION_MARKER` constants.

## Constraints

- `openjiuwen_common::error::{Error, Result}` only: validation failures are `Error::Validation`, filesystem failures `Error::SysOp`; error messages name the source file or document.
- Rendering budgets are measured in characters, not bytes.
- `SkillRail` never fails a run: an empty library or an empty rendering leaves `ctx.messages` untouched, and a message carrying `SKILL_PROMPT_METADATA_KEY` blocks a second injection.
- No stdout, no `unwrap`/`expect` outside tests, no new dependencies (`serde_yaml_ng`, `tokio`, `tracing`, `tempfile` are already declared).
- Only this directory's files are owned here; `src/lib.rs` declares `pub mod skills;`.

## Routing Table

- `./skill.rs` → `Skill` and `SkillFrontMatter`, front-matter/Markdown parsing and rendering
- `./manager.rs` → `SkillManager` registry and directory/file discovery
- `./util.rs` → `SkillUtil` catalog/prompt rendering, truncation, query selection
- `./rail.rs` → `SkillRail` and the prompt-marker/budget constants
