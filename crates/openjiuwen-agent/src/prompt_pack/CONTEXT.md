# prompt_pack

## Intent

Agent prompt configuration in two additive forms over [`crate::ReActAgent`]: a Rust-designed YAML **prompt pack**, and the Python SDK's on-disk **prompt directories**.
Both map onto the same runtime structures (`SectionSpec`, `PromptSectionsRail`, `PromptPack::apply_to`), and neither introduces new runtime behaviour — they only set values the agent already understands through `configure`, `set_tool_exposure_policy`, `set_mcp_tool_allowlist`, `set_tool_resilience`, and `register_rail`.

## API Surface

- `pack.rs` — `PromptPack` (`id`, `name`, `description`, `prompts`, `tools`, `rails`, `metadata`), `PromptSpec`, `SectionSpec` (default priority 50), `ToolSpec`, `AppliedPack`, `PromptPackContext`.
- `PromptPack::from_yaml` / `from_yaml_file` (async) / `to_yaml` / `rail_specs` / `to_tool_policy` / `identity_message` / `apply_to` / `apply_to_with`.
- `rails.rs` — `RailSpec` internally tagged by `type`: `guardrail`, `confirm_interrupt`, `ask_user`, `skill { budget }`, `model_backup`, `custom { name, priority, options }`; `RailSpec::kind` and `RailSpec::build(&PromptPackContext)`.
- `rail.rs` — `PromptSectionsRail` (`name() = "prompt_sections"`), `PROMPT_SECTIONS_METADATA_KEY`, `PROMPT_SECTIONS_RAIL_NAME`.
- `directory.rs` — `PromptDirectory` (`root`, `sections`), `DirectorySection` (`name`, `priority`, `variants`), `PromptFile` (`path`, `name`, `language`, `priority`, `body`, `meta`), `PromptFileMeta`; `PromptDirectory::load` (async), `section`, `len`/`is_empty`, `section_specs(language)`, `prompt_sections(language)`, `render(language)`, `pack(id, language)`.
- `DirectorySection::variant` / `text` / `spec` / `prompt_section` / `languages` / `kind` / `is_evolved`; `PromptFile::body_sha256` / `is_evolved` / `render(&BTreeMap<String, Value>)`.
- Directory helpers re-exported from `mod.rs`: `read_frontmatter`, `body_sha256`, `is_evolved`, `normalize_language`, `DEFAULT_PROMPT_PRIORITY` (100), `SUPPORTED_LANGUAGES` (`["cn", "en"]`), `FALLBACK_LANGUAGES` (`["en", "cn"]`).
- `PromptPack::from_prompt_dir(path, language)` (async) and `PromptPack::from_prompt_directory(&dir, id, language)` fill only `prompts.sections`; the pack id is the directory's file name (or `"prompt_pack"`).

### Directory layouts

- Flat language directories: `{cn,en}/<name>.md`; a `zh` path component aliases `cn`, and the file stem is the section name.
- Suffixed files anywhere below the root: `<name>.<lang>.md`, including `system/`, `identity/`, and `tool/<domain>/` subdirectories; the final two-letter dot segment is stripped from the name.
- A language is chosen from the front-matter `language`, else the `.<lang>` suffix, else the first path component naming `cn`/`en`/`zh`; an unqualified file is a variant with no language.
- Section text resolves by language with the chain *requested → `en` → `cn` → first variant by relative path*.
- Sections order by priority ascending, then name; variants order by relative path.
- Duplicate section name **and** language: the higher priority wins; on a tie the lexicographically later relative path wins.
- Hidden entries (leading `.`), symbolic links, non-`.md` files, and non-UTF-8 names are skipped; an empty directory loads with no sections.
- Errors: [`openjiuwen_common::error::Error::SysOp`] for a missing/unreadable path, `Error::Validation` for a non-directory path, malformed front matter, a non-mapping front-matter root, a wrong-typed known field, or an invalid `priority`.

### Front matter

- Optional YAML block delimited by `---` lines; a document without a leading `---` line, or with no closing `---`, is a hand-written body with empty metadata.
- Fields: `name`, `language`, `priority`, `kind`, `evolved`, `baseline_sha256`, `updated_at`; unknown keys are kept in `PromptFileMeta::extra`.
- `priority` defaults to 100 and accepts an integer or a decimal string; anything else is a `Validation` error naming `priority`.
- Newlines are normalized to `\n`; the body after the closing `---` line is kept byte-faithfully (trailing newlines preserved) so `body_sha256` round-trips.
- Evolved-ness is computed from the body hash: a missing `baseline_sha256` or a mismatch with `body_sha256(body)` means evolved; the declared `evolved` flag is informational only.
- `{{var}}` placeholders are never rendered at load time; `PromptFile::render` performs the single-pass substitution (unknown placeholders stay verbatim).

## Constraints

- The pack is plain data: it never owns runtime state and can be applied to several agents or re-serialized.
- Applying preserves every config field except `prompt_template`: the pack template replaces it when non-empty, otherwise a synthesized identity message is prepended; a directory-derived pack sets no template, name, or description.
- Rails are built only from `PromptPackContext`; a spec the context cannot satisfy is reported in `AppliedPack::deferred_rails`, never dropped silently.
- `custom` rails are always deferred, `skill` requires `PromptPackContext::skills`, `model_backup` requires at least one `PromptPackContext::models` entry.
- `guardrail` builds `GuardrailRail` over an empty `GuardrailPipeline` (a pack cannot describe rules); hosts replace it with a configured rail.
- `PromptSectionsRail` injects one marked `system` message at the front of the window, at most once per window, and never fails the run; section text is trimmed when a spec/section is built, while `PromptFile::body` stays raw for hashing.
- YAML pack parsing semantics are unchanged; the directory loader is additive and has no Python `PromptPack` counterpart.

## Notes for Agents

- Tests are inline: `pack/tests.rs` (schema, parse/serialize, apply, policy, rails, sections, confirm interrupt), `rail.rs` (section rendering/idempotency), `rails.rs` (spec parsing and rail construction), `directory/tests.rs` (both layouts, front matter, hash/evolution, duplicates, ordering, fallbacks, applying to an agent), `directory/front_matter.rs` (framing, typed errors, evolution), `directory/sha256.rs` (FIPS vectors).
- The model-facing effects of the tests are asserted through `ScriptedModel::last_call()` (`request.tools`, `messages`), so the suite stays offline and deterministic.
- `pack/tests.rs` is a child module of `pack.rs` (`#[cfg(test)] mod tests;`) only to keep `pack.rs` small; it is not a separate public module.
