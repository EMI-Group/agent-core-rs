# `spec` — file-based DeepAgent templates

## Intent

Ports the Python `openjiuwen.harness.resources` package, `extension_binder.py` and the `DeepAgentSpec` parts of `factory.py`/`deep_agent.py` as a **data-only** agent-template and plugin loader.
A template describes an agent (identity, prompt, model hints, tools, skills, sub-agents, rails, workspace layout); loading turns it into validated harness parts and then into a `DeepAgent` through the existing builder/factory.
Loading reads files only: no dynamic-library loading, no process spawn, no network, no template/code evaluation — a tool is chosen by its model-facing name via `BuiltinTool::from_name`.

## API Surface

- `DeepAgentSpec` — serde template (`yaml`, `json`, or Markdown with YAML front matter); `from_path`, `parse`, `validate`, `label`, `with_source_path`; `source_path` is skipped by serde.
- Field types: `ModelSpec`, `PromptFragment` (default priority `100`), `SkillSpec` (default `enabled: true`), `SubAgentSpec`, `WorkspaceHints`, `SpecFormat`.
- `KNOWN_RAILS` (accepted rail names) and `SUPPORTED_SPEC_EXTENSIONS` (`yaml`, `yml`, `json`, `md`, `markdown`).
- `SpecRegistry::{discover, from_specs, get, entry, entries, names, resolve}` with `SpecEntry { name, path, source, spec }` and `SpecSource::{Workspace, Home, Builtin}`; `AGENTS_DIR = ".openjiuwen/agents"`.
- `PluginManifest` (`from_path`, `validate`, `label`), `PluginRegistry::{load, resolve, get, plugins, len, is_empty}`, `ResolvedPlugin`, `ResolvedPlugins::{tools, rails, skill_roots, specs, plugins, len, is_empty}`, `load_plugins`, `PLUGIN_MANIFEST_FILENAME = "plugin.yaml"`.
- `ResolvedSpec` (+ `ResolvedSkill`) — `workspace`, `enabled_skill_roots`, `to_config`, `supplemental_tools`, `into_builder`; `resolve`, `resolve_with_source`, `SpecBuildOptions`, and async `build_agent`.
- `BUILTIN_RULES_YAML` and `builtin_specs()` — the embedded catalog, parsed once into `DeepAgentSpec` values.

## File Formats

Agent template (`<name>.yaml|yml|json|md`), snake_case, unknown fields ignored, every field optional:

```yaml
id: team.solver              # defaults to a generated id
name: solver                 # defaults to the file stem
description: solves tasks
language: en                 # defaults to the workspace language
system_prompt: You solve tasks.
prompt_sections:             # rendered by ascending priority, then document order
  - {name: rules, content: Never edit tests., priority: 10}
model: {provider: openai, model: gpt-test, api_base: http://localhost, temperature: 0.2, top_p: 0.9, max_tokens: 512}
tools: [bash, read_file, web_search]   # names resolved by BuiltinTool::from_name
skills:
  - {root: ../skills, enabled: true}   # relative to the template directory
subagents:
  - {name: helper, preset: explore, description: explores, tools: [read_file, grep], model: {model: child}, max_iterations: 8}
rails: [security, skills, subagent]    # core.<name> aliases accepted
workspace: {root: ./, language: en, directories: [notes]}
prompt_mode: full            # full | minimal | none
max_iterations: 12
task_loop: false             # alias: enable_task_loop
restrict_to_work_dir: true
metadata: {owner: platform}
```

A Markdown template is a `---`-fenced YAML header plus a body; the body becomes `system_prompt` (front-matter splitting reuses `skills::parse_skill_front_matter`).

Plugin package `<workspace>/.openjiuwen/plugins/<name>/plugin.yaml`:

```yaml
name: acme                   # defaults to the package directory name
version: "0.1.0"
description: acme helpers
tools: [read_file, web_search]
rails: [security]
skills: [skills]             # package-relative directory, must exist
spec: agent.yaml             # optional package-relative agent template
metadata: {owner: platform}
```

Every plugin path is package-relative (`is_safe_relative_path`); a manifest can never point outside its own directory.
Discovery reads only `plugin.yaml`; `PluginManifest::from_path` also accepts `.yml`/`.json` when used directly.

## Precedence

Discovery order is `<workspace>/.openjiuwen/agents/`, each supplied `<home>/.openjiuwen/agents/` in argument order, then the embedded built-ins; files inside a directory are sorted, so ordering is deterministic.
`workspace > home > built-in`: a file template shadows a same-named built-in silently (debug log).
Two *file* templates with the same name are a typed validation error naming the winner and the loser; re-discovering the identical path is idempotent.
`resolve` resolves relative skill roots against the template's own directory, falling back to the workspace root for a built-in template.

## Constraints

- Never mount the builder's `BuiltinTool::DEFAULT` twice: `ResolvedSpec::supplemental_tools` returns only *extended* tools, a documented deviation from an exclusive tool list.
- `build_agent` only assembles the documented 15 `create_deep_agent` arguments (and `into_builder` the `DeepAgentBuilder` inputs); neither factory signature changes.
- Rails: `security`, `memory`, `skills`, `subagent`, `evolution`, `harness_prompt`, `permission_interrupt`; `core.<name>` aliases normalize to these; unknown names are validation errors listing the known set.
- Security rail: enabled when `security` is declared, disabled when a non-empty rail list omits it, left at default for an empty list.
- Sub-agents accept a preset (`code`, `explore`, `plan`, `research`, `verification`, or their `*_agent` names) or a description; duplicates and unknown presets are validation errors. Feature-gated presets stay unavailable without their cargo feature.
- Errors use `openjiuwen_common::error::{Error, Result}` only (`sys_op` for IO, `validation` for parse/validation, `not_found` for an unknown template); messages name the offending tool/rail/template and file.
- No plugin executes anything: an unknown tool name, rail, unsafe path, missing skill root or missing template is a typed error, never a silent no-op.
- Files stay ≲1000 lines; tests are offline, use `tempfile` and `ScriptedModel`, and never touch the network or the real home directory.

## Routing Table

- `./mod.rs` → module docs, submodule declarations and the public re-exports (plus `SpecRegistry::resolve`).
- `./spec.rs` → `DeepAgentSpec` and its field types, format detection, parsing, validation, rail normalization.
- `./registry.rs` → `SpecRegistry`, `SpecEntry`, `SpecSource`, discovery precedence and duplicate reporting.
- `./plugin.rs` → `PluginManifest`, `PluginRegistry`, `ResolvedPlugin`, `ResolvedPlugins`, package-relative resource checks.
- `./resolve.rs` → `ResolvedSpec`, `ResolvedSkill`, `resolve`/`resolve_with_source`, `SpecBuildOptions`, `build_agent` hand-off.
- `./resources.rs` → `BUILTIN_RULES_YAML` (`include_str!`) and the cached `builtin_specs()`.
- `./resources/builtin_rules.yaml` → the embedded catalog: `deepagent`, `explorer`, `reviewer`.
- `./tests/` → parse, registry, resolve, plugin, builtin-catalog and end-to-end suites (`support.rs` holds the temp-dir fixtures).
