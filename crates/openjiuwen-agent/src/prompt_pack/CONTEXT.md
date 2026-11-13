# prompt_pack

## Intent

YAML prompt packs: one YAML document that carries an agent's prompt template, extra system sections, tool policy, and rails, plus the code that applies it to a live `ReActAgent`.
This is an additive configuration format (no Python analogue): a pack introduces no new runtime behaviour, it only sets values `ReActAgent` already understands through `configure`, `set_tool_exposure_policy`, `set_mcp_tool_allowlist`, `set_tool_resilience`, and `register_rail`.

## API Surface

- `pack.rs` — `PromptPack` (`id`, `name`, `description`, `prompts`, `tools`, `rails`, `metadata`), `PromptSpec`, `SectionSpec` (default priority 50), `ToolSpec`, `AppliedPack`, `PromptPackContext`.
- `PromptPack::from_yaml` / `from_yaml_file` (async) / `to_yaml` / `rail_specs` / `to_tool_policy` / `identity_message` / `apply_to` / `apply_to_with`.
- `rails.rs` — `RailSpec` internally tagged by `type`: `guardrail`, `confirm_interrupt`, `ask_user`, `skill { budget }`, `model_backup`, `custom { name, priority, options }`; `RailSpec::kind` and `RailSpec::build(&PromptPackContext)`.
- `rail.rs` — `PromptSectionsRail` (`name() = "prompt_sections"`), `PROMPT_SECTIONS_METADATA_KEY`, `PROMPT_SECTIONS_RAIL_NAME`.
- A pack needs only `id`; missing id and malformed YAML are `Error::Validation`, unreadable files are `Error::SysOp`.

## Constraints

- The pack is plain data: it never owns runtime state and can be applied to several agents or re-serialized.
- Applying preserves every config field except `prompt_template`: the pack template replaces it when non-empty, otherwise a synthesized identity message is prepended.
- Rails are built only from `PromptPackContext`; a spec the context cannot satisfy is reported in `AppliedPack::deferred_rails`, never dropped silently.
- `custom` rails are always deferred, `skill` requires `PromptPackContext::skills`, `model_backup` requires at least one `PromptPackContext::models` entry.
- `guardrail` builds `GuardrailRail` over an empty `GuardrailPipeline` (a pack cannot describe rules); hosts replace it with a configured rail.
- `PromptSectionsRail` injects one marked `system` message at the front of the window, at most once per window, and never fails the run.

## Notes for Agents

- Tests are inline: `pack/tests.rs` (schema, parse/serialize, apply, policy, rails, sections, confirm interrupt), `rail.rs` (section rendering/idempotency), `rails.rs` (spec parsing and rail construction).
- The model-facing effects are asserted through `ScriptedModel::last_call()` (`request.tools`, `messages`), so the suite stays offline and deterministic.
- `pack/tests.rs` is a child module of `pack.rs` (`#[cfg(test)] mod tests;`) only to keep `pack.rs` small; it is not a separate public module.
