//! YAML prompt packs: load an agent's prompts, tool policy, and rails from one
//! YAML document.
//!
//! A pack is an additive configuration format over [`crate::ReActAgent`]: it
//! introduces no new runtime behaviour, it only describes values the agent
//! already understands. [`PromptPack::apply_to`] copies those values onto an
//! existing agent, so a host can ship prompts, tool visibility, and rails as
//! data instead of code.
//!
//! # Format
//!
//! One YAML document describes one pack:
//!
//! ```yaml
//! id: support-agent
//! name: Support Agent
//! description: Handles support requests
//! prompts:
//!   template:                     # ReActAgentConfig::prompt_template
//!     - { role: system, content: "You are a support agent." }
//!   sections:                     # extra system sections injected before each model call
//!     - { name: policy, content: "Never reveal internal ids.", priority: 20 }
//!     - { name: tone, content: "Be concise.", priority: 30 }
//! tools:
//!   policy: { mode: allowlist, allowed: [search, echo], hide_deferred: true }
//!   mcp_allowlist: { servers: { "kb": ["search"] } }
//! rails:
//!   - { type: guardrail }
//!   - { type: confirm_interrupt }
//!   - { type: ask_user }
//!   - { type: skill, budget: 2000 }
//! metadata: { owner: team-a }
//! ```
//!
//! Only `id` is required; every other key defaults to "not configured", and
//! unknown keys are tolerated so a newer pack can be read by an older build.
//!
//! # Applying a pack
//!
//! - `prompts.template` replaces
//!   [`crate::ReActAgentConfig::prompt_template`] when it is non-empty;
//!   otherwise, when the pack sets `name` or `description`, one synthesized
//!   `system` message (`You are <name>.\n\n<description>`) is prepended to the
//!   agent's existing template. Every other configuration field is preserved.
//! - `prompts.sections` become a [`PromptSectionsRail`]: one `system` message
//!   rendered from the sections before every model call, marked with
//!   [`PROMPT_SECTIONS_METADATA_KEY`] so it is injected at most once per model
//!   window.
//! - `tools.policy`, `tools.mcp_allowlist`, and `tools.resilience` install the
//!   matching policy on the agent; [`PromptPack::to_tool_policy`] exposes the
//!   resolved exposure policy (allow-all when the pack sets none).
//! - `rails` builds the kernel rails a pack can construct on its own
//!   (`guardrail`, `confirm_interrupt`, `ask_user`). Rails that need a host
//!   object are reported as deferred instead of silently dropped: `skill` needs
//!   a [`crate::skills::SkillManager`], `model_backup` needs at least one
//!   backup model, and `custom` is always deferred because only the host can
//!   build it. [`PromptPack::apply_to_with`] supplies the first two through a
//!   [`PromptPackContext`].
//!
//! A pack cannot describe guardrail rules, so `guardrail` builds a
//! [`crate::GuardrailRail`] over an empty
//! [`openjiuwen_security::GuardrailPipeline`]: the rail is registered (and
//! reported as such) as a placeholder a host replaces with a configured one.
//!
//! # Modules
//!
//! - [`pack`] — the pack schema plus parse, serialize, and apply methods.
//! - [`rails`] — [`RailSpec`] and the rail construction.
//! - [`rail`] — the [`PromptSectionsRail`] that injects pack sections.
pub mod pack;
pub mod rail;
pub mod rails;
pub use pack::{AppliedPack, PromptPack, PromptPackContext, PromptSpec, SectionSpec, ToolSpec};
pub use rail::{PromptSectionsRail, PROMPT_SECTIONS_METADATA_KEY};
pub use rails::RailSpec;
