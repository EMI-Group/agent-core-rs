//! Parameter and tool binding: map an optimizer's output back into parameters.
//!
//! Ports Python's `LLMCall`/`TextualParameter` binding layer
//! (`openjiuwen/dev_tools/tune/optimizer/base.py`, `tune/utils.py`) as typed
//! values. Python's optimizers mutate an agent's prompt parameters in place
//! through a mutable `LLMCall`; the Rust optimizers take prompt strings and return
//! an [`OptimizationResult`](crate::tune::OptimizationResult). This module is the
//! missing glue between the two:
//!
//! - [`TextualParameter`] is one named prompt template plus the textual gradients
//!   and candidate values an optimizer stages for it. It owns text, not a live
//!   call, so a parameter can be rendered and compared without side effects.
//! - [`ParameterBinding`] validates a candidate prompt against the parameter's
//!   declared `{{name}}` placeholders and returns a [`ParameterUpdate`] describing
//!   the change instead of performing it — a rewrite that drops, renames, or
//!   invents a placeholder is rejected, because a downstream parameter map would
//!   have no value for it.
//! - [`ToolBinding`] joins the tools a [`Case`](crate::tune::Case) declares to the
//!   runnable [`ToolStubs`](crate::tune::ToolStubs) it may call, reporting what is
//!   bound, what is missing, and — through [`ToolTrace`](crate::tune::ToolTrace) —
//!   what a run used.
//! - [`BoundAgent`] is the agent-side adapter: a name-keyed parameter set whose
//!   `validate_all`/`apply` turn one candidate prompt map into the update list a
//!   tuning driver writes back.
//!
//! # Example
//!
//! ```
//! # use std::collections::BTreeMap;
//! # use openjiuwen_dev_tools::tune::{BoundAgent, ParameterBinding};
//! # fn main() -> Result<(), openjiuwen_common::error::Error> {
//! let binding = ParameterBinding::from_template("system", "Answer {{question}}.");
//! let agent = BoundAgent::new(vec![binding])?;
//!
//! let mut prompts = BTreeMap::new();
//! prompts.insert("system".to_owned(), "Answer {{question}} briefly.".to_owned());
//! let updates = agent.apply(&prompts)?;
//! assert_eq!(updates.len(), 1);
//! assert!(updates[0].changed());
//! assert_eq!(updates[0].preserved_placeholders, vec!["question".to_owned()]);
//! # Ok(())
//! # }
//! ```
//!
//! Rendering is separate from binding: a parameter fills its own placeholders
//! from a value map, leaving unknown placeholders verbatim.
//!
//! ```
//! # use std::collections::BTreeMap;
//! # use openjiuwen_dev_tools::tune::ParameterBinding;
//! # use serde_json::Value;
//! # fn main() -> Result<(), openjiuwen_common::error::Error> {
//! let binding = ParameterBinding::from_template("system", "Answer {{question}} in {{language}}.");
//! let mut values = binding.prompt_values(); // `{{question}}`, `{{language}}`
//! values.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
//! assert_eq!(
//!     binding.parameter().render(&values)?,
//!     "Answer 2 + 2? in {{language}}."
//! );
//! # Ok(())
//! # }
//! ```
//!
//! # Deferred
//!
//! Python's `LLMCall` itself — the traced operator with a mutable system/user
//! prompt, an optimizer callback, and a frozen-prompt accessor — is not ported, and
//! neither is `OptimizeHistory` (the per-case trace of node inputs and outputs) or
//! the traced-node gradient attribution built on it. A binding therefore has no
//! baseline of its own: [`ParameterUpdate::gain`] carries the optimizer's absolute
//! score rather than a measured improvement.
mod agent;
mod apply;
mod parameter;
mod tools;

#[cfg(test)]
mod tests;

pub use agent::BoundAgent;
pub use apply::{ParameterBinding, ParameterUpdate};
pub use parameter::TextualParameter;
pub use tools::{ToolBinding, ToolBindingReport};
