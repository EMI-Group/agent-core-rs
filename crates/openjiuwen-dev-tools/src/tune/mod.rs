//! Prompt-tuning data models and the deferred tuning pipelines.
//!
//! Ports the pure part of `openjiuwen/dev_tools/tune/`: `Case`, `EvaluatedCase`,
//! and `CaseLoader`. The optimizers and the trainer depend on Python
//! optimisation internals and are API-shaped stubs that return
//! [`openjiuwen_common::error::Error::Unsupported`].
