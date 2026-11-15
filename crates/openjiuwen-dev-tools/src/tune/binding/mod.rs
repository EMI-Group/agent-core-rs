//! Parameter and tool binding: map an optimizer's output back into parameters.
//!
//! Ports Python's `LLMCall`/`TextualParameter` binding layer
//! (`tune/optimizer/base.py`, `tune/utils.py`) as typed values: a text parameter
//! holds the template an optimizer rewrites, and a binding applies the winning
//! prompt to a parameter map after validating it against the template's
//! placeholders and the case's declared tools.
