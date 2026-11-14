//! Legacy single-call LLM wrapper with an optimizer trace callback.
//!
//! Ports `openjiuwen/core/operator/legacy/llm_call/`: [`LLMCall`] renders its
//! prompt templates, optionally threads history and tool definitions into one
//! model call, and then reports the exchange to an [`OptimizerCallback`] so an
//! evolution loop can observe live traffic. It is distinct from the tunable
//! [`LLMCallOperator`](crate::operator::llm_call::LLMCallOperator), which owns
//! parameters but executes no call of its own.

pub mod llm_call;

pub use llm_call::{
    LLMCall, OptimizerCallback, OptimizerTrace, OptimizerTraceFuture, DEFAULT_USER_PROMPT,
};
