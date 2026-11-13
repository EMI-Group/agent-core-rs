//! Research optimizers: real textual-gradient search over signals.
//!
//! Ports the portable core of Python
//! `openjiuwen.agent_evolving.optimizer` — the domain optimizers that turn
//! evolution signals into updated operator text:
//!
//! - [`params`] — [`TextualParameter`], [`OperatorState`] and [`OperatorKind`],
//!   the checkpoint-friendly representation of tunable text.
//! - [`base`] — [`TextualStrategy`] and the shared
//!   [`TextualGradientOptimizer`] engine (backward gradients, deterministic and
//!   model-driven proposals, best-so-far restoration).
//! - [`InstructionStrategy`] / [`InstructionOptimizer`] — prompt/instruction text.
//! - [`ToolStrategy`] / [`ToolOptimizer`] — per-tool descriptions.
//! - [`MemoryStrategy`] / [`MemoryOptimizer`] — bounded long-term memory entries.
//! - [`SkillExperienceStrategy`] / [`SkillExperienceOptimizer`] — skill experience
//!   bullets.
//! - [`MetisStrategy`] / [`MetisOptimizer`] — context-section selection by signal
//!   utility (Python's `context_evolve_call`).
//! - [`TtseStrategy`] / [`TtseOptimizer`] — test-time skill tips clustered from
//!   similar feedback.
//!
//! Every optimizer is deterministic without a model; injecting a
//! `openjiuwen_core::foundation::llm::Model` enables model-driven proposals with
//! a documented fallback to the deterministic rewrite.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_evolving::optimizer::{
//!     InstructionOptimizer, OperatorKind, OperatorState, Optimizer,
//! };
//! use openjiuwen_agent_evolving::signal::{EvolutionSignal, EvolutionTarget};
//!
//! # #[tokio::main]
//! # async fn main() {
//! let mut optimizer = InstructionOptimizer::new().with_operator(
//!     OperatorState::new("planner", OperatorKind::Instruction)
//!         .with_parameter("instruction", "Plan the task."),
//! );
//! let signal = EvolutionSignal::new(
//!     "low_score",
//!     EvolutionTarget::Instruction,
//!     "Plans omit a verification step.",
//! )
//! .expect("signal");
//! assert_eq!(optimizer.backward(&[signal]).expect("backward"), 1);
//!
//! let candidates = optimizer.propose(1).await.expect("propose");
//! let value = &candidates[0].parameters["planner"]["instruction"]["value"];
//! assert!(value.as_str().is_some_and(|text| text.contains("verification")));
//! # }
//! ```

macro_rules! optimizer_wrapper {
    ($name:ident, $strategy:ty) => {
        #[doc = concat!(
                                            "The `",
                                            stringify!($strategy),
                                            "` optimizer over the shared textual-gradient engine."
                                        )]
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: $crate::optimizer::research::TextualGradientOptimizer<$strategy>,
        }

        impl $name {
            /// Creates the optimizer with the strategy's defaults and no model.
            #[must_use]
            pub fn new() -> Self {
                Self {
                    inner: $crate::optimizer::research::TextualGradientOptimizer::new(
                        <$strategy>::new(),
                    ),
                }
            }

            /// Registers (or replaces) one operator.
            #[must_use]
            pub fn with_operator(
                mut self,
                operator: $crate::optimizer::research::OperatorState,
            ) -> Self {
                self.inner = self.inner.with_operator(operator);
                self
            }

            /// Injects the model used for proposals.
            #[must_use]
            pub fn with_model(
                mut self,
                model: std::sync::Arc<dyn openjiuwen_core::foundation::llm::Model>,
            ) -> Self {
                self.inner = self.inner.with_model(model);
                self
            }

            /// Replaces the proposal budget.
            #[must_use]
            pub fn with_max_proposals(mut self, max_proposals: usize) -> Self {
                self.inner = self.inner.with_max_proposals(max_proposals);
                self
            }

            /// Replaces the model request configuration.
            #[must_use]
            pub fn with_request(
                mut self,
                request: openjiuwen_core::foundation::llm::ModelRequestConfig,
            ) -> Self {
                self.inner = self.inner.with_request(request);
                self
            }

            /// Replaces the model retry/timeout policy.
            #[must_use]
            pub fn with_policy(mut self, policy: $crate::optimizer::LlmInvokePolicy) -> Self {
                self.inner = self.inner.with_policy(policy);
                self
            }

            /// The shared search engine behind this optimizer.
            #[must_use]
            pub fn inner(
                &self,
            ) -> &$crate::optimizer::research::TextualGradientOptimizer<$strategy> {
                &self.inner
            }

            /// The shared search engine behind this optimizer, mutably.
            pub fn inner_mut(
                &mut self,
            ) -> &mut $crate::optimizer::research::TextualGradientOptimizer<$strategy> {
                &mut self.inner
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::ops::Deref for $name {
            type Target = $crate::optimizer::research::TextualGradientOptimizer<$strategy>;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl std::ops::DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        #[async_trait::async_trait]
        impl $crate::optimizer::Optimizer for $name {
            fn name(&self) -> &str {
                $crate::optimizer::Optimizer::name(&self.inner)
            }

            async fn propose(
                &mut self,
                limit: usize,
            ) -> ::openjiuwen_common::error::Result<Vec<$crate::optimizer::Candidate>> {
                $crate::optimizer::Optimizer::propose(&mut self.inner, limit).await
            }

            fn observe(&mut self, candidate: &$crate::optimizer::Candidate, score: f64) {
                $crate::optimizer::Optimizer::observe(&mut self.inner, candidate, score);
            }

            fn best(&self) -> Option<&$crate::optimizer::Candidate> {
                $crate::optimizer::Optimizer::best(&self.inner)
            }
        }
    };
}

mod base;
mod instruction;
mod memory;
mod metis;
mod params;
mod skill;
mod tool;
mod ttse;

pub use base::{
    parse_tagged, ProposalRecord, TextualGradientOptimizer, TextualStrategy, CANDIDATE_META_KEY,
    DEFAULT_MAX_PROPOSALS, IMPROVED_TAG, PARAMETER_SEPARATOR,
};
pub use instruction::{InstructionOptimizer, InstructionStrategy};
pub use memory::{MemoryOptimizer, MemoryStrategy};
pub use metis::{MetisOptimizer, MetisStrategy, DEFAULT_SECTION_BUDGET};
pub use params::{OperatorKind, OperatorState, TextualParameter, MAX_GRADIENT_CHARS};
pub use skill::{SkillExperienceOptimizer, SkillExperienceStrategy};
pub use tool::{ToolOptimizer, ToolStrategy};
pub use ttse::{TtseOptimizer, TtseStrategy, DEFAULT_CLUSTER_THRESHOLD, DEFAULT_TIP_CAPACITY};
