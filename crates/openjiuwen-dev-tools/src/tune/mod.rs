//! Prompt-tuning data models and the deferred tuning pipelines.
//!
//! Ports the pure part of the Python `openjiuwen/dev_tools/tune/` package:
//!
//! - [`Case`] — one tuning example: the model [`inputs`](Case::inputs), the
//!   expected [`label`](Case::label), optional [`tools`](Case::tools), and a
//!   [`case_id`](Case::case_id).
//! - [`EvaluatedCase`] — a [`Case`] plus the [`answer`](EvaluatedCase::answer),
//!   [`score`](EvaluatedCase::score), and [`reason`](EvaluatedCase::reason)
//!   produced by an evaluator.
//! - [`CaseLoader`] — an in-memory case collection with deterministic
//!   [`shuffle`](CaseLoader::shuffle) and [`split`](CaseLoader::split).
//!
//! The optimizers ([`InstructionOptimizer`], [`ExampleOptimizer`],
//! [`JointOptimizer`], [`PromptSearchOptimizer`]), the [`Trainer`], and the
//! [`optimize_prompt`] convenience entry point depend on Python-only
//! optimisation internals. They keep their API shape and the tuning constants
//! ([`DEFAULT_ITERATION_NUM`], [`MAX_PARALLEL_NUM`], …) but return
//! [`openjiuwen_common::error::Error::Unsupported`]; see `docs/PARITY.md`.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//!
//! use openjiuwen_dev_tools::tune::{Case, CaseLoader};
//! use serde_json::Value;
//!
//! let mut inputs = BTreeMap::new();
//! inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
//! let mut label = BTreeMap::new();
//! label.insert("answer".to_owned(), Value::String("4".to_owned()));
//!
//! let case = Case::new(inputs, label)?;
//! let loader = CaseLoader::new([case]);
//!
//! let (train, test) = loader.split(0.5)?;
//! assert_eq!(train.size() + test.size(), 1);
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

mod base;
mod loader;
mod optimizer;

pub use base::{Case, EvaluatedCase};
pub use loader::CaseLoader;
pub use optimizer::{
    optimize_prompt, ExampleOptimizer, InstructionOptimizer, JointOptimizer, OptimizationResult,
    PromptSearchOptimizer, Trainer,
};
pub use optimizer::{
    DEFAULT_EARLY_STOP_SCORE, DEFAULT_EXAMPLE_NUM, DEFAULT_ITERATION_NUM, DEFAULT_PARALLEL_NUM,
    MAX_EXAMPLE_NUM, MAX_ITERATION_NUM, MAX_PARALLEL_NUM, MIN_EXAMPLE_NUM, MIN_ITERATION_NUM,
    MIN_PARALLEL_NUM,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;

    /// A minimal valid case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(id.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(id.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    #[test]
    fn re_exports_compose_into_a_loader_workflow() {
        let loader = CaseLoader::new([case("a"), case("b")]);
        assert_eq!(loader.size(), 2);

        let shuffled = loader.shuffle(DEFAULT_PARALLEL_NUM as u64);
        assert_eq!(shuffled.size(), 2);

        let (train, test) = shuffled.split(0.5).unwrap();
        assert_eq!(train.size(), 1);
        assert_eq!(test.size(), 1);

        let evaluated = EvaluatedCase::from(test.into_cases().remove(0));
        assert!(evaluated.validate().is_ok());
        assert!(evaluated.score.abs() < f64::EPSILON);
    }

    #[test]
    fn re_exported_constants_are_usable_from_the_module_root() {
        assert_eq!(MIN_ITERATION_NUM, 1);
        assert_eq!(MAX_ITERATION_NUM, 20);
        assert!((DEFAULT_EARLY_STOP_SCORE - 1.0).abs() < f64::EPSILON);
        // Read the thresholds through locals so these stay plain value checks.
        for (min, default, max) in [
            (MIN_ITERATION_NUM, DEFAULT_ITERATION_NUM, MAX_ITERATION_NUM),
            (MIN_PARALLEL_NUM, DEFAULT_PARALLEL_NUM, MAX_PARALLEL_NUM),
            (MIN_EXAMPLE_NUM, DEFAULT_EXAMPLE_NUM, MAX_EXAMPLE_NUM),
        ] {
            assert!(min <= default, "{min} <= {default}");
            assert!(default <= max, "{default} <= {max}");
        }
    }

    #[tokio::test]
    async fn re_exported_pipeline_entry_points_are_deferred() {
        let cases = [case("a")];
        assert!(matches!(
            optimize_prompt("prompt", &cases).await.unwrap_err(),
            openjiuwen_common::error::Error::Unsupported(_)
        ));
        assert!(matches!(
            InstructionOptimizer::new()
                .optimize("prompt", &cases)
                .await
                .unwrap_err(),
            openjiuwen_common::error::Error::Unsupported(_)
        ));
    }
}
