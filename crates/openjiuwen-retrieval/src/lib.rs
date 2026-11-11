//! Documents, chunking, embeddings, vector stores, rerankers, and knowledge base.
//!
//! Maps to the Python package `openjiuwen.core.retrieval` in the openJiuwen
//! reference implementation. The crate is split into independent layers:
//! [`common`] (shared data model, configs, metadata filters), [`chunking`],
//! [`embedding`], [`vector_store`], and [`reranker`].
#![forbid(unsafe_code)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_abs_to_unsigned
)]

pub mod chunking;
pub mod common;
pub mod embedding;
pub mod reranker;
pub mod vector_store;
