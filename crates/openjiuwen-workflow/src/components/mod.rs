//! Built-in workflow components.
//!
//! Every component implements [`crate::component::Component`] and is driven by the
//! workflow engine. Components without external dependencies are fully offline-testable;
//! the leaf components take their runtime collaborators (models, tools, knowledge bases,
//! memory stores) through their constructors.

pub mod branch;
pub mod end;
pub mod http_request;
pub mod llm;
pub mod loop_;
pub mod memory;
pub mod retrieval;
pub mod start;
pub mod tool;

pub use branch::{BranchComponent, BranchRouterAdapter};
pub use end::{EndComponent, EndConfig};
pub use http_request::{
    HttpAdvancedOptionsConfig, HttpAuthConfig, HttpAuthIn, HttpAuthType, HttpBackoffType,
    HttpComponentConfig, HttpContentType, HttpHeadersConfig, HttpRateLimitConfig,
    HttpRateLimitUnit, HttpRequestBodyConfig, HttpRequestComponent, HttpRequestParamConfig,
    HttpResponseFormat, HttpResponseHandlingConfig, HttpResponseMode, HttpRetryConfig,
};
pub use llm::{LlmCompConfig, LlmComponent, LlmResponseFormat};
pub use loop_::{LoopComponent, LoopConfig, LoopType};
pub use memory::{MemoryRecall, MemoryRecord, MemoryRetrievalComponent, MemoryWriteComponent};
pub use retrieval::KnowledgeRetrievalComponent;
pub use start::StartComponent;
pub use tool::ToolComponent;
