//! Span naming rules for agent and workflow spans.
//!
//! Ports the name construction sites of `openjiuwen/extensions/tracer_otel/handler.py`
//! (v0.1.20): `on_llm_start` (`chat {class_name}`), `on_plugin_start` and
//! `_start_non_llm_span` (`execute_tool {name}` or `{prefix}.{name}`),
//! `on_embedding_start` (`embeddings {model}`), and the workflow component
//! branch of `on_call_start` (`chat {model}` / `execute_tool {name}` /
//! `retrieval {name}` / `component.{invoke_id}`), plus the
//! `_LLM_SUBSTRINGS` / `_TOOL_SUBSTRINGS` / `_RETRIEVAL_SUBSTRINGS`
//! component-type classifiers.
//!
//! The rules are pure string building, so they compile with or without the
//! `otel` feature and can be asserted without an SDK. Two joining styles exist,
//! mirroring Python:
//!
//! - Space-joined `GenAI` names (`chat {name}`, `execute_tool {name}`) collapse to
//!   the bare prefix when the name is empty or blank, so no span is named
//!   `"chat "` with a dangling space.
//! - Dot-joined names (`prompt.{name}`, `component.{invoke_id}`, …) always keep
//!   the separator, so an empty name yields `"prompt."` rather than `"prompt"`.
//!
//! Name arguments are never trimmed when they are used; whitespace only decides
//! whether a space-joined prefix stands alone.
//!
//! # Deviations from Python
//!
//! - [`simple_span_name`] mirrors `_start_non_llm_span` for the invoke types the
//!   Rust session tracer models ([`InvokeType`]). Python also names memory spans
//!   `memory.{name}` (`on_memory_start`, a `"memory"` prefix); the Rust
//!   [`InvokeType`] enum has no `Memory` variant — the session tracer models
//!   memory through [`InvokeType::Plugin`] and [`InvokeType::Chain`] — so no
//!   `memory.` branch exists here and none is invented in another crate.
//! - Python's `on_llm_start` falls back to the literal class name `"unknown"`
//!   when the key is missing and keeps an empty class name as `"chat "`. Rust
//!   callers pass the name directly, so an empty or blank class name yields the
//!   bare `"chat"` (the same rule as [`chat_span_name_for_model`]).
//! - [`embeddings_span_name`] keeps Python's raw `f"embeddings {model}"`
//!   formatting: an empty model yields `"embeddings "` with the trailing space.
//!   Python never passes an empty model (`on_embedding_start` defaults it to
//!   `"unknown"`), so this branch is unobservable in the reference and the
//!   literal formatting is the faithful port.

use openjiuwen_core::session::tracer::InvokeType;

/// Component-type substrings that identify LLM-bearing workflow components.
///
/// Python counterpart: `_LLM_SUBSTRINGS` in `tracer_otel/handler.py`.
/// `"LLM"` covers `LLMComponent`; `"IntentDetection"` covers
/// `IntentDetectionComponent`; `"Questioner"` covers `QuestionerComponent`
/// (both invoke an LLM internally). Matching is case-sensitive, like Python's
/// `str.__contains__`.
const LLM_SUBSTRINGS: [&str; 3] = ["LLM", "IntentDetection", "Questioner"];

/// Component-type substrings that identify tool-executing workflow components.
///
/// Python counterpart: `_TOOL_SUBSTRINGS` in `tracer_otel/handler.py`
/// (covers `ToolExecutable` and future `Tool`-named variants).
const TOOL_SUBSTRINGS: [&str; 1] = ["Tool"];

/// Component-type substrings that identify retrieval workflow components.
///
/// Python counterpart: `_RETRIEVAL_SUBSTRINGS` in `tracer_otel/handler.py`
/// (covers `KnowledgeRetrievalExecutable` and `MemoryRetrievalExecutable`).
const RETRIEVAL_SUBSTRINGS: [&str; 1] = ["Retrieval"];

/// The span name for agent LLM work: `chat {class_name}`.
///
/// Python counterpart: `on_llm_start` in `tracer_otel/handler.py`
/// (`name=f"chat {instance_info.get('class_name', 'unknown')}"`). The span is
/// created with `SpanKind::Client` and `gen_ai.operation.name = "chat"`.
///
/// An empty or whitespace-only `class_name` yields the bare `"chat"` rather
/// than a name with a trailing space.
#[must_use]
pub fn chat_span_name(class_name: &str) -> String {
    join_with_space("chat", class_name)
}

/// The span name for a workflow LLM component: `chat {model}`.
///
/// Python counterpart: the `is_llm_component` branch of `on_call_start` in
/// `tracer_otel/handler.py`, which reads `model_name` (or `model`) from the
/// component metadata and uses `"chat"` when neither is set.
///
/// An empty or whitespace-only `model` yields `"chat"`.
#[must_use]
pub fn chat_span_name_for_model(model: &str) -> String {
    chat_span_name(model)
}

/// The span name for tool/plugin execution: `execute_tool {name}`.
///
/// Python counterpart: `_start_non_llm_span` with the
/// `gen_ai.operation.name = "execute_tool"` extra attribute
/// (`on_plugin_start`, and the tool component branch of `on_call_start`).
/// The span carries `SpanKind::Internal` and `gen_ai.operation.name =
/// "execute_tool"` per the `GenAI` semantic conventions.
///
/// An empty or whitespace-only `name` yields the bare `"execute_tool"`.
#[must_use]
pub fn tool_span_name(name: &str) -> String {
    join_with_space("execute_tool", name)
}

/// The legacy session-tracer tool span name: `tool.{name}`.
///
/// Python counterpart: the `"{span_name_prefix}.{instance_name}"` branch of
/// `_start_non_llm_span` when called from the legacy decorator tool event
/// (`InvokeType.PLUGIN` with prefix `"tool"` and no `execute_tool` attribute).
/// Dot-joined names keep the separator, so an empty `name` yields `"tool."`.
#[must_use]
pub fn legacy_tool_span_name(name: &str) -> String {
    join_with_dot("tool", name)
}

/// The span name for a non-LLM agent invocation.
///
/// Python counterpart: `_start_non_llm_span`, whose call sites use the prefixes
/// `"prompt"`, `"chain"`, `"retriever"`, `"evaluator"`, `"workflow"` and
/// `"memory"`.
///
/// Mapping:
///
/// - [`InvokeType::Prompt`] → `prompt.{name}`
/// - [`InvokeType::Chain`] → `chain.{name}`
/// - [`InvokeType::Retriever`] → `retriever.{name}`
/// - [`InvokeType::Evaluator`] → `evaluator.{name}`
/// - [`InvokeType::Workflow`] → `workflow.{name}`
/// - [`InvokeType::Llm`] → [`chat_span_name`] (`chat {name}`), because LLM work
///   uses the `GenAI` chat span name rather than `llm.{name}`
/// - [`InvokeType::Plugin`] → [`tool_span_name`] (`execute_tool {name}`), the
///   `GenAI` tool span name
///
/// The dot-joined branches keep the separator for an empty `name`
/// (`"prompt."`), and the space-joined branches collapse to the bare prefix
/// (`"chat"`, `"execute_tool"`).
///
/// Python also names memory spans `memory.{name}`; see the module docs for why
/// that branch has no [`InvokeType`] equivalent.
#[must_use]
pub fn simple_span_name(invoke_type: InvokeType, name: &str) -> String {
    match invoke_type {
        InvokeType::Prompt => join_with_dot("prompt", name),
        InvokeType::Llm => chat_span_name(name),
        InvokeType::Plugin => tool_span_name(name),
        InvokeType::Workflow => join_with_dot("workflow", name),
        InvokeType::Chain => join_with_dot("chain", name),
        InvokeType::Retriever => join_with_dot("retriever", name),
        InvokeType::Evaluator => join_with_dot("evaluator", name),
    }
}

/// The span name for an embedding call: `embeddings {model}`.
///
/// Python counterpart: `on_embedding_start` in `tracer_otel/handler.py`, which
/// builds `span_name=f"embeddings {model_name}"` with
/// `gen_ai.request.model = model_name` and `gen_ai.operation.name =
/// "embeddings"`.
///
/// Unlike the other space-joined names this follows Python's literal formatting:
/// an empty `model` yields `"embeddings "` (trailing space). Python defaults the
/// model to `"unknown"` before formatting, so the empty case does not occur in
/// the reference; the raw formatting is kept for parity.
#[must_use]
pub fn embeddings_span_name(model: &str) -> String {
    let mut name = String::with_capacity("embeddings".len() + 1 + model.len());
    name.push_str("embeddings ");
    name.push_str(model);
    name
}

/// The fallback span name for a workflow component: `component.{invoke_id}`.
///
/// Python counterpart: the final `else` branch of the component naming in
/// `on_call_start` (`span_name = f"component.{invoke_id}"`), used when the
/// component type is neither LLM-like, tool-like, nor retrieval-like. Dot-joined
/// names keep the separator, so an empty `invoke_id` yields `"component."`.
#[must_use]
pub fn workflow_component_span_name(invoke_id: &str) -> String {
    join_with_dot("component", invoke_id)
}

/// Whether a workflow component type is LLM-bearing.
///
/// Python counterpart: `any(s in component_type for s in _LLM_SUBSTRINGS)` in
/// `on_call_start`. A match selects `SpanKind::Client` and the `chat` span name.
///
/// Case-sensitive, like Python: `"LLMComponent"`, `"IntentDetectionComponent"`
/// and `"QuestionerComponent"` match; `"llm_component"` does not.
#[must_use]
pub fn is_llm_component(component_type: &str) -> bool {
    contains_any(component_type, &LLM_SUBSTRINGS)
}

/// Whether a workflow component type executes a tool.
///
/// Python counterpart: `any(s in component_type for s in _TOOL_SUBSTRINGS)` in
/// `on_call_start`. A match selects the `execute_tool {name}` span name.
///
/// Case-sensitive: `"ToolExecutable"` matches, `"tool_executable"` does not.
#[must_use]
pub fn is_tool_component(component_type: &str) -> bool {
    contains_any(component_type, &TOOL_SUBSTRINGS)
}

/// Whether a workflow component type performs retrieval.
///
/// Python counterpart: `any(s in component_type for s in _RETRIEVAL_SUBSTRINGS)`
/// in `on_call_start` (and `_is_retrieval_component_span` in `handler.py`). A
/// match selects the `retrieval {name}` span name.
///
/// Case-sensitive: `"KnowledgeRetrievalExecutable"` matches,
/// `"knowledge_retrieval"` does not.
#[must_use]
pub fn is_retrieval_component(component_type: &str) -> bool {
    contains_any(component_type, &RETRIEVAL_SUBSTRINGS)
}

/// Join `name` onto `prefix` with a single space, or return the bare `prefix`
/// when `name` is empty or whitespace-only.
fn join_with_space(prefix: &str, name: &str) -> String {
    if name.trim().is_empty() {
        return prefix.to_owned();
    }
    let mut joined = String::with_capacity(prefix.len() + 1 + name.len());
    joined.push_str(prefix);
    joined.push(' ');
    joined.push_str(name);
    joined
}

/// Join `name` onto `prefix` with a single dot, always keeping the separator.
fn join_with_dot(prefix: &str, name: &str) -> String {
    let mut joined = String::with_capacity(prefix.len() + 1 + name.len());
    joined.push_str(prefix);
    joined.push('.');
    joined.push_str(name);
    joined
}

/// Whether `value` contains any of `needles` (Python's `any(s in value for s
/// in needles)`).
fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

// Python counterparts for every case below:
// `tests/unit_tests/extensions/tracer_otel/test_handler.py` (`on_llm_start`,
// `on_plugin_start`, `_start_non_llm_span`, `on_call_start`,
// `on_embedding_start`, `test_workflow_tool_component_tagged_as_execute_tool`,
// and the retrieval/component naming tests).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_span_name_matches_on_llm_start() {
        assert_eq!(chat_span_name("TestModel"), "chat TestModel");
        assert_eq!(chat_span_name("ReActAgent"), "chat ReActAgent");
    }

    #[test]
    fn chat_span_name_collapses_empty_and_blank_class_names() {
        // Python's `"chat {class_name}"` would leave a trailing space for an
        // empty name; the port never emits a dangling separator.
        assert_eq!(chat_span_name(""), "chat");
        assert_eq!(chat_span_name("   "), "chat");
    }

    #[test]
    fn chat_span_name_for_model_matches_component_branch() {
        // `on_call_start`: `f"chat {request_model}" if request_model else "chat"`.
        assert_eq!(chat_span_name_for_model("qwen3-max"), "chat qwen3-max");
        assert_eq!(chat_span_name_for_model(""), "chat");
        assert_eq!(chat_span_name_for_model(" \t"), "chat");
    }

    #[test]
    fn tool_span_name_matches_execute_tool_convention() {
        // test_handler.py asserts `s.name == "execute_tool ChildTool"` and
        // `s.name == "execute_tool echo"`.
        assert_eq!(tool_span_name("ChildTool"), "execute_tool ChildTool");
        assert_eq!(tool_span_name("echo"), "execute_tool echo");
        assert_eq!(tool_span_name(""), "execute_tool");
        assert_eq!(tool_span_name("  "), "execute_tool");
    }

    #[test]
    fn legacy_tool_span_name_keeps_the_dot() {
        // Legacy decorator tool event: `f"tool.{instance_name}"`.
        assert_eq!(legacy_tool_span_name("echo"), "tool.echo");
        assert_eq!(legacy_tool_span_name(""), "tool.");
    }

    #[test]
    fn simple_span_name_maps_every_invoke_type() {
        assert_eq!(
            simple_span_name(InvokeType::Prompt, "answer"),
            "prompt.answer"
        );
        assert_eq!(simple_span_name(InvokeType::Chain, "rag"), "chain.rag");
        assert_eq!(
            simple_span_name(InvokeType::Retriever, "knowledge"),
            "retriever.knowledge"
        );
        assert_eq!(
            simple_span_name(InvokeType::Evaluator, "faithfulness"),
            "evaluator.faithfulness"
        );
        assert_eq!(
            simple_span_name(InvokeType::Workflow, "triage"),
            "workflow.triage"
        );
        // LLM work and plugin calls use the GenAI names, not `llm.*` / `plugin.*`.
        assert_eq!(
            simple_span_name(InvokeType::Llm, "TestModel"),
            "chat TestModel"
        );
        assert_eq!(
            simple_span_name(InvokeType::Plugin, "echo"),
            "execute_tool echo"
        );
    }

    #[test]
    fn simple_span_name_handles_empty_names_per_joiner() {
        // Dot-joined prefixes keep the separator …
        assert_eq!(simple_span_name(InvokeType::Prompt, ""), "prompt.");
        assert_eq!(simple_span_name(InvokeType::Chain, ""), "chain.");
        assert_eq!(simple_span_name(InvokeType::Retriever, ""), "retriever.");
        assert_eq!(simple_span_name(InvokeType::Evaluator, ""), "evaluator.");
        assert_eq!(simple_span_name(InvokeType::Workflow, ""), "workflow.");
        // … while space-joined GenAI prefixes collapse to the bare prefix.
        assert_eq!(simple_span_name(InvokeType::Llm, ""), "chat");
        assert_eq!(simple_span_name(InvokeType::Plugin, ""), "execute_tool");
        assert_eq!(simple_span_name(InvokeType::Llm, "  "), "chat");
        assert_eq!(simple_span_name(InvokeType::Plugin, "  "), "execute_tool");
    }

    #[test]
    fn embeddings_span_name_matches_genai_rule() {
        // test_handler.py: `assert s.name == "embeddings bge-m3"`.
        assert_eq!(embeddings_span_name("bge-m3"), "embeddings bge-m3");
        // Python's literal `f"embeddings {model}"` keeps the trailing space.
        assert_eq!(embeddings_span_name(""), "embeddings ");
    }

    #[test]
    fn workflow_component_span_name_matches_on_call_start_fallback() {
        // test_handler.py: `assert s.name == "component.other_node"`.
        assert_eq!(
            workflow_component_span_name("other_node"),
            "component.other_node"
        );
        assert_eq!(workflow_component_span_name(""), "component.");
    }

    #[test]
    fn llm_component_classifier_matches_python_substrings() {
        assert!(is_llm_component("LLMComponent"));
        assert!(is_llm_component("IntentDetectionComponent"));
        assert!(is_llm_component("QuestionerComponent"));
        assert!(is_llm_component("SubLLMComponent"));
        assert!(!is_llm_component("ConditionComponent"));
        assert!(!is_llm_component("ToolExecutable"));
        assert!(!is_llm_component("KnowledgeRetrievalExecutable"));
        assert!(!is_llm_component(""));
        // Case-sensitive, like Python's `str.__contains__`.
        assert!(!is_llm_component("LlmComponent"));
        assert!(!is_llm_component("intentdetectionComponent"));
    }

    #[test]
    fn tool_component_classifier_matches_python_substrings() {
        assert!(is_tool_component("ToolExecutable"));
        assert!(is_tool_component("WeatherTool"));
        assert!(!is_tool_component("LLMComponent"));
        assert!(!is_tool_component("KnowledgeRetrievalExecutable"));
        assert!(!is_tool_component("ConditionComponent"));
        assert!(!is_tool_component(""));
        assert!(!is_tool_component("toolExecutable"));
    }

    #[test]
    fn retrieval_component_classifier_matches_python_substrings() {
        // test_handler.py: `KnowledgeRetrievalExecutable` is named
        // `retrieval kb_node`.
        assert!(is_retrieval_component("KnowledgeRetrievalExecutable"));
        assert!(is_retrieval_component("MemoryRetrievalExecutable"));
        assert!(!is_retrieval_component("LLMComponent"));
        assert!(!is_retrieval_component("ToolExecutable"));
        assert!(!is_retrieval_component("ConditionComponent"));
        assert!(!is_retrieval_component(""));
        assert!(!is_retrieval_component("knowledgeretrieval"));
    }
}
