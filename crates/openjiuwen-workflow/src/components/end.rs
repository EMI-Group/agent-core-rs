//! Built-in `end` workflow component.
//!
//! Ports Python's `openjiuwen.core.workflow.components.flow.end_comp.End`. The end node
//! has two modes:
//!
//! - **Flat mode** (no `response_template`): `invoke` projects the inputs into
//!   `{"output": …}`, `stream` yields one `{"output": {key: value}}` frame per input
//!   entry, and `collect` flattens nested objects/arrays into path→value pairs.
//! - **Template mode** (`response_template = Some(…)`): `{{path}}` segments resolve
//!   against the inputs. `invoke`/`collect` join the segments into one `{"response":
//!   "<text>"}` value; `stream`/`transform` yield one `{"response": …}` frame per
//!   segment piece, expanding resolved arrays into one frame per element so a buffered
//!   producer's frames flow through in template order.
//!
//! Template segmentation mirrors Python's `render_template_to_list`
//! (`re.split(r'(\{\{[^{}]+\}\})', template)` with empty pieces dropped): a segment that
//! starts with `{{` and ends with `}}` is a variable whose path is the inner text
//! verbatim (no trimming), everything else is literal text.
//!
//! **Deviation.** Python's batch `str(value)` uses the Python `repr` of numbers,
//! arrays, and objects (`3`, `[1, 2]`, `{'a': 1}`); Rust renders non-string batch
//! values as JSON text (`3`, `[1,2]`, `{"a":1}`) via `serde_json::to_string`. Strings are
//! used verbatim in both implementations. Stream frames carry the raw JSON value
//! (exactly like Python's `OutputSchema.payload`).

use std::sync::{Arc, OnceLock};

use futures::stream::BoxStream;
use futures::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::reference;

/// Configuration of the [`EndComponent`] response template.
///
/// The wire shape matches Python's `EndConfig`: the field is `response_template` and
/// the camel-case `responseTemplate` spelling is accepted as an alias.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EndConfig {
    /// Template rendered into `{"response": …}` instead of the flat `{"output": …}`
    /// projection; `None` selects flat mode.
    #[serde(alias = "responseTemplate")]
    pub response_template: Option<String>,
}

impl EndConfig {
    /// Creates a template configuration from `template`.
    #[must_use]
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            response_template: Some(template.into()),
        }
    }
}

/// One piece of a parsed response template.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EndSegment {
    /// Literal text emitted verbatim.
    Literal(String),
    /// A `{{path}}` reference resolved against the component inputs.
    Variable(String),
}

/// The `end` built-in component.
///
/// The parsed template is cached at construction time, so running a workflow twice does
/// not re-segment the template. See the module documentation for the two modes and the
/// batch/stream return shapes.
#[derive(Debug, Clone, Default)]
pub struct EndComponent {
    config: EndConfig,
    segments: Vec<EndSegment>,
}

impl EndComponent {
    /// Creates an end component without a response template (flat mode).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an end component from `config`, segmenting its template once.
    #[must_use]
    pub fn with_config(config: EndConfig) -> Self {
        let segments = config
            .response_template
            .as_deref()
            .map_or_else(Vec::new, parse_template);
        Self { config, segments }
    }

    /// The component configuration.
    #[must_use]
    pub fn config(&self) -> &EndConfig {
        &self.config
    }

    /// Whether the component renders a response template.
    fn has_template(&self) -> bool {
        self.config.response_template.is_some()
    }

    /// Builds the frames `stream`/`transform` yield, in template order.
    fn frames(&self, inputs: &Value) -> Vec<Value> {
        if !self.has_template() {
            return flat_frames(inputs);
        }
        let mut frames = Vec::new();
        for segment in &self.segments {
            match segment {
                EndSegment::Literal(text) => frames.push(json!({"response": text})),
                EndSegment::Variable(path) => match reference::resolve_path(inputs, path) {
                    None | Some(Value::Null) => {}
                    Some(Value::Array(items)) => {
                        frames.extend(items.into_iter().map(|item| json!({"response": item})));
                    }
                    Some(value) => frames.push(json!({"response": value})),
                },
            }
        }
        frames
    }

    /// The stream body shared by `stream` and `transform`.
    fn frame_stream(&self, inputs: &Value) -> BoxStream<'static, Result<Value>> {
        futures::stream::iter(self.frames(inputs).into_iter().map(Ok)).boxed()
    }
}

#[async_trait::async_trait]
impl Component for EndComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("end", "end").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![
            ComponentAbility::Invoke,
            ComponentAbility::Stream,
            ComponentAbility::Collect,
            ComponentAbility::Transform,
        ]
    }

    fn component_type(&self) -> &'static str {
        "end"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        if self.has_template() {
            return Ok(json!({"response": render_batch(&self.segments, &inputs)}));
        }
        Ok(json!({"output": flat_output(inputs)}))
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        self.frame_stream(&inputs)
    }

    async fn collect(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        if self.has_template() {
            return Ok(json!({"response": render_batch(&self.segments, &inputs)}));
        }
        let mut flattened = Map::new();
        flatten_into(&inputs, &mut Vec::new(), &mut flattened);
        Ok(json!({"output": flattened}))
    }

    fn transform(
        self: Arc<Self>,
        inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        self.frame_stream(&inputs)
    }
}

/// The `{{…}}` matcher used by Python's `render_template_to_list`.
///
/// The pattern cannot fail to compile; the `Option` keeps the failure path explicit
/// (the whole template then becomes one literal segment) instead of panicking.
fn segment_regex() -> Option<&'static Regex> {
    static SEGMENT: OnceLock<Option<Regex>> = OnceLock::new();
    SEGMENT
        .get_or_init(|| Regex::new(r"\{\{[^{}]+\}\}").ok())
        .as_ref()
}

/// Splits `template` into literal and variable segments, dropping empty pieces.
fn parse_template(template: &str) -> Vec<EndSegment> {
    let Some(regex) = segment_regex() else {
        return literal_segments(template);
    };
    let mut segments = Vec::new();
    let mut cursor = 0;
    for matched in regex.find_iter(template) {
        push_literal(&mut segments, &template[cursor..matched.start()]);
        let raw = matched.as_str();
        match raw
            .strip_prefix("{{")
            .and_then(|inner| inner.strip_suffix("}}"))
        {
            Some(path) => segments.push(EndSegment::Variable(path.to_owned())),
            None => segments.push(EndSegment::Literal(raw.to_owned())),
        }
        cursor = matched.end();
    }
    push_literal(&mut segments, &template[cursor..]);
    segments
}

/// A template that is not matched by [`segment_regex`] is one literal segment.
fn literal_segments(template: &str) -> Vec<EndSegment> {
    if template.is_empty() {
        Vec::new()
    } else {
        vec![EndSegment::Literal(template.to_owned())]
    }
}

/// Appends `text` as a literal segment unless it is empty (Python's `filter(None, …)`).
fn push_literal(segments: &mut Vec<EndSegment>, text: &str) {
    if !text.is_empty() {
        segments.push(EndSegment::Literal(text.to_owned()));
    }
}

/// Renders the template into one string (batch mode).
fn render_batch(segments: &[EndSegment], inputs: &Value) -> String {
    let mut rendered = String::new();
    for segment in segments {
        match segment {
            EndSegment::Literal(text) => rendered.push_str(text),
            EndSegment::Variable(path) => {
                if let Some(value) =
                    reference::resolve_path(inputs, path).filter(|value| !value.is_null())
                {
                    rendered.push_str(&batch_text(&value));
                }
            }
        }
    }
    rendered
}

/// Renders one resolved value as batch text.
///
/// Strings are used verbatim (Python's `str`); other values become JSON text, which is
/// the documented deviation from Python's `repr`.
fn batch_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        // Serializing a `Value` cannot fail; the fallback keeps the path panic-free.
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Flat-mode `invoke` output: the input object without `null`-valued keys.
fn flat_output(inputs: Value) -> Value {
    match inputs {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, value)| !value.is_null())
                .collect(),
        ),
        Value::Null => Value::Object(Map::new()),
        other => other,
    }
}

/// Flat-mode `stream`/`transform` frames: one `{"output": {key: value}}` per entry.
fn flat_frames(inputs: &Value) -> Vec<Value> {
    match inputs {
        Value::Null => Vec::new(),
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| {
                let mut output = Map::new();
                output.insert(key.clone(), value.clone());
                json!({"output": output})
            })
            .collect(),
        other => vec![json!({"output": other})],
    }
}

/// Walks `value` and records every non-null leaf under its flattened path.
///
/// An object key `a` extends the path with `a`, an array element `i` with `[i]`, and
/// scalars are recorded as `a.b[0].c`. `null` leaves are omitted, matching Python's
/// `extract_leaf_nodes` pruning of empty values (other falsy scalars such as `0`, `""`,
/// and `false` are kept).
fn flatten_into(value: &Value, path: &mut Vec<String>, out: &mut Map<String, Value>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                path.push(key.clone());
                flatten_into(child, path, out);
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                path.push(format!("[{index}]"));
                flatten_into(item, path, out);
                path.pop();
            }
        }
        Value::Null => {}
        leaf => {
            out.insert(format_path(path), leaf.clone());
        }
    }
}

/// Formats a path the way Python's `format_path` does: `[i]` attaches directly and
/// other keys are dot-separated.
fn format_path(path: &[String]) -> String {
    let mut formatted = String::new();
    for segment in path {
        if formatted.is_empty() || segment.starts_with('[') {
            formatted.push_str(segment);
        } else {
            formatted.push('.');
            formatted.push_str(segment);
        }
    }
    formatted
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("end-tests"));
        ComponentContext::new(session, "end", "wf")
    }

    fn arc_context() -> Arc<ComponentContext> {
        Arc::new(context())
    }

    async fn stream_frames(component: EndComponent, inputs: Value) -> Vec<Value> {
        Arc::new(component)
            .stream(inputs, arc_context())
            .map(|frame| frame.unwrap())
            .collect()
            .await
    }

    #[test]
    fn template_segments_keep_order_and_verbatim_paths() {
        assert_eq!(
            parse_template("Hello {{name}}, welcome to {{ place }}!"),
            vec![
                EndSegment::Literal("Hello ".to_owned()),
                EndSegment::Variable("name".to_owned()),
                EndSegment::Literal(", welcome to ".to_owned()),
                EndSegment::Variable(" place ".to_owned()),
                EndSegment::Literal("!".to_owned()),
            ]
        );
        // Adjacent variables produce no empty literal.
        assert_eq!(
            parse_template("{{a}}{{b}}"),
            vec![
                EndSegment::Variable("a".to_owned()),
                EndSegment::Variable("b".to_owned()),
            ]
        );
        assert!(parse_template("").is_empty());
        assert_eq!(
            parse_template("plain text"),
            vec![EndSegment::Literal("plain text".to_owned())]
        );
        assert_eq!(
            parse_template("{{unclosed"),
            vec![EndSegment::Literal("{{unclosed".to_owned())]
        );
    }

    #[tokio::test]
    async fn invoke_without_template_filters_nulls() {
        let component = EndComponent::new();
        let output = component
            .invoke(json!({"a": 1, "b": null, "c": "x"}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"output": {"a": 1, "c": "x"}}));
    }

    #[tokio::test]
    async fn invoke_without_template_handles_null_and_scalars() {
        let component = EndComponent::new();
        assert_eq!(
            component.invoke(Value::Null, &context()).await.unwrap(),
            json!({"output": {}})
        );
        assert_eq!(
            component.invoke(json!(7), &context()).await.unwrap(),
            json!({"output": 7})
        );
        assert_eq!(
            component.invoke(json!([1, 2]), &context()).await.unwrap(),
            json!({"output": [1, 2]})
        );
    }

    #[tokio::test]
    async fn invoke_with_template_stringifies_non_string_values() {
        let inputs = json!({"n": 3, "a": [1, 2], "o": {"a": 1}, "s": "hi"});
        let output = EndComponent::with_config(EndConfig::new("n={{n}} a={{a}} o={{o}} s={{s}}"))
            .invoke(inputs, &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"response": "n=3 a=[1,2] o={\"a\":1} s=hi"}));
    }

    #[tokio::test]
    async fn invoke_with_template_omits_missing_and_null_variables() {
        let component = EndComponent::with_config(EndConfig::new("a{{missing}}b{{present}}c"));
        let output = component
            .invoke(json!({"present": null}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"response": "abc"}));

        let output = component
            .invoke(json!({"present": "!"}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"response": "ab!c"}));
    }

    #[tokio::test]
    async fn stream_yields_frames_in_template_order() {
        let component = EndComponent::with_config(EndConfig::new("x{{n}}{{items}}"));
        let frames = stream_frames(component, json!({"n": 1, "items": [7, 8]})).await;
        assert_eq!(
            frames,
            vec![
                json!({"response": "x"}),
                json!({"response": 1}),
                json!({"response": 7}),
                json!({"response": 8}),
            ]
        );
    }

    #[tokio::test]
    async fn stream_expands_arrays_one_frame_per_element() {
        let component = EndComponent::with_config(EndConfig::new("{{items}}"));
        let frames = stream_frames(component, json!({"items": ["a", "b"]})).await;
        assert_eq!(
            frames,
            vec![json!({"response": "a"}), json!({"response": "b"})]
        );

        let component = EndComponent::with_config(EndConfig::new("{{items}}"));
        let frames = stream_frames(component, json!({"items": []})).await;
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn stream_without_template_yields_one_frame_per_entry() {
        let component = EndComponent::new();
        let frames = stream_frames(component, json!({"a": 1, "b": null})).await;
        assert_eq!(
            frames,
            vec![json!({"output": {"a": 1}}), json!({"output": {"b": null}})]
        );
    }

    #[tokio::test]
    async fn stream_without_template_handles_null_and_scalars() {
        assert!(stream_frames(EndComponent::new(), Value::Null)
            .await
            .is_empty());
        assert_eq!(
            stream_frames(EndComponent::new(), json!("x")).await,
            vec![json!({"output": "x"})]
        );
    }

    #[tokio::test]
    async fn transform_matches_stream() {
        let inputs = json!({"n": 1, "items": [2, 3]});
        let component = EndComponent::with_config(EndConfig::new("{{n}}-{{items}}"));
        let streamed = stream_frames(component.clone(), inputs.clone()).await;
        let transformed: Vec<Value> = Arc::new(component)
            .transform(inputs, arc_context())
            .map(|frame| frame.unwrap())
            .collect()
            .await;
        assert_eq!(streamed, transformed);
    }

    #[tokio::test]
    async fn collect_flattens_nested_structures() {
        let component = EndComponent::new();
        let output = component
            .collect(
                json!({"a": [1, {"b": 2}], "c": {"d": null, "e": true}}),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(
            output,
            json!({"output": {"a[0]": 1, "a[1].b": 2, "c.e": true}})
        );
    }

    #[tokio::test]
    async fn collect_handles_null_input() {
        let output = EndComponent::new()
            .collect(Value::Null, &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"output": {}}));
    }

    #[tokio::test]
    async fn collect_with_template_joins_segments() {
        let component = EndComponent::with_config(EndConfig::new("{{a}}/{{b}}"));
        let output = component
            .collect(json!({"a": 1, "b": [2]}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"response": "1/[2]"}));
    }

    #[test]
    fn config_accepts_the_python_camel_case_alias() {
        let parsed: EndConfig = serde_json::from_value(json!({"responseTemplate": "hi"})).unwrap();
        assert_eq!(parsed.response_template.as_deref(), Some("hi"));
        assert_eq!(
            serde_json::to_value(EndConfig::new("hi")).unwrap(),
            json!({"response_template": "hi"})
        );
        assert_eq!(
            EndConfig::default(),
            EndConfig {
                response_template: None
            }
        );
    }

    #[test]
    fn identity_matches_python() {
        let component = EndComponent::new();
        assert_eq!(component.component_type(), "end");
        assert_eq!(
            component.abilities(),
            vec![
                ComponentAbility::Invoke,
                ComponentAbility::Stream,
                ComponentAbility::Collect,
                ComponentAbility::Transform,
            ]
        );
        let card = component.card();
        assert_eq!(card.id, "end");
        assert_eq!(card.component_type, "end");
        assert_eq!(card.abilities, component.abilities());
        assert_eq!(component.config(), &EndConfig::default());
    }
}
