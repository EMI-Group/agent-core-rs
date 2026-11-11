//! Parameter mapping: split tool inputs into query, path, body and header buckets.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::LazyLock;

use openjiuwen_common::error::{Error, Result};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// URL path-placeholder pattern, e.g. `{id}` in `/users/{id}`.
///
/// The pattern is a compile-time constant, so it cannot fail to compile; the
/// `expect` below documents that invariant instead of hiding it.
#[allow(clippy::expect_used)]
pub(super) static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\{(\w+)\}").expect("the path-placeholder pattern is a valid regex")
});

/// Where an input parameter travels in the HTTP request.
///
/// Python parity: `APIParamLocation` in
/// `openjiuwen/core/foundation/tool/service_api/api_param_mapper.py`. Python's
/// `FORM` variant is not ported: multipart form handling is out of scope, and a
/// `"location": "form"` property is rejected as a validation error.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ParamLocation {
    /// URL query parameter (`?key=value`).
    Query,
    /// URL path parameter (`/users/{id}`).
    Path,
    /// JSON request body; the default when the schema declares no location.
    #[default]
    Body,
    /// HTTP request header.
    Header,
}

impl ParamLocation {
    /// Every location, in a stable order.
    pub const ALL: [Self; 4] = [Self::Path, Self::Query, Self::Header, Self::Body];

    /// Locations whose configured defaults are merged into the mapped values.
    ///
    /// Python parity: `APIParamMapper.map` merges defaults for `PATH`, `QUERY`
    /// and `HEADER` only.
    pub const DEFAULTABLE: [Self; 3] = [Self::Path, Self::Query, Self::Header];

    /// Returns the lower-case wire name, e.g. `"query"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Path => "path",
            Self::Body => "body",
            Self::Header => "header",
        }
    }
}

impl FromStr for ParamLocation {
    type Err = Error;

    /// Parses a location name case-insensitively.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for anything but `query`, `path`, `body` and
    /// `header` — including Python's `form`, which is out of scope.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "query" => Ok(Self::Query),
            "path" => Ok(Self::Path),
            "body" => Ok(Self::Body),
            "header" => Ok(Self::Header),
            other => Err(Error::validation(format!(
                "unsupported parameter location '{other}': only query, path, body and header are supported"
            ))),
        }
    }
}

/// Input parameters grouped by where they travel.
pub type ParamMap = BTreeMap<ParamLocation, BTreeMap<String, Value>>;

/// Splits tool inputs into query, path, body and header buckets.
///
/// Python parity: `APIParamMapper`. The mapper is built from the card's input
/// schema (each property may carry a `"location"` keyword) and the config's
/// default query, header, and path parameters.
///
/// ```
/// use openjiuwen_foundation::tool::restful::{ApiParamMapper, ParamLocation};
/// use serde_json::json;
///
/// let mapper = ApiParamMapper::new(json!({
///     "type": "object",
///     "properties": {
///         "id": {"type": "integer", "location": "path"},
///         "name": {"type": "string", "location": "query"},
///     },
/// }));
/// let params = mapper.map(&json!({"id": 7, "name": "bob"})).unwrap();
/// assert_eq!(params[&ParamLocation::Path]["id"], json!(7));
/// assert_eq!(params[&ParamLocation::Query]["name"], json!("bob"));
/// assert_eq!(
///     mapper.render_url("https://api.example.com/users/{id}", &params).unwrap(),
///     "https://api.example.com/users/7",
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ApiParamMapper {
    schema: Value,
    defaults: BTreeMap<ParamLocation, BTreeMap<String, Value>>,
}

impl ApiParamMapper {
    /// Creates a mapper for `schema` with no configured defaults.
    #[must_use]
    pub fn new(schema: Value) -> Self {
        Self::with_defaults(schema, BTreeMap::new(), BTreeMap::new(), BTreeMap::new())
    }

    /// Creates a mapper with the config's default queries, headers, and paths.
    ///
    /// Defaults are merged into the mapped values of their own location, with
    /// input values winning unless they are `null` or the empty string (Python
    /// parity).
    #[must_use]
    pub fn with_defaults(
        schema: Value,
        queries: BTreeMap<String, Value>,
        headers: BTreeMap<String, Value>,
        paths: BTreeMap<String, Value>,
    ) -> Self {
        let mut defaults = BTreeMap::new();
        defaults.insert(ParamLocation::Query, queries);
        defaults.insert(ParamLocation::Header, headers);
        defaults.insert(ParamLocation::Path, paths);
        Self { schema, defaults }
    }

    /// Returns the input schema this mapper was built from.
    #[must_use]
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Returns the configured defaults for `location`, if the location takes any.
    #[must_use]
    pub fn defaults_for(&self, location: ParamLocation) -> Option<&BTreeMap<String, Value>> {
        self.defaults.get(&location)
    }

    /// Splits `inputs` into the four [`ParamLocation`] buckets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a schema property declares a location
    /// other than `query`, `path`, `body` or `header`; the message names the
    /// property. Python's multipart `form` location falls into this category.
    ///
    /// Only properties declared in `schema.properties` are mapped (Python
    /// iterates the schema, so undeclared input keys are dropped). When the
    /// schema declares no `properties` at all, every input key is sent as a body
    /// parameter — a documented deviation, because Python would silently drop
    /// them.
    pub fn map(&self, inputs: &Value) -> Result<ParamMap> {
        let mut result = empty_param_map();
        let input_object = inputs.as_object();

        match self.schema.get("properties").and_then(Value::as_object) {
            Some(properties) => {
                for (name, property_schema) in properties {
                    let Some(value) = input_object.and_then(|object| object.get(name)) else {
                        continue;
                    };
                    let location = location_for(name, property_schema)?;
                    result
                        .entry(location)
                        .or_default()
                        .insert(name.clone(), value.clone());
                }
            }
            None => {
                if let Some(object) = input_object {
                    let body = result.entry(ParamLocation::Body).or_default();
                    body.extend(
                        object
                            .iter()
                            .map(|(key, value)| (key.clone(), value.clone())),
                    );
                }
            }
        }

        self.merge_defaults(&mut result);
        Ok(result)
    }

    /// Substitutes every `{name}` placeholder in `url` from the path bucket.
    ///
    /// String values are used verbatim; numbers, booleans, arrays and objects are
    /// rendered as compact JSON (Python converts through `str()`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a placeholder has no path value.
    pub fn render_url(&self, url: &str, params: &ParamMap) -> Result<String> {
        let empty = BTreeMap::new();
        let path_params = params.get(&ParamLocation::Path).unwrap_or(&empty);

        let mut missing: Vec<&str> = Vec::new();
        for captures in PLACEHOLDER.captures_iter(url) {
            let Some(name) = captures.get(1).map(|matched| matched.as_str()) else {
                continue;
            };
            if !path_params.contains_key(name) && !missing.contains(&name) {
                missing.push(name);
            }
        }
        if !missing.is_empty() {
            return Err(Error::validation(format!(
                "missing values for URL path parameters {} in '{url}'",
                quoted_set(&missing)
            )));
        }

        let rendered = PLACEHOLDER.replace_all(url, |captures: &regex::Captures<'_>| {
            captures
                .get(1)
                .and_then(|matched| path_params.get(matched.as_str()))
                .map_or_else(String::new, value_to_text)
        });
        Ok(rendered.into_owned())
    }

    /// Merges the configured defaults into `result`.
    ///
    /// Python parity: defaults are loaded first and then overridden by input
    /// values that are neither `null` nor `""`.
    fn merge_defaults(&self, result: &mut ParamMap) {
        for location in ParamLocation::DEFAULTABLE {
            let Some(defaults) = self.defaults.get(&location) else {
                continue;
            };
            let mut merged = defaults.clone();
            let mapped = std::mem::take(result.entry(location).or_default());
            for (key, value) in mapped {
                if !is_blank(&value) {
                    merged.insert(key, value);
                }
            }
            result.insert(location, merged);
        }
    }
}

/// Creates a map with an empty bucket for every [`ParamLocation`].
fn empty_param_map() -> ParamMap {
    ParamLocation::ALL
        .into_iter()
        .map(|location| (location, BTreeMap::new()))
        .collect()
}

/// Returns the location declared by `property_schema`, defaulting to `Body`.
fn location_for(name: &str, property_schema: &Value) -> Result<ParamLocation> {
    match property_schema.get("location") {
        Some(Value::String(raw)) => raw.parse().map_err(|_: Error| {
            Error::validation(format!(
                "unsupported parameter location '{raw}' for parameter '{name}': only query, path, body and header are supported"
            ))
        }),
        _ => Ok(ParamLocation::Body),
    }
}

/// Returns whether `value` is a value that must not override a configured default.
fn is_blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        _ => false,
    }
}

/// Renders a JSON value as request text (Python's `str()`/`urlencode` behavior).
pub(super) fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Renders `"{'a', 'b'}"`, the Python set literal used in the validation messages.
pub(super) fn quoted_set(names: &[&str]) -> String {
    let rendered: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
    format!("{{{}}}", rendered.join(", "))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    fn bucket(params: &ParamMap, location: ParamLocation) -> Value {
        serde_json::to_value(params.get(&location).cloned().unwrap_or_default()).unwrap()
    }

    // port of tests/unit_tests/core/foundation/tool/test_api_param_mapper.py

    #[test]
    fn mapper_routes_every_declared_location() {
        let schema = json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "location": "path"},
                "name": {"type": "string", "location": "query"},
                "data": {"type": "object", "location": "body"},
                "auth_token": {"type": "string", "location": "header"},
                "unlocated": {"type": "string"},
            },
        });
        let mapper = ApiParamMapper::new(schema);

        let params = mapper
            .map(&json!({
                "id": 123,
                "name": "John",
                "data": {"key": "value"},
                "auth_token": "abc123",
                "unlocated": "dropped?",
                "undeclared": "dropped",
            }))
            .unwrap();

        assert_eq!(bucket(&params, ParamLocation::Path), json!({"id": 123}));
        assert_eq!(
            bucket(&params, ParamLocation::Query),
            json!({"name": "John"})
        );
        assert_eq!(
            bucket(&params, ParamLocation::Body),
            json!({"data": {"key": "value"}, "unlocated": "dropped?"})
        );
        assert_eq!(
            bucket(&params, ParamLocation::Header),
            json!({"auth_token": "abc123"})
        );
        assert!(
            !bucket(&params, ParamLocation::Body)
                .as_object()
                .unwrap()
                .contains_key("undeclared"),
            "undeclared input keys are dropped, as in Python"
        );
    }

    #[test]
    fn mapper_forwards_everything_to_the_body_without_properties() {
        let mapper = ApiParamMapper::new(json!({"type": "object"}));
        let params = mapper.map(&json!({"a": 1, "b": "two"})).unwrap();

        assert_eq!(
            bucket(&params, ParamLocation::Body),
            json!({"a": 1, "b": "two"})
        );
        assert_eq!(bucket(&params, ParamLocation::Query), json!({}));
    }

    #[test]
    fn mapper_accepts_upper_case_locations_and_rejects_form() {
        let mapper = ApiParamMapper::new(json!({
            "type": "object",
            "properties": {"q": {"type": "string", "location": "QUERY"}},
        }));
        let params = mapper.map(&json!({"q": "x"})).unwrap();
        assert_eq!(bucket(&params, ParamLocation::Query), json!({"q": "x"}));

        let form = ApiParamMapper::new(json!({
            "type": "object",
            "properties": {"file": {"type": "string", "location": "form"}},
        }));
        let error = form.map(&json!({"file": "a"})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("form"), "{error}");
        assert!(error.to_string().contains("'file'"), "{error}");
    }

    #[test]
    fn mapper_renders_urls_and_reports_missing_path_values() {
        let mapper = ApiParamMapper::new(json!({}));
        let mut params = empty_param_map();
        let path_bucket = params.entry(ParamLocation::Path).or_default();
        path_bucket.insert("id".to_owned(), json!(42));
        path_bucket.insert("flag".to_owned(), json!(true));
        path_bucket.insert("name".to_owned(), json!("a b"));

        assert_eq!(
            mapper
                .render_url("https://api.example.com/users/{id}/{flag}/{name}", &params)
                .unwrap(),
            "https://api.example.com/users/42/true/a b"
        );
        assert_eq!(
            mapper
                .render_url("https://api.example.com/users", &params)
                .unwrap(),
            "https://api.example.com/users"
        );

        let error = mapper
            .render_url("https://api.example.com/users/{id}/{other}", &params)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("'other'"), "{error}");
    }

    #[test]
    fn param_location_serde_is_lower_case() {
        assert_eq!(
            serde_json::to_string(&ParamLocation::Header).unwrap(),
            "\"header\""
        );
        assert_eq!(
            serde_json::from_str::<ParamLocation>("\"query\"").unwrap(),
            ParamLocation::Query
        );
        assert_eq!(ParamLocation::default(), ParamLocation::Body);
        assert_eq!(ParamLocation::as_str(ParamLocation::Path), "path");
        assert!(ParamLocation::from_str("form").is_err());
    }
}
