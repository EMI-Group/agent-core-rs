//! Value templating shared by every `http_request` field.
//!
//! Python's `_resolve_placeholders` replaces `{{key}}` with the top-level input `key`;
//! this module adds embedded `${path}` references (resolved against the component inputs
//! with [`crate::reference::resolve_path`]) as a Rust extension. Both forms are applied to
//! the URL, the method, header values, string query-parameter values, and the body fields.

use serde_json::Value;

use crate::reference;

/// Resolves `${path}` references embedded in `template`; unresolvable ones stay literal.
pub(crate) fn resolve_embedded_refs(template: &str, inputs: &Value) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("${") {
        let (before, candidate) = rest.split_at(start);
        rendered.push_str(before);
        let Some(end) = candidate.find('}') else {
            rendered.push_str(candidate);
            return rendered;
        };
        let path = &candidate[2..end];
        match reference::resolve_path(inputs, path) {
            Some(value) if !path.is_empty() => rendered.push_str(&stringify(&value)),
            _ => rendered.push_str(&candidate[..=end]),
        }
        rest = &candidate[end + 1..];
    }
    rendered.push_str(rest);
    rendered
}

/// Replaces `{{key}}` placeholders with the top-level input values (Python parity).
pub(crate) fn resolve_placeholders(template: &str, inputs: &Value) -> String {
    let Value::Object(map) = inputs else {
        return template.to_owned();
    };
    let mut rendered = template.to_owned();
    for (key, value) in map {
        rendered = rendered.replace(&format!("{{{{{key}}}}}"), &stringify(value));
    }
    rendered
}

/// Resolves `${path}` references first, then `{{key}}` placeholders.
pub(crate) fn resolve_template(template: &str, inputs: &Value) -> String {
    resolve_placeholders(&resolve_embedded_refs(template, inputs), inputs)
}

/// Renders a value as text: strings verbatim, everything else compact JSON.
pub(crate) fn stringify(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        // Serializing a `Value` cannot fail; the fallback keeps the path panic-free.
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn templating_resolves_braces_and_embedded_references() {
        let inputs = json!({"url": "https://example.com", "n": 3, "flag": true});
        assert_eq!(
            resolve_template("{{url}}/items/{{n}}?f={{flag}}", &inputs),
            "https://example.com/items/3?f=true"
        );
        assert_eq!(
            resolve_template("{{url}}/${n}/${missing}", &inputs),
            "https://example.com/3/${missing}"
        );
        // Text that looks like a placeholder but has no matching key stays literal.
        assert_eq!(resolve_template("{{nope}}", &inputs), "{{nope}}");
        // A scalar input object leaves the template untouched.
        assert_eq!(resolve_template("{{a}}", &json!("x")), "{{a}}");
        // An unterminated `${` reference is copied verbatim.
        assert_eq!(resolve_template("${a", &inputs), "${a");
    }

    #[test]
    fn stringify_renders_strings_verbatim_and_rest_as_json() {
        assert_eq!(stringify(&json!("x")), "x");
        assert_eq!(stringify(&json!({"a": 1})), r#"{"a":1}"#);
        assert_eq!(stringify(&json!([1, 2])), "[1,2]");
        assert_eq!(stringify(&Value::Null), "null");
    }
}
