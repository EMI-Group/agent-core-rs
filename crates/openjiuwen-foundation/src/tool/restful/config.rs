//! Transport configuration: [`HttpMethod`] and [`RestfulApiConfig`].

use std::collections::BTreeMap;
use std::str::FromStr;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::Method;
use schemars::JsonSchema;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Default request timeout, in seconds.
///
/// Python parity: `RestfulApiCard.timeout` defaults to `60.0`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Default cap on a response body, in bytes (10 MiB).
///
/// Python parity: `RestfulApiCard.max_response_byte_size` defaults to
/// `10 * 1024 * 1024`.
pub const DEFAULT_MAX_RESPONSE_BYTES: u64 = 10 * 1024 * 1024;

/// HTTP methods accepted by a REST tool.
///
/// Python parity: `RestfulApiCard.SUPPORTED_METHODS`. Serializes as the
/// upper-case wire form (`"GET"`) and deserializes case-insensitively, so
/// `"get"`, `"Get"` and `"GET"` all produce [`HttpMethod::Get`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, JsonSchema)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    /// `GET`.
    Get,
    /// `POST`; the default, matching `RestfulApiCard.method`.
    #[default]
    Post,
    /// `PUT`.
    Put,
    /// `PATCH`.
    Patch,
    /// `DELETE`.
    Delete,
    /// `HEAD`.
    Head,
    /// `OPTIONS`.
    Options,
}

impl HttpMethod {
    /// Returns the method name as used on the wire, e.g. `"GET"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }

    /// Returns whether the mapped body bucket travels as query parameters.
    ///
    /// Python parity: `RestfulApi._async_request` sends the body bucket through
    /// `params` for `GET`, `HEAD`, `OPTIONS` and `DELETE`, and as a JSON body for
    /// `POST`, `PUT` and `PATCH`. This returns `true` for the former group.
    #[must_use]
    pub fn is_bodyless(self) -> bool {
        matches!(self, Self::Get | Self::Head | Self::Options | Self::Delete)
    }
}

impl From<HttpMethod> for Method {
    fn from(method: HttpMethod) -> Self {
        match method {
            HttpMethod::Get => Self::GET,
            HttpMethod::Post => Self::POST,
            HttpMethod::Put => Self::PUT,
            HttpMethod::Patch => Self::PATCH,
            HttpMethod::Delete => Self::DELETE,
            HttpMethod::Head => Self::HEAD,
            HttpMethod::Options => Self::OPTIONS,
        }
    }
}

impl FromStr for HttpMethod {
    type Err = Error;

    /// Parses a method name case-insensitively.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the name is not one of the seven
    /// supported methods.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "GET" => Ok(Self::Get),
            "POST" => Ok(Self::Post),
            "PUT" => Ok(Self::Put),
            "PATCH" => Ok(Self::Patch),
            "DELETE" => Ok(Self::Delete),
            "HEAD" => Ok(Self::Head),
            "OPTIONS" => Ok(Self::Options),
            other => Err(Error::validation(format!(
                "unsupported http method '{other}': only GET, POST, PUT, PATCH, DELETE, HEAD and OPTIONS are supported"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for HttpMethod {
    /// Deserializes a method name case-insensitively.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(de::Error::custom)
    }
}

/// The transport configuration of a REST tool.
///
/// Python parity: the transport fields of `RestfulApiCard`. The card keeps the
/// model-facing description; this struct keeps the url, method, defaults,
/// timeout, and response cap, so it can be persisted and edited independently.
///
/// `timeout` serializes as a number of seconds (integers and floats are both
/// accepted on the way in), matching Python's float `timeout` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RestfulApiConfig {
    /// Absolute `http`/`https` URL; `{name}` placeholders are filled from the
    /// path bucket.
    pub url: String,
    /// HTTP method of the call; `POST` by default.
    pub method: HttpMethod,
    /// Default request headers, merged with the `header`-located inputs.
    pub headers: BTreeMap<String, Value>,
    /// Default query parameters, merged with the `query`-located inputs.
    pub queries: BTreeMap<String, Value>,
    /// Default path parameters, merged with the `path`-located inputs.
    pub paths: BTreeMap<String, Value>,
    /// Total request timeout, in seconds on the wire.
    #[serde(with = "duration_secs")]
    #[schemars(with = "f64")]
    pub timeout: Duration,
    /// Reject responses whose body exceeds this many bytes.
    pub max_response_bytes: u64,
}

impl Default for RestfulApiConfig {
    /// Python defaults: `POST`, no header/query/path defaults, a 60 second
    /// timeout, and a 10 MiB response cap.
    fn default() -> Self {
        Self {
            url: String::new(),
            method: HttpMethod::Post,
            headers: BTreeMap::new(),
            queries: BTreeMap::new(),
            paths: BTreeMap::new(),
            timeout: DEFAULT_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }
}

impl RestfulApiConfig {
    /// Creates a configuration for `url` with every other field defaulted.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Self::default()
        }
    }

    /// Returns this configuration with the given HTTP method.
    #[must_use]
    pub fn with_method(mut self, method: HttpMethod) -> Self {
        self.method = method;
        self
    }

    /// Returns this configuration with one additional default header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: Value) -> Self {
        self.headers.insert(name.into(), value);
        self
    }

    /// Returns this configuration with one additional default query parameter.
    #[must_use]
    pub fn with_query(mut self, name: impl Into<String>, value: Value) -> Self {
        self.queries.insert(name.into(), value);
        self
    }

    /// Returns this configuration with one additional default path parameter.
    #[must_use]
    pub fn with_path(mut self, name: impl Into<String>, value: Value) -> Self {
        self.paths.insert(name.into(), value);
        self
    }

    /// Returns this configuration with the given request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Returns this configuration with the given response-size cap in bytes.
    #[must_use]
    pub fn with_max_response_bytes(mut self, max_response_bytes: u64) -> Self {
        self.max_response_bytes = max_response_bytes;
        self
    }
}

/// Seconds-based `Duration` serde helpers for [`RestfulApiConfig::timeout`].
mod duration_secs {
    use std::fmt;
    use std::time::Duration;

    use serde::de::{self, Unexpected, Visitor};
    use serde::{Deserializer, Serializer};

    /// Upper bound accepted on the way in, so [`Duration::from_secs_f64`] cannot
    /// overflow (about 31.7 billion years).
    const MAX_SECONDS: f64 = 1e18;

    /// Serializes a duration as a whole number of seconds when it is exact, and
    /// as fractional seconds otherwise.
    pub(super) fn serialize<S>(
        duration: &Duration,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if duration.subsec_nanos() == 0 {
            serializer.serialize_u64(duration.as_secs())
        } else {
            serializer.serialize_f64(duration.as_secs_f64())
        }
    }

    /// Deserializes a duration from integer or float seconds.
    pub(super) fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SecondsVisitor;

        impl Visitor<'_> for SecondsVisitor {
            type Value = Duration;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a non-negative number of seconds")
            }

            fn visit_u64<E: de::Error>(self, seconds: u64) -> std::result::Result<Duration, E> {
                Ok(Duration::from_secs(seconds))
            }

            fn visit_i64<E: de::Error>(self, seconds: i64) -> std::result::Result<Duration, E> {
                u64::try_from(seconds)
                    .map(Duration::from_secs)
                    .map_err(|_| E::invalid_value(Unexpected::Signed(seconds), &self))
            }

            fn visit_f64<E: de::Error>(self, seconds: f64) -> std::result::Result<Duration, E> {
                if !seconds.is_finite() || !(0.0..=MAX_SECONDS).contains(&seconds) {
                    return Err(E::invalid_value(Unexpected::Float(seconds), &self));
                }
                Ok(Duration::from_secs_f64(seconds))
            }
        }

        deserializer.deserialize_any(SecondsVisitor)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn http_method_serde_and_schema_use_the_wire_form() {
        assert_eq!(
            serde_json::from_str::<HttpMethod>("\"get\"").unwrap(),
            HttpMethod::Get
        );
        assert_eq!(
            serde_json::from_str::<HttpMethod>("\"GET\"").unwrap(),
            HttpMethod::Get
        );
        assert_eq!(
            serde_json::from_str::<HttpMethod>("\"Options\"").unwrap(),
            HttpMethod::Options
        );
        assert_eq!(
            serde_json::to_string(&HttpMethod::Delete).unwrap(),
            "\"DELETE\""
        );
        assert_eq!(HttpMethod::default(), HttpMethod::Post);
        assert_eq!(HttpMethod::Get.as_str(), "GET");
        assert!(HttpMethod::Get.is_bodyless());
        assert!(!HttpMethod::Patch.is_bodyless());
        assert!(serde_json::from_str::<HttpMethod>("\"TRACE\"").is_err());

        let schema = serde_json::to_value(
            schemars::SchemaGenerator::default().into_root_schema_for::<HttpMethod>(),
        )
        .unwrap();
        // An enum of unit variants is rendered as `oneOf` of `const` values.
        let variants: Vec<Value> = schema
            .get("oneOf")
            .and_then(Value::as_array)
            .map(|variants| {
                variants
                    .iter()
                    .filter_map(|variant| variant.get("const").cloned())
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(variants.len(), 7, "{schema}");
        assert!(variants.contains(&json!("GET")), "{schema}");
        assert!(variants.contains(&json!("OPTIONS")), "{schema}");
    }

    #[test]
    fn config_round_trips_with_seconds_timeout() {
        let config = RestfulApiConfig::new("https://api.example.com/users")
            .with_method(HttpMethod::Get)
            .with_timeout(Duration::from_secs(30))
            .with_max_response_bytes(1024)
            .with_query("lang", json!("en"))
            .with_header("X-Api-Key", json!("key"))
            .with_path("version", json!("v1"));

        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["method"], json!("GET"));
        assert_eq!(value["timeout"], json!(30));
        assert_eq!(value["max_response_bytes"], json!(1024));
        assert_eq!(value["queries"]["lang"], json!("en"));

        let restored: RestfulApiConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);

        let fractional: RestfulApiConfig =
            serde_json::from_value(json!({"url": "https://api.example.com", "timeout": 1.5}))
                .unwrap();
        assert_eq!(fractional.timeout, Duration::from_millis(1500));

        let partial: RestfulApiConfig =
            serde_json::from_value(json!({"url": "https://api.example.com"})).unwrap();
        assert_eq!(partial.method, HttpMethod::Post);
        assert_eq!(partial.timeout, DEFAULT_TIMEOUT);
        assert_eq!(partial.max_response_bytes, DEFAULT_MAX_RESPONSE_BYTES);

        let negative: std::result::Result<RestfulApiConfig, _> =
            serde_json::from_value(json!({"url": "https://api.example.com", "timeout": -1}));
        assert!(negative.is_err(), "a negative timeout must be rejected");
    }
}
