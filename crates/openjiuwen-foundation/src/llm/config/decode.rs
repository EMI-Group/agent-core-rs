//! Serde helpers that let a Python-serialized model config decode.
//!
//! Python writes `timeout`, `stream_first_chunk_timeout` and
//! `stream_idle_timeout` as `float` seconds (`360.0`), `stop` as one optional
//! string, and `custom_headers` as `dict[str, Any]`. Rust's `Option<Duration>`,
//! `Vec<String>` and `BTreeMap<String, String>` reject those shapes, so the
//! fields carry the `deserialize_with` / `serialize_with` helpers below:
//!
//! - durations decode from a JSON number of seconds (integer or float), a
//!   numeric string, or Rust's own `{"secs", "nanos"}` object (backward
//!   compatibility), and serialize as a number of seconds (`360` → `360.0`);
//!   `null` stays `null`, and zero/negative values are rejected the way Python's
//!   `gt=0` rejects them;
//! - `stop` decodes from one string, `null`, or an array of strings, and always
//!   serializes as an array (the documented Rust shape);
//! - headers decode from a map whose values may be strings, numbers, or booleans
//!   (coerced with `to_string()`); `null` values are dropped and arrays/objects
//!   are errors.
//!
//! Every helper is reached only through a field attribute; none of them is part
//! of the crate's public surface.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serializer};
use serde_json::Value;

/// Seconds in one millionth-of-a-second unit, used to render sub-second parts.
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// Decodes an optional timeout given in Python (`float`) seconds.
///
/// Accepts `null` (`None`), an integer or float number of seconds, a string
/// holding a number of seconds, and the `{"secs", "nanos"}` object Rust's
/// derived `Duration` serializer used to write.
///
/// # Errors
///
/// Returns a deserialization error when the number is not finite or not
/// positive, when a string does not hold a number, when the object form is
/// missing `secs` or carries an unknown field, or when the value overflows
/// [`Duration`].
pub(super) fn deserialize_duration<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
where
    D: Deserializer<'de>,
{
    struct DurationVisitor;

    impl<'de> Visitor<'de> for DurationVisitor {
        type Value = Option<Duration>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .write_str("a number of seconds, a numeric string, a {secs, nanos} object, or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D2: Deserializer<'de>>(
            self,
            deserializer: D2,
        ) -> Result<Self::Value, D2::Error> {
            deserialize_duration(deserializer)
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
            seconds_to_duration(value as f64).map(Some)
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
            seconds_to_duration(value as f64).map(Some)
        }

        fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
            seconds_to_duration(value).map(Some)
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
            parse_seconds(value).map(Some)
        }

        fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
            parse_seconds(&value).map(Some)
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut secs: Option<u64> = None;
            let mut nanos: Option<u64> = None;
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "secs" => secs = Some(map.next_value()?),
                    "nanos" => nanos = Some(map.next_value()?),
                    other => return Err(de::Error::unknown_field(other, &["secs", "nanos"])),
                }
            }
            let secs = secs.ok_or_else(|| de::Error::missing_field("secs"))?;
            Duration::from_secs(secs)
                .checked_add(Duration::from_nanos(nanos.unwrap_or(0)))
                .map(Some)
                .ok_or_else(|| de::Error::custom("duration overflows `Duration`"))
        }
    }

    deserializer.deserialize_any(DurationVisitor)
}

/// Writes an optional timeout as a number of seconds (`360.0`), or `null`.
///
/// Serde's `serialize_with` hands the helper a reference to the field, so the
/// `&Option<Duration>` shape is required here.
///
/// # Errors
///
/// Returns the serializer's error for a failing writer.
#[allow(clippy::ref_option)]
pub(super) fn serialize_duration<S>(
    value: &Option<Duration>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        None => serializer.serialize_none(),
        Some(duration) => serializer.serialize_f64(duration_seconds(*duration)),
    }
}

/// Decodes Python's single-or-list `stop` field into the Rust list shape.
///
/// Python types the field `Union[Optional[str], None]`, so a ported config
/// carries one string (or `null`); an array of strings is accepted too because
/// the wire protocols support several sequences.
///
/// # Errors
///
/// Returns a deserialization error for any other JSON type.
pub(super) fn deserialize_stop<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StopVisitor;

    impl<'de> Visitor<'de> for StopVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a stop sequence string, an array of strings, or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_some<D2: Deserializer<'de>>(
            self,
            deserializer: D2,
        ) -> Result<Self::Value, D2::Error> {
            deserialize_stop(deserializer)
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
            Ok(vec![value.to_owned()])
        }

        fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
            Ok(vec![value])
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut stop = Vec::new();
            while let Some(sequence) = seq.next_element::<String>()? {
                stop.push(sequence);
            }
            Ok(stop)
        }
    }

    deserializer.deserialize_any(StopVisitor)
}

/// Decodes Python's `dict[str, Any]` header map into the Rust string map.
///
/// String values are kept verbatim, numbers and booleans are coerced with
/// `to_string()`, and `null` values are dropped (an unset header).
///
/// # Errors
///
/// Returns a deserialization error when the field is not a map or a value is an
/// array/object.
pub(super) fn deserialize_headers<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Option::<BTreeMap<String, Value>>::deserialize(deserializer)?;
    let Some(raw) = raw else {
        return Ok(BTreeMap::new());
    };
    let mut headers = BTreeMap::new();
    for (name, value) in raw {
        match value {
            Value::Null => {}
            Value::String(text) => {
                headers.insert(name, text);
            }
            Value::Number(number) => {
                headers.insert(name, number.to_string());
            }
            Value::Bool(flag) => {
                headers.insert(name, flag.to_string());
            }
            other => {
                return Err(de::Error::custom(format!(
                    "header `{name}` must be a string, number, or boolean; got `{other}`"
                )));
            }
        }
    }
    Ok(headers)
}

/// Converts a number of seconds into a [`Duration`], rejecting non-positive and
/// non-finite values (Python enforces `gt=0` on all three timeouts).
fn seconds_to_duration<E: de::Error>(seconds: f64) -> Result<Duration, E> {
    if !seconds.is_finite() {
        return Err(de::Error::custom(format!(
            "expected a finite number of seconds, got `{seconds}`"
        )));
    }
    if seconds <= 0.0 {
        return Err(de::Error::custom(format!(
            "expected a positive number of seconds (Python requires `gt=0`), got `{seconds}`"
        )));
    }
    Duration::try_from_secs_f64(seconds).map_err(|error| {
        de::Error::custom(format!("invalid number of seconds `{seconds}`: {error}"))
    })
}

/// Parses a string that holds a number of seconds, tolerating surrounding
/// whitespace (Python callers frequently pass `"360"`).
fn parse_seconds<E: de::Error>(text: &str) -> Result<Duration, E> {
    let seconds = text
        .trim()
        .parse::<f64>()
        .map_err(|_| de::Error::custom(format!("expected a number of seconds, got `{text}`")))?;
    seconds_to_duration(seconds)
}

/// Renders a duration as float seconds, so `360 s` serializes as `360.0`.
fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / NANOS_PER_SECOND
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use super::*;

    /// One struct wired exactly like the two config types, so the helpers are
    /// exercised through serde's derive rather than called directly.
    #[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
    struct DurationSample {
        #[serde(
            default,
            deserialize_with = "deserialize_duration",
            serialize_with = "serialize_duration"
        )]
        timeout: Option<Duration>,
    }

    fn decode_duration(value: Value) -> Result<DurationSample, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn durations_decode_from_python_float_seconds() {
        for value in [json!(360), json!(360.0), json!(12.5)] {
            let sample = decode_duration(json!({ "timeout": value }))
                .unwrap_or_else(|error| panic!("{value} decodes: {error}"));
            assert!(sample.timeout.is_some(), "{value} is set");
        }
        assert_eq!(
            decode_duration(json!({"timeout": 360.0})).unwrap().timeout,
            Some(Duration::from_secs(360))
        );
        assert_eq!(
            decode_duration(json!({"timeout": 0.25})).unwrap().timeout,
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn durations_decode_from_numeric_strings() {
        assert_eq!(
            decode_duration(json!({"timeout": "90"})).unwrap().timeout,
            Some(Duration::from_secs(90))
        );
        assert_eq!(
            decode_duration(json!({"timeout": " 1.5 "}))
                .unwrap()
                .timeout,
            Some(Duration::from_millis(1500))
        );
        assert!(decode_duration(json!({"timeout": "soon"})).is_err());
        assert!(decode_duration(json!({"timeout": ""})).is_err());
    }

    #[test]
    fn durations_decode_from_the_rust_object_form_and_null() {
        assert_eq!(
            decode_duration(json!({"timeout": {"secs": 7, "nanos": 500_000_000}}))
                .unwrap()
                .timeout,
            Some(Duration::new(7, 500_000_000))
        );
        assert_eq!(
            decode_duration(json!({"timeout": null})).unwrap().timeout,
            None
        );
        assert_eq!(decode_duration(json!({})).unwrap().timeout, None);
        assert!(decode_duration(json!({"timeout": {"secs": 1, "other": 2}})).is_err());
        assert!(decode_duration(json!({"timeout": {"nanos": 1}})).is_err());
    }

    #[test]
    fn durations_reject_non_positive_numbers() {
        for value in [json!(-1), json!(-0.5), json!(0), json!(0.0), json!("0")] {
            assert!(
                decode_duration(json!({ "timeout": value })).is_err(),
                "{value} must be rejected"
            );
        }
        assert!(decode_duration(json!({"timeout": true})).is_err());
        assert!(decode_duration(json!({"timeout": [1]})).is_err());
    }

    #[test]
    fn durations_serialize_as_seconds() {
        assert_eq!(
            serde_json::to_value(DurationSample {
                timeout: Some(Duration::from_secs(360)),
            })
            .unwrap(),
            json!({"timeout": 360.0})
        );
        assert_eq!(
            serde_json::to_value(DurationSample {
                timeout: Some(Duration::new(1, 500_000_000)),
            })
            .unwrap(),
            json!({"timeout": 1.5})
        );
        assert_eq!(
            serde_json::to_value(DurationSample { timeout: None }).unwrap(),
            json!({"timeout": null})
        );
    }

    #[test]
    fn durations_round_trip_through_seconds() {
        for duration in [
            Duration::from_secs(1),
            Duration::from_millis(250),
            Duration::from_secs(360),
        ] {
            let original = DurationSample {
                timeout: Some(duration),
            };
            let encoded = serde_json::to_value(&original).unwrap();
            let decoded: DurationSample = serde_json::from_value(encoded).unwrap();
            assert_eq!(decoded, original);
        }
    }

    /// Mirror struct for the `stop` helper.
    #[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
    struct StopSample {
        #[serde(default, deserialize_with = "deserialize_stop")]
        stop: Vec<String>,
    }

    #[test]
    fn stop_decodes_from_a_string_null_and_an_array() {
        let one: StopSample = serde_json::from_value(json!({"stop": "\n"})).unwrap();
        assert_eq!(one.stop, vec!["\n".to_owned()]);

        let empty: StopSample = serde_json::from_value(json!({"stop": null})).unwrap();
        assert_eq!(empty.stop, Vec::<String>::new());

        let many: StopSample = serde_json::from_value(json!({"stop": ["END", "STOP"]})).unwrap();
        assert_eq!(many.stop, vec!["END".to_owned(), "STOP".to_owned()]);

        let missing: StopSample = serde_json::from_value(json!({})).unwrap();
        assert_eq!(missing.stop, Vec::<String>::new());
    }

    #[test]
    fn stop_rejects_unknown_types() {
        for value in [json!(7), json!(true), json!({"a": 1}), json!([1])] {
            assert!(
                serde_json::from_value::<StopSample>(json!({ "stop": value })).is_err(),
                "{value} must be rejected"
            );
        }
    }

    #[test]
    fn stop_always_serializes_as_an_array() {
        assert_eq!(
            serde_json::to_value(StopSample {
                stop: vec!["\n".to_owned()],
            })
            .unwrap(),
            json!({"stop": ["\n"]})
        );
    }

    /// Mirror struct for the header helper.
    #[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
    struct HeaderSample {
        #[serde(default, deserialize_with = "deserialize_headers")]
        headers: BTreeMap<String, String>,
    }

    #[test]
    fn headers_coerce_numbers_and_booleans_and_drop_null() {
        let sample: HeaderSample = serde_json::from_value(json!({
            "headers": {
                "X-Text": "value",
                "X-Rate": 10,
                "X-Ratio": 1.5,
                "X-Debug": true,
                "X-Dropped": null,
            }
        }))
        .unwrap();

        assert_eq!(
            sample.headers.get("X-Text").map(String::as_str),
            Some("value")
        );
        assert_eq!(sample.headers.get("X-Rate").map(String::as_str), Some("10"));
        assert_eq!(
            sample.headers.get("X-Ratio").map(String::as_str),
            Some("1.5")
        );
        assert_eq!(
            sample.headers.get("X-Debug").map(String::as_str),
            Some("true")
        );
        assert!(!sample.headers.contains_key("X-Dropped"));
    }

    #[test]
    fn headers_accept_null_and_are_a_string_map_on_the_wire() {
        let sample: HeaderSample = serde_json::from_value(json!({"headers": null})).unwrap();
        assert!(sample.headers.is_empty());

        let encoded = serde_json::to_value(HeaderSample {
            headers: BTreeMap::from([("X-Rate".to_owned(), "10".to_owned())]),
        })
        .unwrap();
        assert_eq!(encoded, json!({"headers": {"X-Rate": "10"}}));
    }

    #[test]
    fn headers_reject_arrays_and_objects() {
        assert!(serde_json::from_value::<HeaderSample>(json!({
            "headers": {"X-Bad": ["a"]}
        }))
        .is_err());
        assert!(serde_json::from_value::<HeaderSample>(json!({
            "headers": {"X-Bad": {"nested": 1}}
        }))
        .is_err());
        assert!(serde_json::from_value::<HeaderSample>(json!({"headers": ["a"]})).is_err());
    }
}
