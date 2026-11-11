//! The capabilities a workflow component declares.
//!
//! A component advertises which of the four execution modes it supports; the builder
//! uses those declarations to validate edges (a streaming source may only feed a
//! streaming or collecting consumer) and the engine uses them to decide how a node is
//! driven.
//!
//! This ports `ComponentAbility` from the Python `openjiuwen.core.workflow.base`
//! module. Python's enum serializes as a tuple (an artifact of being a `str`-valued
//! enum); the Rust wire format is the plain lowercase name, which is what the
//! documented workflow spec JSON uses.
//!
//! ```
//! use openjiuwen_workflow::ability::ComponentAbility;
//!
//! assert!(ComponentAbility::Stream.is_streaming());
//! assert_eq!(ComponentAbility::default().as_str(), "invoke");
//! assert_eq!("collect".parse::<ComponentAbility>().unwrap(), ComponentAbility::Collect);
//! ```

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

/// A capability a component declares, mirroring Python's `ComponentAbility`.
///
/// The four abilities form two axes: streamed versus batch output (see
/// [`ComponentAbility::is_streaming`] and [`ComponentAbility::is_batch_output`]) and
/// whether the component consumes its inputs one-by-one or all at once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentAbility {
    /// Runs once and returns a single output value.
    #[default]
    Invoke,
    /// Produces a stream of frames.
    Stream,
    /// Runs once after all streaming inputs have been buffered.
    Collect,
    /// Consumes a stream and produces a stream.
    Transform,
}

impl ComponentAbility {
    /// The lowercase wire name (`"invoke"`, `"stream"`, `"collect"`, `"transform"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Invoke => "invoke",
            Self::Stream => "stream",
            Self::Collect => "collect",
            Self::Transform => "transform",
        }
    }

    /// Every ability, in declaration order.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::Invoke, Self::Stream, Self::Collect, Self::Transform]
    }

    /// Whether the component emits a stream of frames (`Stream` or `Transform`).
    #[must_use]
    pub const fn is_streaming(self) -> bool {
        matches!(self, Self::Stream | Self::Transform)
    }

    /// Whether the component emits one batch value (`Invoke` or `Collect`).
    #[must_use]
    pub const fn is_batch_output(self) -> bool {
        matches!(self, Self::Invoke | Self::Collect)
    }
}

impl fmt::Display for ComponentAbility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ComponentAbility {
    type Err = Error;

    /// Parses a wire name; an unknown name is a validation error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `s` is not one of the four wire names.
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "invoke" => Ok(Self::Invoke),
            "stream" => Ok(Self::Stream),
            "collect" => Ok(Self::Collect),
            "transform" => Ok(Self::Transform),
            other => Err(Error::validation(format!(
                "unknown component ability: {other}"
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    #[test]
    fn default_is_invoke() {
        assert_eq!(ComponentAbility::default(), ComponentAbility::Invoke);
    }

    #[test]
    fn wire_names_are_snake_case() {
        assert_eq!(
            serde_json::to_string(&ComponentAbility::Invoke).unwrap(),
            "\"invoke\""
        );
        assert_eq!(
            serde_json::to_string(&ComponentAbility::Stream).unwrap(),
            "\"stream\""
        );
        assert_eq!(
            serde_json::to_string(&ComponentAbility::Collect).unwrap(),
            "\"collect\""
        );
        assert_eq!(
            serde_json::to_string(&ComponentAbility::Transform).unwrap(),
            "\"transform\""
        );
    }

    #[test]
    fn serde_round_trips_every_ability() {
        for ability in ComponentAbility::all() {
            let wire = serde_json::to_string(&ability).unwrap();
            let parsed: ComponentAbility = serde_json::from_str(&wire).unwrap();
            assert_eq!(parsed, ability);
            assert_eq!(format!("\"{}\"", ability.as_str()), wire);
        }
    }

    #[test]
    fn from_str_accepts_wire_names() {
        for ability in ComponentAbility::all() {
            assert_eq!(
                ability.as_str().parse::<ComponentAbility>().unwrap(),
                ability
            );
        }
    }

    #[test]
    fn from_str_rejects_unknown_names() {
        let error = "parallel".parse::<ComponentAbility>().unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert_eq!(error.message(), "unknown component ability: parallel");
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(ComponentAbility::Transform.to_string(), "transform");
    }

    #[test]
    fn stream_and_transform_are_streaming() {
        assert!(!ComponentAbility::Invoke.is_streaming());
        assert!(ComponentAbility::Stream.is_streaming());
        assert!(!ComponentAbility::Collect.is_streaming());
        assert!(ComponentAbility::Transform.is_streaming());
    }

    #[test]
    fn invoke_and_collect_are_batch_outputs() {
        assert!(ComponentAbility::Invoke.is_batch_output());
        assert!(!ComponentAbility::Stream.is_batch_output());
        assert!(ComponentAbility::Collect.is_batch_output());
        assert!(!ComponentAbility::Transform.is_batch_output());
        for ability in ComponentAbility::all() {
            assert_ne!(ability.is_streaming(), ability.is_batch_output());
        }
    }
}
