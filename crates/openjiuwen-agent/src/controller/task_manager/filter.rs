//! Task filters: the selection criteria for queries, removals, and pops.
//!
//! Ports `TaskFilter` from
//! `openjiuwen/core/controller/modules/task_manager.py`. Python validates a
//! filter while pydantic constructs it; the Rust filter is a plain data record
//! that deserializes from either wire form of its selectors, so
//! [`TaskFilter::validate`] is called by every manager method that accepts one.
use crate::controller::schema::TaskStatus;
use openjiuwen_common::error::{Error, Result};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Python's message for a filter that names no criterion.
const AT_LEAST_ONE_FILTER: &str = "At least one filter parameter (task_id, session_id, user_id, priority, status, or is_root) must be provided";

/// One task id or an explicit list of task ids.
///
/// Deserializes from `"t1"` and from `["t1", "t2"]`; serializes back to the same
/// shapes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TaskIdSelector {
    /// A single task id.
    One(String),
    /// An explicit list of task ids.
    Many(Vec<String>),
}

impl TaskIdSelector {
    /// Returns the selected ids as an owned vector.
    #[must_use]
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            Self::One(task_id) => vec![task_id.clone()],
            Self::Many(task_ids) => task_ids.clone(),
        }
    }

    /// Whether `task_id` is selected.
    #[must_use]
    pub fn matches(&self, task_id: &str) -> bool {
        match self {
            Self::One(selected) => selected == task_id,
            Self::Many(selected) => selected.iter().any(|entry| entry == task_id),
        }
    }

    /// Number of ids in the selector.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(task_ids) => task_ids.len(),
        }
    }

    /// Whether the selector names no id.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Many(task_ids) if task_ids.is_empty())
    }
}

impl From<&str> for TaskIdSelector {
    fn from(task_id: &str) -> Self {
        Self::One(task_id.to_owned())
    }
}

impl From<String> for TaskIdSelector {
    fn from(task_id: String) -> Self {
        Self::One(task_id)
    }
}

impl From<Vec<String>> for TaskIdSelector {
    fn from(task_ids: Vec<String>) -> Self {
        Self::Many(task_ids)
    }
}

impl<const N: usize> From<[&str; N]> for TaskIdSelector {
    fn from(task_ids: [&str; N]) -> Self {
        Self::Many(task_ids.into_iter().map(str::to_owned).collect())
    }
}

/// One integer priority level, or the `"highest"` marker.
///
/// Deserializes from `2` and from `"highest"`, matching Python's
/// `Union[int, Literal["highest"]]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrioritySelector {
    /// A concrete priority level.
    Level(i32),
    /// The largest priority level present, resolved by
    /// [`pop_task`](super::TaskManager::pop_task).
    Highest,
}

impl PrioritySelector {
    /// Returns the concrete level, or `None` for [`PrioritySelector::Highest`].
    #[must_use]
    pub const fn level(self) -> Option<i32> {
        match self {
            Self::Level(level) => Some(level),
            Self::Highest => None,
        }
    }

    /// Whether the selector is the `"highest"` marker.
    #[must_use]
    pub const fn is_highest(self) -> bool {
        matches!(self, Self::Highest)
    }
}

impl From<i32> for PrioritySelector {
    fn from(priority: i32) -> Self {
        Self::Level(priority)
    }
}

impl Serialize for PrioritySelector {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Level(priority) => serializer.serialize_i32(*priority),
            Self::Highest => serializer.serialize_str("highest"),
        }
    }
}

impl<'de> Deserialize<'de> for PrioritySelector {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SelectorVisitor;

        impl Visitor<'_> for SelectorVisitor {
            type Value = PrioritySelector;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an integer priority or \"highest\"")
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                i32::try_from(value)
                    .map(PrioritySelector::Level)
                    .map_err(|_| E::custom(format!("priority {value} is out of range")))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                i64::try_from(value)
                    .map_err(|_| E::custom(format!("priority {value} is out of range")))
                    .and_then(|value| self.visit_i64(value))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.eq_ignore_ascii_case("highest") {
                    Ok(PrioritySelector::Highest)
                } else {
                    value
                        .parse::<i32>()
                        .map(PrioritySelector::Level)
                        .map_err(|_| E::custom(format!("invalid priority selector {value:?}")))
                }
            }
        }

        deserializer.deserialize_any(SelectorVisitor)
    }
}

/// Selection criteria for [`TaskManager`](super::TaskManager) operations.
///
/// Every field is optional and they combine: the primary selectors
/// (`task_id`, `session_id`, `priority`, `is_root`) are unioned, then `status`
/// and `user_id` narrow the result, and `with_children` finally appends every
/// transitive child task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskFilter {
    /// Restrict the result to these task ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskIdSelector>,
    /// Restrict the result to tasks of this session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Restrict the result to tasks whose `metadata["user_id"]` is this user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Restrict the result to one priority level, or the highest one present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<PrioritySelector>,
    /// Restrict the result to this lifecycle status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<TaskStatus>,
    /// Append every transitive child of each matched task.
    #[serde(default)]
    pub with_children: bool,
    /// Restrict the result to root tasks (tasks without a parent).
    #[serde(default)]
    pub is_root: bool,
}

impl TaskFilter {
    /// Builds an empty filter; it fails [`validate`](Self::validate) until a
    /// selector is set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the filter to one task id.
    #[must_use]
    pub fn by_id(task_id: impl Into<String>) -> Self {
        Self {
            task_id: Some(TaskIdSelector::One(task_id.into())),
            ..Self::default()
        }
    }

    /// Restricts the filter to a list of task ids.
    #[must_use]
    pub fn by_ids<I, S>(task_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            task_id: Some(TaskIdSelector::Many(
                task_ids.into_iter().map(Into::into).collect(),
            )),
            ..Self::default()
        }
    }

    /// Restricts the filter to one session.
    #[must_use]
    pub fn by_session(session_id: impl Into<String>) -> Self {
        Self {
            session_id: Some(session_id.into()),
            ..Self::default()
        }
    }

    /// Restricts the filter to one user (read from `metadata["user_id"]`).
    #[must_use]
    pub fn by_user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            ..Self::default()
        }
    }

    /// Restricts the filter to one status.
    #[must_use]
    pub fn by_status(status: TaskStatus) -> Self {
        Self {
            status: Some(status),
            ..Self::default()
        }
    }

    /// Restricts the filter to one priority level.
    #[must_use]
    pub fn by_priority(priority: i32) -> Self {
        Self {
            priority: Some(PrioritySelector::Level(priority)),
            ..Self::default()
        }
    }

    /// Restricts the filter to the highest priority level present.
    #[must_use]
    pub fn highest_priority() -> Self {
        Self {
            priority: Some(PrioritySelector::Highest),
            ..Self::default()
        }
    }

    /// Restricts the filter to root tasks.
    #[must_use]
    pub fn roots() -> Self {
        Self {
            is_root: true,
            ..Self::default()
        }
    }

    /// Sets [`with_children`](Self::with_children).
    #[must_use]
    pub const fn with_children(mut self, with_children: bool) -> Self {
        self.with_children = with_children;
        self
    }

    /// Appends child tasks to the result.
    #[must_use]
    pub const fn include_children(self) -> Self {
        self.with_children(true)
    }

    /// Sets [`is_root`](Self::is_root).
    #[must_use]
    pub const fn roots_only(mut self, is_root: bool) -> Self {
        self.is_root = is_root;
        self
    }

    /// Whether any primary selector is set.
    pub(super) fn has_primary_selector(&self) -> bool {
        self.task_id.is_some()
            || self.session_id.is_some()
            || self.priority.is_some()
            || self.is_root
    }

    /// Whether the filter names at least one criterion.
    #[must_use]
    pub fn has_selector(&self) -> bool {
        self.has_primary_selector() || self.status.is_some() || self.user_id.is_some()
    }

    /// Whether the filter names neither a primary selector nor a narrowing one.
    ///
    /// A filter that does not select anything is rejected by
    /// [`validate`](Self::validate), mirroring Python's pydantic validator.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with Python's message when none of
    /// `task_id`, `session_id`, `user_id`, `priority`, or `status` is set and
    /// `is_root` is false.
    pub fn validate(&self) -> Result<()> {
        if self.has_selector() {
            Ok(())
        } else {
            Err(Error::validation(AT_LEAST_ONE_FILTER))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_filter_is_rejected_with_python_message() {
        let error = TaskFilter::new().validate().unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert_eq!(
            error.to_string(),
            format!("validation: {AT_LEAST_ONE_FILTER}")
        );
        assert!(TaskFilter::default().validate().is_err());
    }

    #[test]
    fn every_single_selector_passes_validation() {
        for filter in [
            TaskFilter::by_id("t1"),
            TaskFilter::by_ids(["t1", "t2"]),
            TaskFilter::by_session("s1"),
            TaskFilter::by_user("u1"),
            TaskFilter::by_status(TaskStatus::Working),
            TaskFilter::by_priority(2),
            TaskFilter::highest_priority(),
            TaskFilter::roots(),
        ] {
            assert!(filter.validate().is_ok(), "{filter:?}");
            assert!(filter.has_selector());
        }
        assert!(TaskFilter::new().with_children(true).validate().is_err());
    }

    #[test]
    fn filter_deserializes_from_every_selector_shape() {
        let one: TaskFilter = serde_json::from_value(json!({"task_id": "t1"})).unwrap();
        assert_eq!(one.task_id, Some(TaskIdSelector::One("t1".to_owned())));

        let many: TaskFilter = serde_json::from_value(json!({"task_id": ["t1", "t2"]})).unwrap();
        assert_eq!(
            many.task_id,
            Some(TaskIdSelector::Many(vec!["t1".to_owned(), "t2".to_owned()]))
        );

        let integer: TaskFilter = serde_json::from_value(json!({"priority": 2})).unwrap();
        assert_eq!(integer.priority, Some(PrioritySelector::Level(2)));

        let highest: TaskFilter = serde_json::from_value(json!({"priority": "highest"})).unwrap();
        assert_eq!(highest.priority, Some(PrioritySelector::Highest));

        let roots: TaskFilter = serde_json::from_value(json!({"is_root": true})).unwrap();
        assert!(roots.is_root);

        let session: TaskFilter = serde_json::from_value(
            json!({"session_id": "s1", "user_id": "u1", "status": "working"}),
        )
        .unwrap();
        assert_eq!(session.session_id.as_deref(), Some("s1"));
        assert_eq!(session.user_id.as_deref(), Some("u1"));
        assert_eq!(session.status, Some(TaskStatus::Working));
        assert!(!session.with_children);
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let filter: TaskFilter =
            serde_json::from_value(json!({"task_id": "t1", "future_field": [1, 2]})).unwrap();
        assert_eq!(filter.task_id, Some(TaskIdSelector::One("t1".to_owned())));
    }

    #[test]
    fn filter_round_trips_through_json() {
        let filter = TaskFilter::by_ids(["t1", "t2"])
            .include_children()
            .roots_only(true)
            .with_children(true);
        let wire = serde_json::to_value(&filter).unwrap();
        assert_eq!(
            wire,
            json!({"task_id": ["t1", "t2"], "with_children": true, "is_root": true})
        );
        let parsed: TaskFilter = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, filter);
    }

    #[test]
    fn priority_selector_round_trips_and_rejects_nonsense() {
        assert_eq!(
            serde_json::to_value(PrioritySelector::Level(3)).unwrap(),
            json!(3)
        );
        assert_eq!(
            serde_json::to_value(PrioritySelector::Highest).unwrap(),
            json!("highest")
        );
        let highest: PrioritySelector = serde_json::from_value(json!("highest")).unwrap();
        assert_eq!(highest, PrioritySelector::Highest);
        assert_eq!(highest.level(), None);
        assert!(highest.is_highest());
        let level: PrioritySelector = serde_json::from_value(json!(0)).unwrap();
        assert_eq!(level.level(), Some(0));
        assert!(!level.is_highest());
        let negative: PrioritySelector = serde_json::from_value(json!(-1)).unwrap();
        assert_eq!(negative.level(), Some(-1));
        assert!(serde_json::from_value::<PrioritySelector>(json!("lowest")).is_err());
        assert!(serde_json::from_value::<PrioritySelector>(json!(1.5)).is_err());
        assert!(serde_json::from_value::<PrioritySelector>(json!(i64::MAX)).is_err());
    }

    #[test]
    fn task_id_selector_conversions_and_helpers() {
        let one = TaskIdSelector::from("t1");
        assert_eq!(one.to_vec(), vec!["t1".to_owned()]);
        assert!(one.matches("t1"));
        assert!(!one.matches("t2"));
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());

        let many = TaskIdSelector::from(vec!["t1".to_owned(), "t2".to_owned()]);
        assert_eq!(many.to_vec(), vec!["t1".to_owned(), "t2".to_owned()]);
        assert!(many.matches("t2"));
        assert_eq!(many.len(), 2);

        let array = TaskIdSelector::from(["a", "b"]);
        assert_eq!(
            array,
            TaskIdSelector::Many(vec!["a".to_owned(), "b".to_owned()])
        );

        let owned = TaskIdSelector::from(String::from("t9"));
        assert_eq!(owned, TaskIdSelector::One("t9".to_owned()));

        let empty = TaskIdSelector::Many(Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(!empty.matches("t1"));
    }
}
