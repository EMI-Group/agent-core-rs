//! Behaviour tests for the shared [`TaskManager`](super::TaskManager).
use super::*;
use crate::controller::config::ControllerConfig;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The ids of `tasks`, in order.
fn ids(tasks: &[Task]) -> Vec<&str> {
    tasks.iter().map(|task| task.task_id.as_str()).collect()
}

/// Asserts that `tasks` holds nothing.
fn assert_no_tasks(tasks: &[Task]) {
    assert_eq!(tasks, Vec::<Task>::new(), "expected no tasks");
}

/// A three-level chain: `root` -> `child` -> `grandchild`.
fn hierarchy() -> TaskManager {
    let manager = TaskManager::new();
    manager
        .add_tasks(vec![
            Task::new("s1", "root", "agent"),
            Task::new("s1", "child", "agent").with_parent("root"),
            Task::new("s1", "grandchild", "agent").with_parent("child"),
        ])
        .unwrap();
    manager
}

#[test]
fn add_and_query_tasks() {
    let manager = TaskManager::new();
    assert!(manager.is_empty());
    assert_eq!(manager.len(), 0);
    assert_eq!(manager.next_created_seq(), 1);

    manager
        .add_task(Task::submitted("s1", "t9", "agent", "nine"))
        .unwrap();
    manager
        .add_task(Task::submitted("s1", "t1", "agent", "one"))
        .unwrap();
    manager
        .add_task(Task::submitted("s2", "t5", "agent", "five"))
        .unwrap();

    assert_eq!(manager.len(), 3);
    assert!(!manager.is_empty());
    // Creation order, not id order.
    assert_eq!(ids(&manager.tasks()), vec!["t9", "t1", "t5"]);
    assert_eq!(
        ids(&manager.get_task(None).unwrap()),
        vec!["t9", "t1", "t5"]
    );
    assert_eq!(ids(&manager.session_tasks("s1")), vec!["t9", "t1"]);
    assert_no_tasks(&manager.session_tasks("nope"));
    assert_eq!(
        manager.task("t1").unwrap().description.as_deref(),
        Some("one")
    );
    assert!(manager.task("nope").is_none());
    assert_eq!(manager.next_created_seq(), 4);
}

#[test]
fn duplicate_ids_are_rejected_across_sessions() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    let error = manager
        .add_task(Task::new("s1", "t1", "agent"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: t1 already exists!");
    let other_session = manager
        .add_task(Task::new("s2", "t1", "agent"))
        .unwrap_err();
    assert_eq!(other_session.to_string(), "validation: t1 already exists!");
    assert_eq!(manager.len(), 1);
}

#[test]
fn add_tasks_is_all_or_nothing() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    let collision = manager
        .add_tasks(vec![
            Task::new("s1", "t2", "agent"),
            Task::new("s1", "t1", "agent"),
        ])
        .unwrap_err();
    assert_eq!(collision.to_string(), "validation: t1 already exists!");
    assert_eq!(manager.len(), 1);

    let fresh = TaskManager::new();
    let duplicated = fresh
        .add_tasks(vec![
            Task::new("s1", "a", "agent"),
            Task::new("s1", "a", "agent"),
        ])
        .unwrap_err();
    assert_eq!(duplicated.to_string(), "validation: a already exists!");
    assert!(fresh.is_empty());
}

#[test]
fn submitted_callback_fires_on_insert_and_on_status_change() {
    let manager = TaskManager::new();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    manager.set_on_task_submitted(Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    }));

    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    manager
        .add_task(Task::new("s1", "t2", "agent").with_status(TaskStatus::Working))
        .unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    manager
        .add_tasks(vec![
            Task::new("s1", "t3", "agent"),
            Task::new("s1", "t4", "agent").with_status(TaskStatus::Working),
        ])
        .unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 2);

    manager.update_task_status("t2", TaskStatus::Submitted, false, false, None);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
    manager.update_task_status("t2", TaskStatus::Working, false, false, None);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
    manager.update_task_status("missing", TaskStatus::Submitted, false, false, None);
    assert_eq!(hits.load(Ordering::SeqCst), 3);

    manager.clear_on_task_submitted();
    manager.update_task_status("t2", TaskStatus::Submitted, false, false, None);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
}

#[test]
fn update_reports_unknown_ids_and_still_applies_known_ones() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    let mut known = manager.task("t1").unwrap();
    known.status = TaskStatus::Working;
    assert!(manager.update_task(known.clone()));
    assert_eq!(manager.task("t1").unwrap().status, TaskStatus::Working);

    assert!(!manager.update_task(Task::new("s1", "ghost", "agent")));
    let mut ghost = Task::new("s1", "ghost", "agent");
    ghost.description = Some("never stored".to_owned());
    let updated = manager.update_tasks(vec![known, ghost]);
    assert!(!updated);
    assert_eq!(manager.len(), 1);
    assert!(manager.task("ghost").is_none());

    manager.add_task(Task::new("s1", "t2", "agent")).unwrap();
    let mut landed = manager.task("t1").unwrap();
    landed.priority = 4;
    let mut second = manager.task("t2").unwrap();
    second.status = TaskStatus::Completed;
    assert!(manager.update_tasks(vec![landed, second]));
    assert_eq!(manager.task("t1").unwrap().priority, 4);
    assert_eq!(manager.task("t2").unwrap().status, TaskStatus::Completed);
    assert_eq!(manager.len(), 2);
}

#[test]
fn update_moves_the_priority_and_hierarchy_indexes() {
    let manager = hierarchy();
    let mut child = manager.task("child").unwrap();
    child.parent_task_id = None;
    child.priority = 6;
    assert!(manager.update_task(child));
    assert_eq!(
        ids(&manager.get_child_task("root", true)),
        Vec::<&str>::new()
    );
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::by_priority(6))).unwrap()),
        vec!["child"]
    );
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::roots())).unwrap()),
        vec!["root", "child"]
    );
}

#[test]
fn returned_tasks_are_deep_copies() {
    let manager = TaskManager::new();
    manager
        .add_task(
            Task::new("s1", "t1", "agent")
                .with_description("original")
                .with_metadata("user_id", "u1"),
        )
        .unwrap();

    let mut copy = manager.task("t1").unwrap();
    copy.description = Some("mutated".to_owned());
    copy.status = TaskStatus::Completed;
    copy.metadata.insert("user_id".to_owned(), json!("u2"));
    let mut listed = manager.tasks();
    listed[0].priority = 99;
    let mut filtered = manager
        .get_task(Some(&TaskFilter::by_session("s1")))
        .unwrap();
    filtered[0].task_type = "mutated".to_owned();

    let stored = manager.task("t1").unwrap();
    assert_eq!(stored.description.as_deref(), Some("original"));
    assert_eq!(stored.status, TaskStatus::Submitted);
    assert_eq!(stored.priority, 1);
    assert_eq!(stored.task_type, "agent");
    assert_eq!(stored.metadata.get("user_id"), Some(&json!("u1")));
}

#[test]
fn primary_filters_union_and_narrow() {
    let manager = TaskManager::new();
    manager
        .add_tasks(vec![
            Task::new("s1", "t1", "agent").with_metadata("user_id", "u1"),
            Task::new("s1", "t2", "agent").with_metadata("user_id", "u2"),
            Task::new("s2", "t3", "agent").with_metadata("user_id", "u1"),
            Task::new("s2", "t4", "agent").with_priority(5),
        ])
        .unwrap();

    assert_eq!(
        ids(&manager
            .get_task(Some(&TaskFilter::by_session("s1")))
            .unwrap()),
        vec!["t1", "t2"]
    );
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::by_id("t4"))).unwrap()),
        vec!["t4"]
    );

    let union = TaskFilter {
        task_id: Some(TaskIdSelector::Many(vec!["t4".to_owned()])),
        session_id: Some("s1".to_owned()),
        ..TaskFilter::default()
    };
    assert_eq!(
        ids(&manager.get_task(Some(&union)).unwrap()),
        vec!["t1", "t2", "t4"]
    );

    manager.update_task_status("t2", TaskStatus::Working, false, false, None);
    let working = TaskFilter {
        session_id: Some("s1".to_owned()),
        status: Some(TaskStatus::Working),
        ..TaskFilter::default()
    };
    assert_eq!(ids(&manager.get_task(Some(&working)).unwrap()), vec!["t2"]);

    let by_user = TaskFilter {
        session_id: Some("s2".to_owned()),
        user_id: Some("u1".to_owned()),
        ..TaskFilter::default()
    };
    assert_eq!(ids(&manager.get_task(Some(&by_user)).unwrap()), vec!["t3"]);
    // A task without `metadata["user_id"]` never matches a user filter.
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::by_user("u1"))).unwrap()),
        vec!["t1", "t3"]
    );
    assert_no_tasks(
        &manager
            .get_task(Some(&TaskFilter::by_user("nobody")))
            .unwrap(),
    );

    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::by_priority(5))).unwrap()),
        vec!["t4"]
    );
    assert_no_tasks(&manager.get_task(Some(&TaskFilter::by_priority(2))).unwrap());
    assert_no_tasks(
        &manager
            .get_task(Some(&TaskFilter::by_id("missing")))
            .unwrap(),
    );
}

#[test]
fn with_children_expands_transitively_and_roots_filter() {
    let manager = hierarchy();
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::roots())).unwrap()),
        vec!["root"]
    );
    assert_eq!(
        ids(&manager
            .get_task(Some(&TaskFilter::by_id("root").include_children()))
            .unwrap()),
        vec!["root", "child", "grandchild"]
    );
    assert_eq!(
        ids(&manager
            .get_task(Some(&TaskFilter::by_id("child").with_children(true)))
            .unwrap()),
        vec!["child", "grandchild"]
    );
    assert_eq!(
        ids(&manager
            .get_task(Some(
                &TaskFilter::by_ids(["root", "grandchild"]).include_children()
            ))
            .unwrap()),
        vec!["root", "child", "grandchild"]
    );
}

#[test]
fn empty_filter_is_rejected_by_every_query() {
    let manager = hierarchy();
    let empty = TaskFilter::default();
    for error in [
        manager.get_task(Some(&empty)).unwrap_err(),
        manager.remove_task(Some(&empty)).unwrap_err(),
        manager.pop_task(Some(&empty)).unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("At least one filter parameter"));
    }
    assert_eq!(manager.len(), 3);
}

#[test]
fn none_filter_is_rejected_by_remove_and_pop() {
    let manager = hierarchy();
    assert_eq!(
        manager.remove_task(None).unwrap_err().to_string(),
        "validation: task_filter cannot be None in remove_task"
    );
    assert_eq!(
        manager.pop_task(None).unwrap_err().to_string(),
        "validation: task_filter cannot be None in pop_task"
    );
    assert_eq!(manager.len(), 3);
}

#[test]
fn highest_priority_is_a_pop_only_selector() {
    let manager = TaskManager::new();
    manager
        .add_tasks(vec![
            Task::new("s1", "low", "agent").with_priority(1),
            Task::new("s1", "high", "agent").with_priority(7),
            Task::new("s1", "mid", "agent").with_priority(3),
        ])
        .unwrap();

    assert_eq!(
        manager
            .get_task(Some(&TaskFilter::highest_priority()))
            .unwrap_err()
            .to_string(),
        "validation: Priority 'highest' is not supported in get_task, use pop_task instead"
    );
    assert_eq!(
        manager
            .remove_task(Some(&TaskFilter::highest_priority()))
            .unwrap_err()
            .to_string(),
        "validation: Priority 'highest' is not supported in remove_task"
    );

    let popped = manager
        .pop_task(Some(&TaskFilter::highest_priority()))
        .unwrap();
    assert_eq!(ids(&popped), vec!["high"]);
    assert_eq!(manager.len(), 2);
    let next = manager
        .pop_task(Some(&TaskFilter::highest_priority()))
        .unwrap();
    assert_eq!(ids(&next), vec!["mid"]);

    manager.pop_task(Some(&TaskFilter::roots())).unwrap();
    assert!(manager.is_empty());
    assert_no_tasks(
        &manager
            .pop_task(Some(&TaskFilter::highest_priority()))
            .unwrap(),
    );
}

#[test]
fn get_child_task_is_direct_or_recursive() {
    let manager = hierarchy();
    manager
        .add_task(Task::new("s1", "sibling", "agent"))
        .unwrap();

    assert_eq!(ids(&manager.get_child_task("root", false)), vec!["child"]);
    assert_eq!(
        ids(&manager.get_child_task("root", true)),
        vec!["child", "grandchild"]
    );
    assert_no_tasks(&manager.get_child_task("grandchild", true));
    assert_no_tasks(&manager.get_child_task("missing", true));
}

#[test]
fn claim_is_atomic_and_one_shot() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    manager
        .add_task(Task::new("s1", "done", "agent").with_status(TaskStatus::Completed))
        .unwrap();

    assert!(manager.claim_task("t1"));
    assert_eq!(manager.task("t1").unwrap().status, TaskStatus::Working);
    assert!(!manager.claim_task("t1"));
    assert!(!manager.claim_task("done"));
    assert!(!manager.claim_task("missing"));

    manager.update_task_status("t1", TaskStatus::Completed, false, false, None);
    assert!(!manager.claim_task("t1"));
}

#[test]
fn status_updates_reach_children_and_failed_stores_a_message() {
    let manager = hierarchy();
    manager.update_task_status("root", TaskStatus::Failed, true, true, None);
    for task_id in ["root", "child", "grandchild"] {
        let task = manager.task(task_id).unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.error_message.as_deref(), Some("Task execution failed"));
    }

    manager.update_task_status(
        "root",
        TaskStatus::Failed,
        false,
        false,
        Some("boom".to_owned()),
    );
    assert_eq!(
        manager.task("root").unwrap().error_message.as_deref(),
        Some("boom")
    );

    manager.update_task_status("root", TaskStatus::Paused, true, false, None);
    assert_eq!(manager.task("root").unwrap().status, TaskStatus::Paused);
    assert_eq!(manager.task("child").unwrap().status, TaskStatus::Paused);
    assert_eq!(
        manager.task("grandchild").unwrap().status,
        TaskStatus::Failed
    );

    // A missing id is a no-op.
    manager.update_task_status("missing", TaskStatus::Completed, true, true, None);
    assert_eq!(manager.len(), 3);
}

#[test]
fn set_priority_moves_a_task_between_levels() {
    let manager = hierarchy();
    manager.set_priority("root", 5, false, false);
    assert_eq!(manager.task("root").unwrap().priority, 5);
    assert_eq!(manager.task("child").unwrap().priority, 1);
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::by_priority(5))).unwrap()),
        vec!["root"]
    );

    manager.set_priority("root", 5, false, false);
    assert_eq!(
        ids(&manager.get_task(Some(&TaskFilter::by_priority(5))).unwrap()),
        vec!["root"]
    );

    manager.set_priority("root", 2, true, true);
    for task_id in ["root", "child", "grandchild"] {
        assert_eq!(manager.task(task_id).unwrap().priority, 2);
    }
    assert_no_tasks(&manager.get_task(Some(&TaskFilter::by_priority(5))).unwrap());
    assert_eq!(
        manager
            .get_task(Some(&TaskFilter::by_priority(2)))
            .unwrap()
            .len(),
        3
    );

    manager.set_priority("root", 8, true, false);
    assert_eq!(manager.task("root").unwrap().priority, 8);
    assert_eq!(manager.task("child").unwrap().priority, 8);
    assert_eq!(manager.task("grandchild").unwrap().priority, 2);
}

#[test]
fn snapshot_and_restore_round_trip_through_json() {
    let manager = TaskManager::new();
    manager
        .add_tasks(vec![
            Task::new("s1", "t1", "agent")
                .with_priority(3)
                .with_metadata("user_id", "u1"),
            Task::new("s1", "t2", "agent")
                .with_parent("t1")
                .with_status(TaskStatus::Working),
        ])
        .unwrap();

    let snapshot = manager.snapshot_session("s1");
    assert_eq!(snapshot.task_ids(), vec!["t1".to_owned(), "t2".to_owned()]);
    let wire = serde_json::to_string(&snapshot).unwrap();
    let parsed: TaskManagerState = serde_json::from_str(&wire).unwrap();

    let restored = TaskManager::new();
    restored.restore_session("s1", Some(&parsed)).unwrap();
    assert_eq!(restored.len(), 2);
    assert_eq!(ids(&restored.tasks()), vec!["t1", "t2"]);
    let first = restored.task("t1").unwrap();
    assert_eq!(first.priority, 3);
    assert_eq!(first.metadata.get("user_id"), Some(&json!("u1")));
    let second = restored.task("t2").unwrap();
    assert_eq!(second.parent_task_id.as_deref(), Some("t1"));
    assert_eq!(second.status, TaskStatus::Working);
    assert_eq!(ids(&restored.get_child_task("t1", true)), vec!["t2"]);
    assert_eq!(
        ids(&restored.get_task(Some(&TaskFilter::roots())).unwrap()),
        vec!["t1"]
    );
    assert_eq!(restored.next_created_seq(), manager.next_created_seq());
}

#[test]
fn restore_is_session_scoped_and_clears_with_none() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("a", "a1", "agent")).unwrap();
    manager.add_task(Task::new("b", "b1", "agent")).unwrap();
    let snapshot = manager.snapshot_session("a");
    manager.add_task(Task::new("a", "a2", "agent")).unwrap();

    manager.restore_session("a", Some(&snapshot)).unwrap();
    assert_eq!(ids(&manager.tasks()), vec!["b1", "a1"]);
    assert!(manager.task("a2").is_none());
    assert!(manager.task("b1").is_some());

    manager.restore_session("a", None).unwrap();
    assert_eq!(ids(&manager.tasks()), vec!["b1"]);
    // Clearing an absent session is not an error.
    manager.restore_session("c", None).unwrap();
    assert_eq!(manager.len(), 1);
}

#[test]
fn restore_rejects_an_id_owned_by_another_session() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("b", "shared", "agent")).unwrap();
    manager.add_task(Task::new("a", "a1", "agent")).unwrap();
    let state = TaskManagerState::new(
        vec![
            Task::new("a", "a1", "agent").with_status(TaskStatus::Working),
            Task::new("a", "shared", "agent"),
        ],
        4,
    );

    let error = manager.restore_session("a", Some(&state)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: Task ID already belongs to another session: shared"
    );
    // The rejected restore left both sessions untouched.
    assert_eq!(ids(&manager.tasks()), vec!["shared", "a1"]);
    assert_eq!(manager.task("a1").unwrap().status, TaskStatus::Submitted);
}

#[test]
fn active_helpers_track_submitted_and_working() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    manager
        .add_task(Task::new("s2", "t2", "agent").with_status(TaskStatus::Completed))
        .unwrap();
    assert!(manager.has_active_tasks());
    assert!(manager.session_has_active_tasks("s1"));
    assert!(!manager.session_has_active_tasks("s2"));
    assert!(!manager.session_has_active_tasks("nope"));

    manager.update_task_status("t1", TaskStatus::Completed, false, false, None);
    assert!(!manager.has_active_tasks());
    manager.update_task_status("t1", TaskStatus::Working, false, false, None);
    assert!(manager.has_active_tasks());
    assert!(manager.session_has_active_tasks("s1"));
}

#[test]
fn clones_share_one_store() {
    let manager = TaskManager::new();
    let clone = manager.clone();
    clone.add_task(Task::new("s1", "t1", "agent")).unwrap();
    assert_eq!(manager.len(), 1);
    clone.clear();
    assert!(manager.is_empty());
}

#[test]
fn config_round_trips() {
    let manager = TaskManager::new();
    assert_eq!(manager.config(), ControllerConfig::default());

    let config = ControllerConfig::new()
        .with_max_tasks(9)
        .with_enable_task_persistence(false);
    let configured = TaskManager::with_config(config.clone());
    assert_eq!(configured.config(), config);
    configured.set_config(ControllerConfig::default());
    assert_eq!(configured.config(), ControllerConfig::default());
}

#[test]
fn clear_keeps_the_sequence_monotonic() {
    let manager = TaskManager::new();
    manager.add_task(Task::new("s1", "t1", "agent")).unwrap();
    manager.clear();
    assert!(manager.is_empty());
    assert_no_tasks(&manager.tasks());
    manager.add_task(Task::new("s1", "t2", "agent")).unwrap();
    assert_eq!(manager.next_created_seq(), 3);
}
