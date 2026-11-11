//! Integration tests for the structured task manager.
//!
//! Port of `tests/unit_tests/core/common/test_task_manager.py` scenarios that
//! exercise the manager end to end.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use openjiuwen_common::task::{TaskManager, TaskStatus};

#[tokio::test]
async fn many_concurrent_tasks_all_complete() {
    let manager = TaskManager::new();
    let mut handles = Vec::new();
    for index in 0..20 {
        handles.push(manager.spawn(format!("worker-{index}"), async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            index
        }));
    }
    manager
        .wait_all(Some(Duration::from_secs(5)))
        .await
        .unwrap();
    let mut total = 0;
    for handle in handles {
        total += handle.join().await.unwrap();
    }
    assert_eq!(total, (0..20).sum::<i32>());
    let statuses = manager.list();
    assert_eq!(statuses.len(), 20);
    assert!(statuses
        .iter()
        .all(|info| info.status == TaskStatus::Completed));
    assert_eq!(manager.remove_finished(), 20);
    assert!(manager.is_empty());
}

#[tokio::test]
async fn cancel_all_then_wait_all_leaves_every_task_terminal() {
    let manager = TaskManager::new();
    for index in 0..5 {
        manager
            .spawn(format!("sleeper-{index}"), async {
                tokio::time::sleep(Duration::from_secs(30)).await;
            })
            .detach();
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(manager.cancel_all("test teardown"), 5);
    manager
        .wait_all(Some(Duration::from_secs(5)))
        .await
        .unwrap();
    assert!(manager.list().iter().all(|info| info.status.is_terminal()));
}

#[tokio::test]
async fn shutdown_stops_everything_and_does_not_hang() {
    let manager = TaskManager::new();
    for index in 0..3 {
        manager
            .spawn(format!("stubborn-{index}"), async {
                tokio::time::sleep(Duration::from_secs(300)).await;
            })
            .detach();
    }
    tokio::time::timeout(Duration::from_secs(5), manager.shutdown())
        .await
        .unwrap();
    assert!(manager.is_empty());
    assert_eq!(manager.len(), 0);
}

#[tokio::test]
async fn nested_tasks_record_parent_and_cascade() {
    let manager = TaskManager::new();
    let manager_for_parent = manager.clone();
    let parent = manager.spawn("parent", async move {
        for index in 0..3 {
            manager_for_parent
                .spawn(format!("child-{index}"), async {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                })
                .detach();
        }
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    let parent_id = parent.id().clone();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let children = manager
            .list()
            .into_iter()
            .filter(|info| info.parent_id.as_ref() == Some(&parent_id))
            .collect::<Vec<_>>();
        if children.len() == 3 {
            assert!(children
                .iter()
                .all(|info| info.status == TaskStatus::Running));
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "children never spawned"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    assert_eq!(manager.cascade_cancel(&parent_id, "parent_cancelled"), 4);
    assert!(manager
        .list()
        .iter()
        .all(|info| info.status == TaskStatus::Cancelled));
    // Cancellation is cooperative, so stop the sleeping parent explicitly
    // before awaiting it; its status is already terminal.
    parent.abort();
    assert!(parent.join().await.unwrap_err().is_cancelled());
}
