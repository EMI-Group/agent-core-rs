//! Unit tests of the todo store and the four todo tools.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_foundation::tool::{Tool, ToolContext};

use super::*;
use crate::schema::{TodoItem, TodoStatus};
use crate::workspace::Workspace;

const SESSION: &str = "s1";
const INVALID_STATUS: &str =
    "Invalid status 'bogus'. Valid values: [\"pending\", \"in_progress\", \"completed\", \"cancelled\"]";
const INSERT_SHAPE: &str =
    "Invalid input for insert action: 'todo_data' must be an object with 'target_id' and 'items'";

fn setup() -> (Arc<TodoStore>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = TodoStore::new(dir.path());
    (store, dir)
}

fn ctx(session: &str) -> ToolContext {
    ToolContext::new().with_session(session)
}

async fn invoke(tool: &dyn Tool, inputs: Value, session: &str) -> Value {
    tool.invoke(inputs, &ctx(session)).await.unwrap()
}

fn create_inputs() -> Value {
    json!({"todos": [
        {"id": "a", "content": "do a", "activeForm": "doing a", "description": "first"},
        {"id": "b", "content": "do b", "activeForm": "doing b", "description": "second"}
    ]})
}

fn task(id: &str) -> Value {
    json!({"id": id, "content": format!("do {id}"), "activeForm": format!("doing {id}"),
           "description": "inserted", "status": "pending"})
}

/// Creates the two shared tasks and returns the store, a modify tool and the temp dir.
async fn seeded() -> (Arc<TodoStore>, TodoModifyTool, tempfile::TempDir) {
    let (store, dir) = setup();
    let create = TodoCreateTool::new(Arc::clone(&store), "en");
    let out = invoke(&create, create_inputs(), SESSION).await;
    assert_eq!(out["success"], json!(true), "{out}");
    let modify = TodoModifyTool::new(Arc::clone(&store), "en");
    (store, modify, dir)
}

async fn ids(store: &TodoStore) -> Vec<String> {
    store
        .load(SESSION)
        .await
        .unwrap()
        .into_iter()
        .map(|item| item.id)
        .collect()
}

/// Runs table-driven `(inputs, expected error)` cases against `tool`.
async fn assert_failures(tool: &dyn Tool, cases: Vec<(Value, &str)>) {
    for (inputs, expected) in cases {
        let out = invoke(tool, inputs, SESSION).await;
        assert_eq!(out["success"], json!(false), "{out}");
        assert_eq!(out["error"], json!(expected));
    }
}

#[test]
fn cards_expose_python_names_and_parallel_safety() {
    let create = TodoCreateTool::card_for("en");
    assert_eq!(
        (create.name.as_str(), create.parallel_safe),
        ("todo_create", true)
    );
    assert_eq!(create.input_params["required"], json!(["todos"]));
    assert_eq!(
        create.input_params["properties"]["todos"]["type"],
        json!("array")
    );

    let list = TodoListTool::card_for("en");
    assert_eq!(
        (list.name.as_str(), list.parallel_safe),
        ("todo_list", true)
    );
    assert_eq!(
        list.input_params["properties"].as_object().unwrap().len(),
        0
    );

    let get = TodoGetTool::card_for("en");
    assert_eq!((get.name.as_str(), get.parallel_safe), ("todo_get", true));
    assert_eq!(get.input_params["required"], json!(["id"]));

    let modify = TodoModifyTool::card_for("en");
    assert_eq!(
        (modify.name.as_str(), modify.parallel_safe),
        ("todo_modify", true)
    );
    for key in ["action", "ids", "todos", "todo_data"] {
        assert!(
            modify.input_params["properties"].get(key).is_some(),
            "missing {key}"
        );
    }
}

/// Returns the sorted property names of a schema object.
fn property_names(schema: &Value) -> Vec<&str> {
    let mut names: Vec<&str> = schema["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    names.sort_unstable();
    names
}

#[test]
fn modify_card_requires_the_six_action_enum() {
    let schema = TodoModifyTool::card_for("en").input_params;
    assert_eq!(schema["required"], json!(["action"]));
    assert_eq!(
        schema["properties"]["action"],
        json!({
            "type": "string",
            "enum": ["update", "delete", "cancel", "append", "insert_after", "insert_before"],
            "description": "One of `update`, `delete`, `cancel`, `append`, `insert_after`, \
                            `insert_before`.",
        })
    );
}

#[test]
fn item_schemas_carry_python_fields_required_lists_and_status_enum() {
    // Python publishes nine item fields; only the required list differs per tool.
    let all_fields = [
        "activeForm",
        "content",
        "depends_on",
        "description",
        "id",
        "meta_data",
        "result_summary",
        "selected_model_id",
        "status",
    ];
    let create = TodoCreateTool::card_for("en").input_params;
    let create_items = &create["properties"]["todos"]["items"];
    assert_eq!(
        create_items["required"],
        json!(["id", "content", "activeForm", "description"])
    );
    assert_eq!(property_names(create_items), all_fields.to_vec());
    assert_eq!(
        create_items["properties"]["status"],
        json!({
            "type": "string",
            "description": "Current lifecycle status.",
            "enum": ["pending", "in_progress", "completed", "cancelled"],
        })
    );

    let modify = TodoModifyTool::card_for("en").input_params;
    let modify_items = &modify["properties"]["todos"]["items"];
    assert_eq!(modify_items["required"], json!(["id"]));
    assert_eq!(property_names(modify_items), all_fields.to_vec());
    assert_eq!(
        modify_items["properties"]["status"]["enum"],
        json!(["pending", "in_progress", "completed", "cancelled"])
    );

    let todo_data = modify["properties"]["todo_data"]["anyOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|branch| branch["type"] == json!("object"))
        .expect("todo_data object branch");
    assert_eq!(todo_data["required"], json!(["target_id", "items"]));
    assert_eq!(
        todo_data["properties"]["items"]["items"]["required"],
        json!(["id"])
    );
    assert_eq!(
        property_names(&todo_data["properties"]["items"]["items"]),
        all_fields.to_vec()
    );
}

#[test]
fn every_card_schema_inlines_its_references() {
    let schemas = [
        TodoCreateTool::card_for("en").input_params,
        TodoListTool::card_for("en").input_params,
        TodoGetTool::card_for("en").input_params,
        TodoModifyTool::card_for("en").input_params,
    ];
    for schema in schemas {
        let serialized = serde_json::to_string(&schema).unwrap();
        assert!(!serialized.contains("$ref"), "dangling ref: {serialized}");
        assert!(!serialized.contains("$defs"), "leftover defs: {serialized}");
        assert!(
            !serialized.contains("\"default\""),
            "schema-only default: {serialized}"
        );
    }
}

#[tokio::test]
async fn paths_sessions_and_store_round_trip_are_confined() {
    let (store, dir) = setup();
    assert_eq!(
        store.path_for(SESSION),
        dir.path().join("s1").join("todo.json")
    );
    assert_eq!(
        TodoStore::for_workspace(&Workspace::new(dir.path())).dir(),
        dir.path().join("todo")
    );
    assert_eq!(session_key(&ToolContext::new()), "default");
    assert_eq!(session_key(&ctx("normal")), "normal");
    for unsafe_id in ["", "..", "../x", "a/b", "a\\b"] {
        assert_eq!(session_key(&ctx(unsafe_id)), "default", "{unsafe_id:?}");
        assert_eq!(
            store.path_for(unsafe_id),
            dir.path().join("default").join("todo.json"),
            "unexpected path for {unsafe_id:?}"
        );
    }

    assert_eq!(store.load(SESSION).await.unwrap(), Vec::new());
    let item = TodoItem {
        id: "a".to_owned(),
        content: "do a".to_owned(),
        active_form: "doing a".to_owned(),
        description: "first".to_owned(),
        ..TodoItem::default()
    };
    store
        .save(SESSION, std::slice::from_ref(&item))
        .await
        .unwrap();
    assert_eq!(store.load(SESSION).await.unwrap(), vec![item]);
    let raw = std::fs::read_to_string(store.path_for(SESSION)).unwrap();
    assert!(raw.contains('\n'), "payload must be pretty printed");
    assert!(raw.contains("\"activeForm\": \"doing a\""));
}

#[tokio::test]
async fn create_list_get_round_trip() {
    let (store, dir) = setup();
    let create = TodoCreateTool::new(Arc::clone(&store), "en");
    let out = invoke(&create, create_inputs(), SESSION).await;
    assert_eq!(out["success"], json!(true), "{out}");
    let message = out["data"]["message"].as_str().unwrap();
    assert_eq!(
        message,
        "Successfully created 2 task(s):\n  [→] task_id: a , content: do a\n  \
         [ ] task_id: b , content: do b\n\nNext step: Immediately execute task 'do a'"
    );
    assert_eq!(create.render_for_llm(&out), message);

    let raw = std::fs::read_to_string(dir.path().join(SESSION).join("todo.json")).unwrap();
    let items = serde_json::from_str::<Value>(&raw).unwrap();
    assert_eq!(
        (
            &items[0]["activeForm"],
            &items[0]["status"],
            &items[1]["status"]
        ),
        (&json!("doing a"), &json!("in_progress"), &json!("pending"))
    );

    let list = TodoListTool::new(Arc::clone(&store), "en");
    let out = invoke(&list, json!({}), SESSION).await;
    assert_eq!(
        out["data"]["todos"],
        json!([
            {"id": "a", "content": "do a", "status": "in_progress", "depends_on": []},
            {"id": "b", "content": "do b", "status": "pending", "depends_on": []}
        ])
    );
    assert_eq!(
        list.render_for_llm(&out),
        "- [in_progress] a: do a\n- [pending] b: do b"
    );

    let get = TodoGetTool::new(Arc::clone(&store), "en");
    let out = invoke(&get, json!({"id": "b"}), SESSION).await;
    assert_eq!(out["data"]["todo"]["activeForm"], json!("doing b"));
    assert_eq!(
        get.render_for_llm(&out),
        "id: b\ncontent: do b\nactiveForm: doing b\ndescription: second\nstatus: pending\ndepends_on: []"
    );
}

#[tokio::test]
async fn create_validation_messages() {
    let (store, _dir) = setup();
    let create = TodoCreateTool::new(store, "en");
    assert_failures(
        &create,
        vec![
            (
                json!({}),
                "'todos' parameter is required and must be a JSON array",
            ),
            (
                json!({"todos": "nope"}),
                "'todos' parameter is required and must be a JSON array",
            ),
            (json!({"todos": []}), "Task list cannot be empty"),
            (
                json!({"todos": [{"id": "a", "activeForm": "x", "description": "d"}]}),
                "Task at index 0 is missing a 'content' field",
            ),
            (
                json!({"todos": [{"id": "a", "content": "c", "description": "d"}]}),
                "Task at index 0 is missing a 'activeForm' field",
            ),
            (
                json!({"todos": [{"id": "a", "content": "c", "activeForm": "x"}]}),
                "Task at index 0 is missing a 'description' field",
            ),
            (
                json!({"todos": [
                    {"id": "a", "content": "c", "activeForm": "x", "description": "d"},
                    {"id": "a", "content": "c2", "activeForm": "x2", "description": "d2"}
                ]}),
                "Duplicate task id 'a' at index 1",
            ),
        ],
    )
    .await;

    // A missing id gets a uuid and a bogus status is ignored.
    let inputs = json!({"todos": [{"content": "c", "activeForm": "x", "description": "d",
                                   "status": "bogus"}]});
    let out = invoke(&create, inputs, SESSION).await;
    assert_eq!(out["success"], json!(true), "{out}");
    assert!(out["data"]["message"]
        .as_str()
        .unwrap()
        .contains("[→] task_id: "));
}

#[tokio::test]
async fn list_hides_finished_items_and_get_reports_unknown_ids() {
    let (store, _dir) = setup();
    let list = TodoListTool::new(Arc::clone(&store), "en");
    let out = invoke(&list, json!({}), SESSION).await;
    assert_eq!(out["data"]["todos"], json!([]));
    assert_eq!(list.render_for_llm(&out), "No active tasks.");

    let create = TodoCreateTool::new(Arc::clone(&store), "en");
    invoke(&create, create_inputs(), SESSION).await;
    let modify = TodoModifyTool::new(Arc::clone(&store), "en");
    let inputs = json!({"action": "update", "todos": [
        {"id": "a", "status": "completed"}, {"id": "b", "status": "cancelled"}
    ]});
    assert_eq!(
        invoke(&modify, inputs, SESSION).await["success"],
        json!(true)
    );
    let out = invoke(&list, json!({}), SESSION).await;
    assert_eq!(out["data"]["todos"], json!([]));
    assert_eq!(list.render_for_llm(&out), "No active tasks.");

    let get = TodoGetTool::new(store, "en");
    assert_failures(
        &get,
        vec![
            (json!({}), "Task ID is required"),
            (json!({"id": ""}), "Task ID is required"),
            (json!({"id": "ghost"}), "Task with id 'ghost' not found"),
        ],
    )
    .await;
}

#[tokio::test]
async fn modify_updates_fields_and_validates_the_patch() {
    let (store, modify, _dir) = seeded().await;
    let inputs = json!({"action": "update", "todos": [
        {"id": "a", "status": "completed"},
        {"id": "b", "status": "in_progress", "content": "do b now"}
    ]});
    let out = invoke(&modify, inputs, SESSION).await;
    assert_eq!(
        out["data"]["message"],
        json!("Successfully updated 2 task(s)")
    );
    assert_eq!(
        modify.render_for_llm(&out),
        "Successfully updated 2 task(s)"
    );
    let todos = store.load(SESSION).await.unwrap();
    assert_eq!(
        (todos[0].status, todos[1].status),
        (TodoStatus::Completed, TodoStatus::InProgress)
    );
    assert_eq!(
        (todos[1].content.as_str(), todos[1].active_form.as_str()),
        ("do b now", "doing b")
    );

    assert_failures(
        &modify,
        vec![
            (
                json!({"action": "update"}),
                "Invalid input for update: 'todos' must be a non-empty array of task objects. \
                 Example: {\"todos\": [{\"id\": \"task_id\", \"status\": \"completed\"}]}",
            ),
            (
                json!({"action": "update", "todos": [{"status": "completed"}]}),
                "Batch update failed: Missing required field: 'id'",
            ),
            (
                json!({"action": "update", "todos": [{"id": "ghost"}]}),
                "Batch update failed: Task with ID 'ghost' not found",
            ),
            (
                json!({"action": "update", "todos": [{"id": "b", "status": "bogus"}]}),
                INVALID_STATUS,
            ),
            (
                json!({"action": "update", "todos": [{"id": "a", "status": "in_progress"}]}),
                "More than one task is marked as 'in_progress' (only one allowed)",
            ),
        ],
    )
    .await;
}

#[tokio::test]
async fn modify_appends_deletes_and_cancels() {
    let (store, modify, _dir) = seeded().await;
    assert_failures(
        &modify,
        vec![
            (
                json!({"action": "append"}),
                "Invalid input for append: 'todos' must be a non-empty array of task objects. \
                 Example: {\"todos\": [{\"id\": \"task_id\", \"content\": \"...\", \
                 \"activeForm\": \"...\", \"description\": \"...\", \"status\": \"pending\"}]}",
            ),
            (
                json!({"action": "append", "todos": [{"id": "c", "content": "do c"}]}),
                "Todo data validation error: Missing required field: 'activeForm'; \
                 Missing required field: 'description'; Missing required field: 'status'",
            ),
            (
                json!({"action": "cancel", "ids": []}),
                "Invalid input for cancel action: 'ids' must be a non-empty list of task IDs",
            ),
        ],
    )
    .await;

    let out = invoke(
        &modify,
        json!({"action": "append", "todos": [task("c")]}),
        SESSION,
    )
    .await;
    assert_eq!(
        out["data"]["message"],
        json!("Successfully appended 1 task(s)")
    );
    assert_eq!(ids(&store).await, vec!["a", "b", "c"]);
    let out = invoke(
        &modify,
        json!({"action": "append", "todos": [task("c")]}),
        SESSION,
    )
    .await;
    assert_eq!(
        out["error"],
        json!("Batch append failed: Task with ID 'c' is duplicated")
    );

    let out = invoke(
        &modify,
        json!({"action": "delete", "ids": ["b", "ghost"]}),
        SESSION,
    )
    .await;
    assert_eq!(
        out["data"]["message"],
        json!("Successfully deleted 1 task(s) (IDs: b)")
    );
    let out = invoke(
        &modify,
        json!({"action": "delete", "ids": ["ghost"]}),
        SESSION,
    )
    .await;
    assert_eq!(
        out["data"]["message"],
        json!("No tasks deleted: None of the provided IDs (ghost) were found")
    );
    let out = invoke(
        &modify,
        json!({"action": "cancel", "ids": ["a", "ghost"]}),
        SESSION,
    )
    .await;
    assert_eq!(
        out["data"]["message"],
        json!("Successfully cancelled 1 task(s) (IDs: a)")
    );
    assert_eq!(
        store.load(SESSION).await.unwrap()[0].status,
        TodoStatus::Cancelled
    );
}

#[tokio::test]
async fn modify_inserts_around_pending_targets() {
    let (store, modify, _dir) = seeded().await;
    let insert = |action: &str, target: &str, id: &str| json!({"action": action, "todo_data": {"target_id": target, "items": [task(id)]}});
    let out = invoke(&modify, insert("insert_after", "a", "c"), SESSION).await;
    assert_eq!(
        out["data"]["message"],
        json!("Successfully inserted 1 task(s) after target task, id: 'a'")
    );
    assert_eq!(ids(&store).await, vec!["a", "c", "b"]);
    assert_failures(
        &modify,
        vec![
            (
                insert("insert_before", "a", "z"),
                "Target task status 'in_progress' doesn't allow insertion.",
            ),
            (
                insert("insert_after", "a", "c"),
                "Insert failed: Task with ID 'c' already exists",
            ),
            (
                insert("insert_after", "ghost", "q"),
                "Target task with ID 'ghost' not found in current todo list",
            ),
        ],
    )
    .await;
    let out = invoke(&modify, insert("insert_before", "b", "z"), SESSION).await;
    assert_eq!(
        out["data"]["message"],
        json!("Successfully inserted 1 task(s) before target task, id: 'b'")
    );
    assert_eq!(ids(&store).await, vec!["a", "c", "z", "b"]);
}

#[tokio::test]
async fn modify_validates_insert_shape_and_actions() {
    let (_store, modify, _dir) = seeded().await;
    let with_data = |target: Value, items: Value| json!({"action": "insert_after", "todo_data": {"target_id": target, "items": items}});
    assert_failures(
        &modify,
        vec![
            (json!({}), "Invalid input: 'action' field is required"),
            (
                json!({"action": ""}),
                "Invalid input: 'action' field is required",
            ),
            (json!({"action": "explode"}), "Invalid action: explode"),
            (json!({"action": "insert_after"}), INSERT_SHAPE),
            (
                json!({"action": "insert_before", "todo_data": {}}),
                INSERT_SHAPE,
            ),
            (
                with_data(json!(""), json!([{}])),
                "Invalid input: todo_data 'target_id' must be a non-empty string",
            ),
            (
                with_data(json!("a"), json!([])),
                "Invalid input: todo_data 'items' must be a non-empty list of todo objects",
            ),
            (
                with_data(json!("a"), json!([{"id": "x"}])),
                "Todo data validation error: Missing required field: 'content'; \
                 Missing required field: 'activeForm'; Missing required field: 'description'; \
                 Missing required field: 'status'",
            ),
            (
                with_data(
                    json!("a"),
                    json!([{"id": "x", "content": "c", "activeForm": "x",
                            "description": "d", "status": "bogus"}]),
                ),
                INVALID_STATUS,
            ),
        ],
    )
    .await;
}

#[tokio::test]
async fn cancelled_context_returns_cancelled() {
    let (store, _dir) = setup();
    let create = TodoCreateTool::new(store, "en");
    let context = ToolContext::new().with_session(SESSION);
    context.cancellation.cancel();
    let error = create.invoke(create_inputs(), &context).await.unwrap_err();
    assert!(matches!(error, openjiuwen_common::error::Error::Cancelled));
}
