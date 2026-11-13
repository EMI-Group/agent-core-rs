//! Controller data model.
//!
//! Ports `openjiuwen/core/controller/schema/`: payload [`DataFrame`]s are the
//! unit of data exchange, [`Event`]s carry input and task signals, [`Intent`]s
//! route an event to an operation, [`Task`]s hold scheduled work, and
//! [`ControllerOutputChunk`]/[`ControllerOutput`] carry the results back.
pub mod dataframe;
pub mod event;
pub mod intent;
pub mod output;
pub mod task;
pub use dataframe::{frames_text, DataFrame, FileDataFrame, JsonDataFrame, TextDataFrame};
pub use event::{Event, EventType};
pub use intent::{Intent, IntentType};
pub use output::{
    ControllerOutput, ControllerOutputChunk, ControllerOutputPayload, OutputKind,
    ALL_TASKS_PROCESSED, CONTROLLER_OUTPUT_TYPE, TASK_PROCESSING,
};
pub use task::{Task, TaskResult, TaskStatus};
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn reexports_cover_the_python_schema_surface() {
        let event = Event::from_user_inputs(&json!("hi")).unwrap();
        assert_eq!(event.event_type, EventType::Input);
        let intent = Intent::default_for(event.clone());
        assert!(intent.validate().is_ok());
        let task = Task::submitted("s1", "t1", "agent", "do it").with_input(event);
        assert!(task.validate().is_ok());
        let output = ControllerOutput::new("done", TaskStatus::Completed).with_frames(vec![
            ControllerOutputChunk::new(
                0,
                ControllerOutputPayload::new(
                    OutputKind::TaskCompletion,
                    vec![DataFrame::text(task.input_text().unwrap_or_default())],
                ),
            ),
        ]);
        assert_eq!(output.frames.len(), 1);
        assert_eq!(
            frames_text(&output.frames[0].payload.clone().unwrap().data).as_deref(),
            Some("hi")
        );
    }
}
