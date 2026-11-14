//! Prompt text and response schema for the model-backed consolidation pipeline.
//!
//! Python's `openjiuwen.core.memory.dreaming` package has no consolidation prompt at all:
//! it ships only the idle scheduler, and the LLM-side merge/summarise decisions live in
//! `process/extract` for fact extraction. The prompt and schema below are therefore
//! owned by this crate; they mirror the analyzer conventions (one bounded JSON input, one
//! JSON-only answer, ids never invented) rather than transliterating a Python prompt.
//!
//! [`CONSOLIDATION_SYSTEM_PROMPT`] is the system message; [`CONSOLIDATION_RESPONSE_SCHEMA`]
//! is the JSON Schema of the answer and is appended to the system message so the shape
//! travels with the instructions. The request body is built in
//! [`crate::dreaming::pipeline`] and documented on
//! [`LlmConsolidatorConfig`](crate::dreaming::LlmConsolidatorConfig).
/// System instructions for one consolidation call.
///
/// The rules are deliberately conservative: a model that is unsure must keep a record,
/// because a wrongly merged memory is unrecoverable while a kept duplicate is cheap.
pub const CONSOLIDATION_SYSTEM_PROMPT: &str = "\
You are the memory-consolidation engine of a long-running agent.
You receive one dream job and a bounded list of memory records that belong to one scope.
Decide which records to merge, keep, summarize, or forget.

Rules:
- Only use record ids that appear in the input; never invent an id.
- Merge only records that are redundant or near-duplicates of each other.
- Summarize only clusters of related records whose individual entries are redundant.
- Forget only records that are stale, superseded, or duplicated beyond a merge.
- When unsure, keep the record unchanged.
- Answer with JSON only: no prose, no code fences, no comments.

Action shapes:
- {\"action\":\"merge\",\"target_id\":\"<kept id>\",\"source_ids\":[\"<absorbed id>\", ...],\"content\":\"<merged text>\",\"reason\":\"...\"}
- {\"action\":\"keep\",\"id\":\"<record id>\",\"reason\":\"...\"}
- {\"action\":\"summarize\",\"source_ids\":[\"<id>\", ...],\"content\":\"<summary>\",\"reason\":\"...\"}
- {\"action\":\"forget\",\"ids\":[\"<id>\", ...],\"reason\":\"...\"}
";

/// JSON Schema of the consolidation answer, appended to the system message.
///
/// The parser is more tolerant than this schema (it accepts `id`/`ids`, `op`/`type`
/// action aliases, and text inside code fences), but the schema is what the model is
/// asked to produce, so a conforming answer always parses.
pub const CONSOLIDATION_RESPONSE_SCHEMA: &str = "\
Answer schema:
{
  \"type\": \"object\",
  \"required\": [\"actions\"],
  \"properties\": {
    \"actions\": {
      \"type\": \"array\",
      \"items\": {
        \"type\": \"object\",
        \"required\": [\"action\"],
        \"properties\": {
          \"action\": { \"enum\": [\"merge\", \"keep\", \"summarize\", \"forget\"] },
          \"id\": { \"type\": \"string\", \"description\": \"target id for merge or keep\" },
          \"target_id\": { \"type\": \"string\", \"description\": \"merge target; defaults to the first id\" },
          \"ids\": { \"type\": \"array\", \"items\": { \"type\": \"string\" } },
          \"source_ids\": { \"type\": \"array\", \"items\": { \"type\": \"string\" } },
          \"content\": { \"type\": \"string\", \"description\": \"merged text or summary; defaults to the longest source text\" },
          \"reason\": { \"type\": \"string\" }
        }
      }
    }
  }
}
";

/// The action names the consolidation schema accepts, in documentation order.
pub const CONSOLIDATION_ACTION_NAMES: [&str; 4] = ["merge", "keep", "summarize", "forget"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_names_every_action_the_schema_allows() {
        for action in CONSOLIDATION_ACTION_NAMES {
            assert!(
                CONSOLIDATION_SYSTEM_PROMPT.contains(action),
                "the system prompt should explain `{action}`"
            );
            assert!(
                CONSOLIDATION_RESPONSE_SCHEMA.contains(action),
                "the schema should allow `{action}`"
            );
        }
    }

    #[test]
    fn the_schema_is_valid_json_once_its_header_is_stripped() {
        let body = CONSOLIDATION_RESPONSE_SCHEMA
            .trim()
            .strip_prefix("Answer schema:")
            .expect("a header")
            .trim();
        let value: serde_json::Value = serde_json::from_str(body).expect("valid JSON schema");
        assert_eq!(value["type"], serde_json::json!("object"));
        assert_eq!(
            value["required"],
            serde_json::json!(["actions"]),
            "the only required field is the action list"
        );
    }
}
