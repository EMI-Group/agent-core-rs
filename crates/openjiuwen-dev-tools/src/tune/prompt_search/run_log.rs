//! The JSONL run log for prompt-search runs.
//!
//! Ports `openjiuwen/dev_tools/tune/optimizer/prompt_search/run_log.py`: an
//! append-only JSONL stream with one record per iteration (the iteration index,
//! the candidate, its rewards, the convergence state, and the adoption decision)
//! plus a reader that tolerates truncated or foreign lines.
