# openjiuwen-agent-teams

## Intent

Leader/teammate runtime, worktrees, HITL, and the skill CLI/MCP server.

## API Surface

`openjiuwen-agent-teams` exposes the `openjiuwen_agent_teams` module root; concrete items are defined by the crate architecture.

## Constraints

Follows the shared contracts in `docs/ARCHITECTURE.md`: async-first tokio I/O, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, and `tracing` logging.

## Routing Table

Modules are defined by the crate architecture.
