# openjiuwen-agent/src/steering

## Intent

Steering queue for the ReAct loop: hosts and rails push instructions while a run is in flight, and the loop drains them at its safe point between steps.
Ports the queue half of Python `openjiuwen/core/single_agent/rail/base.py` (`push_steering`/`drain_steering`) plus the `react_agent.py` batch rendering; the loop wiring lives in `../react/`.

## API Surface

- `SteeringMessage` — `{text, source, metadata}` snake_case serde type; `new`, `with_source`, `with_metadata`, `is_empty`, `to_message` (a `user` `BaseMessage`; metadata rides under `STEERING_METADATA_KEY = "_openjiuwen_steering"`).
- `SteeringQueue` — cheaply cloned `Arc<Mutex<VecDeque<..>>>` FIFO; `new`, `push`, `push_text`, `drain`, `drain_limit`, `drain_messages`, `snapshot`, `len`, `is_empty`, `clear`, `drain_prompt`.
- `drain_limit(Option<usize>)` removes at most `max(1, n)` messages, oldest first, so `Some(0)` still makes progress and `None` drains the whole backlog (Python's `drain_steering(limit)` cap).
- A capped drain skips blank messages without spending the cap, and every message beyond the cap stays queued in order for the next `drain`/`drain_limit`.
- `drain_prompt` joins drained texts with `"\n\n"` and returns `None` when nothing was queued; it is the standalone text-block renderer over `drain`.
- The `ReAct` loop drains through `drain_limit` at its steering-drain point, so a cap supplied by a rail is honoured.

## Constraints

- The queue is infallible: no `Result`, no panics, no `unwrap` outside tests.
- Blank/whitespace-only messages are dropped on push and skipped on drain.
- Thread-safe through `parking_lot::Mutex`; clones share one backlog.
- Do not add loop behavior here — the queue is self-contained and the ReAct loop consumes it from `../react/`.
