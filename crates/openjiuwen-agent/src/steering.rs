//! Steering: user messages injected into a running agent at safe points.
//!
//! Ports the steering-queue half of Python's `ReActAgent`: a host pushes a
//! message while a run is in flight and the loop drains it between steps.
