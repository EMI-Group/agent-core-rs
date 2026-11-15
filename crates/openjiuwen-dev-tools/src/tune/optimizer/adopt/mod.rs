//! The adopt family: decide whether a tuned prompt is worth installing.
//!
//! Ports `openjiuwen/dev_tools/tune/optimizer/adopt/` (`adopt_optimizer.py`,
//! `utils.py`) as a deterministic decision layer over the existing prompt-search
//! seams: a policy turns a reward history into an adoption decision, the memory
//! records what was adopted, and the service composes drift, convergence, and
//! memory so a caller can adopt a candidate without re-implementing them.
