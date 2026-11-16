//! Experts for teams: the locating protocol, `AgentGroup` packages and template → member assembly.
//!
//! Python parity source: `openjiuwen/agent_teams/experts/`.
//!
//! Three concerns are deliberately separate:
//!
//! * **locating** — [`ExpertCatalog`] answers which packages are installed and where; it never
//!   opens a manifest;
//! * **package parsing** — [`load_agent_group`] reads and validates an `AgentGroup` package and
//!   exposed its display cards;
//! * **assembly** — [`compose_template_member`] turns a template into a
//!   [`DeepAgentSpec`](crate::schema::DeepAgentSpec), and
//!   [`bind_group`] describes the spawnable members of a loaded group, recording every roster
//!   member it did not bind in [`BoundAgentGroup::skipped`].
//!
//! The process-wide catalog ([`set_process_expert_catalog`] / [`get_process_expert_catalog`]) is
//! one of the crate's two documented singletons; clear it in tests.
//!
//! ```
//! use openjiuwen_agent_teams::experts::{compose_template_member, AgentTemplateCard, ExpertRef};
//! use openjiuwen_agent_teams::schema::DeepAgentSpec;
//!
//! let template = AgentTemplateCard::new("writer");
//! let spec =
//!     compose_template_member(&DeepAgentSpec::default(), &template, "# Writer", Some("en"));
//! assert_eq!(spec.language.as_deref(), Some("en"));
//! assert_eq!(ExpertRef::new("writer").label(), "writer");
//! ```

pub mod agent_group;
pub mod catalog;
pub mod template;

#[cfg(test)]
pub(crate) mod test_support;

pub use agent_group::{
    load_agent_group, AgentGroupCard, AgentGroupMemberCard, AgentTemplateCard,
    LoadedAgentGroupPackage, LocalizedText, LEADER_MEMBER,
};
pub use catalog::{
    clear_process_expert_catalog, get_process_expert_catalog, set_process_expert_catalog,
    ExpertCatalog, ExpertRef, PackageLocation,
};
pub use template::{
    bind_group, compose_template_member, render_persona, BoundAgentGroup, BoundGroupMember,
    SkipReason, SkippedMember,
};
