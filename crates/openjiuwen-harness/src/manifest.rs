//! Harness element catalog — the Rust counterpart of Python's `@harness_element`.
//!
//! Python declares every built-in rail / tool / sub-agent through
//! `openjiuwen.harness.manifest.harness_element`, which records a serializable
//! [`HarnessElementDescriptor`] in a process-global catalog
//! (`openjiuwen/harness/manifest/catalog.py`). The Rust port keeps the descriptor
//! data model and the catalog operations, but makes the catalog an ordinary owned
//! value ([`Catalog`]) instead of a module-level singleton: the harness builder
//! wires elements explicitly, and tests can build isolated catalogs without
//! touching shared mutable state.
//!
//! Python resolves a descriptor's `factory_ref` reflectively through entry points;
//! Rust has no reflection, so `factory_ref` is optional metadata and the concrete
//! construction path is always supplied by the caller (see `DeepAgentBuilder`).
//!
//! Wire form is `snake_case` and round-trips through serde; unknown fields are
//! ignored so descriptors written by a newer Python release still load.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Kind of harness element a descriptor describes.
///
/// Python parity: `manifest.models.ElementKind(str, Enum)`.
/// The kind doubles as the registration target: tools, rails and sub-agents each
/// have their own registry in the Python provider layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    /// A model-facing tool; wire name `tool`.
    Tool,
    /// A task-loop or agent rail; wire name `rail`.
    Rail,
    /// A sub-agent definition; wire name `subagent`.
    Subagent,
}

impl ElementKind {
    /// Returns the serde wire name of the kind (`"tool"`, `"rail"` or `"subagent"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Rail => "rail",
            Self::Subagent => "subagent",
        }
    }

    /// Parses an exact wire name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "tool" => Some(Self::Tool),
            "rail" => Some(Self::Rail),
            "subagent" => Some(Self::Subagent),
            _ => None,
        }
    }
}

/// One public method or property a built element exposes.
///
/// Python parity: `manifest.models.InterfaceMethod`; `description` defaults to
/// the empty string and `is_async` to `false`, as in pydantic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InterfaceMethod {
    /// Method or property name.
    pub name: String,
    /// One-line description of what the method does.
    #[serde(default)]
    pub description: String,
    /// Whether the method is `async`.
    #[serde(default)]
    pub is_async: bool,
}

impl InterfaceMethod {
    /// Creates a synchronous interface method with an empty description.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            is_async: false,
        }
    }

    /// Sets the one-line description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Marks the method as `async` (or synchronous) when `is_async` is `true`.
    #[must_use]
    pub const fn with_async(mut self, is_async: bool) -> Self {
        self.is_async = is_async;
        self
    }
}

/// Serializable, LLM-tool-style definition of one harness element.
///
/// Python parity: `manifest.models.HarnessElementDescriptor`. A descriptor is
/// pure data: it carries no live handles and never triggers registration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct HarnessElementDescriptor {
    /// The element kind (tool / rail / sub-agent).
    pub kind: ElementKind,
    /// Unique element name, doubling as the spec `type` used for resolution.
    pub name: String,
    /// Human-readable summary of what the element provides.
    #[serde(default)]
    pub description: String,
    /// Entry-point-style reference to the construction factory, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factory_ref: Option<String>,
    /// JSON schema of the construction inputs, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// Reference to the input model, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_model_ref: Option<String>,
    /// Public interface methods the built instance exposes.
    #[serde(default)]
    pub interface_methods: Vec<InterfaceMethod>,
}

impl HarnessElementDescriptor {
    /// Creates a descriptor with an empty description and no metadata.
    #[must_use]
    pub fn new(kind: ElementKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
            description: String::new(),
            factory_ref: None,
            input_schema: None,
            input_model_ref: None,
            interface_methods: Vec::new(),
        }
    }

    /// Sets the human-readable description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the entry-point-style factory reference.
    #[must_use]
    pub fn with_factory_ref(mut self, factory_ref: impl Into<String>) -> Self {
        self.factory_ref = Some(factory_ref.into());
        self
    }

    /// Sets the JSON schema of the construction inputs.
    #[must_use]
    pub fn with_input_schema(mut self, input_schema: serde_json::Value) -> Self {
        self.input_schema = Some(input_schema);
        self
    }

    /// Sets the reference to the construction-input model.
    #[must_use]
    pub fn with_input_model_ref(mut self, input_model_ref: impl Into<String>) -> Self {
        self.input_model_ref = Some(input_model_ref.into());
        self
    }

    /// Appends one interface method, preserving declaration order.
    #[must_use]
    pub fn with_interface_method(mut self, method: InterfaceMethod) -> Self {
        self.interface_methods.push(method);
        self
    }
}

/// Owned, ordered catalog of harness-element descriptors.
///
/// Python keeps a module-global `_CATALOG`; Rust deliberately keeps the catalog a
/// plain value owned by the harness builder so there is no undocumented global
/// mutable state, tests stay isolated, and two `DeepAgent` instances can carry
/// different element sets.
///
/// Ordering is insertion order, which makes [`Catalog::list`] and
/// [`Catalog::with_prefix`] deterministic (Python relied on `dict` ordering).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Catalog {
    entries: Vec<HarnessElementDescriptor>,
}

impl Catalog {
    /// Creates an empty catalog.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Registers one descriptor, rejecting a duplicate `(kind, name)` pair.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with the Python message text when an element
    /// of the same kind and name is already registered.
    pub fn register(&mut self, descriptor: HarnessElementDescriptor) -> Result<()> {
        if self.get(descriptor.kind, &descriptor.name).is_some() {
            let name = &descriptor.name;
            return Err(Error::validation(format!(
                "Duplicate harness element name: '{name}'"
            )));
        }
        self.entries.push(descriptor);
        Ok(())
    }

    /// Looks up an element by kind and exact name.
    #[must_use]
    pub fn get(&self, kind: ElementKind, name: &str) -> Option<&HarnessElementDescriptor> {
        self.entries
            .iter()
            .find(|entry| entry.kind == kind && entry.name == name)
    }

    /// Returns every descriptor in stable insertion order.
    #[must_use]
    pub fn list(&self) -> Vec<&HarnessElementDescriptor> {
        self.entries.iter().collect()
    }

    /// Returns the descriptors whose name starts with `prefix`, in insertion order.
    #[must_use]
    pub fn with_prefix(&self, prefix: &str) -> Vec<&HarnessElementDescriptor> {
        self.entries
            .iter()
            .filter(|entry| entry.name.starts_with(prefix))
            .collect()
    }

    /// Returns the number of registered descriptors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no descriptor is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Returns the built-in element descriptors declared by the Python harness.
///
/// Every built-in element lives under the `core.` namespace (parallel to a
/// platform's `swarm.` namespace), so a spec `type` unambiguously names its
/// owning layer. The list is a partial snapshot of `manifest/builtin_elements.py`:
/// the `lsp`, `worktree`, `goal`, `heartbeat`, progressive-discovery, web,
/// media, cron and MCP-resource surfaces have no descriptor here.
#[must_use]
pub fn builtin_elements() -> Vec<HarnessElementDescriptor> {
    vec![
        rail(
            "core.task_planning",
            "Task-planning rail that maintains the todo list and feeds it into the system prompt.",
        ),
        rail(
            "core.skill_use",
            "Skill-discovery and skill-execution rail backed by the skill library.",
        ),
        rail(
            "core.subagent",
            "Sub-agent rail that exposes the task tool for delegating work.",
        ),
        rail(
            "core.sys_operation",
            "System-operation rail providing sandboxed file, shell and code access.",
        ),
        rail(
            "core.security",
            "Security rail that guards prompts, tool input and tool output.",
        ),
        rail(
            "core.memory",
            "Long-term memory rail exposing the memory search/read/write tools.",
        ),
        rail(
            "core.task_completion",
            "Task-completion rail that decides when the task loop may stop.",
        ),
        rail(
            "core.confirm_interrupt",
            "Human-in-the-loop rail that interrupts on tools requiring confirmation.",
        ),
        tool(
            "core.ask_user",
            "Tool that asks the user a question and resumes the loop with the answer.",
        ),
        rail(
            "core.verification",
            "Verification rail that reviews the agent's work before completion.",
        ),
        rail(
            "core.agent_mode",
            "Agent-mode rail that switches between normal and plan execution.",
        ),
        subagent(
            "core.subagent.explore_agent",
            "Read-only sub-agent that explores the workspace and reports findings.",
        ),
        subagent(
            "core.subagent.plan_agent",
            "Sub-agent that produces an implementation plan for a task.",
        ),
        subagent(
            "core.subagent.research_agent",
            "Sub-agent that gathers external research for a task.",
        ),
        subagent(
            "core.subagent.verification_agent",
            "Sub-agent that verifies a completed change against its requirements.",
        ),
        rail(
            "core.observability",
            "Observability rail that emits task-loop and tool spans.",
        ),
    ]
}

/// Builds a rail descriptor.
fn rail(name: &str, description: &str) -> HarnessElementDescriptor {
    HarnessElementDescriptor::new(ElementKind::Rail, name).with_description(description)
}

/// Builds a tool descriptor.
fn tool(name: &str, description: &str) -> HarnessElementDescriptor {
    HarnessElementDescriptor::new(ElementKind::Tool, name).with_description(description)
}

/// Builds a sub-agent descriptor.
fn subagent(name: &str, description: &str) -> HarnessElementDescriptor {
    HarnessElementDescriptor::new(ElementKind::Subagent, name).with_description(description)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::{
        builtin_elements, Catalog, ElementKind, HarnessElementDescriptor, InterfaceMethod,
    };
    use openjiuwen_common::error::Error;

    #[test]
    fn element_kind_wire_names_round_trip() {
        for kind in [ElementKind::Tool, ElementKind::Rail, ElementKind::Subagent] {
            let wire = serde_json::to_string(&kind).expect("serialize kind");
            assert_eq!(wire, format!("\"{}\"", kind.as_str()));
            let back: ElementKind = serde_json::from_str(&wire).expect("deserialize kind");
            assert_eq!(back, kind);
            assert_eq!(ElementKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(ElementKind::parse("Subagent"), None);
        assert_eq!(ElementKind::parse(""), None);
    }

    #[test]
    fn descriptor_builders_set_every_field() {
        let descriptor = HarnessElementDescriptor::new(ElementKind::Rail, "core.security")
            .with_description("Security rail")
            .with_factory_ref("openjiuwen.harness.rails:SecurityRail")
            .with_input_schema(json!({"type": "object"}))
            .with_input_model_ref("openjiuwen.harness.manifest.inputs:EmptyInput")
            .with_interface_method(
                InterfaceMethod::new("before_task_iteration").with_description("Guard a round."),
            )
            .with_interface_method(InterfaceMethod::new("after_model_call").with_async(true));

        assert_eq!(descriptor.kind, ElementKind::Rail);
        assert_eq!(descriptor.name, "core.security");
        assert_eq!(descriptor.description, "Security rail");
        assert_eq!(
            descriptor.factory_ref.as_deref(),
            Some("openjiuwen.harness.rails:SecurityRail")
        );
        assert_eq!(descriptor.input_schema, Some(json!({"type": "object"})));
        assert_eq!(
            descriptor.input_model_ref.as_deref(),
            Some("openjiuwen.harness.manifest.inputs:EmptyInput")
        );
        assert_eq!(descriptor.interface_methods.len(), 2);
        assert_eq!(
            descriptor.interface_methods[0].name,
            "before_task_iteration"
        );
        assert_eq!(
            descriptor.interface_methods[0].description,
            "Guard a round."
        );
        assert!(!descriptor.interface_methods[0].is_async);
        assert!(descriptor.interface_methods[1].is_async);
    }

    #[test]
    fn descriptor_new_defaults_are_empty() {
        let descriptor = HarnessElementDescriptor::new(ElementKind::Tool, "core.ask_user");
        assert_eq!(descriptor.description, "");
        assert_eq!(descriptor.factory_ref, None);
        assert_eq!(descriptor.input_schema, None);
        assert_eq!(descriptor.input_model_ref, None);
        assert_eq!(descriptor.interface_methods.len(), 0);
    }

    #[test]
    fn descriptor_round_trips_through_json() {
        let descriptor = HarnessElementDescriptor::new(ElementKind::Subagent, "core.subagent.plan")
            .with_description("Plan agent")
            .with_input_schema(
                json!({"type": "object", "properties": {"depth": {"type": "integer"}}}),
            )
            .with_interface_method(
                InterfaceMethod::new("run")
                    .with_description("Run the agent.")
                    .with_async(true),
            );

        let json = serde_json::to_value(&descriptor).expect("serialize descriptor");
        assert_eq!(json["kind"], json!("subagent"));
        assert_eq!(json["name"], json!("core.subagent.plan"));
        assert_eq!(json["interface_methods"][0]["name"], json!("run"));
        assert_eq!(json["interface_methods"][0]["is_async"], json!(true));

        let back: HarnessElementDescriptor =
            serde_json::from_value(json).expect("deserialize descriptor");
        assert_eq!(back, descriptor);
    }

    #[test]
    fn descriptor_tolerates_unknown_fields_and_omitted_optionals() {
        let raw = json!({
            "kind": "rail",
            "name": "core.memory",
            "description": "Memory rail",
            "future_field": {"nested": true},
            "interface_methods": [{"name": "search", "extra": 1}]
        });

        let descriptor: HarnessElementDescriptor =
            serde_json::from_value(raw).expect("deserialize descriptor");
        assert_eq!(descriptor.kind, ElementKind::Rail);
        assert_eq!(descriptor.factory_ref, None);
        assert_eq!(descriptor.input_schema, None);
        assert_eq!(descriptor.input_model_ref, None);
        assert_eq!(descriptor.interface_methods.len(), 1);
        assert_eq!(descriptor.interface_methods[0].name, "search");
        assert_eq!(descriptor.interface_methods[0].description, "");
        assert!(!descriptor.interface_methods[0].is_async);

        let encoded = serde_json::to_value(&descriptor).expect("serialize descriptor");
        assert!(encoded.get("factory_ref").is_none());
        assert!(encoded.get("input_schema").is_none());
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let descriptor = HarnessElementDescriptor::new(ElementKind::Rail, "core.security");
        let mut catalog = Catalog::new();
        catalog
            .register(descriptor.clone())
            .expect("first register");

        let error = catalog
            .register(descriptor)
            .expect_err("duplicate must fail");
        match error {
            Error::Validation(message) => {
                assert_eq!(message, "Duplicate harness element name: 'core.security'");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        assert_eq!(catalog.len(), 1);
    }

    #[test]
    fn same_name_different_kind_is_allowed() {
        let mut catalog = Catalog::new();
        catalog
            .register(HarnessElementDescriptor::new(
                ElementKind::Rail,
                "core.security",
            ))
            .expect("register rail");
        catalog
            .register(HarnessElementDescriptor::new(
                ElementKind::Tool,
                "core.security",
            ))
            .expect("register tool");

        assert_eq!(catalog.len(), 2);
        assert_eq!(
            catalog
                .get(ElementKind::Tool, "core.security")
                .map(|d| d.kind),
            Some(ElementKind::Tool)
        );
        assert_eq!(
            catalog
                .get(ElementKind::Subagent, "core.security")
                .map(|d| d.kind),
            None
        );
    }

    #[test]
    fn listing_is_stable_insertion_order() {
        let mut catalog = Catalog::new();
        catalog
            .register(HarnessElementDescriptor::new(ElementKind::Rail, "core.b"))
            .expect("register b");
        catalog
            .register(HarnessElementDescriptor::new(ElementKind::Tool, "core.a"))
            .expect("register a");
        catalog
            .register(HarnessElementDescriptor::new(
                ElementKind::Subagent,
                "core.c",
            ))
            .expect("register c");

        let names: Vec<&str> = catalog.list().iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["core.b", "core.a", "core.c"]);
    }

    #[test]
    fn prefix_filter_keeps_order_and_vets_non_matches() {
        let mut catalog = Catalog::new();
        for name in [
            "core.observability",
            "core.subagent.b",
            "team.leader",
            "core.subagent.a",
        ] {
            catalog
                .register(HarnessElementDescriptor::new(ElementKind::Rail, name))
                .expect("register");
        }

        let subagent_names: Vec<&str> = catalog
            .with_prefix("core.subagent.")
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        assert_eq!(subagent_names, vec!["core.subagent.b", "core.subagent.a"]);

        assert_eq!(catalog.with_prefix("core.").len(), 3);
        assert_eq!(catalog.with_prefix("swarm.").len(), 0);
        assert_eq!(catalog.with_prefix("").len(), 4);
    }

    #[test]
    fn empty_catalog_reports_empty() {
        let catalog = Catalog::new();
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
        assert_eq!(catalog.list().len(), 0);
        assert!(catalog
            .get(ElementKind::Rail, "core.task_planning")
            .is_none());
        assert_eq!(catalog, Catalog::default());
    }

    #[test]
    fn builtin_elements_register_without_duplicates() {
        let mut catalog = Catalog::new();
        for descriptor in builtin_elements() {
            catalog.register(descriptor).expect("builtin must register");
        }

        assert_eq!(catalog.len(), 16);
        assert!(!catalog.is_empty());
        assert_eq!(
            catalog
                .get(ElementKind::Tool, "core.ask_user")
                .map(|d| d.kind),
            Some(ElementKind::Tool)
        );
        assert_eq!(catalog.with_prefix("core.subagent.").len(), 4);
        assert_eq!(catalog.with_prefix("core.").len(), catalog.len());
        assert_eq!(
            catalog
                .get(ElementKind::Rail, "core.observability")
                .map(|d| d.kind),
            Some(ElementKind::Rail)
        );
        assert!(catalog
            .list()
            .iter()
            .all(|descriptor| !descriptor.description.is_empty()));
    }
}
