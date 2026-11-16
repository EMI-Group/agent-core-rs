//! Component registry: a name → factory table for rebuilding components from JSON.
//!
//! The workflow builder stores `Arc<dyn Component>` handles directly, so the registry is
//! the seam for callers that serialize component configuration (cards, workflow specs,
//! the CLI) and need to construct components from a name plus a JSON config.
//!
//! [`ComponentRegistry::builtin`] registers the nine dependency-free control-flow
//! components: `start`, `end`, `branch`, `loop`, `loop_group`, `advanced_loop`,
//! `loop_break`, `loop_set_variable`, and `http_request`. The leaf components
//! (`llm`, `tool`, `retrieval`, `memory`) own runtime collaborators (models, tools,
//! knowledge bases, memory stores) and are therefore registered by the caller with
//! [`ComponentRegistry::register_component`] or [`ComponentRegistry::register`].
//!
//! ```ignore
//! use openjiuwen_workflow::registry::ComponentRegistry;
//!
//! let registry = ComponentRegistry::builtin();
//! let end = registry.create("end", json!({"response_template": "answer: {{answer}}"}))?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::component::Component;
use crate::components::{
    AdvancedLoopComponent, AdvancedLoopConfig, BranchComponent, EndComponent, EndConfig,
    HttpComponentConfig, HttpRequestComponent, LoopBreakComponent, LoopBreakConfig, LoopComponent,
    LoopConfig, LoopGroupComponent, LoopGroupConfig, LoopSetVariableComponent,
    LoopSetVariableConfig, StartComponent,
};
use crate::condition::{Branch, BranchRouter, Condition};
use crate::error::{
    branch_param_invalid, component_tool_input_param_error, end_param_invalid, loop_param_invalid,
};

/// Builds a component from its JSON configuration.
pub type ComponentFactory = Arc<dyn Fn(Value) -> Result<Arc<dyn Component>> + Send + Sync>;

/// A name → factory table used to construct components from serialized configuration.
///
/// The registry is a plain value: cloning it clones the factory handles (cheap, they are
/// reference counted), and it holds no global state.
#[derive(Clone, Default)]
pub struct ComponentRegistry {
    factories: HashMap<String, ComponentFactory>,
}

impl ComponentRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding the nine control-flow built-ins: `start`, `end`, `branch`,
    /// `loop`, `loop_group`, `advanced_loop`, `loop_break`, `loop_set_variable`, and
    /// `http_request`.
    ///
    /// Every built-in parses its JSON configuration; see [`Self::create`]. `start`
    /// ignores its configuration, `end`, `loop`, and `http_request` accept `null` or `{}`
    /// as the default configuration, and `branch` requires a non-empty `branches` array.
    /// `http_request` is config-only: it needs no runtime collaborator.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self::new();
        registry.insert("start", factory(|_config| Ok(boxed(StartComponent::new()))));
        registry.insert("end", factory(|config| Ok(boxed(build_end(config)?))));
        registry.insert("branch", factory(|config| Ok(boxed(build_branch(config)?))));
        registry.insert("loop", factory(|config| Ok(boxed(build_loop(config)?))));
        registry.insert(
            "loop_group",
            factory(|config| Ok(boxed(build_loop_group(config)?))),
        );
        registry.insert(
            "advanced_loop",
            factory(|config| Ok(boxed(build_advanced_loop(config)?))),
        );
        registry.insert(
            "loop_break",
            factory(|config| Ok(boxed(build_loop_break(config)?))),
        );
        registry.insert(
            "loop_set_variable",
            factory(|config| Ok(boxed(build_loop_set_variable(config)?))),
        );
        registry.insert(
            "http_request",
            factory(|config| Ok(boxed(build_http_request(config)?))),
        );
        registry
    }

    /// Registers `factory` under `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `name` is already registered.
    pub fn register(&mut self, name: impl Into<String>, factory: ComponentFactory) -> Result<()> {
        let name = name.into();
        if self.factories.contains_key(&name) {
            return Err(Error::validation(format!(
                "component '{name}' is already registered"
            )));
        }
        let _ = self.factories.insert(name, factory);
        Ok(())
    }

    /// Registers `build` under `name`, boxing the component it returns.
    ///
    /// This is the convenience form of [`Self::register`] for components that are built
    /// directly rather than through a hand-written [`ComponentFactory`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `name` is already registered.
    pub fn register_component<C>(
        &mut self,
        name: impl Into<String>,
        build: impl Fn(Value) -> Result<C> + Send + Sync + 'static,
    ) -> Result<()>
    where
        C: Component + 'static,
    {
        let factory = factory(move |config| Ok(boxed(build(config)?)));
        self.register(name, factory)
    }

    /// Whether a factory is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Every registered name, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }

    /// Builds the component registered under `name` from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `name` is not registered, and propagates the
    /// factory's configuration error when `config` is invalid for that component.
    pub fn create(&self, name: &str, config: Value) -> Result<Arc<dyn Component>> {
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| Error::not_found(format!("component '{name}' is not registered")))?;
        factory(config)
    }

    /// The number of registered factories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Whether no factory is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Inserts without the duplicate check; only used while building a fresh registry.
    fn insert(&mut self, name: &str, factory: ComponentFactory) {
        let _ = self.factories.insert(name.to_owned(), factory);
    }
}

/// Boxes `build` as a [`ComponentFactory`].
fn factory<F>(build: F) -> ComponentFactory
where
    F: Fn(Value) -> Result<Arc<dyn Component>> + Send + Sync + 'static,
{
    Arc::new(build)
}

/// Type-erases a concrete component.
fn boxed<C: Component + 'static>(component: C) -> Arc<dyn Component> {
    Arc::new(component)
}

/// Parses an [`EndConfig`]; `null` selects the default (flat) configuration.
fn build_end(config: Value) -> Result<EndComponent> {
    let config = if config.is_null() {
        EndConfig::default()
    } else {
        serde_json::from_value::<EndConfig>(config)
            .map_err(|error| end_param_invalid(&format!("conf is invalid, {error}")))?
    };
    Ok(EndComponent::with_config(config))
}

/// Parses a [`HttpComponentConfig`]; `null` selects the default configuration.
fn build_http_request(config: Value) -> Result<HttpRequestComponent> {
    let config = if config.is_null() {
        HttpComponentConfig::default()
    } else {
        serde_json::from_value::<HttpComponentConfig>(config).map_err(|error| {
            component_tool_input_param_error(&format!("conf is invalid, {error}"))
        })?
    };
    Ok(HttpRequestComponent::new(config))
}

/// The JSON configuration of a registry-built `branch` component.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct BranchConfig {
    /// Ordered branches; the first matching branch wins.
    branches: Vec<BranchSpec>,
    /// Targets used when no branch condition holds.
    #[serde(rename = "default")]
    default_targets: Option<Vec<String>>,
}

/// One `branch` entry: its optional tracing name, targets, and condition source.
#[derive(Debug, Deserialize)]
struct BranchSpec {
    /// Optional identifier used for tracing.
    #[serde(default)]
    name: Option<String>,
    /// Component ids routed to when the condition holds.
    targets: Vec<String>,
    /// Condition expression source.
    condition: String,
}

/// Parses a [`BranchConfig`] into a component with an ordered [`BranchRouter`].
fn build_branch(config: Value) -> Result<BranchComponent> {
    let config: BranchConfig = serde_json::from_value(config)
        .map_err(|error| branch_param_invalid(&format!("conf is invalid, {error}")))?;
    if config.branches.is_empty() {
        return Err(Error::validation(
            "branch config requires at least one 'branches' entry",
        ));
    }
    let mut router = BranchRouter::new();
    for spec in config.branches {
        let condition = Condition::new(spec.condition.as_str())?;
        let mut branch = Branch::new(condition, spec.targets);
        if let Some(name) = spec.name {
            branch = branch.with_name(name);
        }
        let _ = router.add_branch(branch);
    }
    if let Some(targets) = config.default_targets {
        let _ = router.default_target(targets);
    }
    Ok(BranchComponent::new(router))
}

/// Parses a [`LoopConfig`]; `null` selects the default configuration.
///
/// A registry-built loop has an empty body: body components are attached
/// programmatically through [`LoopComponent::new`], so an empty body echoes the iteration
/// inputs. The parsed configuration is validated, so a loop without a bound is rejected
/// here with `[101030]` instead of failing at invoke time.
fn build_loop(config: Value) -> Result<LoopComponent> {
    let config = if config.is_null() {
        LoopConfig::default()
    } else {
        serde_json::from_value::<LoopConfig>(config)
            .map_err(|error| loop_param_invalid(&format!("conf is invalid, {error}")))?
    };
    let component = LoopComponent::new(config, Vec::new());
    component.validate_config()?;
    Ok(component)
}

/// Parses a [`LoopGroupConfig`]; `null` selects the default configuration.
///
/// A registry-built group has an empty body and no way to add one, so only the bounds are
/// checked here: a configuration that cannot run or cannot terminate is rejected with
/// `[101030]`.
fn build_loop_group(config: Value) -> Result<LoopGroupComponent> {
    let config = if config.is_null() {
        LoopGroupConfig::default()
    } else {
        serde_json::from_value::<LoopGroupConfig>(config)
            .map_err(|error| loop_param_invalid(&format!("conf is invalid, {error}")))?
    };
    let component = LoopGroupComponent::new(config, Vec::new());
    component.validate_config()?;
    Ok(component)
}

/// Parses an [`AdvancedLoopConfig`]; `null` selects the default configuration.
fn build_advanced_loop(config: Value) -> Result<AdvancedLoopComponent> {
    let config = if config.is_null() {
        AdvancedLoopConfig::default()
    } else {
        serde_json::from_value::<AdvancedLoopConfig>(config)
            .map_err(|error| loop_param_invalid(&format!("conf is invalid, {error}")))?
    };
    let component = AdvancedLoopComponent::new(config, Vec::new());
    component.validate_config()?;
    Ok(component)
}

/// Parses a [`LoopBreakConfig`]; `null` selects the default (unconditional) configuration.
fn build_loop_break(config: Value) -> Result<LoopBreakComponent> {
    let config = if config.is_null() {
        LoopBreakConfig::default()
    } else {
        serde_json::from_value::<LoopBreakConfig>(config)
            .map_err(|error| loop_param_invalid(&format!("conf is invalid, {error}")))?
    };
    let component = LoopBreakComponent::with_config(config);
    component.validate_config()?;
    Ok(component)
}

/// Parses a [`LoopSetVariableConfig`]; `null` is rejected because the mapping is required.
fn build_loop_set_variable(config: Value) -> Result<LoopSetVariableComponent> {
    let config = if config.is_null() {
        LoopSetVariableConfig::default()
    } else {
        serde_json::from_value::<LoopSetVariableConfig>(config)
            .map_err(|error| loop_param_invalid(&format!("conf is invalid, {error}")))?
    };
    let component = LoopSetVariableComponent::with_config(config);
    component.validate_config()?;
    Ok(component)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;
    use crate::context::ComponentContext;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("registry-tests"));
        ComponentContext::new(session, "end", "wf")
    }

    #[test]
    fn builtin_registers_the_control_flow_components() {
        let registry = ComponentRegistry::builtin();
        assert_eq!(registry.len(), 9);
        assert!(!registry.is_empty());
        assert_eq!(
            registry.names(),
            vec![
                "advanced_loop".to_owned(),
                "branch".to_owned(),
                "end".to_owned(),
                "http_request".to_owned(),
                "loop".to_owned(),
                "loop_break".to_owned(),
                "loop_group".to_owned(),
                "loop_set_variable".to_owned(),
                "start".to_owned()
            ]
        );
        for name in [
            "start",
            "end",
            "branch",
            "loop",
            "loop_group",
            "advanced_loop",
            "loop_break",
            "loop_set_variable",
            "http_request",
        ] {
            assert!(registry.contains(name), "missing builtin '{name}'");
        }
        assert!(!registry.contains("llm"));
    }

    #[test]
    fn new_registry_is_empty() {
        let registry = ComponentRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.names().is_empty());
    }

    #[test]
    fn builtin_components_report_their_component_types() {
        let registry = ComponentRegistry::builtin();
        let cases = [
            ("start", json!({"ignored": true}), "start"),
            ("end", json!({}), "end"),
            (
                "branch",
                json!({"branches": [{"targets": ["a"], "condition": "${score} > 0"}]}),
                "branch",
            ),
            (
                "loop",
                json!({"loop_type": "number", "loop_number": 2}),
                "loop",
            ),
            (
                "loop_group",
                json!({"loop_type": "number", "loop_number": 2}),
                "loop_group",
            ),
            ("advanced_loop", json!({"loop_number": 2}), "advanced_loop"),
            (
                "loop_break",
                json!({"condition": "${index} > 1"}),
                "loop_break",
            ),
            (
                "loop_set_variable",
                json!({"variable_mapping": {"${loop.count}": "${index}"}}),
                "loop_set_variable",
            ),
        ];
        for (name, config, expected) in cases {
            let component = registry.create(name, config).unwrap();
            assert_eq!(component.component_type(), expected, "component '{name}'");
        }
    }

    #[test]
    fn end_config_accepts_null_and_object_shapes() {
        let registry = ComponentRegistry::builtin();
        assert_eq!(
            registry
                .create("end", Value::Null)
                .unwrap()
                .component_type(),
            "end"
        );
        assert_eq!(
            registry.create("end", json!({})).unwrap().component_type(),
            "end"
        );
        // An invalid config is reported as an end parameter error, not a panic.
        let error = registry
            .create("end", json!({"response_template": 3}))
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("component end params is invalid"));
    }

    #[tokio::test]
    async fn end_config_applies_the_response_template() {
        let registry = ComponentRegistry::builtin();
        let component = registry
            .create("end", json!({"response_template": "hi {{x}}"}))
            .unwrap();
        let output = component
            .invoke(json!({"x": "there"}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"response": "hi there"}));

        // The parsed template is observable through the concrete builder.
        let end = build_end(json!({"response_template": "hi {{x}}"})).unwrap();
        assert_eq!(end.config().response_template.as_deref(), Some("hi {{x}}"));
    }

    #[test]
    fn branch_config_routes_the_first_match() {
        let registry = ComponentRegistry::builtin();
        let component = registry
            .create(
                "branch",
                json!({
                    "branches": [
                        {"name": "high", "targets": ["a"], "condition": "${score} > 10"},
                        {"targets": ["b"], "condition": "${score} > 0"}
                    ],
                    "default": ["c"]
                }),
            )
            .unwrap();
        assert_eq!(
            component.route(&json!({"score": 20})).unwrap().unwrap(),
            vec!["a".to_owned()]
        );
        assert_eq!(
            component.route(&json!({"score": 5})).unwrap().unwrap(),
            vec!["b".to_owned()]
        );
        assert_eq!(
            component.route(&json!({"score": 0})).unwrap().unwrap(),
            vec!["c".to_owned()]
        );
    }

    #[test]
    fn branch_config_requires_branches() {
        let registry = ComponentRegistry::builtin();
        for config in [json!({}), json!({"branches": [], "default": ["a"]})] {
            let error = registry.create("branch", config).err().unwrap();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.message().contains("at least one 'branches' entry"));
        }
    }

    #[test]
    fn branch_config_reports_bad_conditions_and_shapes() {
        let registry = ComponentRegistry::builtin();
        // A malformed condition expression is a branch parameter error.
        let error = registry
            .create(
                "branch",
                json!({"branches": [{"targets": ["a"], "condition": "${score} >"}]}),
            )
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("1010"));
        // A malformed JSON shape is also a branch parameter error.
        let error = registry
            .create("branch", json!({"branches": [{"condition": "${a}"}]}))
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Workflow);
    }

    #[test]
    fn loop_config_round_trips() {
        let config = json!({
            "loop_type": "condition",
            "bool_expression": "${go} == true",
            "max_iterations": 7
        });
        let loop_component = build_loop(config.clone()).unwrap();
        assert_eq!(
            loop_component.config().loop_type,
            crate::components::LoopType::Condition
        );
        assert_eq!(
            loop_component.config().bool_expression.as_deref(),
            Some("${go} == true")
        );
        assert_eq!(loop_component.config().max_iterations, 7);
        assert_eq!(loop_component.body_len(), 0);

        let round_tripped = serde_json::to_value(loop_component.config()).unwrap();
        let reparsed = build_loop(round_tripped).unwrap();
        assert_eq!(reparsed.config(), loop_component.config());
    }

    #[test]
    fn loop_config_rejects_a_missing_bound_and_bad_shapes() {
        let registry = ComponentRegistry::builtin();
        // `null` selects the default configuration, which has no bound: the construction
        // path validates, so the gap is reported here as `[101030]`.
        let error = registry.create("loop", Value::Null).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101030]"), "{error}");
        assert!(error.message().contains("loop_number variable not found"));
        // A bounded configuration is accepted.
        assert_eq!(
            registry
                .create("loop", json!({"loop_number": 2}))
                .unwrap()
                .component_type(),
            "loop"
        );
        let error = registry
            .create("loop", json!({"loop_type": "forever"}))
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error
            .message()
            .contains("component loop_group params is invalid"));
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut registry = ComponentRegistry::builtin();
        let custom: ComponentFactory = factory(|_config| Ok(boxed(StartComponent::new())));
        registry.register("custom", Arc::clone(&custom)).unwrap();
        let error = registry.register("custom", custom).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "component 'custom' is already registered");
        // Registering over a builtin is rejected the same way.
        let duplicate: ComponentFactory = factory(|_config| Ok(boxed(StartComponent::new())));
        let error = registry.register("start", duplicate).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn unknown_component_is_not_found() {
        let registry = ComponentRegistry::builtin();
        let error = registry.create("llm", json!({})).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(error.message(), "component 'llm' is not registered");
    }

    #[test]
    fn loop_group_factory_parses_bounds_and_rejects_bad_ones() {
        let registry = ComponentRegistry::builtin();
        let group = registry
            .create(
                "loop_group",
                json!({
                    "loop_type": "array",
                    "loop_array": "${start.items}",
                    "max_iterations": 4,
                    "loop_variables": {"total": 0},
                }),
            )
            .unwrap();
        assert_eq!(group.component_type(), "loop_group");

        // An unbounded group needs a break component, which a config-only factory cannot
        // attach, so it is rejected.
        let error = registry
            .create("loop_group", json!({"loop_type": "until_break"}))
            .err()
            .unwrap();
        assert!(error.message().contains("cannot terminate"), "{error}");

        let error = registry
            .create("loop_group", json!({"loop_type": "condition"}))
            .err()
            .unwrap();
        assert!(error
            .message()
            .contains("bool_expression variable not found"));
    }

    #[test]
    fn loop_break_factory_parses_its_condition() {
        let registry = ComponentRegistry::builtin();
        let component = registry
            .create("loop_break", json!({"condition": "${index} > 1"}))
            .unwrap();
        assert_eq!(component.component_type(), "loop_break");
        assert_eq!(
            registry
                .create("loop_break", Value::Null)
                .unwrap()
                .component_type(),
            "loop_break"
        );
        let error = registry
            .create("loop_break", json!({"condition": "${index} >"}))
            .err()
            .unwrap();
        assert!(
            error.message().contains("break condition is invalid"),
            "{error}"
        );
    }

    #[test]
    fn loop_set_variable_factory_requires_a_mapping() {
        let registry = ComponentRegistry::builtin();
        let component = registry
            .create(
                "loop_set_variable",
                json!({"variable_mapping": {"${loop.count}": "${index}"}}),
            )
            .unwrap();
        assert_eq!(component.component_type(), "loop_set_variable");

        for config in [Value::Null, json!({}), json!({"variable_mapping": {}})] {
            let error = registry.create("loop_set_variable", config).err().unwrap();
            assert_eq!(error.kind(), ErrorKind::Workflow);
            assert!(error.message().contains("[101031]"), "{error}");
        }
    }

    #[test]
    fn advanced_loop_factory_parses_bounds() {
        let registry = ComponentRegistry::builtin();
        let component = registry
            .create(
                "advanced_loop",
                json!({"loop_condition": "${total} < 3", "loop_number_max_limit": 5}),
            )
            .unwrap();
        assert_eq!(component.component_type(), "advanced_loop");
        let error = registry.create("advanced_loop", json!({})).err().unwrap();
        assert!(error.message().contains("cannot terminate"), "{error}");
    }

    #[test]
    fn custom_registration_creates_components() {
        let mut registry = ComponentRegistry::new();
        registry
            .register_component("start-copy", |_config| Ok(StartComponent::new()))
            .unwrap();
        assert!(registry.contains("start-copy"));
        let component = registry
            .create("start-copy", json!({"query": "hi"}))
            .unwrap();
        assert_eq!(component.component_type(), "start");
    }
}
