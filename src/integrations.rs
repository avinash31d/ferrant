//! Curated, capability-indexed integration catalog and factory registry.
//!
//! The catalog separates discovery metadata from factories. A framework can
//! advertise its curated integrations without constructing credentials or
//! pulling every provider into the dependency graph, then attach a factory
//! only for integrations compiled into the application.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

fn read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationCategory {
    ModelProvider,
    ToolProvider,
    McpTransport,
    Embedder,
    VectorStore,
    DocumentLoader,
    Reranker,
    WorkflowStore,
    Observability,
    EvaluationStore,
    UsageStore,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationCapability {
    Chat,
    ToolCalling,
    StructuredOutput,
    StreamingText,
    StreamingToolCalls,
    MultimodalInput,
    MultimodalOutput,
    UsageAccounting,
    Embeddings,
    DenseRetrieval,
    SparseRetrieval,
    HybridRetrieval,
    MetadataFiltering,
    Reranking,
    DurableState,
    DistributedState,
    Traces,
    Metrics,
    EvaluationReports,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationStability {
    Experimental,
    Beta,
    Stable,
    Deprecated,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ConfigurationSource {
    Config,
    Environment,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigurationValueKind {
    String,
    Boolean,
    Integer,
    Number,
    Object,
    Array,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigurationField {
    pub name: String,
    pub description: String,
    pub source: ConfigurationSource,
    pub value_kind: ConfigurationValueKind,
    pub required: bool,
    pub secret: bool,
}

impl ConfigurationField {
    pub fn environment(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            source: ConfigurationSource::Environment,
            value_kind: ConfigurationValueKind::String,
            required: true,
            secret: true,
        }
    }

    pub fn config(
        name: impl Into<String>,
        description: impl Into<String>,
        value_kind: ConfigurationValueKind,
        required: bool,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            source: ConfigurationSource::Config,
            value_kind,
            required,
            secret: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrationDescriptor {
    /// Stable namespaced identifier, for example `model.openai`.
    pub id: String,
    pub display_name: String,
    pub version: String,
    pub category: IntegrationCategory,
    pub stability: IntegrationStability,
    pub capabilities: BTreeSet<IntegrationCapability>,
    #[serde(default)]
    pub configuration: Vec<ConfigurationField>,
    pub documentation_url: Option<String>,
    pub deprecation_message: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl IntegrationDescriptor {
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        version: impl Into<String>,
        category: IntegrationCategory,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            version: version.into(),
            category,
            stability: IntegrationStability::Stable,
            capabilities: BTreeSet::new(),
            configuration: Vec::new(),
            documentation_url: None,
            deprecation_message: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn capability(mut self, capability: IntegrationCapability) -> Self {
        self.capabilities.insert(capability);
        self
    }

    pub fn configuration(mut self, field: ConfigurationField) -> Self {
        self.configuration.push(field);
        self
    }

    pub fn stability(mut self, stability: IntegrationStability) -> Self {
        self.stability = stability;
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.id.is_empty()
            || !self.id.chars().all(|character| {
                character.is_ascii_lowercase()
                    || character.is_ascii_digit()
                    || matches!(character, '.' | '-' | '_')
            })
        {
            anyhow::bail!(
                "integration id '{}' must contain lowercase letters, numbers, '.', '-' or '_'",
                self.id
            );
        }
        if self.display_name.trim().is_empty() {
            anyhow::bail!("integration '{}' has an empty display name", self.id);
        }
        if self.version.trim().is_empty() {
            anyhow::bail!("integration '{}' has an empty version", self.id);
        }
        if self.capabilities.is_empty() {
            anyhow::bail!("integration '{}' has no declared capabilities", self.id);
        }
        let mut fields = BTreeSet::new();
        for field in &self.configuration {
            if field.name.trim().is_empty() {
                anyhow::bail!("integration '{}' has an empty configuration field", self.id);
            }
            if !fields.insert((field.source, field.name.as_str())) {
                anyhow::bail!(
                    "integration '{}' has duplicate configuration field '{}'",
                    self.id,
                    field.name
                );
            }
        }
        if self.stability == IntegrationStability::Deprecated
            && self
                .deprecation_message
                .as_deref()
                .unwrap_or_default()
                .is_empty()
        {
            anyhow::bail!(
                "deprecated integration '{}' requires a deprecation message",
                self.id
            );
        }
        Ok(())
    }

    /// Validate explicit configuration. Environment fields are reported by
    /// [`missing_environment`](Self::missing_environment), allowing callers to
    /// supply secrets through their own secret manager instead of `std::env`.
    pub fn validate_config(&self, config: &Value) -> anyhow::Result<()> {
        let object = config
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("integration config must be a JSON object"))?;
        for field in self
            .configuration
            .iter()
            .filter(|field| field.source == ConfigurationSource::Config)
        {
            let value = object.get(&field.name);
            if field.required && value.is_none_or(Value::is_null) {
                anyhow::bail!(
                    "integration '{}' requires config field '{}'",
                    self.id,
                    field.name
                );
            }
            if let Some(value) = value.filter(|value| !value.is_null()) {
                let valid = match field.value_kind {
                    ConfigurationValueKind::String => value.is_string(),
                    ConfigurationValueKind::Boolean => value.is_boolean(),
                    ConfigurationValueKind::Integer => value.as_i64().is_some(),
                    ConfigurationValueKind::Number => value.is_number(),
                    ConfigurationValueKind::Object => value.is_object(),
                    ConfigurationValueKind::Array => value.is_array(),
                };
                if !valid {
                    anyhow::bail!(
                        "integration '{}' config field '{}' has the wrong type",
                        self.id,
                        field.name
                    );
                }
            }
        }
        Ok(())
    }

    pub fn missing_environment<F>(&self, lookup: F) -> Vec<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        self.configuration
            .iter()
            .filter(|field| {
                field.source == ConfigurationSource::Environment
                    && field.required
                    && lookup(&field.name).is_none_or(|value| value.is_empty())
            })
            .map(|field| field.name.clone())
            .collect()
    }
}

/// Type-erased component returned by a factory. Applications downcast to the
/// documented concrete or trait-object wrapper type.
#[derive(Clone)]
pub struct IntegrationComponent {
    inner: Arc<dyn Any + Send + Sync>,
}

impl IntegrationComponent {
    pub fn new<T: Any + Send + Sync>(component: T) -> Self {
        Self {
            inner: Arc::new(component),
        }
    }

    pub fn downcast<T: Any + Send + Sync>(&self) -> anyhow::Result<Arc<T>> {
        self.inner
            .clone()
            .downcast::<T>()
            .map_err(|_| anyhow::anyhow!("integration component type mismatch"))
    }
}

pub trait IntegrationFactory: Send + Sync {
    fn create(&self, config: &Value) -> anyhow::Result<IntegrationComponent>;
}

pub struct FunctionIntegrationFactory<F> {
    function: F,
}

impl<F> FunctionIntegrationFactory<F> {
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F> IntegrationFactory for FunctionIntegrationFactory<F>
where
    F: Fn(&Value) -> anyhow::Result<IntegrationComponent> + Send + Sync,
{
    fn create(&self, config: &Value) -> anyhow::Result<IntegrationComponent> {
        (self.function)(config)
    }
}

#[derive(Clone)]
struct RegistryEntry {
    descriptor: IntegrationDescriptor,
    factory: Option<Arc<dyn IntegrationFactory>>,
}

pub struct IntegrationInstance {
    pub descriptor: IntegrationDescriptor,
    pub component: IntegrationComponent,
}

#[derive(Clone, Default)]
pub struct IntegrationRegistry {
    entries: Arc<RwLock<BTreeMap<String, RegistryEntry>>>,
}

impl IntegrationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Catalog with the framework's reviewed first-party integration surface.
    /// Factories are deliberately unattached; provider modules can attach
    /// only what was compiled into the final application.
    pub fn curated() -> Self {
        let registry = Self::new();
        for descriptor in curated_descriptors() {
            // Built-in descriptors are covered by tests and are valid.
            registry
                .register_descriptor(descriptor)
                .expect("invalid built-in integration descriptor");
        }
        registry
    }

    pub fn register_descriptor(&self, descriptor: IntegrationDescriptor) -> anyhow::Result<()> {
        descriptor.validate()?;
        let mut entries = write(&self.entries);
        if entries.contains_key(&descriptor.id) {
            anyhow::bail!("integration '{}' is already registered", descriptor.id);
        }
        entries.insert(
            descriptor.id.clone(),
            RegistryEntry {
                descriptor,
                factory: None,
            },
        );
        Ok(())
    }

    pub fn register(
        &self,
        descriptor: IntegrationDescriptor,
        factory: impl IntegrationFactory + 'static,
    ) -> anyhow::Result<()> {
        descriptor.validate()?;
        let mut entries = write(&self.entries);
        if entries.contains_key(&descriptor.id) {
            anyhow::bail!("integration '{}' is already registered", descriptor.id);
        }
        entries.insert(
            descriptor.id.clone(),
            RegistryEntry {
                descriptor,
                factory: Some(Arc::new(factory)),
            },
        );
        Ok(())
    }

    pub fn attach_factory(
        &self,
        id: &str,
        factory: impl IntegrationFactory + 'static,
    ) -> anyhow::Result<()> {
        let mut entries = write(&self.entries);
        let entry = entries
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("integration '{id}' is not registered"))?;
        if entry.factory.is_some() {
            anyhow::bail!("integration '{id}' already has a factory");
        }
        entry.factory = Some(Arc::new(factory));
        Ok(())
    }

    pub fn descriptor(&self, id: &str) -> Option<IntegrationDescriptor> {
        read(&self.entries)
            .get(id)
            .map(|entry| entry.descriptor.clone())
    }

    pub fn list(&self) -> Vec<IntegrationDescriptor> {
        read(&self.entries)
            .values()
            .map(|entry| entry.descriptor.clone())
            .collect()
    }

    pub fn by_capability(&self, capability: IntegrationCapability) -> Vec<IntegrationDescriptor> {
        read(&self.entries)
            .values()
            .filter(|entry| entry.descriptor.capabilities.contains(&capability))
            .map(|entry| entry.descriptor.clone())
            .collect()
    }

    pub fn by_category(&self, category: IntegrationCategory) -> Vec<IntegrationDescriptor> {
        read(&self.entries)
            .values()
            .filter(|entry| entry.descriptor.category == category)
            .map(|entry| entry.descriptor.clone())
            .collect()
    }

    pub fn is_available(&self, id: &str) -> bool {
        read(&self.entries)
            .get(id)
            .is_some_and(|entry| entry.factory.is_some())
    }

    pub fn create(&self, id: &str, config: &Value) -> anyhow::Result<IntegrationInstance> {
        let entry = read(&self.entries)
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("integration '{id}' is not registered"))?;
        entry.descriptor.validate_config(config)?;
        let factory = entry.factory.ok_or_else(|| {
            anyhow::anyhow!("integration '{id}' is catalogued but its factory is not installed")
        })?;
        Ok(IntegrationInstance {
            descriptor: entry.descriptor,
            component: factory.create(config)?,
        })
    }
}

fn capabilities(values: &[IntegrationCapability]) -> BTreeSet<IntegrationCapability> {
    values.iter().copied().collect()
}

fn curated_descriptors() -> Vec<IntegrationDescriptor> {
    use IntegrationCapability as Capability;
    use IntegrationCategory as Category;

    vec![
        IntegrationDescriptor {
            id: "model.openai".into(),
            display_name: "OpenAI".into(),
            version: "1".into(),
            category: Category::ModelProvider,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[
                Capability::Chat,
                Capability::ToolCalling,
                Capability::StructuredOutput,
                Capability::StreamingText,
                Capability::StreamingToolCalls,
                Capability::MultimodalInput,
                Capability::MultimodalOutput,
                Capability::UsageAccounting,
            ]),
            configuration: vec![ConfigurationField::environment(
                "OPENAI_API_KEY",
                "OpenAI API key",
            )],
            documentation_url: Some("https://platform.openai.com/docs".into()),
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
        IntegrationDescriptor {
            id: "model.anthropic".into(),
            display_name: "Anthropic".into(),
            version: "1".into(),
            category: Category::ModelProvider,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[
                Capability::Chat,
                Capability::ToolCalling,
                Capability::StructuredOutput,
                Capability::StreamingText,
                Capability::StreamingToolCalls,
                Capability::MultimodalInput,
                Capability::UsageAccounting,
            ]),
            configuration: vec![ConfigurationField::environment(
                "ANTHROPIC_API_KEY",
                "Anthropic API key",
            )],
            documentation_url: Some("https://docs.anthropic.com".into()),
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
        IntegrationDescriptor {
            id: "mcp.stdio".into(),
            display_name: "MCP stdio".into(),
            version: "1".into(),
            category: Category::McpTransport,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[Capability::ToolCalling]),
            configuration: vec![ConfigurationField::config(
                "command",
                "Server executable",
                ConfigurationValueKind::String,
                true,
            )],
            documentation_url: Some("https://modelcontextprotocol.io".into()),
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
        IntegrationDescriptor {
            id: "embedder.hash".into(),
            display_name: "Local hash embedder".into(),
            version: "1".into(),
            category: Category::Embedder,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[Capability::Embeddings]),
            configuration: vec![ConfigurationField::config(
                "dimensions",
                "Embedding dimensions",
                ConfigurationValueKind::Integer,
                false,
            )],
            documentation_url: None,
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
        IntegrationDescriptor {
            id: "vector.memory".into(),
            display_name: "In-memory vector store".into(),
            version: "1".into(),
            category: Category::VectorStore,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[
                Capability::DenseRetrieval,
                Capability::MetadataFiltering,
            ]),
            configuration: Vec::new(),
            documentation_url: None,
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
        IntegrationDescriptor {
            id: "storage.jsonl".into(),
            display_name: "Durable JSONL store".into(),
            version: "1".into(),
            category: Category::WorkflowStore,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[Capability::DurableState]),
            configuration: vec![ConfigurationField::config(
                "path",
                "Store file path",
                ConfigurationValueKind::String,
                true,
            )],
            documentation_url: None,
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
        IntegrationDescriptor {
            id: "observability.opentelemetry".into(),
            display_name: "OpenTelemetry adapter".into(),
            version: "1".into(),
            category: Category::Observability,
            stability: IntegrationStability::Stable,
            capabilities: capabilities(&[Capability::Traces, Capability::Metrics]),
            configuration: Vec::new(),
            documentation_url: Some("https://opentelemetry.io/docs/".into()),
            deprecation_message: None,
            metadata: BTreeMap::new(),
        },
    ]
}
