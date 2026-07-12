//! **ferrant** — a lightweight, multi-provider AI agent framework in pure Rust,
//! inspired by [Agno](https://github.com/agno-agi/agno).
//!
//! Core pieces:
//! - [`llm::Model`] — trait for any LLM provider (OpenAI + Anthropic included).
//! - [`tool::Tool`] — trait for anything the agent can call (with [`tool::FunctionTool`]
//!   for defining tools from plain closures).
//! - [`agent::Agent`] — the reasoning loop that ties a model + tools + memory together.
//! - [`memory::Storage`] — pluggable session persistence.
//!
//! ```no_run
//! use ferrant::agent::Agent;
//! use ferrant::llm::openai::OpenAiModel;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let model = OpenAiModel::new("gpt-4o-mini", std::env::var("OPENAI_API_KEY")?);
//! let mut agent = Agent::builder(model)
//!     .instructions("You are a concise, helpful assistant.")
//!     .build();
//!
//! let answer = agent.run("What is the capital of France?").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod cli;
pub mod error;
pub mod evaluation;
pub mod graph;
pub mod integrations;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod message;
pub mod observability;
pub mod orchestration;
pub mod persistence;
pub mod rag;
pub mod runtime;
pub mod skills;
pub mod structured;
pub mod tool;
pub mod tracing;
pub mod workflow;

pub use agent::{Agent, AgentBuilder};
pub use error::{AgentError, Result};
pub use evaluation::{
    AgentEvaluationTarget, EvaluationCase, EvaluationConfig, EvaluationDataset, EvaluationReport,
    EvaluationRunner, ExactMatchScorer, FunctionEvaluationTarget, FunctionScorer, RegressionCheck,
    RegressionThresholds, Scorer,
};
pub use graph::{
    FileGraphStore, Graph, GraphBuilder, GraphCheckpoint, GraphCheckpointStore, GraphError,
    GraphFailure, GraphFailureKind, GraphResult, GraphStatus, GraphStoreError, GraphStoreResult,
    GraphValidationCode, GraphValidationError, GraphValidationIssue, InMemoryGraphStore,
    NodeContext, NodeExecutionStatus, NodeInvocation, NodeOutput, NodeRetryPolicy, StateUpdate,
};
pub use integrations::{
    FunctionIntegrationFactory, IntegrationCapability, IntegrationCategory, IntegrationDescriptor,
    IntegrationRegistry, IntegrationStability,
};
pub use llm::{ModelResponse, Usage};
pub use mcp::McpClient;
pub use memory::{FileStorage, InMemoryStorage, Storage};
pub use message::{ContentPart, Message, Role};
pub use observability::{
    CompositeTracer, InMemoryMetricsCollector, InMemorySpanExporter, InMemoryUsageCollector,
    MetricRecord, MetricsRecorder, ModelPricing, OpenTelemetryAdapter, OpenTelemetryExporter,
    OpenTelemetrySpan, PricingTable, PricingUsageRecorder, UsageRecord, UsageRecorder,
};
pub use orchestration::{AgentTeam, AgentTool};
pub use persistence::{
    AppendOutcome, AtomicJsonFile, DurableJsonlStore, EvaluationBaselineStore,
    EvaluationReportLogStore, MetricsLogStore, OpenTelemetryLogStore, PersistentRecordStore,
    StoredRecord, TraceLogStore, UsageLogStore,
};
pub use rag::{
    Citation, ContextFormatter, Document, DocumentLoader, Embedder, FileVectorStore,
    FormattedContext, FunctionQueryTransformer, HashEmbedder, InMemoryVectorStore,
    IngestionPipeline, LexicalReranker, MetadataFilter, OpenAiEmbedder, QueryTransformer, Reranker,
    RetrievalOptions, RetrievalStrategy, Retriever, RetrieverTool, TextChunker, TextFileLoader,
    VectorStore,
};
pub use runtime::{ExecutionPolicy, StreamEvent};
pub use skills::{Skill, SkillCatalog, SkillError, SkillLimits, SkillMetadata, SkillSource};
pub use structured::{parse_structured, validate_json};
pub use tool::{FunctionTool, Tool, ToolSpec};
#[cfg(feature = "opentelemetry")]
pub use tracing::OpenTelemetryTracer;
pub use tracing::{InMemoryTracer, NoopTracer, TraceEvent, Tracer};
pub use workflow::{FileWorkflowStore, Workflow, WorkflowState, WorkflowStore};
