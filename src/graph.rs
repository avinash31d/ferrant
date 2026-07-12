//! Durable graph-based workflow orchestration.
//!
//! A graph runs in checkpointed supersteps. Every node in the current frontier
//! observes the same state snapshot and may run concurrently. Individual node
//! results are checkpointed as they finish, while state updates and routing are
//! committed only after the entire frontier succeeds. This makes recovery
//! deterministic and avoids re-running successful siblings when one branch
//! fails.

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};

/// The durable lifecycle of a graph execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
}

/// The durable lifecycle of one node invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeExecutionStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
}

/// A state mutation returned by a node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum StateUpdate {
    /// Leave graph state unchanged.
    #[default]
    None,
    /// Recursively merge an object into graph state. Non-object values replace
    /// the value at their path.
    Merge(Value),
    /// Replace graph state in its entirety.
    Replace(Value),
}

/// The serializable result of a graph node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeOutput {
    pub update: StateUpdate,
    /// Route labels selected by this node. Route-labelled edges whose label is
    /// present here are followed in addition to unconditional edges.
    #[serde(default)]
    pub routes: Vec<String>,
    /// Pause after committing this superstep. The computed next frontier is
    /// checkpointed and will execute when the graph is resumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_reason: Option<String>,
}

impl NodeOutput {
    pub fn no_update() -> Self {
        Self::default()
    }

    pub fn merge(value: Value) -> Self {
        Self {
            update: StateUpdate::Merge(value),
            ..Self::default()
        }
    }

    pub fn replace(value: Value) -> Self {
        Self {
            update: StateUpdate::Replace(value),
            ..Self::default()
        }
    }

    pub fn route(mut self, route: impl Into<String>) -> Self {
        self.routes.push(route.into());
        self
    }

    pub fn routes(mut self, routes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.routes.extend(routes.into_iter().map(Into::into));
        self
    }

    pub fn and_pause(mut self, reason: impl Into<String>) -> Self {
        self.pause_reason = Some(reason.into());
        self
    }
}

/// Stable context supplied to a node invocation.
#[derive(Debug, Clone)]
pub struct NodeContext {
    pub execution_id: String,
    pub node_name: String,
    /// Stable across retries and crash recovery. Side-effecting nodes should
    /// pass this key to idempotent downstream APIs.
    pub idempotency_key: String,
    /// One-based attempt number across all recoveries of this invocation.
    pub attempt: u32,
    /// Read-only state snapshot shared by every node in this superstep.
    pub state: Value,
}

/// Per-node retry and timeout configuration.
#[derive(Debug, Clone)]
pub struct NodeRetryPolicy {
    /// Total attempts made during one call to `run`, `resume`, or `recover`.
    pub max_attempts: u32,
    pub initial_backoff: Duration,
    pub backoff_multiplier: f64,
    pub max_backoff: Duration,
    pub attempt_timeout: Option<Duration>,
}

impl Default for NodeRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(5),
            attempt_timeout: None,
        }
    }
}

impl NodeRetryPolicy {
    pub fn attempts(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Self::default()
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.attempt_timeout = Some(timeout);
        self
    }

    pub fn with_backoff(mut self, initial: Duration, maximum: Duration) -> Self {
        self.initial_backoff = initial;
        self.max_backoff = maximum;
        self
    }
}

/// Failure metadata retained in checkpoints for inspection and recovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphFailure {
    pub kind: GraphFailureKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphFailureKind {
    Node,
    Routing,
    StateConflict,
    MaxSteps,
    Deadlock,
}

/// Durable state for one node invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInvocation {
    pub sequence: u64,
    pub node: String,
    #[serde(default)]
    pub sources: Vec<String>,
    pub idempotency_key: String,
    pub status: NodeExecutionStatus,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<NodeOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Ensures an `interrupt_before` breakpoint fires only once for this
    /// invocation, including across process restarts.
    #[serde(default)]
    pub breakpoint_passed: bool,
}

/// Complete, serializable checkpoint for an execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphCheckpoint {
    pub graph_name: String,
    pub graph_version: String,
    pub execution_id: String,
    /// Optimistic-concurrency revision, incremented on every durable write.
    pub revision: u64,
    pub status: GraphStatus,
    pub state: Value,
    #[serde(default)]
    pub frontier: Vec<NodeInvocation>,
    /// Join target -> predecessors that have arrived at its durable barrier.
    #[serde(default)]
    pub pending_joins: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    pub completed_nodes: BTreeMap<String, u64>,
    /// Number of successfully committed node invocations.
    pub steps: u64,
    pub next_sequence: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_failure: Option<GraphFailure>,
}

/// Errors produced by checkpoint implementations.
#[derive(Debug, Error)]
pub enum GraphStoreError {
    #[error("checkpoint revision conflict (expected {expected:?}, actual {actual:?})")]
    Conflict {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    #[error("invalid execution id '{0}'; use ASCII letters, numbers, '-', or '_'")]
    InvalidExecutionId(String),
    #[error("invalid checkpoint revision: expected {expected}, got {actual}")]
    InvalidRevision { expected: u64, actual: u64 },
    #[error("checkpoint I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("checkpoint serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type GraphStoreResult<T> = std::result::Result<T, GraphStoreError>;

/// Durable checkpoint backend with optimistic concurrency.
///
/// `expected_revision == None` means create-only. Replacements must pass the
/// revision returned by `load`, and the new checkpoint must increment it by
/// exactly one.
#[async_trait]
pub trait GraphCheckpointStore: Send + Sync {
    async fn load(&self, execution_id: &str) -> GraphStoreResult<Option<GraphCheckpoint>>;

    async fn save(
        &self,
        checkpoint: &GraphCheckpoint,
        expected_revision: Option<u64>,
    ) -> GraphStoreResult<()>;

    async fn delete(&self, execution_id: &str) -> GraphStoreResult<()>;
}

/// Thread-safe in-memory backend, useful for tests and ephemeral workers.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraphStore {
    checkpoints: Arc<RwLock<HashMap<String, GraphCheckpoint>>>,
}

#[async_trait]
impl GraphCheckpointStore for InMemoryGraphStore {
    async fn load(&self, execution_id: &str) -> GraphStoreResult<Option<GraphCheckpoint>> {
        validate_execution_id(execution_id)?;
        Ok(self.checkpoints.read().await.get(execution_id).cloned())
    }

    async fn save(
        &self,
        checkpoint: &GraphCheckpoint,
        expected_revision: Option<u64>,
    ) -> GraphStoreResult<()> {
        validate_execution_id(&checkpoint.execution_id)?;
        let mut checkpoints = self.checkpoints.write().await;
        let actual = checkpoints
            .get(&checkpoint.execution_id)
            .map(|current| current.revision);
        validate_revision(checkpoint.revision, expected_revision, actual)?;
        checkpoints.insert(checkpoint.execution_id.clone(), checkpoint.clone());
        Ok(())
    }

    async fn delete(&self, execution_id: &str) -> GraphStoreResult<()> {
        validate_execution_id(execution_id)?;
        self.checkpoints.write().await.remove(execution_id);
        Ok(())
    }
}

/// Atomic JSON-file checkpoint backend that survives process restarts.
///
/// Writes use a temporary file, `sync_all`, and an atomic rename. Optimistic
/// revision checks prevent lost updates between callers sharing this backend.
#[derive(Debug, Clone)]
pub struct FileGraphStore {
    directory: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl FileGraphStore {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    fn path(&self, execution_id: &str) -> GraphStoreResult<PathBuf> {
        validate_execution_id(execution_id)?;
        Ok(self.directory.join(format!("{execution_id}.json")))
    }

    async fn load_unlocked(&self, execution_id: &str) -> GraphStoreResult<Option<GraphCheckpoint>> {
        let path = self.path(execution_id)?;
        match tokio::fs::read(path).await {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
}

async fn sync_graph_directory(path: &Path) -> io::Result<()> {
    match tokio::fs::File::open(path).await {
        Ok(directory) => match directory.sync_all().await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => Ok(()),
            Err(error) => Err(error),
        },
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => Ok(()),
        Err(error) => Err(error),
    }
}

#[async_trait]
impl GraphCheckpointStore for FileGraphStore {
    async fn load(&self, execution_id: &str) -> GraphStoreResult<Option<GraphCheckpoint>> {
        let _guard = self.lock.lock().await;
        self.load_unlocked(execution_id).await
    }

    async fn save(
        &self,
        checkpoint: &GraphCheckpoint,
        expected_revision: Option<u64>,
    ) -> GraphStoreResult<()> {
        validate_execution_id(&checkpoint.execution_id)?;
        let _guard = self.lock.lock().await;
        let actual = self
            .load_unlocked(&checkpoint.execution_id)
            .await?
            .map(|current| current.revision);
        validate_revision(checkpoint.revision, expected_revision, actual)?;

        tokio::fs::create_dir_all(&self.directory).await?;
        let target = self.path(&checkpoint.execution_id)?;
        let temporary = self.directory.join(format!(
            ".{}.{}.{}.tmp",
            checkpoint.execution_id,
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let bytes = serde_json::to_vec_pretty(checkpoint)?;
        let write_result = async {
            let mut file = tokio::fs::File::create(&temporary).await?;
            file.write_all(&bytes).await?;
            file.flush().await?;
            file.sync_all().await?;
            drop(file);
            tokio::fs::rename(&temporary, &target).await?;
            // Persist the directory entry where supported.
            sync_graph_directory(&self.directory).await?;
            Ok::<(), io::Error>(())
        }
        .await;
        if let Err(error) = write_result {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(error.into());
        }
        Ok(())
    }

    async fn delete(&self, execution_id: &str) -> GraphStoreResult<()> {
        let _guard = self.lock.lock().await;
        let path = self.path(execution_id)?;
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

fn validate_execution_id(execution_id: &str) -> GraphStoreResult<()> {
    if execution_id.is_empty()
        || !execution_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(GraphStoreError::InvalidExecutionId(execution_id.to_owned()));
    }
    Ok(())
}

fn validate_revision(
    new_revision: u64,
    expected_revision: Option<u64>,
    actual_revision: Option<u64>,
) -> GraphStoreResult<()> {
    if expected_revision != actual_revision {
        return Err(GraphStoreError::Conflict {
            expected: expected_revision,
            actual: actual_revision,
        });
    }
    let required = expected_revision.unwrap_or(0).saturating_add(1);
    if new_revision != required {
        return Err(GraphStoreError::InvalidRevision {
            expected: required,
            actual: new_revision,
        });
    }
    Ok(())
}

/// Static graph validation issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphValidationIssue {
    pub code: GraphValidationCode,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphValidationCode {
    InvalidGraphName,
    InvalidVersion,
    MissingEntry,
    MissingNode,
    DuplicateNode,
    InvalidNodeName,
    InvalidRoute,
    DuplicateEdge,
    InvalidJoin,
    UnreachableNode,
    InvalidRetryPolicy,
    InvalidMaxSteps,
}

#[derive(Debug, Clone, Error)]
#[error("graph definition is invalid: {summary}")]
pub struct GraphValidationError {
    pub issues: Vec<GraphValidationIssue>,
    summary: String,
}

impl GraphValidationError {
    fn new(issues: Vec<GraphValidationIssue>) -> Self {
        let summary = issues
            .iter()
            .map(|issue| issue.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        Self { issues, summary }
    }
}

/// Runtime graph errors. Execution failures are checkpointed before being
/// returned, unless the checkpoint backend itself failed.
#[derive(Debug, Error)]
pub enum GraphError {
    #[error(transparent)]
    Store(#[from] GraphStoreError),
    #[error("execution '{0}' is already running; use recover only after its worker is gone")]
    AlreadyRunning(String),
    #[error("execution '{execution_id}' cannot be resumed from status {status:?}")]
    CannotResume {
        execution_id: String,
        status: GraphStatus,
    },
    #[error(
        "checkpoint belongs to graph '{actual_name}' version '{actual_version}', not '{expected_name}' version '{expected_version}'"
    )]
    DefinitionMismatch {
        expected_name: String,
        expected_version: String,
        actual_name: String,
        actual_version: String,
    },
    #[error("checkpoint for execution '{execution_id}' is invalid: {message}")]
    InvalidCheckpoint {
        execution_id: String,
        message: String,
    },
    #[error("node '{node}' failed in execution '{execution_id}': {message}")]
    NodeFailed {
        execution_id: String,
        node: String,
        message: String,
    },
    #[error("routing failed in execution '{execution_id}': {message}")]
    Routing {
        execution_id: String,
        message: String,
    },
    #[error("parallel state updates conflict in execution '{execution_id}': {message}")]
    StateConflict {
        execution_id: String,
        message: String,
    },
    #[error("execution '{execution_id}' exceeded the graph limit of {max_steps} steps")]
    MaxStepsExceeded {
        execution_id: String,
        max_steps: u64,
    },
    #[error("execution '{execution_id}' cannot satisfy join barriers: {waiting_for}")]
    JoinDeadlock {
        execution_id: String,
        waiting_for: String,
    },
}

pub type GraphResult<T> = std::result::Result<T, GraphError>;

type NodeFn = Arc<
    dyn Fn(NodeContext) -> BoxFuture<'static, anyhow::Result<NodeOutput>> + Send + Sync + 'static,
>;
type RoutePredicate =
    Arc<dyn Fn(&Value, &NodeOutput) -> anyhow::Result<bool> + Send + Sync + 'static>;

struct NodeSpec {
    function: NodeFn,
    retry: NodeRetryPolicy,
}

enum EdgeMatcher {
    Always,
    Route(String),
    Predicate(RoutePredicate),
}

struct EdgeSpec {
    from: String,
    to: String,
    matcher: EdgeMatcher,
    join: bool,
}

/// Builder for a validated graph definition.
pub struct GraphBuilder {
    name: String,
    version: String,
    entry: Option<String>,
    nodes: BTreeMap<String, Arc<NodeSpec>>,
    duplicate_nodes: BTreeSet<String>,
    edges: Vec<EdgeSpec>,
    joins: BTreeMap<String, BTreeSet<String>>,
    duplicate_joins: BTreeSet<String>,
    interrupt_before: BTreeSet<String>,
    max_steps: u64,
    store: Arc<dyn GraphCheckpointStore>,
}

impl GraphBuilder {
    pub fn new(name: impl Into<String>, store: impl GraphCheckpointStore + 'static) -> Self {
        Self::with_store(name, Arc::new(store))
    }

    pub fn with_store(name: impl Into<String>, store: Arc<dyn GraphCheckpointStore>) -> Self {
        Self {
            name: name.into(),
            version: "1".to_owned(),
            entry: None,
            nodes: BTreeMap::new(),
            duplicate_nodes: BTreeSet::new(),
            edges: Vec::new(),
            joins: BTreeMap::new(),
            duplicate_joins: BTreeSet::new(),
            interrupt_before: BTreeSet::new(),
            max_steps: 1_000,
            store,
        }
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn entry(mut self, node: impl Into<String>) -> Self {
        self.entry = Some(node.into());
        self
    }

    pub fn max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn node<F, Fut>(self, name: impl Into<String>, function: F) -> Self
    where
        F: Fn(NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<NodeOutput>> + Send + 'static,
    {
        self.node_with_retry(name, NodeRetryPolicy::default(), function)
    }

    pub fn node_with_retry<F, Fut>(
        mut self,
        name: impl Into<String>,
        retry: NodeRetryPolicy,
        function: F,
    ) -> Self
    where
        F: Fn(NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<NodeOutput>> + Send + 'static,
    {
        let name = name.into();
        if self.nodes.contains_key(&name) {
            self.duplicate_nodes.insert(name.clone());
        }
        let function = Arc::new(move |context| {
            Box::pin(function(context)) as BoxFuture<'static, anyhow::Result<NodeOutput>>
        });
        self.nodes
            .insert(name, Arc::new(NodeSpec { function, retry }));
        self
    }

    /// Follow an unconditional edge. Multiple matching outgoing edges fan out.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(EdgeSpec {
            from: from.into(),
            to: to.into(),
            matcher: EdgeMatcher::Always,
            join: false,
        });
        self
    }

    /// Follow this edge when the node emits `route` in its `NodeOutput`.
    pub fn route(
        mut self,
        from: impl Into<String>,
        route: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        self.edges.push(EdgeSpec {
            from: from.into(),
            to: to.into(),
            matcher: EdgeMatcher::Route(route.into()),
            join: false,
        });
        self
    }

    /// Follow this edge when a synchronous predicate matches the committed
    /// superstep state and this node's output.
    pub fn conditional_edge<F>(
        self,
        from: impl Into<String>,
        to: impl Into<String>,
        predicate: F,
    ) -> Self
    where
        F: Fn(&Value, &NodeOutput) -> bool + Send + Sync + 'static,
    {
        self.try_conditional_edge(from, to, move |state, output| Ok(predicate(state, output)))
    }

    /// Fallible conditional edge, useful when routing parses typed state.
    pub fn try_conditional_edge<F>(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        predicate: F,
    ) -> Self
    where
        F: Fn(&Value, &NodeOutput) -> anyhow::Result<bool> + Send + Sync + 'static,
    {
        self.edges.push(EdgeSpec {
            from: from.into(),
            to: to.into(),
            matcher: EdgeMatcher::Predicate(Arc::new(predicate)),
            join: false,
        });
        self
    }

    /// Add a durable AND-join. The target is scheduled once every listed
    /// predecessor has arrived. The barrier survives pauses and restarts.
    pub fn join<I, S>(mut self, predecessors: I, target: impl Into<String>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let target = target.into();
        let predecessors = predecessors
            .into_iter()
            .map(Into::into)
            .collect::<BTreeSet<_>>();
        if self.joins.contains_key(&target) {
            self.duplicate_joins.insert(target.clone());
        }
        for predecessor in &predecessors {
            self.edges.push(EdgeSpec {
                from: predecessor.clone(),
                to: target.clone(),
                matcher: EdgeMatcher::Always,
                join: true,
            });
        }
        self.joins.insert(target, predecessors);
        self
    }

    /// Pause immediately before each invocation of this node. A durable flag
    /// ensures resume passes the breakpoint exactly once for that invocation.
    pub fn interrupt_before(mut self, node: impl Into<String>) -> Self {
        self.interrupt_before.insert(node.into());
        self
    }

    pub fn build(self) -> std::result::Result<Graph, GraphValidationError> {
        let issues = self.validation_issues();
        if !issues.is_empty() {
            return Err(GraphValidationError::new(issues));
        }
        Ok(Graph {
            name: self.name,
            version: self.version,
            entry: self.entry.expect("entry validated"),
            nodes: self.nodes,
            edges: Arc::new(self.edges),
            joins: Arc::new(self.joins),
            interrupt_before: Arc::new(self.interrupt_before),
            max_steps: self.max_steps,
            store: self.store,
        })
    }

    fn validation_issues(&self) -> Vec<GraphValidationIssue> {
        let mut issues = Vec::new();
        let mut issue = |code, message: String| {
            issues.push(GraphValidationIssue { code, message });
        };

        if self.name.trim().is_empty() {
            issue(
                GraphValidationCode::InvalidGraphName,
                "graph name cannot be empty".to_owned(),
            );
        }
        if self.version.trim().is_empty() {
            issue(
                GraphValidationCode::InvalidVersion,
                "graph version cannot be empty".to_owned(),
            );
        }
        if self.max_steps == 0 {
            issue(
                GraphValidationCode::InvalidMaxSteps,
                "max_steps must be at least one".to_owned(),
            );
        }
        if self.entry.is_none() {
            issue(
                GraphValidationCode::MissingEntry,
                "graph entry node is not configured".to_owned(),
            );
        }
        for duplicate in &self.duplicate_nodes {
            issue(
                GraphValidationCode::DuplicateNode,
                format!("node '{duplicate}' is defined more than once"),
            );
        }
        for (name, node) in &self.nodes {
            if name.trim().is_empty() {
                issue(
                    GraphValidationCode::InvalidNodeName,
                    "node names cannot be empty".to_owned(),
                );
            }
            if node.retry.max_attempts == 0
                || !node.retry.backoff_multiplier.is_finite()
                || node.retry.backoff_multiplier < 1.0
                || node.retry.max_backoff < node.retry.initial_backoff
                || node.retry.attempt_timeout == Some(Duration::ZERO)
            {
                issue(
                    GraphValidationCode::InvalidRetryPolicy,
                    format!("node '{name}' has an invalid retry policy"),
                );
            }
        }
        if let Some(entry) = &self.entry {
            if !self.nodes.contains_key(entry) {
                issue(
                    GraphValidationCode::MissingNode,
                    format!("entry node '{entry}' is not defined"),
                );
            }
        }
        for node in &self.interrupt_before {
            if !self.nodes.contains_key(node) {
                issue(
                    GraphValidationCode::MissingNode,
                    format!("interrupt node '{node}' is not defined"),
                );
            }
        }
        for duplicate in &self.duplicate_joins {
            issue(
                GraphValidationCode::InvalidJoin,
                format!("join target '{duplicate}' is configured more than once"),
            );
        }
        for (target, predecessors) in &self.joins {
            if predecessors.len() < 2 {
                issue(
                    GraphValidationCode::InvalidJoin,
                    format!("join '{target}' requires at least two distinct predecessors"),
                );
            }
            if predecessors.contains(target) {
                issue(
                    GraphValidationCode::InvalidJoin,
                    format!("join '{target}' cannot list itself as a predecessor"),
                );
            }
        }

        let mut simple_edges = BTreeSet::new();
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from) {
                issue(
                    GraphValidationCode::MissingNode,
                    format!("edge source '{}' is not defined", edge.from),
                );
            }
            if !self.nodes.contains_key(&edge.to) {
                issue(
                    GraphValidationCode::MissingNode,
                    format!("edge target '{}' is not defined", edge.to),
                );
            }
            if let EdgeMatcher::Route(route) = &edge.matcher {
                if route.trim().is_empty() {
                    issue(
                        GraphValidationCode::InvalidRoute,
                        format!(
                            "route from '{}' to '{}' cannot be empty",
                            edge.from, edge.to
                        ),
                    );
                }
            }
            let discriminator = match &edge.matcher {
                EdgeMatcher::Always => Some("always".to_owned()),
                EdgeMatcher::Route(route) => Some(format!("route:{route}")),
                EdgeMatcher::Predicate(_) => None,
            };
            if let Some(discriminator) = discriminator {
                let key = (edge.from.clone(), edge.to.clone(), discriminator, edge.join);
                if !simple_edges.insert(key) {
                    issue(
                        GraphValidationCode::DuplicateEdge,
                        format!("duplicate edge from '{}' to '{}'", edge.from, edge.to),
                    );
                }
            }
            if self.joins.contains_key(&edge.to) && !edge.join {
                issue(
                    GraphValidationCode::InvalidJoin,
                    format!(
                        "join target '{}' also has a non-join incoming edge from '{}'",
                        edge.to, edge.from
                    ),
                );
            }
        }

        if let Some(entry) = &self.entry {
            if self.nodes.contains_key(entry) {
                let mut reachable = BTreeSet::new();
                let mut queue = VecDeque::from([entry.clone()]);
                while let Some(node) = queue.pop_front() {
                    if !reachable.insert(node.clone()) {
                        continue;
                    }
                    for edge in self.edges.iter().filter(|edge| edge.from == node) {
                        queue.push_back(edge.to.clone());
                    }
                }
                for node in self.nodes.keys() {
                    if !reachable.contains(node) {
                        issue(
                            GraphValidationCode::UnreachableNode,
                            format!("node '{node}' is unreachable from entry '{entry}'"),
                        );
                    }
                }
            }
        }

        issues
    }
}

/// A validated, reusable graph definition.
#[derive(Clone)]
pub struct Graph {
    name: String,
    version: String,
    entry: String,
    nodes: BTreeMap<String, Arc<NodeSpec>>,
    edges: Arc<Vec<EdgeSpec>>,
    joins: Arc<BTreeMap<String, BTreeSet<String>>>,
    interrupt_before: Arc<BTreeSet<String>>,
    max_steps: u64,
    store: Arc<dyn GraphCheckpointStore>,
}

impl Graph {
    pub fn builder(
        name: impl Into<String>,
        store: impl GraphCheckpointStore + 'static,
    ) -> GraphBuilder {
        GraphBuilder::new(name, store)
    }

    pub fn builder_with_store(
        name: impl Into<String>,
        store: Arc<dyn GraphCheckpointStore>,
    ) -> GraphBuilder {
        GraphBuilder::with_store(name, store)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub async fn checkpoint(&self, execution_id: &str) -> GraphResult<Option<GraphCheckpoint>> {
        let checkpoint = self.store.load(execution_id).await?;
        if let Some(checkpoint) = &checkpoint {
            self.ensure_checkpoint(checkpoint, execution_id)?;
        }
        Ok(checkpoint)
    }

    /// Start a new execution. Repeating `run` for a completed id returns its
    /// durable result; paused and failed executions require explicit `resume`.
    pub async fn run(
        &self,
        execution_id: &str,
        initial_state: Value,
    ) -> GraphResult<GraphCheckpoint> {
        validate_execution_id(execution_id)?;
        if let Some(checkpoint) = self.store.load(execution_id).await? {
            self.ensure_checkpoint(&checkpoint, execution_id)?;
            return match checkpoint.status {
                GraphStatus::Completed | GraphStatus::Paused => Ok(checkpoint),
                GraphStatus::Running => Err(GraphError::AlreadyRunning(execution_id.to_owned())),
                GraphStatus::Failed | GraphStatus::Pending => Err(GraphError::CannotResume {
                    execution_id: execution_id.to_owned(),
                    status: checkpoint.status,
                }),
            };
        }

        let entry = self.new_invocation(execution_id, &self.entry, Vec::new(), 0);
        let mut checkpoint = GraphCheckpoint {
            graph_name: self.name.clone(),
            graph_version: self.version.clone(),
            execution_id: execution_id.to_owned(),
            revision: 0,
            status: GraphStatus::Running,
            state: initial_state,
            frontier: vec![entry],
            pending_joins: BTreeMap::new(),
            completed_nodes: BTreeMap::new(),
            steps: 0,
            next_sequence: 1,
            pause_reason: None,
            last_failure: None,
        };
        self.persist(&mut checkpoint).await?;
        self.drive(checkpoint).await
    }

    /// Resume a paused or failed execution. Failed node invocations retain the
    /// same idempotency key and receive a fresh retry window.
    pub async fn resume(&self, execution_id: &str) -> GraphResult<GraphCheckpoint> {
        self.resume_with(execution_id, None).await
    }

    /// Resume while merging human or external input into durable state.
    pub async fn resume_with(
        &self,
        execution_id: &str,
        state_update: Option<Value>,
    ) -> GraphResult<GraphCheckpoint> {
        let mut checkpoint =
            self.store
                .load(execution_id)
                .await?
                .ok_or_else(|| GraphError::CannotResume {
                    execution_id: execution_id.to_owned(),
                    status: GraphStatus::Pending,
                })?;
        self.ensure_checkpoint(&checkpoint, execution_id)?;
        match checkpoint.status {
            GraphStatus::Paused | GraphStatus::Failed => {}
            GraphStatus::Completed => return Ok(checkpoint),
            status => {
                return Err(GraphError::CannotResume {
                    execution_id: execution_id.to_owned(),
                    status,
                })
            }
        }
        self.prepare_continuation(&mut checkpoint, state_update);
        self.persist(&mut checkpoint).await?;
        self.drive(checkpoint).await
    }

    /// Recover a checkpoint left `Running` by a dead worker. Calling this while
    /// the original worker is alive can duplicate side effects, so callers must
    /// first establish ownership (for example with a distributed lease).
    pub async fn recover(&self, execution_id: &str) -> GraphResult<GraphCheckpoint> {
        let mut checkpoint =
            self.store
                .load(execution_id)
                .await?
                .ok_or_else(|| GraphError::CannotResume {
                    execution_id: execution_id.to_owned(),
                    status: GraphStatus::Pending,
                })?;
        self.ensure_checkpoint(&checkpoint, execution_id)?;
        if checkpoint.status == GraphStatus::Completed {
            return Ok(checkpoint);
        }
        self.prepare_continuation(&mut checkpoint, None);
        self.persist(&mut checkpoint).await?;
        self.drive(checkpoint).await
    }

    /// Persist an offline pause request. If a worker is currently executing,
    /// optimistic concurrency makes either this request or its next node result
    /// win explicitly instead of silently losing an update.
    pub async fn pause(
        &self,
        execution_id: &str,
        reason: impl Into<String>,
    ) -> GraphResult<GraphCheckpoint> {
        let mut checkpoint =
            self.store
                .load(execution_id)
                .await?
                .ok_or_else(|| GraphError::CannotResume {
                    execution_id: execution_id.to_owned(),
                    status: GraphStatus::Pending,
                })?;
        self.ensure_checkpoint(&checkpoint, execution_id)?;
        if checkpoint.status == GraphStatus::Completed {
            return Ok(checkpoint);
        }
        checkpoint.status = GraphStatus::Paused;
        checkpoint.pause_reason = Some(reason.into());
        self.persist(&mut checkpoint).await?;
        Ok(checkpoint)
    }

    pub async fn delete_checkpoint(&self, execution_id: &str) -> GraphResult<()> {
        Ok(self.store.delete(execution_id).await?)
    }

    fn prepare_continuation(&self, checkpoint: &mut GraphCheckpoint, state_update: Option<Value>) {
        for invocation in &mut checkpoint.frontier {
            if matches!(
                invocation.status,
                NodeExecutionStatus::Running | NodeExecutionStatus::Failed
            ) {
                invocation.status = NodeExecutionStatus::Pending;
                invocation.error = None;
            }
        }
        if let Some(update) = state_update {
            merge_value(&mut checkpoint.state, update);
        }
        checkpoint.status = GraphStatus::Running;
        checkpoint.pause_reason = None;
        checkpoint.last_failure = None;
    }

    fn ensure_definition(&self, checkpoint: &GraphCheckpoint) -> GraphResult<()> {
        if checkpoint.graph_name != self.name || checkpoint.graph_version != self.version {
            return Err(GraphError::DefinitionMismatch {
                expected_name: self.name.clone(),
                expected_version: self.version.clone(),
                actual_name: checkpoint.graph_name.clone(),
                actual_version: checkpoint.graph_version.clone(),
            });
        }
        Ok(())
    }

    fn ensure_checkpoint(
        &self,
        checkpoint: &GraphCheckpoint,
        expected_execution_id: &str,
    ) -> GraphResult<()> {
        self.ensure_definition(checkpoint)?;
        let invalid = |message: String| GraphError::InvalidCheckpoint {
            execution_id: expected_execution_id.to_owned(),
            message,
        };
        if checkpoint.execution_id != expected_execution_id {
            return Err(invalid(format!(
                "stored execution id is '{}'",
                checkpoint.execution_id
            )));
        }
        if checkpoint.revision == 0 {
            return Err(invalid(
                "persisted revision must be greater than zero".to_owned(),
            ));
        }

        let mut sequences = BTreeSet::new();
        for invocation in &checkpoint.frontier {
            if !self.nodes.contains_key(&invocation.node) {
                return Err(invalid(format!(
                    "frontier references unknown node '{}'",
                    invocation.node
                )));
            }
            if !sequences.insert(invocation.sequence) {
                return Err(invalid(format!(
                    "frontier contains duplicate sequence {}",
                    invocation.sequence
                )));
            }
            let expected_key = format!(
                "{}:{}:{}",
                checkpoint.execution_id, invocation.sequence, invocation.node
            );
            if invocation.idempotency_key != expected_key {
                return Err(invalid(format!(
                    "node '{}' has an invalid idempotency key",
                    invocation.node
                )));
            }
            match invocation.status {
                NodeExecutionStatus::Succeeded if invocation.output.is_none() => {
                    return Err(invalid(format!(
                        "successful node '{}' has no output",
                        invocation.node
                    )))
                }
                NodeExecutionStatus::Failed if invocation.error.is_none() => {
                    return Err(invalid(format!(
                        "failed node '{}' has no error",
                        invocation.node
                    )))
                }
                NodeExecutionStatus::Pending | NodeExecutionStatus::Running
                    if invocation.output.is_some() =>
                {
                    return Err(invalid(format!(
                        "unfinished node '{}' already has an output",
                        invocation.node
                    )))
                }
                _ => {}
            }
            if invocation.sequence >= checkpoint.next_sequence {
                return Err(invalid(format!(
                    "next_sequence {} does not follow invocation sequence {}",
                    checkpoint.next_sequence, invocation.sequence
                )));
            }
        }

        for (target, arrived) in &checkpoint.pending_joins {
            let Some(expected) = self.joins.get(target) else {
                return Err(invalid(format!(
                    "checkpoint references unknown join target '{target}'"
                )));
            };
            if !arrived.is_subset(expected) {
                return Err(invalid(format!(
                    "join '{target}' contains an unexpected predecessor"
                )));
            }
        }
        if checkpoint.status == GraphStatus::Completed
            && (!checkpoint.frontier.is_empty() || !checkpoint.pending_joins.is_empty())
        {
            return Err(invalid(
                "completed checkpoint still contains runnable or joined work".to_owned(),
            ));
        }
        Ok(())
    }

    async fn persist(&self, checkpoint: &mut GraphCheckpoint) -> GraphResult<()> {
        let previous = checkpoint.revision;
        checkpoint.revision = previous.saturating_add(1);
        let expected = (previous != 0).then_some(previous);
        if let Err(error) = self.store.save(checkpoint, expected).await {
            checkpoint.revision = previous;
            return Err(error.into());
        }
        Ok(())
    }

    async fn drive(&self, mut checkpoint: GraphCheckpoint) -> GraphResult<GraphCheckpoint> {
        loop {
            if checkpoint.frontier.is_empty() {
                if checkpoint.pending_joins.is_empty() {
                    checkpoint.status = GraphStatus::Completed;
                    checkpoint.pause_reason = None;
                    self.persist(&mut checkpoint).await?;
                    return Ok(checkpoint);
                }
                let waiting_for = format_pending_joins(&checkpoint.pending_joins, &self.joins);
                checkpoint.status = GraphStatus::Failed;
                checkpoint.last_failure = Some(GraphFailure {
                    kind: GraphFailureKind::Deadlock,
                    node: None,
                    message: waiting_for.clone(),
                });
                self.persist(&mut checkpoint).await?;
                return Err(GraphError::JoinDeadlock {
                    execution_id: checkpoint.execution_id,
                    waiting_for,
                });
            }

            let breakpoint_nodes = checkpoint
                .frontier
                .iter()
                .filter(|invocation| {
                    invocation.status == NodeExecutionStatus::Pending
                        && !invocation.breakpoint_passed
                        && self.interrupt_before.contains(&invocation.node)
                })
                .map(|invocation| invocation.node.clone())
                .collect::<BTreeSet<_>>();
            if !breakpoint_nodes.is_empty() {
                for invocation in &mut checkpoint.frontier {
                    if breakpoint_nodes.contains(&invocation.node) {
                        invocation.breakpoint_passed = true;
                    }
                }
                checkpoint.status = GraphStatus::Paused;
                checkpoint.pause_reason = Some(format!(
                    "interrupted before {}",
                    breakpoint_nodes.into_iter().collect::<Vec<_>>().join(", ")
                ));
                self.persist(&mut checkpoint).await?;
                return Ok(checkpoint);
            }

            let frontier_size = checkpoint.frontier.len() as u64;
            if checkpoint.steps.saturating_add(frontier_size) > self.max_steps {
                checkpoint.status = GraphStatus::Failed;
                checkpoint.last_failure = Some(GraphFailure {
                    kind: GraphFailureKind::MaxSteps,
                    node: None,
                    message: format!("maximum of {} steps exceeded", self.max_steps),
                });
                self.persist(&mut checkpoint).await?;
                return Err(GraphError::MaxStepsExceeded {
                    execution_id: checkpoint.execution_id,
                    max_steps: self.max_steps,
                });
            }

            let pending = checkpoint
                .frontier
                .iter()
                .enumerate()
                .filter_map(|(index, invocation)| {
                    (invocation.status == NodeExecutionStatus::Pending).then_some(index)
                })
                .collect::<Vec<_>>();

            if !pending.is_empty() {
                for index in &pending {
                    checkpoint.frontier[*index].status = NodeExecutionStatus::Running;
                }
                self.persist(&mut checkpoint).await?;

                let state = checkpoint.state.clone();
                let execution_id = checkpoint.execution_id.clone();
                let mut executions = futures::stream::FuturesUnordered::new();
                for index in pending {
                    let invocation = checkpoint.frontier[index].clone();
                    let node = self
                        .nodes
                        .get(&invocation.node)
                        .expect("validated node")
                        .clone();
                    let state = state.clone();
                    let execution_id = execution_id.clone();
                    executions.push(async move {
                        let result = execute_node(node, &execution_id, &invocation, state).await;
                        (index, result)
                    });
                }

                while let Some((index, result)) = executions.next().await {
                    let invocation = &mut checkpoint.frontier[index];
                    invocation.attempts = invocation.attempts.saturating_add(result.attempts);
                    match result.output {
                        Ok(output) => {
                            invocation.status = NodeExecutionStatus::Succeeded;
                            invocation.output = Some(output);
                            invocation.error = None;
                        }
                        Err(message) => {
                            invocation.status = NodeExecutionStatus::Failed;
                            invocation.error = Some(message);
                            invocation.output = None;
                        }
                    }
                    self.persist(&mut checkpoint).await?;
                }
            }

            if let Some(failed) = checkpoint
                .frontier
                .iter()
                .filter(|invocation| invocation.status == NodeExecutionStatus::Failed)
                .min_by_key(|invocation| invocation.sequence)
            {
                let node = failed.node.clone();
                let message = failed
                    .error
                    .clone()
                    .unwrap_or_else(|| "node failed without an error".to_owned());
                checkpoint.status = GraphStatus::Failed;
                checkpoint.last_failure = Some(GraphFailure {
                    kind: GraphFailureKind::Node,
                    node: Some(node.clone()),
                    message: message.clone(),
                });
                self.persist(&mut checkpoint).await?;
                return Err(GraphError::NodeFailed {
                    execution_id: checkpoint.execution_id,
                    node,
                    message,
                });
            }

            if checkpoint
                .frontier
                .iter()
                .any(|invocation| invocation.status != NodeExecutionStatus::Succeeded)
            {
                let message = "checkpoint contains an unfinished invocation".to_owned();
                checkpoint.status = GraphStatus::Failed;
                checkpoint.last_failure = Some(GraphFailure {
                    kind: GraphFailureKind::Node,
                    node: None,
                    message: message.clone(),
                });
                self.persist(&mut checkpoint).await?;
                return Err(GraphError::Routing {
                    execution_id: checkpoint.execution_id,
                    message,
                });
            }

            match self.commit_superstep(&mut checkpoint) {
                Ok(()) => {
                    self.persist(&mut checkpoint).await?;
                    if checkpoint.status == GraphStatus::Paused
                        || checkpoint.status == GraphStatus::Completed
                    {
                        return Ok(checkpoint);
                    }
                }
                Err(CommitError::StateConflict(message)) => {
                    checkpoint.status = GraphStatus::Failed;
                    checkpoint.last_failure = Some(GraphFailure {
                        kind: GraphFailureKind::StateConflict,
                        node: None,
                        message: message.clone(),
                    });
                    self.persist(&mut checkpoint).await?;
                    return Err(GraphError::StateConflict {
                        execution_id: checkpoint.execution_id,
                        message,
                    });
                }
                Err(CommitError::Routing(message)) => {
                    checkpoint.status = GraphStatus::Failed;
                    checkpoint.last_failure = Some(GraphFailure {
                        kind: GraphFailureKind::Routing,
                        node: None,
                        message: message.clone(),
                    });
                    self.persist(&mut checkpoint).await?;
                    return Err(GraphError::Routing {
                        execution_id: checkpoint.execution_id,
                        message,
                    });
                }
            }
        }
    }

    fn commit_superstep(
        &self,
        checkpoint: &mut GraphCheckpoint,
    ) -> std::result::Result<(), CommitError> {
        let mut invocations = checkpoint.frontier.clone();
        invocations.sort_by_key(|invocation| invocation.sequence);

        let mut next_state = checkpoint.state.clone();
        apply_parallel_updates(&mut next_state, &invocations)?;

        let mut pending_joins = checkpoint.pending_joins.clone();
        let mut scheduled: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for invocation in &invocations {
            let output = invocation.output.as_ref().expect("successful output");
            for edge in self
                .edges
                .iter()
                .filter(|edge| edge.from == invocation.node)
            {
                let selected = match &edge.matcher {
                    EdgeMatcher::Always => true,
                    EdgeMatcher::Route(route) => output.routes.iter().any(|item| item == route),
                    EdgeMatcher::Predicate(predicate) => {
                        predicate(&next_state, output).map_err(|error| {
                            CommitError::Routing(format!(
                                "edge '{}' -> '{}' predicate failed: {error}",
                                edge.from, edge.to
                            ))
                        })?
                    }
                };
                if !selected {
                    continue;
                }
                if edge.join {
                    pending_joins
                        .entry(edge.to.clone())
                        .or_default()
                        .insert(edge.from.clone());
                } else {
                    scheduled
                        .entry(edge.to.clone())
                        .or_default()
                        .insert(edge.from.clone());
                }
            }
        }

        for (target, expected) in self.joins.iter() {
            let ready = pending_joins
                .get(target)
                .map(|arrived| expected.is_subset(arrived))
                .unwrap_or(false);
            if ready {
                scheduled
                    .entry(target.clone())
                    .or_default()
                    .extend(expected.iter().cloned());
                pending_joins.remove(target);
            }
        }

        for invocation in &invocations {
            *checkpoint
                .completed_nodes
                .entry(invocation.node.clone())
                .or_default() += 1;
        }
        checkpoint.steps = checkpoint.steps.saturating_add(invocations.len() as u64);
        checkpoint.state = next_state;
        checkpoint.pending_joins = pending_joins;

        let mut frontier = Vec::with_capacity(scheduled.len());
        for (node, sources) in scheduled {
            let sequence = checkpoint.next_sequence;
            checkpoint.next_sequence = checkpoint.next_sequence.saturating_add(1);
            frontier.push(self.new_invocation(
                &checkpoint.execution_id,
                &node,
                sources.into_iter().collect(),
                sequence,
            ));
        }
        checkpoint.frontier = frontier;

        let pause_reason = invocations.iter().find_map(|invocation| {
            invocation
                .output
                .as_ref()
                .and_then(|output| output.pause_reason.clone())
        });
        checkpoint.last_failure = None;
        if let Some(reason) = pause_reason {
            checkpoint.status = GraphStatus::Paused;
            checkpoint.pause_reason = Some(reason);
        } else if checkpoint.frontier.is_empty() && checkpoint.pending_joins.is_empty() {
            checkpoint.status = GraphStatus::Completed;
            checkpoint.pause_reason = None;
        } else {
            checkpoint.status = GraphStatus::Running;
            checkpoint.pause_reason = None;
        }
        Ok(())
    }

    fn new_invocation(
        &self,
        execution_id: &str,
        node: &str,
        sources: Vec<String>,
        sequence: u64,
    ) -> NodeInvocation {
        NodeInvocation {
            sequence,
            node: node.to_owned(),
            sources,
            idempotency_key: format!("{execution_id}:{sequence}:{node}"),
            status: NodeExecutionStatus::Pending,
            attempts: 0,
            output: None,
            error: None,
            breakpoint_passed: false,
        }
    }
}

struct NodeRunResult {
    attempts: u32,
    output: std::result::Result<NodeOutput, String>,
}

async fn execute_node(
    node: Arc<NodeSpec>,
    execution_id: &str,
    invocation: &NodeInvocation,
    state: Value,
) -> NodeRunResult {
    let mut backoff = node.retry.initial_backoff;
    let mut last_error = "node did not execute".to_owned();
    for local_attempt in 1..=node.retry.max_attempts {
        let context = NodeContext {
            execution_id: execution_id.to_owned(),
            node_name: invocation.node.clone(),
            idempotency_key: invocation.idempotency_key.clone(),
            attempt: invocation.attempts.saturating_add(local_attempt),
            state: state.clone(),
        };
        match invoke_node(node.function.clone(), context, node.retry.attempt_timeout).await {
            Ok(output) => {
                return NodeRunResult {
                    attempts: local_attempt,
                    output: Ok(output),
                }
            }
            Err(error) => last_error = error,
        }
        if local_attempt < node.retry.max_attempts && !backoff.is_zero() {
            tokio::time::sleep(backoff).await;
        }
        backoff = multiply_duration(
            backoff,
            node.retry.backoff_multiplier,
            node.retry.max_backoff,
        );
    }
    NodeRunResult {
        attempts: node.retry.max_attempts,
        output: Err(last_error),
    }
}

async fn invoke_node(
    function: NodeFn,
    context: NodeContext,
    timeout: Option<Duration>,
) -> std::result::Result<NodeOutput, String> {
    let future = std::panic::AssertUnwindSafe((function)(context)).catch_unwind();
    let caught = if let Some(timeout) = timeout {
        tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| format!("node attempt timed out after {timeout:?}"))?
    } else {
        future.await
    };
    match caught {
        Ok(result) => result.map_err(|error| format!("{error:#}")),
        Err(payload) => Err(format!("node panicked: {}", panic_message(payload))),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn multiply_duration(current: Duration, multiplier: f64, maximum: Duration) -> Duration {
    let seconds = (current.as_secs_f64() * multiplier).min(maximum.as_secs_f64());
    if !seconds.is_finite() {
        maximum
    } else {
        Duration::from_secs_f64(seconds).min(maximum)
    }
}

enum CommitError {
    StateConflict(String),
    Routing(String),
}

fn apply_parallel_updates(
    state: &mut Value,
    invocations: &[NodeInvocation],
) -> std::result::Result<(), CommitError> {
    let updates = invocations
        .iter()
        .filter_map(|invocation| {
            let update = &invocation
                .output
                .as_ref()
                .expect("successful output")
                .update;
            (!matches!(update, StateUpdate::None)).then_some((invocation, update))
        })
        .collect::<Vec<_>>();

    if updates.len() > 1
        && updates
            .iter()
            .any(|(_, update)| matches!(update, StateUpdate::Replace(_)))
    {
        return Err(CommitError::StateConflict(
            "a full-state replacement cannot be combined with another update in the same fan-out"
                .to_owned(),
        ));
    }

    let mut writes: BTreeMap<String, (String, Value)> = BTreeMap::new();
    for (invocation, update) in &updates {
        if let StateUpdate::Merge(value) = update {
            let mut leaves = Vec::new();
            collect_write_paths(value, "", &mut leaves);
            for (path, value) in leaves {
                for (existing_path, (existing_node, existing_value)) in &writes {
                    if paths_overlap(existing_path, &path)
                        && !(existing_path == &path && existing_value == &value)
                    {
                        return Err(CommitError::StateConflict(format!(
                            "nodes '{}' and '{}' both update overlapping path '{}' / '{}'",
                            existing_node, invocation.node, existing_path, path
                        )));
                    }
                }
                writes.insert(path, (invocation.node.clone(), value));
            }
        }
    }

    for (_, update) in updates {
        match update {
            StateUpdate::None => {}
            StateUpdate::Merge(value) => merge_value(state, value.clone()),
            StateUpdate::Replace(value) => *state = value.clone(),
        }
    }
    Ok(())
}

fn collect_write_paths(value: &Value, prefix: &str, output: &mut Vec<(String, Value)>) {
    if let Value::Object(object) = value {
        if object.is_empty() {
            return;
        }
        for (key, value) in object {
            let escaped = key.replace('~', "~0").replace('/', "~1");
            let path = format!("{prefix}/{escaped}");
            collect_write_paths(value, &path, output);
        }
    } else {
        output.push((
            if prefix.is_empty() {
                "/".to_owned()
            } else {
                prefix.to_owned()
            },
            value.clone(),
        ));
    }
}

fn paths_overlap(left: &str, right: &str) -> bool {
    if left == "/" || right == "/" {
        return true;
    }
    left == right
        || right.starts_with(&format!("{left}/"))
        || left.starts_with(&format!("{right}/"))
}

fn merge_value(target: &mut Value, update: Value) {
    match (target, update) {
        (Value::Object(target), Value::Object(update)) => {
            for (key, value) in update {
                if let Some(current) = target.get_mut(&key) {
                    merge_value(current, value);
                } else {
                    target.insert(key, value);
                }
            }
        }
        (target, update) => *target = update,
    }
}

fn format_pending_joins(
    pending: &BTreeMap<String, BTreeSet<String>>,
    joins: &BTreeMap<String, BTreeSet<String>>,
) -> String {
    pending
        .iter()
        .map(|(target, arrived)| {
            let missing = joins
                .get(target)
                .map(|expected| expected.difference(arrived).cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            format!("{target} missing [{}]", missing.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

// Keep accidental debug formatting of closure-bearing definitions out of the
// public API while still allowing useful diagnostics in internal assertions.
impl std::fmt::Debug for Graph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Graph")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("entry", &self.entry)
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("max_steps", &self.max_steps)
            .finish_non_exhaustive()
    }
}

// The graph and both bundled stores must remain safe to share across workers.
#[allow(dead_code)]
fn assert_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<Graph>();
    check::<InMemoryGraphStore>();
    check::<FileGraphStore>();
}
