//! Durable, resumable sequential workflows.

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub workflow_id: String,
    pub next_step: usize,
    pub data: Value,
    pub completed: bool,
}

#[async_trait]
pub trait WorkflowStore: Send + Sync {
    async fn load(&self, id: &str) -> anyhow::Result<Option<WorkflowState>>;
    async fn save(&self, state: &WorkflowState) -> anyhow::Result<()>;
    async fn clear(&self, id: &str) -> anyhow::Result<()>;
}

/// JSON-file checkpoint storage that survives process restarts.
pub struct FileWorkflowStore {
    directory: PathBuf,
}
impl FileWorkflowStore {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }
    fn path(&self, id: &str) -> anyhow::Result<PathBuf> {
        if id.is_empty()
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            anyhow::bail!("workflow id may contain only letters, numbers, '-' and '_'");
        }
        Ok(self.directory.join(format!("{id}.json")))
    }
}

#[async_trait]
impl WorkflowStore for FileWorkflowStore {
    async fn load(&self, id: &str) -> anyhow::Result<Option<WorkflowState>> {
        let path = self.path(id)?;
        match tokio::fs::read(path).await {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
    async fn save(&self, state: &WorkflowState) -> anyhow::Result<()> {
        tokio::fs::create_dir_all(&self.directory).await?;
        let path = self.path(&state.workflow_id)?;
        let temporary = path.with_extension("json.tmp");
        tokio::fs::write(&temporary, serde_json::to_vec_pretty(state)?).await?;
        tokio::fs::rename(temporary, path).await?;
        Ok(())
    }
    async fn clear(&self, id: &str) -> anyhow::Result<()> {
        let path = self.path(id)?;
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

pub type WorkflowFn = Arc<dyn Fn(Value) -> BoxFuture<'static, anyhow::Result<Value>> + Send + Sync>;

pub struct Workflow {
    steps: Vec<(String, WorkflowFn)>,
    store: Arc<dyn WorkflowStore>,
}

impl Workflow {
    pub fn new(store: impl WorkflowStore + 'static) -> Self {
        Self {
            steps: Vec::new(),
            store: Arc::new(store),
        }
    }
    pub fn step<F>(mut self, name: impl Into<String>, function: F) -> Self
    where
        F: Fn(Value) -> BoxFuture<'static, anyhow::Result<Value>> + Send + Sync + 'static,
    {
        self.steps.push((name.into(), Arc::new(function)));
        self
    }
    /// Start or resume a workflow. State is checkpointed after every step, so
    /// a failed process continues from the last successful boundary.
    pub async fn run(&self, id: &str, initial: Value) -> anyhow::Result<WorkflowState> {
        let mut state = self.store.load(id).await?.unwrap_or(WorkflowState {
            workflow_id: id.to_owned(),
            next_step: 0,
            data: initial,
            completed: false,
        });
        while state.next_step < self.steps.len() {
            let (_, function) = &self.steps[state.next_step];
            state.data = function(state.data.clone()).await?;
            state.next_step += 1;
            state.completed = state.next_step == self.steps.len();
            self.store.save(&state).await?;
        }
        Ok(state)
    }
}
