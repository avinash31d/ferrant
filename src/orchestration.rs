//! Multi-agent composition and coordinator-based orchestration.

use crate::{Agent, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Adapts a specialist [`Agent`] into a tool another agent can delegate to.
pub struct AgentTool {
    name: String,
    description: String,
    agent: Mutex<Agent>,
}

impl AgentTool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent: Mutex::new(agent),
        }
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> Value {
        json!({
            "type":"object",
            "properties":{"task":{"type":"string","description":"The task to delegate"}},
            "required":["task"]
        })
    }
    async fn execute(&self, args: Value) -> anyhow::Result<String> {
        let task = args
            .get("task")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("agent tool requires a string 'task'"))?;
        Ok(self.agent.lock().await.run(task).await?)
    }
}

/// A team in which a coordinator model decides which specialist agents to
/// call, can call several of them, and synthesizes their results.
pub struct AgentTeam {
    coordinator: Agent,
}

impl AgentTeam {
    pub fn new(coordinator_model: impl crate::llm::Model + 'static) -> Self {
        let coordinator = Agent::builder(coordinator_model)
            .instructions("You coordinate a team of specialist agents. Delegate tasks to the most relevant specialists, call multiple specialists when useful, and synthesize a clear final answer.")
            .build();
        Self { coordinator }
    }

    pub fn member(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Agent,
    ) -> Self {
        self.coordinator
            .add_shared_tool(Arc::new(AgentTool::new(name, description, agent)));
        self
    }

    pub async fn run(&mut self, task: impl Into<String>) -> crate::Result<String> {
        self.coordinator.run(task).await
    }
}
