pub mod anthropic;
pub mod openai;

use crate::error::Result;
use crate::message::{ContentPart, Message, ToolCall};
use crate::runtime::StreamEvent;
use crate::tool::ToolSpec;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

/// Provider-normalized token accounting for a single model request.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Usage {
    pub fn add_assign(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

/// The result of one call to a model: either free-text content, one or more
/// tool calls the agent should execute, or both.
#[derive(Debug, Clone, Default)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub content_parts: Vec<ContentPart>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
}

/// Abstraction over any LLM provider (OpenAI, Anthropic, local models, etc).
/// Implement this trait to plug in a new provider.
#[async_trait]
pub trait Model: Send + Sync {
    /// Human readable identifier, e.g. "gpt-4o" or "claude-sonnet-4-6".
    fn id(&self) -> &str;

    fn provider(&self) -> &str {
        "custom"
    }

    /// Run one generation step given the full message history and the set of
    /// tools currently available to the agent.
    async fn generate(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<ModelResponse>;

    /// Ask the provider to enforce a JSON schema natively when supported.
    /// Custom providers retain compatibility and use post-generation
    /// validation by default.
    async fn generate_structured(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        _schema: &Value,
    ) -> Result<ModelResponse> {
        self.generate(messages, tools).await
    }

    /// Stream generation events. Providers may override this for native
    /// token streaming; the default preserves compatibility by emitting the
    /// completed response as one delta.
    async fn generate_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        events: mpsc::Sender<Result<StreamEvent>>,
    ) -> Result<ModelResponse> {
        let response = self.generate(messages, tools).await?;
        if let Some(content) = &response.content {
            let _ = events
                .send(Ok(StreamEvent::ContentDelta {
                    delta: content.clone(),
                }))
                .await;
        }
        for part in &response.content_parts {
            let _ = events
                .send(Ok(StreamEvent::ContentPart { part: part.clone() }))
                .await;
        }
        if !response.tool_calls.is_empty() {
            let _ = events
                .send(Ok(StreamEvent::ToolCalls {
                    calls: response.tool_calls.clone(),
                }))
                .await;
        }
        if response.usage.total_tokens > 0 {
            let _ = events
                .send(Ok(StreamEvent::Usage {
                    usage: response.usage.clone(),
                }))
                .await;
        }
        Ok(response)
    }
}
