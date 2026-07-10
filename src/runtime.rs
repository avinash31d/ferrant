use crate::llm::Usage;
use crate::message::{ContentPart, ToolCall};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Retry and timeout policy applied to model and tool calls.
#[derive(Debug, Clone)]
pub struct ExecutionPolicy {
    pub max_attempts: usize,
    pub initial_backoff: Duration,
    pub request_timeout: Duration,
    pub tool_timeout: Duration,
    pub parallel_tool_calls: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(200),
            request_timeout: Duration::from_secs(60),
            tool_timeout: Duration::from_secs(30),
            parallel_tool_calls: true,
        }
    }
}

/// Incremental events emitted by streaming model and agent runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    ContentDelta {
        delta: String,
    },
    ContentPart {
        part: ContentPart,
    },
    ToolCalls {
        calls: Vec<ToolCall>,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    Usage {
        usage: Usage,
    },
    Done,
}
