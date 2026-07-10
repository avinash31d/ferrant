use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("model provider error: {0}")]
    Provider(String),

    #[error("tool '{0}' not found")]
    ToolNotFound(String),

    #[error("tool '{name}' execution failed: {source}")]
    ToolExecution {
        name: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("exceeded max reasoning steps ({0})")]
    MaxStepsExceeded(usize),

    #[error("operation timed out: {0}")]
    Timeout(String),

    #[error("structured output failed schema validation: {0}")]
    SchemaValidation(String),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("other: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;
