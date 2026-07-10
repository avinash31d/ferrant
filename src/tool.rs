use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::Value;
use std::sync::Arc;

/// JSON-schema-described spec of a tool, as sent to model providers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Anything that can be called by an agent as a "tool" / "function".
///
/// Implement this directly for complex tools, or use [`FunctionTool`] to
/// wrap a plain async closure.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON schema (an object with "type": "object", "properties": {...}, "required": [...])
    fn parameters(&self) -> Value;

    async fn execute(&self, args: Value) -> anyhow::Result<String>;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}

pub type ToolFn = Arc<dyn Fn(Value) -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync>;

/// A [`Tool`] built from a plain closure, so you don't need to define a new
/// struct + impl for every simple function.
///
/// ```ignore
/// let tool = FunctionTool::new(
///     "get_weather",
///     "Get the current weather for a city",
///     serde_json::json!({
///         "type": "object",
///         "properties": { "city": { "type": "string" } },
///         "required": ["city"]
///     }),
///     |args| Box::pin(async move {
///         let city = args["city"].as_str().unwrap_or_default();
///         Ok(format!("It's sunny in {city}"))
///     }),
/// );
/// ```
pub struct FunctionTool {
    name: String,
    description: String,
    parameters: Value,
    func: ToolFn,
}

impl FunctionTool {
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        func: F,
    ) -> Self
    where
        F: Fn(Value) -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            func: Arc::new(func),
        }
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(&self, args: Value) -> anyhow::Result<String> {
        (self.func)(args).await
    }
}

/// Convenience macro to build a [`FunctionTool`] with less boilerplate.
///
/// ```ignore
/// let tool = function_tool!(
///     "add",
///     "Add two numbers",
///     json!({"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}),
///     |args| async move {
///         let a = args["a"].as_f64().unwrap_or(0.0);
///         let b = args["b"].as_f64().unwrap_or(0.0);
///         Ok((a + b).to_string())
///     }
/// );
/// ```
#[macro_export]
macro_rules! function_tool {
    ($name:expr, $desc:expr, $params:expr, $func:expr) => {
        $crate::tool::FunctionTool::new($name, $desc, $params, move |args| Box::pin($func(args)))
    };
}
