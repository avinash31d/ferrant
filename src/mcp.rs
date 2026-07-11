//! Model Context Protocol (MCP) client support.
//!
//! This module connects to a local MCP server over the standard stdio
//! transport, discovers its tools, and exposes them as regular [`Tool`]s.

use crate::tool::Tool;
use anyhow::{anyhow, Context};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const PROTOCOL_VERSION: &str = "2025-06-18";

struct Connection {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Connection {
    async fn send(&mut self, message: &Value) -> anyhow::Result<()> {
        let mut encoded = serde_json::to_vec(message)?;
        encoded.push(b'\n');
        self.stdin.write_all(&encoded).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}))
            .await?;

        loop {
            let mut line = String::new();
            if self.stdout.read_line(&mut line).await? == 0 {
                return Err(anyhow!(
                    "MCP server closed stdout while waiting for {method}"
                ));
            }
            let message: Value = serde_json::from_str(&line)
                .with_context(|| format!("invalid JSON from MCP server: {}", line.trim()))?;
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                // Notifications and unrelated responses may be interleaved.
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(anyhow!("MCP {method} failed: {error}"));
            }
            return message
                .get("result")
                .cloned()
                .ok_or_else(|| anyhow!("MCP {method} response did not contain a result"));
        }
    }
}

/// A connected stdio MCP server. Clone this value to retain the same session.
#[derive(Clone)]
pub struct McpClient {
    connection: Arc<Mutex<Connection>>,
}

impl McpClient {
    /// Spawn and initialize an MCP server process.
    pub async fn connect<I, S>(
        program: impl AsRef<std::ffi::OsStr>,
        args: I,
    ) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .context("failed to start MCP server")?;
        let stdin = child
            .stdin
            .take()
            .context("MCP server stdin was unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("MCP server stdout was unavailable")?;
        let client = Self {
            connection: Arc::new(Mutex::new(Connection {
                _child: child,
                stdin,
                stdout: BufReader::new(stdout),
                next_id: 1,
            })),
        };

        let mut connection = client.connection.lock().await;
        connection
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name":"ferragent", "version":env!("CARGO_PKG_VERSION")}
                }),
            )
            .await?;
        connection
            .send(&json!({
                "jsonrpc":"2.0", "method":"notifications/initialized"
            }))
            .await?;
        drop(connection);
        Ok(client)
    }

    /// Discover all server tools, following MCP cursor pagination.
    pub async fn tools(&self) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |c| json!({"cursor":c}));
            let result = self
                .connection
                .lock()
                .await
                .request("tools/list", params)
                .await?;
            for value in result
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("MCP tools/list result did not contain a tools array"))?
            {
                tools.push(Arc::new(McpTool {
                    client: self.clone(),
                    name: value
                        .get("name")
                        .and_then(Value::as_str)
                        .context("MCP tool has no name")?
                        .to_owned(),
                    description: value
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("MCP server tool")
                        .to_owned(),
                    parameters: value
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type":"object"})),
                }));
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> anyhow::Result<String> {
        let result = self
            .connection
            .lock()
            .await
            .request(
                "tools/call",
                json!({
                    "name": name, "arguments": arguments
                }),
            )
            .await?;
        if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(anyhow!(
                "MCP tool {name} returned an error: {}",
                render_content(&result)
            ));
        }
        Ok(render_content(&result))
    }
}

fn render_content(result: &Value) -> String {
    if let Some(structured) = result.get("structuredContent") {
        return structured.to_string();
    }
    result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .or_else(|| Some(block.to_string()))
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| result.to_string())
}

struct McpTool {
    client: McpClient,
    name: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl Tool for McpTool {
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
        self.client.call_tool(&self.name, args).await
    }
}
