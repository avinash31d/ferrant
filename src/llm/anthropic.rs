use super::{Model, ModelResponse, Usage};
use crate::error::{AgentError, Result};
use crate::message::{ContentPart, Message, Role, ToolCall};
use crate::runtime::StreamEvent;
use crate::tool::ToolSpec;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

/// Works with Anthropic's `/v1/messages` API.
pub struct AnthropicModel {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    max_tokens: u32,
    system: Option<String>,
}

impl AnthropicModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            client: reqwest::Client::new(),
            max_tokens: 2048,
            system: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Separately-set system prompt overrides any `Role::System` messages in history
    /// (Anthropic takes `system` as a top-level field, not part of `messages`).
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    fn split(
        messages: &[Message],
        system_override: &Option<String>,
    ) -> (Option<String>, Vec<Value>) {
        let mut system = system_override.clone();
        let mut out = Vec::new();

        for m in messages {
            match m.role {
                Role::System => {
                    if system.is_none() {
                        system = m.content.clone();
                    }
                }
                Role::User => {
                    let content = if m.content_parts.is_empty() {
                        json!(m.content.clone().unwrap_or_default())
                    } else {
                        Value::Array(m.content_parts.iter().filter_map(|part| match part {
                            ContentPart::Text { text } => Some(json!({"type":"text", "text":text})),
                            ContentPart::ImageData { data, media_type, .. } => Some(json!({
                                "type":"image", "source":{"type":"base64", "media_type":media_type, "data":data}
                            })),
                            ContentPart::ImageUrl { url, .. } => Some(json!({
                                "type":"image", "source":{"type":"url", "url":url}
                            })),
                            ContentPart::File { data: Some(data), media_type: Some(media_type), .. } => Some(json!({
                                "type":"document", "source":{"type":"base64", "media_type":media_type, "data":data}
                            })),
                            _ => None,
                        }).collect())
                    };
                    out.push(json!({ "role": "user", "content": content }));
                }
                Role::Assistant => {
                    if !m.tool_calls.is_empty() {
                        let blocks: Vec<Value> = m
                            .tool_calls
                            .iter()
                            .map(|tc| json!({ "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments }))
                            .collect();
                        out.push(json!({ "role": "assistant", "content": blocks }));
                    } else {
                        out.push(json!({ "role": "assistant", "content": m.content.clone().unwrap_or_default() }));
                    }
                }
                Role::Tool => {
                    let block = json!([{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content.clone().unwrap_or_default(),
                    }]);
                    out.push(json!({ "role": "user", "content": block }));
                }
            }
        }
        (system, out)
    }

    fn to_anthropic_tools(tools: &[ToolSpec]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.parameters }))
            .collect()
    }

    fn usage(data: &Value) -> Usage {
        let input = data
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = data
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cached_input_tokens: data
                .pointer("/usage/cache_read_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            reasoning_tokens: 0,
        }
    }

    fn parse_response(data: &Value) -> ModelResponse {
        let blocks = data["content"].as_array().cloned().unwrap_or_default();
        let mut content = String::new();
        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        for block in &blocks {
            match block["type"].as_str() {
                Some("text") => {
                    let text = block["text"].as_str().unwrap_or_default();
                    content.push_str(text);
                    content_parts.push(ContentPart::text(text));
                }
                Some("tool_use") => tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or_default().to_owned(),
                    name: block["name"].as_str().unwrap_or_default().to_owned(),
                    arguments: block["input"].clone(),
                }),
                _ => {}
            }
        }
        ModelResponse {
            content: (!content.is_empty()).then_some(content),
            content_parts,
            tool_calls,
            usage: Self::usage(data),
        }
    }

    async fn send(&self, body: &Value) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Provider(format!(
                "Anthropic API error ({status}): {text}"
            )));
        }
        Ok(response.json().await?)
    }
}

#[async_trait]
impl Model for AnthropicModel {
    fn id(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "anthropic"
    }

    async fn generate(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<ModelResponse> {
        let (system, anthropic_messages) = Self::split(messages, &self.system);

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        });
        if let Some(s) = system {
            body["system"] = json!(s);
        }
        if !tools.is_empty() {
            body["tools"] = json!(Self::to_anthropic_tools(tools));
        }

        let data = self.send(&body).await?;
        Ok(Self::parse_response(&data))
    }

    async fn generate_structured(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        schema: &Value,
    ) -> Result<ModelResponse> {
        let (mut system, anthropic_messages) = Self::split(messages, &self.system);
        let mut provider_tools = Self::to_anthropic_tools(tools);
        provider_tools.push(json!({
            "name":"ferrant_structured_output",
            "description":"Return the final response in the required structure",
            "input_schema":schema
        }));
        let mut body = json!({
            "model":self.model, "max_tokens":self.max_tokens, "messages":anthropic_messages,
            "tools":provider_tools
        });
        if tools.is_empty() {
            body["tool_choice"] = json!({"type":"tool","name":"ferrant_structured_output"});
        } else {
            let instruction = "Use ordinary tools when needed. You must return the final answer by calling ferrant_structured_output.";
            system = Some(match system {
                Some(existing) => format!("{existing}\n\n{instruction}"),
                None => instruction.into(),
            });
        }
        if let Some(system) = system {
            body["system"] = json!(system);
        }
        let data = self.send(&body).await?;
        let mut response = Self::parse_response(&data);
        if let Some(call) = response
            .tool_calls
            .iter()
            .find(|call| call.name == "ferrant_structured_output")
        {
            response.content = Some(call.arguments.to_string());
            response.content_parts = vec![ContentPart::text(call.arguments.to_string())];
            response
                .tool_calls
                .retain(|call| call.name != "ferrant_structured_output");
        }
        Ok(response)
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        events: mpsc::Sender<Result<StreamEvent>>,
    ) -> Result<ModelResponse> {
        let (system, anthropic_messages) = Self::split(messages, &self.system);
        let mut body = json!({"model":self.model,"max_tokens":self.max_tokens,"messages":anthropic_messages,"stream":true});
        if let Some(system) = system {
            body["system"] = json!(system);
        }
        if !tools.is_empty() {
            body["tools"] = json!(Self::to_anthropic_tools(tools));
        }
        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Provider(format!(
                "Anthropic API error ({status}): {text}"
            )));
        }
        let mut stream = response.bytes_stream();
        let mut pending = String::new();
        let mut content = String::new();
        let mut calls: Vec<(String, String, String)> = Vec::new();
        let mut usage = Usage::default();
        while let Some(chunk) = stream.next().await {
            pending.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(newline) = pending.find('\n') {
                let line = pending[..newline].trim().to_owned();
                pending.drain(..=newline);
                let Some(raw) = line.strip_prefix("data: ") else {
                    continue;
                };
                let event: Value = serde_json::from_str(raw)?;
                match event.get("type").and_then(Value::as_str) {
                    Some("message_start") => {
                        usage = Self::usage(&event["message"]);
                    }
                    Some("content_block_start") => {
                        let index =
                            event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                        while calls.len() <= index {
                            calls.push((String::new(), String::new(), String::new()));
                        }
                        if event.pointer("/content_block/type").and_then(Value::as_str)
                            == Some("tool_use")
                        {
                            calls[index].0 = event
                                .pointer("/content_block/id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                            calls[index].1 = event
                                .pointer("/content_block/name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                        }
                    }
                    Some("content_block_delta") => {
                        let index =
                            event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                        if let Some(delta) = event.pointer("/delta/text").and_then(Value::as_str) {
                            content.push_str(delta);
                            let _ = events
                                .send(Ok(StreamEvent::ContentDelta {
                                    delta: delta.to_owned(),
                                }))
                                .await;
                        }
                        if let Some(delta) =
                            event.pointer("/delta/partial_json").and_then(Value::as_str)
                        {
                            while calls.len() <= index {
                                calls.push((String::new(), String::new(), String::new()));
                            }
                            calls[index].2.push_str(delta);
                            let _ = events
                                .send(Ok(StreamEvent::ToolCallDelta {
                                    index,
                                    id: None,
                                    name: None,
                                    arguments_delta: delta.to_owned(),
                                }))
                                .await;
                        }
                    }
                    Some("message_delta") => {
                        usage.output_tokens = event
                            .pointer("/usage/output_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(usage.output_tokens);
                        usage.total_tokens = usage.input_tokens + usage.output_tokens;
                    }
                    _ => {}
                }
            }
        }
        let tool_calls = calls
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, args)| {
                let arguments = serde_json::from_str(if args.is_empty() { "{}" } else { &args })
                    .map_err(AgentError::Serde)?;
                Ok(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !tool_calls.is_empty() {
            let _ = events
                .send(Ok(StreamEvent::ToolCalls {
                    calls: tool_calls.clone(),
                }))
                .await;
        }
        if usage.total_tokens > 0 {
            let _ = events
                .send(Ok(StreamEvent::Usage {
                    usage: usage.clone(),
                }))
                .await;
        }
        Ok(ModelResponse {
            content: (!content.is_empty()).then_some(content.clone()),
            content_parts: if content.is_empty() {
                vec![]
            } else {
                vec![ContentPart::text(content)]
            },
            tool_calls,
            usage,
        })
    }
}
