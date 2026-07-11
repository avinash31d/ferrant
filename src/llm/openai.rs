use super::{Model, ModelResponse, Usage};
use crate::error::{AgentError, Result};
use crate::message::{ContentPart, Message, Role, ToolCall};
use crate::runtime::StreamEvent;
use crate::tool::ToolSpec;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

/// Works with OpenAI's `/v1/chat/completions` API and any OpenAI-compatible
/// endpoint (Azure OpenAI, Groq, Ollama's OpenAI shim, OpenRouter, etc) —
/// just override `base_url`.
pub struct OpenAiModel {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    temperature: Option<f32>,
    modalities: Option<Vec<String>>,
    audio: Option<Value>,
}

impl OpenAiModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
            // Let each model use its provider-defined default unless explicitly
            // overridden. Some models (including GPT-5) reject custom values.
            temperature: None,
            modalities: None,
            audio: None,
        }
    }

    /// Request one or more output modalities, for example `["text", "audio"]`.
    pub fn with_modalities<I, S>(mut self, modalities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.modalities = Some(modalities.into_iter().map(Into::into).collect());
        self
    }

    /// Configure audio output for audio-capable Chat Completions models.
    pub fn with_audio_output(
        mut self,
        format: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        self.audio = Some(json!({"format": format.into(), "voice": voice.into()}));
        self
    }

    fn content_part(part: &ContentPart) -> Value {
        match part {
            ContentPart::Text { text } => json!({"type":"text", "text":text}),
            ContentPart::ImageUrl { url, detail } => json!({
                "type":"image_url", "image_url":{"url":url, "detail":detail.as_deref().unwrap_or("auto")}
            }),
            ContentPart::ImageData {
                data,
                media_type,
                detail,
            } => json!({
                "type":"image_url", "image_url":{
                    "url":format!("data:{media_type};base64,{data}"),
                    "detail":detail.as_deref().unwrap_or("auto")
                }
            }),
            ContentPart::Audio { data, format, .. } => json!({
                "type":"input_audio", "input_audio":{"data":data, "format":format}
            }),
            ContentPart::File {
                data,
                file_id,
                filename,
                ..
            } => json!({
                "type":"file", "file":{
                    "file_data":data, "file_id":file_id, "filename":filename
                }
            }),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let mut obj = json!({ "role": role });
                if !m.content_parts.is_empty() {
                    obj["content"] =
                        Value::Array(m.content_parts.iter().map(Self::content_part).collect());
                } else if let Some(content) = &m.content {
                    obj["content"] = json!(content);
                } else if m.role != Role::Assistant {
                    obj["content"] = json!("");
                }
                if !m.tool_calls.is_empty() {
                    obj["tool_calls"] = json!(m
                        .tool_calls
                        .iter()
                        .map(|tc| json!({
                            "id": tc.id,
                            "type": "function",
                            "function": { "name": tc.name, "arguments": tc.arguments.to_string() }
                        }))
                        .collect::<Vec<_>>());
                }
                if let Some(id) = &m.tool_call_id {
                    obj["tool_call_id"] = json!(id);
                }
                obj
            })
            .collect()
    }

    fn to_openai_tools(tools: &[ToolSpec]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect()
    }

    fn request_body(&self, messages: &[Message], tools: &[ToolSpec]) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": Self::to_openai_messages(messages),
        });
        if let Some(temperature) = self.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(modalities) = &self.modalities {
            body["modalities"] = json!(modalities);
        }
        if let Some(audio) = &self.audio {
            body["audio"] = audio.clone();
        }
        if !tools.is_empty() {
            body["tools"] = json!(Self::to_openai_tools(tools));
        }
        body
    }

    async fn send(&self, body: &Value) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Provider(format!(
                "OpenAI API error ({status}): {text}"
            )));
        }
        Ok(response.json().await?)
    }

    fn usage(data: &Value) -> Usage {
        Usage {
            input_tokens: data
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            output_tokens: data
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            total_tokens: data
                .pointer("/usage/total_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cached_input_tokens: data
                .pointer("/usage/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            reasoning_tokens: data
                .pointer("/usage/completion_tokens_details/reasoning_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        }
    }

    fn parse_response(&self, data: &Value) -> Result<ModelResponse> {
        let choice = data["choices"]
            .get(0)
            .ok_or_else(|| AgentError::Provider(format!("no choices returned: {data}")))?;
        let message = &choice["message"];
        if let Some(refusal) = message.get("refusal").and_then(Value::as_str) {
            return Err(AgentError::Provider(format!(
                "model refused structured response: {refusal}"
            )));
        }
        let mut content = message
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut content_parts = Vec::new();
        if let Some(parts) = message.get("content").and_then(Value::as_array) {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => content_parts.push(ContentPart::text(
                        part.get("text").and_then(Value::as_str).unwrap_or_default(),
                    )),
                    Some("image_url") => {
                        if let Some(url) = part.pointer("/image_url/url").and_then(Value::as_str) {
                            content_parts.push(ContentPart::image_url(url));
                        }
                    }
                    _ => {}
                }
            }
            let text = content_parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                content = Some(text);
            }
        }
        if let Some(audio) = message.get("audio") {
            content_parts.push(ContentPart::Audio {
                data: audio
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                format: self
                    .audio
                    .as_ref()
                    .and_then(|a| a.get("format"))
                    .and_then(Value::as_str)
                    .unwrap_or("wav")
                    .to_owned(),
                transcript: audio
                    .get("transcript")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
            if content.is_none() {
                content = audio
                    .get("transcript")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
        }
        let mut tool_calls = Vec::new();
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let arguments = serde_json::from_str(
                    call.pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}"),
                )
                .map_err(AgentError::Serde)?;
                tool_calls.push(ToolCall {
                    id: call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name: call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments,
                });
            }
        }
        Ok(ModelResponse {
            content,
            content_parts,
            tool_calls,
            usage: Self::usage(data),
        })
    }
}

#[async_trait]
impl Model for OpenAiModel {
    fn id(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "openai"
    }

    async fn generate(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<ModelResponse> {
        let data = self.send(&self.request_body(messages, tools)).await?;
        self.parse_response(&data)
    }

    async fn generate_structured(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        schema: &Value,
    ) -> Result<ModelResponse> {
        let mut body = self.request_body(messages, tools);
        body["response_format"] = json!({
            "type":"json_schema",
            "json_schema":{"name":"ferragent_response", "strict":true, "schema":schema}
        });
        let data = self.send(&body).await?;
        self.parse_response(&data)
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        events: mpsc::Sender<Result<StreamEvent>>,
    ) -> Result<ModelResponse> {
        let mut body = self.request_body(messages, tools);
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage":true});
        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Provider(format!(
                "OpenAI API error ({status}): {text}"
            )));
        }
        let mut bytes = response.bytes_stream();
        let mut pending = String::new();
        let mut content = String::new();
        let mut calls: Vec<(String, String, String)> = Vec::new();
        let mut usage = Usage::default();
        while let Some(chunk) = bytes.next().await {
            pending.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(newline) = pending.find('\n') {
                let line = pending[..newline].trim().to_owned();
                pending.drain(..=newline);
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let event: Value = serde_json::from_str(data)?;
                if event.get("usage").is_some_and(|value| !value.is_null()) {
                    usage = Self::usage(&event);
                    let _ = events
                        .send(Ok(StreamEvent::Usage {
                            usage: usage.clone(),
                        }))
                        .await;
                }
                if let Some(delta) = event
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                {
                    content.push_str(delta);
                    let _ = events
                        .send(Ok(StreamEvent::ContentDelta {
                            delta: delta.to_owned(),
                        }))
                        .await;
                }
                if let Some(tool_deltas) = event
                    .pointer("/choices/0/delta/tool_calls")
                    .and_then(Value::as_array)
                {
                    for tool_delta in tool_deltas {
                        let index =
                            tool_delta.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                        while calls.len() <= index {
                            calls.push((String::new(), String::new(), String::new()));
                        }
                        let id = tool_delta
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                        let name = tool_delta
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                        let arguments_delta = tool_delta
                            .pointer("/function/arguments")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if let Some(id) = &id {
                            calls[index].0.push_str(id);
                        }
                        if let Some(name) = &name {
                            calls[index].1.push_str(name);
                        }
                        calls[index].2.push_str(arguments_delta);
                        let _ = events
                            .send(Ok(StreamEvent::ToolCallDelta {
                                index,
                                id,
                                name,
                                arguments_delta: arguments_delta.to_owned(),
                            }))
                            .await;
                    }
                }
            }
        }
        let tool_calls = calls
            .into_iter()
            .map(|(id, name, arguments)| {
                let arguments = serde_json::from_str(if arguments.is_empty() {
                    "{}"
                } else {
                    &arguments
                })
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
