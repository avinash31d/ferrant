use ferragent::llm::anthropic::AnthropicModel;
use ferragent::llm::openai::OpenAiModel;
use ferragent::llm::Model;
use ferragent::{Message, StreamEvent, ToolSpec};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn mock_server(
    response_body: String,
    content_type: &'static str,
) -> (String, tokio::sync::oneshot::Receiver<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        let header_end;
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            request.extend_from_slice(&buffer[..read]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                header_end = position + 4;
                break;
            }
        }
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|value| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);
        while request.len() < header_end + content_length {
            let read = socket.read(&mut buffer).await.unwrap();
            request.extend_from_slice(&buffer[..read]);
        }
        let body: Value =
            serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap();
        let _ = tx.send(body);
        let headers = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response_body.len());
        socket.write_all(headers.as_bytes()).await.unwrap();
        socket.write_all(response_body.as_bytes()).await.unwrap();
    });
    (format!("http://{address}"), rx)
}

#[tokio::test]
async fn openai_stream_assembles_tool_arguments_and_usage() {
    let sse = [
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\"a\":"}}]}}]}"#,
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
        r#"data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}"#,
        "data: [DONE]",
    ].join("\n\n");
    let (base_url, _) = mock_server(sse, "text/event-stream").await;
    let model = OpenAiModel::new("test", "key").with_base_url(base_url);
    let tools = vec![ToolSpec {
        name: "add".into(),
        description: "add".into(),
        parameters: json!({"type":"object"}),
    }];
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let response = model
        .generate_stream(&[Message::user("add")], &tools, tx)
        .await
        .unwrap();
    assert_eq!(response.tool_calls[0].arguments, json!({"a":1}));
    assert_eq!(response.usage.total_tokens, 10);
    let mut deltas = 0;
    while let Some(event) = rx.recv().await {
        if matches!(event.unwrap(), StreamEvent::ToolCallDelta { .. }) {
            deltas += 1;
        }
    }
    assert_eq!(deltas, 2);
}

#[tokio::test]
async fn openai_structured_output_uses_native_json_schema() {
    let response = json!({
        "choices":[{"message":{"role":"assistant","content":"{\"answer\":42}"}}],
        "usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}
    })
    .to_string();
    let (base_url, request) = mock_server(response, "application/json").await;
    let model = OpenAiModel::new("test", "key").with_base_url(base_url);
    let schema = json!({"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false});
    let result = model
        .generate_structured(&[Message::user("answer")], &[], &schema)
        .await
        .unwrap();
    assert_eq!(result.content.as_deref(), Some("{\"answer\":42}"));
    let body = request.await.unwrap();
    assert_eq!(
        body.pointer("/response_format/type"),
        Some(&json!("json_schema"))
    );
    assert_eq!(
        body.pointer("/response_format/json_schema/strict"),
        Some(&json!(true))
    );
    assert_eq!(
        body.pointer("/response_format/json_schema/schema"),
        Some(&schema)
    );
}

#[test]
fn full_json_schema_validation_catches_advanced_constraints() {
    let schema = json!({
        "type":"object",
        "properties":{"code":{"type":"string","pattern":"^[A-Z]{3}$"}},
        "required":["code"], "additionalProperties":false
    });
    assert!(ferragent::validate_json(&schema, &json!({"code":"ABC"})).is_ok());
    assert!(ferragent::validate_json(&schema, &json!({"code":"bad","extra":1})).is_err());
}

#[tokio::test]
async fn anthropic_stream_assembles_partial_json_tool_call() {
    let sse = [
        concat!("event: message_start\n", r#"data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}"#),
        concat!("event: content_block_start\n", r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}"#),
        concat!("event: content_block_delta\n", r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#),
        concat!("event: content_block_delta\n", r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}"#),
        concat!("event: message_delta\n", r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}"#),
        concat!("event: message_stop\n", r#"data: {"type":"message_stop"}"#),
    ].join("\n\n");
    let (base_url, _) = mock_server(sse, "text/event-stream").await;
    let model = AnthropicModel::new("test", "key").with_base_url(base_url);
    let tools = vec![ToolSpec {
        name: "lookup".into(),
        description: "lookup".into(),
        parameters: json!({"type":"object"}),
    }];
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let response = model
        .generate_stream(&[Message::user("find")], &tools, tx)
        .await
        .unwrap();
    assert_eq!(response.tool_calls[0].arguments, json!({"q":"rust"}));
    assert_eq!(response.usage.input_tokens, 5);
    assert_eq!(response.usage.output_tokens, 4);
    let mut deltas = 0;
    while let Some(event) = rx.recv().await {
        if matches!(event.unwrap(), StreamEvent::ToolCallDelta { .. }) {
            deltas += 1;
        }
    }
    assert_eq!(deltas, 2);
}

#[tokio::test]
async fn anthropic_structured_output_uses_forced_schema_tool() {
    let response = json!({
        "content":[{"type":"tool_use","id":"tool_1","name":"ferragent_structured_output","input":{"answer":42}}],
        "usage":{"input_tokens":5,"output_tokens":4}
    }).to_string();
    let (base_url, request) = mock_server(response, "application/json").await;
    let model = AnthropicModel::new("test", "key").with_base_url(base_url);
    let schema =
        json!({"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]});
    let output = model
        .generate_structured(&[Message::user("answer")], &[], &schema)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(output.content.as_deref().unwrap()).unwrap(),
        json!({"answer":42})
    );
    let body = request.await.unwrap();
    assert_eq!(
        body.pointer("/tool_choice/name"),
        Some(&json!("ferragent_structured_output"))
    );
    assert_eq!(body.pointer("/tools/0/input_schema"), Some(&schema));
}
