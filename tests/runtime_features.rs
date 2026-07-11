use async_trait::async_trait;
use ferrant::llm::{Model, ModelResponse};
use ferrant::message::ToolCall;
use ferrant::rag::Chunker;
use ferrant::*;
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct FixedModel {
    response: ModelResponse,
}
#[async_trait]
impl Model for FixedModel {
    fn id(&self) -> &str {
        "fixed"
    }
    async fn generate(&self, _: &[Message], _: &[ToolSpec]) -> Result<ModelResponse> {
        Ok(self.response.clone())
    }
}

struct RetryModel {
    calls: Arc<AtomicUsize>,
}

struct NativeStructuredModel {
    used: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl Model for NativeStructuredModel {
    fn id(&self) -> &str {
        "native-structured"
    }
    async fn generate(&self, _: &[Message], _: &[ToolSpec]) -> Result<ModelResponse> {
        Ok(ModelResponse {
            content: Some("not json".into()),
            ..Default::default()
        })
    }
    async fn generate_structured(
        &self,
        _: &[Message],
        _: &[ToolSpec],
        schema: &serde_json::Value,
    ) -> Result<ModelResponse> {
        assert_eq!(schema["required"], json!(["name", "age"]));
        self.used.store(true, Ordering::SeqCst);
        Ok(ModelResponse {
            content: Some(r#"{"name":"Ada","age":36}"#.into()),
            ..Default::default()
        })
    }
}
#[async_trait]
impl Model for RetryModel {
    fn id(&self) -> &str {
        "retry"
    }
    async fn generate(&self, _: &[Message], _: &[ToolSpec]) -> Result<ModelResponse> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(AgentError::Provider("temporary".into()))
        } else {
            Ok(ModelResponse {
                content: Some("ok".into()),
                ..Default::default()
            })
        }
    }
}

#[tokio::test]
async fn retries_and_traces_model_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let tracer = InMemoryTracer::default();
    let mut agent = Agent::builder(RetryModel {
        calls: calls.clone(),
    })
    .execution_policy(ExecutionPolicy {
        initial_backoff: Duration::ZERO,
        ..Default::default()
    })
    .tracer(tracer.clone())
    .build();
    assert_eq!(agent.run("hello").await.unwrap(), "ok");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(tracer.events().iter().any(|e| e.kind == "run_completed"));
}

#[derive(Debug, Deserialize, PartialEq)]
struct Person {
    name: String,
    age: u64,
}

#[tokio::test]
async fn validates_and_deserializes_structured_output() {
    let response = ModelResponse {
        content: Some(r#"{"name":"Ada","age":36}"#.into()),
        ..Default::default()
    };
    let mut agent = Agent::builder(FixedModel { response }).build();
    let schema = json!({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]});
    assert_eq!(
        agent
            .run_structured::<Person>("person", &schema)
            .await
            .unwrap(),
        Person {
            name: "Ada".into(),
            age: 36
        }
    );
}

#[tokio::test]
async fn agent_requests_provider_native_structured_output() {
    let used = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut agent = Agent::builder(NativeStructuredModel { used: used.clone() }).build();
    let schema = json!({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]});
    let value = agent
        .run_structured::<Person>("person", &schema)
        .await
        .unwrap();
    assert_eq!(value.name, "Ada");
    assert!(used.load(Ordering::SeqCst));
}

struct ToolModel {
    calls: AtomicUsize,
}
#[async_trait]
impl Model for ToolModel {
    fn id(&self) -> &str {
        "tools"
    }
    async fn generate(&self, _: &[Message], _: &[ToolSpec]) -> Result<ModelResponse> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(ModelResponse {
                tool_calls: vec![
                    ToolCall {
                        id: "1".into(),
                        name: "slow_a".into(),
                        arguments: json!({}),
                    },
                    ToolCall {
                        id: "2".into(),
                        name: "slow_b".into(),
                        arguments: json!({}),
                    },
                ],
                ..Default::default()
            })
        } else {
            Ok(ModelResponse {
                content: Some("done".into()),
                ..Default::default()
            })
        }
    }
}

#[tokio::test]
async fn executes_independent_tool_calls_in_parallel() {
    let slow = |name| {
        FunctionTool::new(name, "slow", json!({"type":"object"}), |_| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok("ok".into())
            })
        })
    };
    let mut agent = Agent::builder(ToolModel {
        calls: AtomicUsize::new(0),
    })
    .tool(slow("slow_a"))
    .tool(slow("slow_b"))
    .build();
    let started = Instant::now();
    assert_eq!(agent.run("go").await.unwrap(), "done");
    assert!(started.elapsed() < Duration::from_millis(180));
}

#[tokio::test]
async fn streams_default_model_events() {
    let mut agent = Agent::builder(FixedModel {
        response: ModelResponse {
            content: Some("hello".into()),
            ..Default::default()
        },
    })
    .build();
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let response = agent.run_stream("go", tx).await.unwrap();
    assert_eq!(response.content.as_deref(), Some("hello"));
    assert!(matches!(
        rx.recv().await.unwrap().unwrap(),
        StreamEvent::ContentDelta { .. }
    ));
}

#[tokio::test]
async fn workflow_resumes_and_rag_retrieves() {
    let directory = std::env::temp_dir().join(format!("ferrant-test-{}", uuid::Uuid::new_v4()));
    let workflow = Workflow::new(FileWorkflowStore::new(&directory))
        .step("increment", |value| {
            Box::pin(async move { Ok(json!(value.as_u64().unwrap() + 1)) })
        })
        .step("double", |value| {
            Box::pin(async move { Ok(json!(value.as_u64().unwrap() * 2)) })
        });
    let state = workflow.run("run_1", json!(2)).await.unwrap();
    assert_eq!(state.data, json!(6));
    assert_eq!(
        workflow.run("run_1", json!(999)).await.unwrap().data,
        json!(6)
    );

    let chunker = TextChunker::new(20, 5);
    assert!(
        chunker
            .chunk(&Document {
                id: "x".into(),
                text: "a long document that needs chunks".into(),
                metadata: json!({})
            })
            .len()
            > 1
    );
    let retriever = Retriever::new(HashEmbedder::new(128), InMemoryVectorStore::default());
    retriever
        .index(vec![
            Document {
                id: "rust".into(),
                text: "Rust provides memory safety and ownership".into(),
                metadata: json!({}),
            },
            Document {
                id: "cooking".into(),
                text: "Pasta is cooked in salted boiling water".into(),
                metadata: json!({}),
            },
        ])
        .await
        .unwrap();
    assert_eq!(
        retriever.retrieve("Rust ownership", 1).await.unwrap()[0]
            .document
            .id,
        "rust"
    );
    let _ = tokio::fs::remove_dir_all(directory).await;
}
