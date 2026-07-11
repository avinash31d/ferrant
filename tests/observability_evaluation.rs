use async_trait::async_trait;
use ferragent::evaluation::{EvaluationOutput, EvaluationTarget};
use ferragent::integrations::{ConfigurationValueKind, IntegrationComponent};
use ferragent::llm::{Model, ModelResponse};
use ferragent::observability::{
    ModelPricing, OperationKind, OperationOutcome, PricingTable, PricingUsageRecorder, SpanStatus,
    TraceMetricsAdapter,
};
use ferragent::persistence::{AppendOutcome, AtomicJsonFile, DurableJsonlStore};
use ferragent::*;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

#[test]
fn aggregates_usage_latency_errors_and_open_telemetry_spans() {
    let usage = InMemoryUsageCollector::default();
    let mut first = UsageRecord::new(
        "run-1",
        "openai",
        "test-model",
        Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            cached_input_tokens: 2,
            reasoning_tokens: 1,
        },
    );
    first.latency_ms = 20;
    first.estimated_cost_microusd = Some(7);
    usage.record_usage(first);
    let mut second = UsageRecord::new("run-2", "openai", "test-model", Usage::default());
    second.success = false;
    second.error_kind = Some("rate_limit".into());
    second.latency_ms = 10;
    usage.record_usage(second);
    let summary = usage.summary();
    assert_eq!(summary.totals.requests, 2);
    assert_eq!(summary.totals.failed_requests, 1);
    assert_eq!(summary.totals.usage.total_tokens, 15);
    assert_eq!(summary.totals.estimated_cost_microusd, 7);
    assert_eq!(summary.totals.average_latency_ms(), 15.0);

    let priced_sink = InMemoryUsageCollector::default();
    let priced = PricingUsageRecorder::new(
        priced_sink.clone(),
        PricingTable::new().with_model(
            "provider",
            "model",
            ModelPricing {
                input_microusd_per_million_tokens: 2_000_000,
                cached_input_microusd_per_million_tokens: Some(500_000),
                output_microusd_per_million_tokens: 6_000_000,
                reasoning_microusd_per_million_tokens: Some(10_000_000),
            },
        ),
    );
    priced.record_usage(UsageRecord::new(
        "priced",
        "provider",
        "model",
        Usage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            total_tokens: 1_500_000,
            cached_input_tokens: 200_000,
            reasoning_tokens: 100_000,
        },
    ));
    assert_eq!(
        priced_sink.records()[0].estimated_cost_microusd,
        Some(5_100_000)
    );

    let metrics = TraceMetricsAdapter::default();
    let collector = metrics.collector();
    let exporter = InMemorySpanExporter::default();
    let telemetry = OpenTelemetryAdapter::new(exporter.clone());
    let events = [
        event(100, "run_started", json!({"model":"test-model"})),
        event(102, "tool_started", json!({"id":"call-1","name":"search"})),
        event(
            109,
            "tool_completed",
            json!({"id":"call-1","name":"search","ok":false,"error":"offline"}),
        ),
        event(115, "run_completed", json!({})),
    ];
    for trace in events {
        metrics.record(trace.clone());
        telemetry.record(trace);
    }

    let snapshot = collector.snapshot();
    assert_eq!(snapshot.overall.count, 2);
    assert_eq!(snapshot.overall.errors, 1);
    assert_eq!(snapshot.by_kind[&OperationKind::Tool].max_ms, Some(7));
    assert_eq!(collector.records()[0].outcome, OperationOutcome::Error);

    let spans = exporter.spans();
    assert_eq!(spans.len(), 2);
    let tool = spans.iter().find(|span| span.name == "search").unwrap();
    let run = spans.iter().find(|span| span.name == "run").unwrap();
    assert_eq!(tool.status, SpanStatus::Error);
    assert_eq!(tool.parent_span_id.as_deref(), Some(run.span_id.as_str()));
    assert_eq!(tool.trace_id, run.trace_id);
    assert!(telemetry.export_errors().is_empty());
}

fn event(timestamp_ms: u128, kind: &str, fields: Value) -> TraceEvent {
    TraceEvent {
        timestamp_ms,
        run_id: "run-1".into(),
        kind: kind.into(),
        fields,
    }
}

#[tokio::test]
async fn evaluation_runner_reports_failures_and_enforces_regression_gates() {
    let active = Arc::new(AtomicUsize::new(0));
    let maximum_active = Arc::new(AtomicUsize::new(0));
    let target = FunctionEvaluationTarget::new({
        let active = active.clone();
        let maximum_active = maximum_active.clone();
        move |case| {
            let active = active.clone();
            let maximum_active = maximum_active.clone();
            Box::pin(async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum_active.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(15)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                if case.id == "provider-error" {
                    anyhow::bail!("simulated provider failure");
                }
                Ok(EvaluationOutput {
                    value: case.input,
                    latency_ms: 15,
                    usage: Usage {
                        input_tokens: 2,
                        output_tokens: 1,
                        total_tokens: 3,
                        ..Usage::default()
                    },
                    ..EvaluationOutput::default()
                })
            })
        }
    });
    let dataset = EvaluationDataset::new(
        "regression-suite",
        "1",
        vec![
            EvaluationCase::new("one", json!("one")).expected(json!("one")),
            EvaluationCase::new("two", json!("two")).expected(json!("two")),
            EvaluationCase::new("provider-error", json!("x")).expected(json!("x")),
        ],
    );
    let report = EvaluationRunner::new(target)
        .scorer(ExactMatchScorer)
        .config(EvaluationConfig {
            concurrency: 3,
            target_timeout: Duration::from_secs(1),
            scorer_timeout: Duration::from_secs(1),
        })
        .run(&dataset)
        .await
        .unwrap();

    assert_eq!(report.total_cases, 3);
    assert_eq!(report.passed_cases, 2);
    assert_eq!(report.target_errors, 1);
    assert_eq!(report.usage.total_tokens, 6);
    assert!(maximum_active.load(Ordering::SeqCst) > 1);
    assert_eq!(report.cases[2].case_id, "provider-error");

    let mut candidate = report.clone();
    candidate.mean_score = 0.5;
    candidate.pass_rate = 0.4;
    candidate.mean_latency_ms = report.mean_latency_ms * 2.0;
    let check = candidate.check_regression(
        Some(&report),
        &RegressionThresholds {
            maximum_score_drop: Some(0.1),
            maximum_pass_rate_drop: Some(0.1),
            maximum_latency_increase_ratio: Some(0.25),
            require_same_dataset: true,
            ..RegressionThresholds::default()
        },
    );
    assert!(!check.passed);
    assert_eq!(check.violations.len(), 3);
    assert!(check.enforce().is_err());
}

struct FixedModel;

#[async_trait]
impl Model for FixedModel {
    fn id(&self) -> &str {
        "fixed"
    }

    async fn generate(&self, _: &[Message], _: &[ToolSpec]) -> Result<ModelResponse> {
        Ok(ModelResponse {
            content: Some("{\"answer\":42}".into()),
            usage: Usage {
                input_tokens: 4,
                output_tokens: 3,
                total_tokens: 7,
                ..Usage::default()
            },
            ..ModelResponse::default()
        })
    }
}

#[tokio::test]
async fn agent_evaluation_target_preserves_provider_usage() {
    let target =
        AgentEvaluationTarget::new(Agent::builder(FixedModel).build()).parse_json_output(true);
    let output = target
        .evaluate(&EvaluationCase::new("agent", json!("question")))
        .await
        .unwrap();
    assert_eq!(output.value, json!({"answer":42}));
    assert_eq!(output.usage.total_tokens, 7);
}

#[derive(Debug, PartialEq)]
struct ExampleIntegration {
    dimensions: usize,
}

#[test]
fn curated_registry_discovers_capabilities_and_constructs_typed_components() {
    let registry = IntegrationRegistry::curated();
    assert!(registry
        .by_capability(IntegrationCapability::StreamingToolCalls)
        .iter()
        .any(|descriptor| descriptor.id == "model.openai"));
    assert!(!registry.is_available("embedder.hash"));

    registry
        .attach_factory(
            "embedder.hash",
            FunctionIntegrationFactory::new(|config: &Value| {
                let dimensions = config
                    .get("dimensions")
                    .and_then(Value::as_u64)
                    .unwrap_or(128) as usize;
                Ok(IntegrationComponent::new(ExampleIntegration { dimensions }))
            }),
        )
        .unwrap();
    let instance = registry
        .create("embedder.hash", &json!({"dimensions":256}))
        .unwrap();
    assert_eq!(
        *instance.component.downcast::<ExampleIntegration>().unwrap(),
        ExampleIntegration { dimensions: 256 }
    );
    assert!(registry
        .create("embedder.hash", &json!({"dimensions":"wrong"}))
        .is_err());

    let descriptor = registry.descriptor("embedder.hash").unwrap();
    assert_eq!(
        descriptor.configuration[0].value_kind,
        ConfigurationValueKind::Integer
    );
}

#[tokio::test]
async fn durable_stores_recover_torn_tail_reject_middle_corruption_and_dedupe() {
    let directory = std::env::temp_dir().join(format!(
        "ferragent-observability-test-{}",
        uuid::Uuid::new_v4()
    ));
    let snapshot = AtomicJsonFile::<Value>::new(directory.join("snapshot.json"));
    snapshot.save(&json!({"generation":1})).await.unwrap();
    snapshot.save(&json!({"generation":2})).await.unwrap();
    assert_eq!(
        snapshot.load().await.unwrap(),
        Some(json!({"generation":2}))
    );

    let path = directory.join("records.jsonl");
    let store = DurableJsonlStore::<String>::new(&path);
    assert_eq!(
        store
            .append_idempotent("record-1", "one".into())
            .await
            .unwrap(),
        AppendOutcome::Appended { sequence: 1 }
    );
    assert_eq!(
        store
            .append_idempotent("record-1", "ignored retry".into())
            .await
            .unwrap(),
        AppendOutcome::AlreadyExists { sequence: 1 }
    );
    store
        .append_idempotent("record-2", "two".into())
        .await
        .unwrap();

    let mut raw = tokio::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .await
        .unwrap();
    raw.write_all(b"{\"version\":1,\"sequence\":3")
        .await
        .unwrap();
    raw.flush().await.unwrap();
    drop(raw);
    assert_eq!(store.payloads().await.unwrap(), vec!["one", "two"]);
    assert!(tokio::fs::read(&path).await.unwrap().ends_with(b"\n"));
    assert_eq!(
        store.append("three".into()).await.unwrap(),
        AppendOutcome::Appended { sequence: 3 }
    );

    let corrupt_path = directory.join("corrupt.jsonl");
    let corrupt = DurableJsonlStore::<String>::new(&corrupt_path);
    corrupt.append("alpha".into()).await.unwrap();
    corrupt.append("beta".into()).await.unwrap();
    let mut bytes = tokio::fs::read(&corrupt_path).await.unwrap();
    let offset = bytes
        .windows(b"alpha".len())
        .position(|window| window == b"alpha")
        .unwrap();
    bytes[offset] = b'A';
    tokio::fs::write(&corrupt_path, bytes).await.unwrap();
    let error = corrupt.records().await.unwrap_err().to_string();
    assert!(error.contains("corrupt JSONL record"));

    tokio::fs::remove_dir_all(directory).await.unwrap();
}
