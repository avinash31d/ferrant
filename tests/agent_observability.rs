use async_trait::async_trait;
use liteagent::llm::{Model, ModelResponse, Usage};
use liteagent::memory::Storage;
use liteagent::observability::{
    InMemoryMetricsCollector, InMemoryUsageCollector, ModelPricing, PricingTable,
    PricingUsageRecorder, UsageRecord, UsageRecorder,
};
use liteagent::{Agent, FileStorage, Message, Result, ToolSpec};

struct AccountedModel;

#[async_trait]
impl Model for AccountedModel {
    fn id(&self) -> &str {
        "accounted"
    }
    fn provider(&self) -> &str {
        "test-provider"
    }
    async fn generate(&self, _: &[Message], _: &[ToolSpec]) -> Result<ModelResponse> {
        Ok(ModelResponse {
            content: Some("ok".into()),
            usage: Usage {
                input_tokens: 11,
                output_tokens: 3,
                total_tokens: 14,
                cached_input_tokens: 2,
                reasoning_tokens: 1,
            },
            ..Default::default()
        })
    }
}

#[tokio::test]
async fn agent_records_normalized_usage_and_operational_metrics() {
    let usage = InMemoryUsageCollector::default();
    let metrics = InMemoryMetricsCollector::default();
    let mut agent = Agent::builder(AccountedModel)
        .usage_recorder(usage.clone())
        .metrics_recorder(metrics.clone())
        .build();
    assert_eq!(agent.run("hello").await.unwrap(), "ok");
    let summary = usage.summary();
    assert_eq!(summary.totals.usage.total_tokens, 14);
    assert_eq!(summary.by_model["test-provider/accounted"].requests, 1);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.overall.count, 2); // model + complete agent run
    assert_eq!(snapshot.overall.errors, 0);
}

#[tokio::test]
async fn file_session_storage_survives_reopen_and_rejects_unsafe_ids() {
    let directory =
        std::env::temp_dir().join(format!("liteagent-sessions-{}", uuid::Uuid::new_v4()));
    let storage = FileStorage::new(&directory);
    storage
        .save("session_1", &[Message::user("durable")])
        .await
        .unwrap();
    drop(storage);
    let reopened = FileStorage::new(&directory);
    assert_eq!(
        reopened.load("session_1").await.unwrap()[0]
            .content
            .as_deref(),
        Some("durable")
    );
    assert!(reopened.save("../escape", &[]).await.is_err());
    reopened.clear("session_1").await.unwrap();
    assert!(reopened.load("session_1").await.unwrap().is_empty());
    let _ = tokio::fs::remove_dir_all(directory).await;
}

#[test]
fn configurable_pricing_accounts_for_cached_and_reasoning_tokens() {
    let collector = InMemoryUsageCollector::default();
    let pricing = PricingTable::new().with_model(
        "test-provider",
        "accounted",
        ModelPricing {
            input_microusd_per_million_tokens: 5_000_000,
            cached_input_microusd_per_million_tokens: Some(1_000_000),
            output_microusd_per_million_tokens: 10_000_000,
            reasoning_microusd_per_million_tokens: Some(20_000_000),
        },
    );
    let recorder = PricingUsageRecorder::new(collector.clone(), pricing);
    recorder.record_usage(UsageRecord::new(
        "priced",
        "test-provider",
        "accounted",
        Usage {
            input_tokens: 1_000_000,
            cached_input_tokens: 250_000,
            output_tokens: 500_000,
            reasoning_tokens: 100_000,
            total_tokens: 1_500_000,
        },
    ));
    // 750k*5 + 250k*1 + 400k*10 + 100k*20 = $10.00.
    assert_eq!(
        collector.records()[0].estimated_cost_microusd,
        Some(10_000_000)
    );
}
