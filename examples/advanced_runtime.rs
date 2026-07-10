//! End-to-end production-runtime example.
//!
//! This example combines persistent RAG, streaming tool calls, provider-native
//! structured output, durable graph orchestration, usage/metrics/traces,
//! evaluation gates, and integration discovery. It calls OpenAI, so set
//! `OPENAI_API_KEY` before running it.

use liteagent::evaluation::{
    ContainsScorer, EvaluationCase, EvaluationDataset, EvaluationOutput, EvaluationRunner,
    FunctionEvaluationTarget, RegressionThresholds,
};
use liteagent::graph::{FileGraphStore, Graph, GraphStatus, NodeOutput, NodeRetryPolicy};
use liteagent::integrations::{IntegrationCapability, IntegrationRegistry};
use liteagent::llm::openai::OpenAiModel;
use liteagent::observability::{
    CompositeTracer, InMemoryMetricsCollector, InMemorySpanExporter, InMemoryUsageCollector,
    OpenTelemetryAdapter,
};
use liteagent::rag::{
    ContextFormatter, Document, Embedder, FileVectorStore, HashEmbedder, IngestionPipeline,
    LexicalReranker, MetadataFilter, RetrievalOptions, RetrievalStrategy, Retriever, RetrieverTool,
    TextChunker, VectorStore,
};
use liteagent::{
    Agent, EvaluationBaselineStore, EvaluationReportLogStore, ExecutionPolicy, FileStorage,
    StreamEvent, UsageLogStore,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
struct GroundedAnswer {
    answer: String,
    citation_ids: Vec<String>,
}

fn answer_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" },
            "citation_ids": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["answer", "citation_ids"],
        "additionalProperties": false
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let data_dir = std::env::temp_dir().join("liteagent-advanced-runtime");

    // The vector snapshot is atomically replaced and recovered from a backup
    // after an interrupted write. Arc trait objects let ingestion, retrieval,
    // graph nodes, and the agent's RetrieverTool share one index.
    let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(256));
    let vector_store: Arc<dyn VectorStore> =
        Arc::new(FileVectorStore::open(data_dir.join("knowledge.json"))?);
    let ingestion = IngestionPipeline::from_shared(
        embedder.clone(),
        vector_store.clone(),
        Arc::new(TextChunker::new(220, 30)),
    );
    let report = ingestion
        .ingest(vec![
            Document {
                id: "graph-guide".into(),
                text: "Liteagent checkpoints graph supersteps. Failed nodes resume with stable \
                       idempotency keys, while joins and interrupts survive process restarts."
                    .into(),
                metadata: json!({"source":"graph-guide.md", "tenant":"demo"}),
            },
            Document {
                id: "observability-guide".into(),
                text: "Usage collectors aggregate provider token counts. Metrics record latency \
                       and failures, and tracing can export OpenTelemetry-compatible spans."
                    .into(),
                metadata: json!({"source":"observability-guide.md", "tenant":"demo"}),
            },
        ])
        .await?;
    println!("indexed {} chunks", report.chunks_indexed);

    let retriever = Arc::new(
        Retriever::from_shared(embedder, vector_store)
            .hybrid(0.65, 0.35)
            .with_reranker(LexicalReranker::default()),
    );

    // Record exact token usage and operation metrics. The adapter converts
    // lifecycle events to OTel-shaped spans; a production exporter can batch
    // these to an OTLP/OpenTelemetry SDK pipeline.
    let usage = InMemoryUsageCollector::default();
    let metrics = InMemoryMetricsCollector::default();
    let span_exporter = InMemorySpanExporter::default();
    let tracer = CompositeTracer::default().push(OpenTelemetryAdapter::new(span_exporter.clone()));
    #[cfg(feature = "opentelemetry")]
    let tracer = tracer.push(liteagent::OpenTelemetryTracer);

    let model = OpenAiModel::new("gpt-4o-mini", api_key);
    let agent = Agent::builder(model)
        .instructions("Use retrieval for framework facts and cite document IDs.")
        .tool(RetrieverTool::new(retriever.clone(), 4))
        .storage(FileStorage::new(data_dir.join("sessions")))
        .execution_policy(ExecutionPolicy {
            max_attempts: 3,
            request_timeout: Duration::from_secs(45),
            tool_timeout: Duration::from_secs(10),
            parallel_tool_calls: true,
            ..ExecutionPolicy::default()
        })
        .usage_recorder(usage.clone())
        .metrics_recorder(metrics.clone())
        .tracer(tracer)
        .build();
    let agent = Arc::new(Mutex::new(agent));

    // Native provider streams emit partial tool arguments as ToolCallDelta,
    // followed by the assembled ToolCalls event and the final answer.
    let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
    let event_printer = tokio::spawn(async move {
        while let Some(event) = events_rx.recv().await {
            match event? {
                StreamEvent::ContentDelta { delta } => print!("{delta}"),
                StreamEvent::ToolCallDelta {
                    name,
                    arguments_delta,
                    ..
                } => println!("\ntool delta {name:?}: {arguments_delta}"),
                StreamEvent::Done => println!(),
                _ => {}
            }
        }
        Ok::<_, liteagent::AgentError>(())
    });
    agent
        .lock()
        .await
        .run_stream(
            "Call retrieve_documents and explain how graph recovery works.",
            events_tx,
        )
        .await?;
    event_printer.await??;

    let question = "How does liteagent make runs durable and observable?";
    let schema = Arc::new(answer_schema());

    // The graph persists every superstep. `retrieve` conditionally fans out
    // to two parallel nodes, their disjoint state updates meet at a durable
    // AND-join, and an interrupt creates a human-approval boundary.
    let retrieval_node = retriever.clone();
    let answer_agent = agent.clone();
    let answer_node_schema = schema.clone();
    let graph = Graph::builder(
        "advanced-runtime",
        FileGraphStore::new(data_dir.join("graphs")),
    )
    .version("1")
    .entry("retrieve")
    .node_with_retry(
        "retrieve",
        NodeRetryPolicy::attempts(3)
            .with_backoff(Duration::from_millis(50), Duration::from_secs(1))
            .with_timeout(Duration::from_secs(10)),
        move |context| {
            let retriever = retrieval_node.clone();
            async move {
                let query = context.state["question"].as_str().unwrap_or_default();
                let documents = retriever
                    .retrieve_with_options(
                        query,
                        RetrievalOptions {
                            limit: 4,
                            filter: Some(MetadataFilter::eq("tenant", "demo")),
                            strategy: RetrievalStrategy::Hybrid {
                                vector_weight: 0.65,
                                lexical_weight: 0.35,
                            },
                            ..RetrievalOptions::default()
                        },
                    )
                    .await?;
                let formatted = ContextFormatter::default().format(&documents);
                Ok(NodeOutput::merge(json!({"retrieval": formatted})).routes(["answer", "audit"]))
            }
        },
    )
    .node("answer", move |context| {
        let agent = answer_agent.clone();
        let schema = answer_node_schema.clone();
        async move {
            let prompt = format!(
                "Answer this question from the supplied context. Question: {}\n\nContext:\n{}",
                context.state["question"].as_str().unwrap_or_default(),
                context
                    .state
                    .pointer("/retrieval/context")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            );
            let answer: GroundedAnswer = agent.lock().await.run_structured(prompt, &schema).await?;
            Ok(NodeOutput::merge(json!({"answer": answer})))
        }
    })
    .node("audit", |context| async move {
        let citations = context
            .state
            .pointer("/retrieval/citations")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        Ok(NodeOutput::merge(json!({
            "audit": {"has_context": citations > 0, "citation_count": citations}
        })))
    })
    .node("publish", |_context| async move {
        Ok(NodeOutput::merge(json!({"published": true})))
    })
    .route("retrieve", "answer", "answer")
    .route("retrieve", "audit", "audit")
    .join(["answer", "audit"], "publish")
    .interrupt_before("publish")
    .build()?;

    let mut checkpoint = graph
        .run("advanced_demo_v1", json!({"question": question}))
        .await?;
    if checkpoint.status == GraphStatus::Paused {
        println!("approval boundary: {:?}", checkpoint.pause_reason);
        checkpoint = graph.resume("advanced_demo_v1").await?;
    }
    println!(
        "graph status: {:?}\n{}",
        checkpoint.status, checkpoint.state
    );

    // Run a deterministic retrieval regression suite without spending model
    // tokens. Real suites can instead use AgentEvaluationTarget.
    let evaluation_retriever = retriever.clone();
    let target = FunctionEvaluationTarget::new(move |case| {
        let retriever = evaluation_retriever.clone();
        Box::pin(async move {
            let query = case.input.as_str().unwrap_or_default();
            let text = retriever
                .retrieve(query, 4)
                .await?
                .into_iter()
                .map(|result| result.document.text)
                .collect::<Vec<_>>()
                .join("\n");
            Ok(EvaluationOutput::new(json!(text)))
        })
    });
    let dataset = EvaluationDataset::new(
        "advanced-runtime-retrieval",
        "1",
        vec![
            EvaluationCase::new("recovers-graphs", json!("How do failed graphs resume?"))
                .expected(json!("stable idempotency keys")),
        ],
    );
    let evaluation = EvaluationRunner::new(target)
        .scorer(ContainsScorer::new(false))
        .run(&dataset)
        .await?;
    let baseline_store = EvaluationBaselineStore::new(data_dir.join("evaluation-baseline.json"));
    let previous_baseline = baseline_store.load().await?;
    evaluation
        .check_regression(
            previous_baseline.as_ref(),
            &RegressionThresholds {
                minimum_pass_rate: Some(1.0),
                maximum_error_rate: Some(0.0),
                maximum_pass_rate_drop: Some(0.0),
                require_same_dataset: true,
                ..RegressionThresholds::default()
            },
        )
        .enforce()?;

    // Keep append-only evidence and the accepted baseline across restarts.
    let usage_log = UsageLogStore::new(data_dir.join("usage.jsonl"));
    for (index, record) in usage.records().into_iter().enumerate() {
        usage_log
            .append_idempotent(format!("{}-{index}", record.run_id), record)
            .await?;
    }
    EvaluationReportLogStore::new(data_dir.join("evaluations.jsonl"))
        .append_idempotent(evaluation.id.clone(), evaluation.clone())
        .await?;
    baseline_store.save(&evaluation).await?;

    let registry = IntegrationRegistry::curated();
    let streaming_integrations = registry.by_capability(IntegrationCapability::StreamingToolCalls);
    println!(
        "evaluation pass rate: {:.0}%; streaming-tool integrations: {}; tokens: {}; operations: {}; spans: {}",
        evaluation.pass_rate * 100.0,
        streaming_integrations.len(),
        usage.summary().totals.usage.total_tokens,
        metrics.snapshot().overall.count,
        span_exporter.spans().len(),
    );

    Ok(())
}
