//! Durable fan-out, retry, join, approval, resume, and recovery.

use ferrant::graph::{FileGraphStore, Graph, GraphStatus, NodeOutput, NodeRetryPolicy};
use serde_json::json;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let directory = std::env::temp_dir().join("ferrant-advanced-workflow");
    let graph = Graph::builder("release-workflow", FileGraphStore::new(directory))
        .version("1")
        .entry("plan")
        .node("plan", |_context| async {
            Ok(NodeOutput::merge(json!({"plan": "review in parallel"}))
                .routes(["research", "risk"]))
        })
        .node("research", |_context| async {
            Ok(NodeOutput::merge(json!({
                "research": {"finding": "staged rollout reduces blast radius"}
            })))
        })
        .node_with_retry(
            "risk",
            NodeRetryPolicy::attempts(3)
                .with_backoff(Duration::from_millis(10), Duration::from_millis(100))
                .with_timeout(Duration::from_secs(5)),
            |context| async move {
                if context.attempt == 1 {
                    anyhow::bail!("simulated transient dependency failure");
                }
                Ok(NodeOutput::merge(json!({
                    "risk": {"finding": "retain rollback capacity"}
                })))
            },
        )
        .node("publish", |_context| async {
            Ok(NodeOutput::merge(json!({"published": true})))
        })
        .route("plan", "research", "research")
        .route("plan", "risk", "risk")
        .join(["research", "risk"], "publish")
        .interrupt_before("publish")
        .build()?;

    let mut checkpoint = graph.run("release-42", json!({"release": "v2"})).await?;
    println!(
        "approval checkpoint: {:?} {}",
        checkpoint.status, checkpoint.state
    );
    if checkpoint.status == GraphStatus::Paused {
        checkpoint = graph
            .resume_with("release-42", Some(json!({"approved_by": "operator"})))
            .await?;
    }
    println!("completed: {:?} {}", checkpoint.status, checkpoint.state);
    println!("recovered: {:?}", graph.recover("release-42").await?.status);
    Ok(())
}
