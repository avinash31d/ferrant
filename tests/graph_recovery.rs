use async_trait::async_trait;
use ferragent::graph::{
    FileGraphStore, Graph, GraphCheckpoint, GraphCheckpointStore, GraphError, GraphStatus,
    GraphStoreError, GraphStoreResult, GraphValidationCode, InMemoryGraphStore,
    NodeExecutionStatus, NodeOutput, NodeRetryPolicy,
};
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Barrier;

async fn no_op(_: ferragent::graph::NodeContext) -> anyhow::Result<NodeOutput> {
    Ok(NodeOutput::no_update())
}

#[test]
fn rejects_missing_invalid_and_unreachable_nodes() {
    let missing_entry = Graph::builder("invalid", InMemoryGraphStore::default())
        .node("only", no_op)
        .build()
        .unwrap_err();
    assert!(missing_entry
        .issues
        .iter()
        .any(|issue| issue.code == GraphValidationCode::MissingEntry));

    let invalid = Graph::builder("invalid", InMemoryGraphStore::default())
        .entry("start")
        .node("start", no_op)
        .node("orphan", no_op)
        .edge("start", "not_defined")
        .build()
        .unwrap_err();
    assert!(invalid
        .issues
        .iter()
        .any(|issue| issue.code == GraphValidationCode::MissingNode));
    assert!(invalid
        .issues
        .iter()
        .any(|issue| issue.code == GraphValidationCode::UnreachableNode));

    let invalid_entry = Graph::builder("invalid", InMemoryGraphStore::default())
        .entry("ghost")
        .node("real", no_op)
        .build()
        .unwrap_err();
    assert!(invalid_entry.issues.iter().any(|issue| {
        issue.code == GraphValidationCode::MissingNode
            && issue.message.contains("entry node 'ghost'")
    }));
}

#[tokio::test]
async fn executes_cycles_parallel_fanout_durable_join_and_pause_resume() {
    let barrier = Arc::new(Barrier::new(2));
    let left_barrier = barrier.clone();
    let right_barrier = barrier.clone();

    let graph = Graph::builder("branching", InMemoryGraphStore::default())
        .entry("increment")
        .max_steps(20)
        .node("increment", |context| async move {
            let count = context.state["count"].as_u64().unwrap_or_default() + 1;
            Ok(NodeOutput::merge(json!({"count": count})))
        })
        .node("router", |_| async { Ok(NodeOutput::no_update()) })
        .node("left", move |_| {
            let barrier = left_barrier.clone();
            async move {
                barrier.wait().await;
                Ok(NodeOutput::merge(json!({"left": true})))
            }
        })
        .node("right", move |_| {
            let barrier = right_barrier.clone();
            async move {
                barrier.wait().await;
                Ok(NodeOutput::merge(json!({"right": true})))
            }
        })
        .node("join", |context| async move {
            assert_eq!(context.state["left"], json!(true));
            assert_eq!(context.state["right"], json!(true));
            Ok(NodeOutput::merge(json!({"joined": true})).and_pause("human approval"))
        })
        .node("finish", |_| async {
            Ok(NodeOutput::merge(json!({"finished": true})))
        })
        .edge("increment", "router")
        .conditional_edge("router", "increment", |state, _| {
            state["count"].as_u64().unwrap_or_default() < 2
        })
        .conditional_edge("router", "left", |state, _| state["count"] == json!(2))
        .conditional_edge("router", "right", |state, _| state["count"] == json!(2))
        .join(["left", "right"], "join")
        .edge("join", "finish")
        .build()
        .unwrap();

    let paused = tokio::time::timeout(
        Duration::from_secs(2),
        graph.run("branch_run", json!({"count": 0})),
    )
    .await
    .expect("fan-out nodes must run concurrently")
    .unwrap();
    assert_eq!(paused.status, GraphStatus::Paused);
    assert_eq!(paused.pause_reason.as_deref(), Some("human approval"));
    assert_eq!(paused.state["count"], json!(2));
    assert_eq!(paused.state["left"], json!(true));
    assert_eq!(paused.state["right"], json!(true));
    assert_eq!(paused.state["joined"], json!(true));
    assert_eq!(paused.frontier.len(), 1);
    assert_eq!(paused.frontier[0].node, "finish");

    let completed = graph.resume("branch_run").await.unwrap();
    assert_eq!(completed.status, GraphStatus::Completed);
    assert_eq!(completed.state["finished"], json!(true));
    assert_eq!(completed.completed_nodes["increment"], 2);
    assert_eq!(completed.completed_nodes["router"], 2);
    assert_eq!(completed.steps, 8);
}

#[tokio::test]
async fn interrupt_before_is_durable_and_fires_once_per_invocation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let node_calls = calls.clone();
    let graph = Graph::builder("interrupt", InMemoryGraphStore::default())
        .entry("first")
        .node("first", |_| async { Ok(NodeOutput::no_update()) })
        .node("approval", move |_| {
            let calls = node_calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(NodeOutput::merge(json!({"approved": true})))
            }
        })
        .edge("first", "approval")
        .interrupt_before("approval")
        .build()
        .unwrap();

    let paused = graph.run("interrupt_run", json!({})).await.unwrap();
    assert_eq!(paused.status, GraphStatus::Paused);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(paused.frontier[0].breakpoint_passed);

    let completed = graph
        .resume_with("interrupt_run", Some(json!({"human": "yes"})))
        .await
        .unwrap();
    assert_eq!(completed.status, GraphStatus::Completed);
    assert_eq!(completed.state["human"], json!("yes"));
    assert_eq!(completed.state["approved"], json!(true));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Completed runs are idempotent and never invoke a node again.
    assert_eq!(
        graph
            .run("interrupt_run", json!({"ignored": true}))
            .await
            .unwrap(),
        completed
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn retries_nodes_and_recovers_only_failed_parallel_siblings() {
    let retry_attempts = Arc::new(Mutex::new(Vec::new()));
    let observed_attempts = retry_attempts.clone();
    let retry_calls = Arc::new(AtomicUsize::new(0));
    let calls = retry_calls.clone();
    let retry = NodeRetryPolicy {
        max_attempts: 3,
        initial_backoff: Duration::ZERO,
        backoff_multiplier: 1.0,
        max_backoff: Duration::ZERO,
        attempt_timeout: Some(Duration::from_secs(1)),
    };
    let retry_graph = Graph::builder("retry", InMemoryGraphStore::default())
        .entry("flaky")
        .node_with_retry("flaky", retry, move |context| {
            let calls = calls.clone();
            let attempts = observed_attempts.clone();
            async move {
                attempts.lock().unwrap().push(context.attempt);
                if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    anyhow::bail!("temporary")
                }
                Ok(NodeOutput::merge(json!({"retried": true})))
            }
        })
        .build()
        .unwrap();
    let retried = retry_graph.run("retry_run", json!({})).await.unwrap();
    assert_eq!(retried.status, GraphStatus::Completed);
    assert_eq!(*retry_attempts.lock().unwrap(), vec![1, 2, 3]);

    let store = InMemoryGraphStore::default();
    let left_calls = Arc::new(AtomicUsize::new(0));
    let right_calls = Arc::new(AtomicUsize::new(0));
    let right_keys = Arc::new(Mutex::new(Vec::new()));
    let left_counter = left_calls.clone();
    let right_counter = right_calls.clone();
    let keys = right_keys.clone();
    let graph = Graph::builder("recovery", store.clone())
        .entry("fanout")
        .node("fanout", |_| async { Ok(NodeOutput::no_update()) })
        .node("left", move |_| {
            let calls = left_counter.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(NodeOutput::merge(json!({"left": "done"})))
            }
        })
        .node("right", move |context| {
            let calls = right_counter.clone();
            let keys = keys.clone();
            async move {
                keys.lock().unwrap().push(context.idempotency_key);
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    anyhow::bail!("dependency unavailable")
                }
                Ok(NodeOutput::merge(json!({"right": "done"})))
            }
        })
        .node("join", |_| async {
            Ok(NodeOutput::merge(json!({"joined": true})))
        })
        .edge("fanout", "left")
        .edge("fanout", "right")
        .join(["left", "right"], "join")
        .build()
        .unwrap();

    assert!(matches!(
        graph.run("recover_run", json!({})).await,
        Err(GraphError::NodeFailed { ref node, .. }) if node == "right"
    ));
    let failed = store.load("recover_run").await.unwrap().unwrap();
    assert_eq!(failed.status, GraphStatus::Failed);
    assert_eq!(failed.frontier.len(), 2);
    assert_eq!(
        failed
            .frontier
            .iter()
            .find(|node| node.node == "left")
            .unwrap()
            .status,
        NodeExecutionStatus::Succeeded
    );
    assert_eq!(
        failed
            .frontier
            .iter()
            .find(|node| node.node == "right")
            .unwrap()
            .status,
        NodeExecutionStatus::Failed
    );

    let completed = graph.resume("recover_run").await.unwrap();
    assert_eq!(completed.status, GraphStatus::Completed);
    assert_eq!(completed.state["left"], json!("done"));
    assert_eq!(completed.state["right"], json!("done"));
    assert_eq!(completed.state["joined"], json!(true));
    assert_eq!(left_calls.load(Ordering::SeqCst), 1);
    assert_eq!(right_calls.load(Ordering::SeqCst), 2);
    let keys = right_keys.lock().unwrap();
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0], keys[1]);
}

#[derive(Clone, Default)]
struct FailSuccessfulCheckpointOnce {
    inner: InMemoryGraphStore,
    failed: Arc<AtomicBool>,
}

#[async_trait]
impl GraphCheckpointStore for FailSuccessfulCheckpointOnce {
    async fn load(&self, execution_id: &str) -> GraphStoreResult<Option<GraphCheckpoint>> {
        self.inner.load(execution_id).await
    }

    async fn save(
        &self,
        checkpoint: &GraphCheckpoint,
        expected_revision: Option<u64>,
    ) -> GraphStoreResult<()> {
        let saving_success = checkpoint
            .frontier
            .iter()
            .any(|node| node.status == NodeExecutionStatus::Succeeded);
        if saving_success && !self.failed.swap(true, Ordering::SeqCst) {
            return Err(GraphStoreError::Io(std::io::Error::other(
                "simulated worker crash before success checkpoint",
            )));
        }
        self.inner.save(checkpoint, expected_revision).await
    }

    async fn delete(&self, execution_id: &str) -> GraphStoreResult<()> {
        self.inner.delete(execution_id).await
    }
}

#[tokio::test]
async fn recover_reuses_idempotency_key_after_lost_node_result() {
    let store = FailSuccessfulCheckpointOnce::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let effects = Arc::new(AtomicUsize::new(0));
    let seen_keys = Arc::new(Mutex::new(BTreeSet::new()));
    let all_keys = Arc::new(Mutex::new(Vec::new()));
    let node_calls = calls.clone();
    let node_effects = effects.clone();
    let node_seen = seen_keys.clone();
    let node_keys = all_keys.clone();
    let graph = Graph::builder("crash", store.clone())
        .entry("side_effect")
        .node("side_effect", move |context| {
            let calls = node_calls.clone();
            let effects = node_effects.clone();
            let seen = node_seen.clone();
            let keys = node_keys.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                keys.lock().unwrap().push(context.idempotency_key.clone());
                if seen.lock().unwrap().insert(context.idempotency_key) {
                    effects.fetch_add(1, Ordering::SeqCst);
                }
                Ok(NodeOutput::merge(json!({"effect": "recorded"})))
            }
        })
        .build()
        .unwrap();

    assert!(matches!(
        graph.run("crash_run", json!({})).await,
        Err(GraphError::Store(GraphStoreError::Io(_)))
    ));
    let stranded = store.load("crash_run").await.unwrap().unwrap();
    assert_eq!(stranded.status, GraphStatus::Running);
    assert_eq!(stranded.frontier[0].status, NodeExecutionStatus::Running);

    let recovered = graph.recover("crash_run").await.unwrap();
    assert_eq!(recovered.status, GraphStatus::Completed);
    assert_eq!(recovered.state["effect"], json!("recorded"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(effects.load(Ordering::SeqCst), 1);
    let keys = all_keys.lock().unwrap();
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0], keys[1]);
}

#[tokio::test]
async fn file_store_survives_graph_reconstruction_and_checks_revisions() {
    let directory =
        std::env::temp_dir().join(format!("ferragent-graph-recovery-{}", uuid::Uuid::new_v4()));
    let available = Arc::new(AtomicBool::new(false));
    let first_available = available.clone();
    let first_store = FileGraphStore::new(&directory);
    let first_graph = Graph::builder("persistent", first_store.clone())
        .version("2026-01")
        .entry("work")
        .node("work", move |_| {
            let available = first_available.clone();
            async move {
                if !available.load(Ordering::SeqCst) {
                    anyhow::bail!("database offline")
                }
                Ok(NodeOutput::merge(json!({"persisted": true})))
            }
        })
        .build()
        .unwrap();
    assert!(matches!(
        first_graph.run("file_run", json!({"seed": 7})).await,
        Err(GraphError::NodeFailed { .. })
    ));

    available.store(true, Ordering::SeqCst);
    let second_available = available.clone();
    let second_store = FileGraphStore::new(&directory);
    let second_graph = Graph::builder("persistent", second_store.clone())
        .version("2026-01")
        .entry("work")
        .node("work", move |_| {
            let available = second_available.clone();
            async move {
                if !available.load(Ordering::SeqCst) {
                    anyhow::bail!("database offline")
                }
                Ok(NodeOutput::merge(json!({"persisted": true})))
            }
        })
        .build()
        .unwrap();
    let completed = second_graph.resume("file_run").await.unwrap();
    assert_eq!(completed.status, GraphStatus::Completed);
    assert_eq!(completed.state, json!({"seed": 7, "persisted": true}));

    let mut first_update = completed.clone();
    first_update.revision += 1;
    second_store
        .save(&first_update, Some(completed.revision))
        .await
        .unwrap();
    let mut stale_update = completed.clone();
    stale_update.revision += 1;
    assert!(matches!(
        second_store
            .save(&stale_update, Some(completed.revision))
            .await,
        Err(GraphStoreError::Conflict { .. })
    ));

    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[tokio::test]
async fn bounds_cycles_and_rejects_conflicting_parallel_updates() {
    let cycle = Graph::builder("cycle", InMemoryGraphStore::default())
        .entry("again")
        .max_steps(3)
        .node("again", |_| async { Ok(NodeOutput::no_update()) })
        .edge("again", "again")
        .build()
        .unwrap();
    assert!(matches!(
        cycle.run("cycle_run", json!({})).await,
        Err(GraphError::MaxStepsExceeded { max_steps: 3, .. })
    ));
    let cycle_state = cycle.checkpoint("cycle_run").await.unwrap().unwrap();
    assert_eq!(cycle_state.status, GraphStatus::Failed);
    assert_eq!(cycle_state.steps, 3);

    let conflicts = Graph::builder("conflict", InMemoryGraphStore::default())
        .entry("fanout")
        .node("fanout", |_| async { Ok(NodeOutput::no_update()) })
        .node("one", |_| async {
            Ok(NodeOutput::merge(json!({"shared": 1})))
        })
        .node("two", |_| async {
            Ok(NodeOutput::merge(json!({"shared": 2})))
        })
        .edge("fanout", "one")
        .edge("fanout", "two")
        .build()
        .unwrap();
    assert!(matches!(
        conflicts.run("conflict_run", json!({})).await,
        Err(GraphError::StateConflict { .. })
    ));
    let conflict_state = conflicts.checkpoint("conflict_run").await.unwrap().unwrap();
    assert_eq!(conflict_state.status, GraphStatus::Failed);
    assert_eq!(
        conflict_state.last_failure.unwrap().kind,
        ferragent::graph::GraphFailureKind::StateConflict
    );
}

#[tokio::test]
async fn times_out_nodes_and_rejects_corrupt_checkpoints_without_panicking() {
    let timeout_policy = NodeRetryPolicy {
        max_attempts: 1,
        initial_backoff: Duration::ZERO,
        backoff_multiplier: 1.0,
        max_backoff: Duration::ZERO,
        attempt_timeout: Some(Duration::from_millis(10)),
    };
    let timeout_graph = Graph::builder("timeout", InMemoryGraphStore::default())
        .entry("slow")
        .node_with_retry("slow", timeout_policy, |_| async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            Ok(NodeOutput::no_update())
        })
        .build()
        .unwrap();
    assert!(matches!(
        timeout_graph.run("timeout_run", json!({})).await,
        Err(GraphError::NodeFailed { ref message, .. }) if message.contains("timed out")
    ));

    let store = InMemoryGraphStore::default();
    let graph = Graph::builder("integrity", store.clone())
        .entry("pause")
        .node("pause", |_| async {
            Ok(NodeOutput::no_update().and_pause("inspect"))
        })
        .node("finish", |_| async { Ok(NodeOutput::no_update()) })
        .edge("pause", "finish")
        .build()
        .unwrap();
    let mut checkpoint = graph.run("corrupt_run", json!({})).await.unwrap();
    assert_eq!(checkpoint.status, GraphStatus::Paused);
    checkpoint.frontier[0].node = "not_in_definition".to_owned();
    checkpoint.revision += 1;
    store
        .save(&checkpoint, Some(checkpoint.revision - 1))
        .await
        .unwrap();
    assert!(matches!(
        graph.resume("corrupt_run").await,
        Err(GraphError::InvalidCheckpoint { ref message, .. })
            if message.contains("unknown node")
    ));
}
