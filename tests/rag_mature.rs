use async_trait::async_trait;
use ferrant::rag::*;
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn document(id: &str, text: &str, metadata: serde_json::Value) -> Document {
    Document {
        id: id.to_owned(),
        text: text.to_owned(),
        metadata,
    }
}

fn temporary_store_path() -> std::path::PathBuf {
    std::env::temp_dir()
        .join(format!("ferrant-rag-{}", uuid::Uuid::new_v4()))
        .join("vectors.json")
}

struct FailsOnSecondBatch {
    calls: AtomicUsize,
}

#[async_trait]
impl Embedder for FailsOnSecondBatch {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
            anyhow::bail!("simulated embedding outage");
        }
        Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}

#[tokio::test]
async fn ingestion_is_all_or_nothing_and_tracks_chunk_lineage() {
    let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::default());
    let failing = IngestionPipeline::from_shared(
        Arc::new(FailsOnSecondBatch {
            calls: AtomicUsize::new(0),
        }),
        store.clone(),
        Arc::new(TextChunker::new(5, 0)),
    )
    .batch_size(1);

    let error = failing
        .ingest(vec![document(
            "source",
            "first second",
            json!({"tenant":{"id":"acme"}}),
        )])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("simulated embedding outage"));
    assert!(store.documents().await.unwrap().is_empty());

    let pipeline = IngestionPipeline::from_shared(
        Arc::new(HashEmbedder::new(32)),
        store.clone(),
        Arc::new(TextChunker::new(5, 0)),
    );
    let report = pipeline
        .ingest(vec![document(
            "source",
            "first second",
            json!({"tenant":{"id":"acme"},"tags":["public","guide"]}),
        )])
        .await
        .unwrap();
    assert_eq!(report.source_documents, 1);
    assert_eq!(report.chunks_indexed, 3);

    let chunks = store.documents().await.unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].metadata["_rag"]["source_id"], "source");
    assert_eq!(chunks[0].metadata["_rag"]["chunk_index"], 0);
    assert!(MetadataFilter::And {
        filters: vec![
            MetadataFilter::eq("tenant.id", "acme"),
            MetadataFilter::Contains {
                field: "tags".into(),
                value: json!("guide"),
            },
        ],
    }
    .matches(&chunks[0].metadata));

    assert_eq!(pipeline.delete_source("source").await.unwrap(), 3);
    assert!(store.documents().await.unwrap().is_empty());
}

#[tokio::test]
async fn upsert_delete_filters_and_thresholds_are_enforced() {
    let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::default());
    let retriever = Retriever::from_shared(Arc::new(HashEmbedder::new(128)), store.clone());
    retriever
        .upsert(vec![
            document(
                "rust",
                "Rust ownership memory safety",
                json!({"tenant":"acme","visibility":"public"}),
            ),
            document(
                "pasta",
                "Pasta boiling water",
                json!({"tenant":"other","visibility":"private"}),
            ),
        ])
        .await
        .unwrap();
    retriever
        .upsert(vec![document(
            "rust",
            "Rust ownership without garbage collection",
            json!({"tenant":"acme","visibility":"public","revision":2}),
        )])
        .await
        .unwrap();

    let documents = store.documents().await.unwrap();
    assert_eq!(documents.len(), 2);
    assert_eq!(
        documents
            .iter()
            .find(|candidate| candidate.id == "rust")
            .unwrap()
            .metadata["revision"],
        2
    );

    let filtered = retriever
        .retrieve_with_options(
            "ownership",
            RetrievalOptions {
                limit: 5,
                min_score: Some(0.01),
                filter: Some(MetadataFilter::eq("tenant", "acme")),
                ..RetrievalOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].document.id, "rust");

    let above_possible_score = retriever
        .retrieve_with_options(
            "ownership",
            RetrievalOptions {
                limit: 5,
                min_score: Some(1.1),
                ..RetrievalOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(above_possible_score.is_empty());

    assert_eq!(retriever.delete(&["rust".to_owned()]).await.unwrap(), 1);
    assert_eq!(store.documents().await.unwrap()[0].id, "pasta");
}

#[tokio::test]
async fn file_store_persists_recovers_and_rejects_partial_mutations() {
    let path = temporary_store_path();
    {
        let store = FileVectorStore::open(&path).unwrap();
        store
            .upsert(
                vec![document(
                    "durable",
                    "persistent retrieval state",
                    json!({"source":"handbook.md"}),
                )],
                vec![vec![1.0, 0.0, 0.0]],
            )
            .await
            .unwrap();

        let error = store
            .add(
                vec![document("bad", "wrong dimensions", json!({}))],
                vec![vec![1.0, 0.0]],
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("dimension mismatch"));
        assert_eq!(store.documents().await.unwrap().len(), 1);
    }

    let reopened = FileVectorStore::open(&path).unwrap();
    assert_eq!(reopened.documents().await.unwrap()[0].id, "durable");
    drop(reopened);

    std::fs::write(&path, b"not json").unwrap();
    let recovered = FileVectorStore::open(&path).unwrap();
    assert_eq!(recovered.documents().await.unwrap()[0].id, "durable");
    assert!(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&path).unwrap()).is_ok());
    drop(recovered);

    let backup = std::path::PathBuf::from(format!("{}.bak", path.display()));
    std::fs::write(&path, b"broken primary").unwrap();
    std::fs::write(&backup, b"broken backup").unwrap();
    let error = match FileVectorStore::open(&path) {
        Ok(_) => panic!("both corrupt snapshots must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("backup is invalid"));

    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

struct ConstantEmbedder;

#[async_trait]
impl Embedder for ConstantEmbedder {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0]).collect())
    }
}

#[tokio::test]
async fn hybrid_multi_query_reranking_and_citations_work_together() {
    let transformer = FunctionQueryTransformer::new(|query| {
        Box::pin(async move {
            assert_eq!(query, "distant object");
            Ok(vec!["quasar astronomy".to_owned()])
        })
    });
    let retriever = Retriever::new(ConstantEmbedder, InMemoryVectorStore::default())
        .hybrid(0.1, 0.9)
        .with_query_transformer(transformer)
        .with_reranker(LexicalReranker::new(0.25));
    retriever
        .upsert(vec![
            document(
                "a-general",
                "A general cooking handbook",
                json!({"source":"cooking.md","category":"food"}),
            ),
            document(
                "z-quasar",
                "A quasar is a luminous distant astronomy object",
                json!({"source":"space.md","category":"science"}),
            ),
        ])
        .await
        .unwrap();

    let results = retriever
        .retrieve_with_options(
            "distant object",
            RetrievalOptions {
                limit: 2,
                min_score: Some(0.2),
                filter: Some(MetadataFilter::eq("category", "science")),
                strategy: RetrievalStrategy::Hybrid {
                    vector_weight: 0.1,
                    lexical_weight: 0.9,
                },
                candidate_multiplier: 3,
            },
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id, "z-quasar");

    let formatted = ContextFormatter {
        max_chars: 200,
        include_scores: true,
    }
    .format(&results);
    assert!(formatted.context.starts_with("[1] (score:"));
    assert!(formatted.context.contains("quasar"));
    assert_eq!(formatted.citations[0].document_id, "z-quasar");
    assert_eq!(formatted.citations[0].source.as_deref(), Some("space.md"));
    assert!(formatted.context.chars().count() <= 200);
}

#[tokio::test]
async fn persistent_delete_and_upsert_survive_reopen() {
    let path = temporary_store_path();
    {
        let store = FileVectorStore::open(&path).unwrap();
        store
            .add(
                vec![
                    document("one", "old", json!({})),
                    document("two", "remove", json!({})),
                ],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )
            .await
            .unwrap();
        store
            .upsert(
                vec![document("one", "new", json!({"revision":2}))],
                vec![vec![0.8, 0.2]],
            )
            .await
            .unwrap();
        assert_eq!(store.delete(&["two".to_owned()]).await.unwrap(), 1);
    }
    let reopened = FileVectorStore::open(&path).unwrap();
    let documents = reopened.documents().await.unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "new");
    assert_eq!(documents[0].metadata["revision"], 2);
    let _ = std::fs::remove_dir_all(Path::new(&path).parent().unwrap());
}
