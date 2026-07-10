//! Provider-neutral retrieval-augmented generation primitives.
//!
//! The module deliberately keeps the core traits small while providing a
//! complete local implementation: ingestion, metadata filtering, mutable and
//! persistent vector stores, hybrid retrieval, query expansion, reranking,
//! and citation-ready context formatting.

use crate::Tool;
use anyhow::Context;
use async_trait::async_trait;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievedDocument {
    pub document: Document,
    pub score: f32,
}

#[async_trait]
pub trait DocumentLoader: Send + Sync {
    async fn load(&self) -> anyhow::Result<Vec<Document>>;
}

/// Loads UTF-8 text files into documents with their canonical path recorded
/// as source metadata.
pub struct TextFileLoader {
    paths: Vec<PathBuf>,
}

impl TextFileLoader {
    pub fn new(paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self {
            paths: paths.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl DocumentLoader for TextFileLoader {
    async fn load(&self) -> anyhow::Result<Vec<Document>> {
        let mut documents = Vec::with_capacity(self.paths.len());
        for path in &self.paths {
            let canonical = tokio::fs::canonicalize(path)
                .await
                .with_context(|| format!("failed to resolve {}", path.display()))?;
            let text = tokio::fs::read_to_string(&canonical)
                .await
                .with_context(|| {
                    format!("failed to read UTF-8 document {}", canonical.display())
                })?;
            documents.push(Document {
                id: canonical.to_string_lossy().into_owned(),
                text,
                metadata: json!({"source":canonical.to_string_lossy(),"loader":"text_file"}),
            });
        }
        Ok(documents)
    }
}

pub trait Chunker: Send + Sync {
    fn chunk(&self, document: &Document) -> Vec<Document>;
}

/// Character-aware chunking with a bounded overlap.
pub struct TextChunker {
    pub max_chars: usize,
    pub overlap: usize,
}

impl TextChunker {
    pub fn new(max_chars: usize, overlap: usize) -> Self {
        Self { max_chars, overlap }
    }
}

impl Chunker for TextChunker {
    fn chunk(&self, document: &Document) -> Vec<Document> {
        if self.max_chars == 0 || document.text.is_empty() {
            return vec![];
        }
        let chars: Vec<char> = document.text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.max_chars).min(chars.len());
            chunks.push(Document {
                id: format!("{}#{}", document.id, chunks.len()),
                text: chars[start..end].iter().collect(),
                metadata: document.metadata.clone(),
            });
            if end == chars.len() {
                break;
            }
            start = end.saturating_sub(self.overlap.min(self.max_chars.saturating_sub(1)));
        }
        chunks
    }
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// OpenAI-compatible embeddings backend. `base_url` may point at OpenAI,
/// Azure-compatible gateways, Ollama, vLLM, or another `/v1/embeddings`
/// implementation.
pub struct OpenAiEmbedder {
    api_key: String,
    model: String,
    base_url: String,
    dimensions: Option<usize>,
    client: reqwest::Client,
}

impl OpenAiEmbedder {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com/v1".into(),
            dimensions: None,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions.max(1));
        self
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut body = json!({"model":self.model,"input":texts,"encoding_format":"float"});
        if let Some(dimensions) = self.dimensions {
            body["dimensions"] = json!(dimensions);
        }
        let response = self
            .client
            .post(format!(
                "{}/embeddings",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("embedding provider error ({status}): {body}");
        }
        let data: Value = response.json().await?;
        let mut rows = data
            .get("data")
            .and_then(Value::as_array)
            .context("embedding response did not contain a data array")?
            .iter()
            .map(|row| {
                let index = row
                    .get("index")
                    .and_then(Value::as_u64)
                    .context("embedding row has no index")? as usize;
                let vector = row
                    .get("embedding")
                    .and_then(Value::as_array)
                    .context("embedding row has no vector")?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|number| number as f32)
                            .context("embedding value is not numeric")
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                Ok((index, vector))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        rows.sort_by_key(|(index, _)| *index);
        if rows.len() != texts.len()
            || rows
                .iter()
                .enumerate()
                .any(|(expected, (actual, _))| expected != *actual)
        {
            anyhow::bail!("embedding response indices did not match the input batch");
        }
        Ok(rows.into_iter().map(|(_, vector)| vector).collect())
    }
}

/// Dependency-free local embedding useful for tests and small lexical RAG.
pub struct HashEmbedder {
    dimensions: usize,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions: dimensions.max(1),
        }
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut vector = vec![0.0; self.dimensions];
                for word in tokenize(text) {
                    let mut hash = 1469598103934665603u64;
                    for byte in word.bytes() {
                        hash ^= byte as u64;
                        hash = hash.wrapping_mul(1099511628211);
                    }
                    vector[hash as usize % self.dimensions] += 1.0;
                }
                normalize(vector)
            })
            .collect())
    }
}

/// A composable predicate over document metadata. Dot-separated fields access
/// nested objects (for example, `tenant.id`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MetadataFilter {
    Eq { field: String, value: Value },
    NotEq { field: String, value: Value },
    In { field: String, values: Vec<Value> },
    Contains { field: String, value: Value },
    Exists { field: String },
    And { filters: Vec<MetadataFilter> },
    Or { filters: Vec<MetadataFilter> },
    Not { filter: Box<MetadataFilter> },
}

impl MetadataFilter {
    pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Eq {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Self::Eq { field, value } => metadata_value(metadata, field) == Some(value),
            Self::NotEq { field, value } => metadata_value(metadata, field) != Some(value),
            Self::In { field, values } => metadata_value(metadata, field)
                .is_some_and(|candidate| values.iter().any(|value| value == candidate)),
            Self::Contains { field, value } => {
                metadata_value(metadata, field).is_some_and(|candidate| match (candidate, value) {
                    (Value::Array(items), value) => items.iter().any(|item| item == value),
                    (Value::String(text), Value::String(fragment)) => text.contains(fragment),
                    (Value::Object(object), Value::String(key)) => object.contains_key(key),
                    _ => false,
                })
            }
            Self::Exists { field } => metadata_value(metadata, field).is_some(),
            Self::And { filters } => filters.iter().all(|filter| filter.matches(metadata)),
            Self::Or { filters } => filters.iter().any(|filter| filter.matches(metadata)),
            Self::Not { filter } => !filter.matches(metadata),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub min_score: Option<f32>,
    pub filter: Option<MetadataFilter>,
}

impl SearchOptions {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            ..Self::default()
        }
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 5,
            min_score: None,
            filter: None,
        }
    }
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add(&self, documents: Vec<Document>, embeddings: Vec<Vec<f32>>) -> anyhow::Result<()>;

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<RetrievedDocument>>;

    /// Insert new IDs and replace existing IDs. The default preserves source
    /// compatibility for minimal stores, but stores should override it to make
    /// replacement atomic.
    async fn upsert(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> anyhow::Result<()> {
        self.add(documents, embeddings).await
    }

    async fn delete(&self, _ids: &[String]) -> anyhow::Result<usize> {
        anyhow::bail!("this vector store does not support deletion")
    }

    async fn documents(&self) -> anyhow::Result<Vec<Document>> {
        anyhow::bail!("this vector store does not support document enumeration")
    }

    async fn search_with_options(
        &self,
        embedding: &[f32],
        options: &SearchOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        let fetch_limit = if options.filter.is_some() {
            usize::MAX
        } else {
            options.limit
        };
        let mut results = self.search(embedding, fetch_limit).await?;
        apply_search_options(&mut results, options);
        Ok(results)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredEntry {
    document: Document,
    embedding: Vec<f32>,
}

#[derive(Default)]
pub struct InMemoryVectorStore {
    entries: RwLock<Vec<StoredEntry>>,
}

impl InMemoryVectorStore {
    fn search_entries(
        &self,
        embedding: &[f32],
        options: &SearchOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        let entries = self
            .entries
            .read()
            .map_err(|_| anyhow::anyhow!("in-memory vector store lock was poisoned"))?;
        validate_query_dimension(&entries, embedding)?;
        let mut results = score_entries(&entries, embedding, options.filter.as_ref());
        apply_search_options(&mut results, options);
        Ok(results)
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add(&self, documents: Vec<Document>, embeddings: Vec<Vec<f32>>) -> anyhow::Result<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| anyhow::anyhow!("in-memory vector store lock was poisoned"))?;
        let incoming = make_entries(documents, embeddings, expected_dimension(&entries))?;
        ensure_ids_absent(&entries, &incoming)?;
        entries.extend(incoming);
        Ok(())
    }

    async fn upsert(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> anyhow::Result<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| anyhow::anyhow!("in-memory vector store lock was poisoned"))?;
        let incoming = make_entries(documents, embeddings, expected_dimension(&entries))?;
        let ids = incoming
            .iter()
            .map(|entry| entry.document.id.as_str())
            .collect::<HashSet<_>>();
        entries.retain(|entry| !ids.contains(entry.document.id.as_str()));
        entries.extend(incoming);
        Ok(())
    }

    async fn delete(&self, ids: &[String]) -> anyhow::Result<usize> {
        let ids = ids.iter().map(String::as_str).collect::<HashSet<_>>();
        let mut entries = self
            .entries
            .write()
            .map_err(|_| anyhow::anyhow!("in-memory vector store lock was poisoned"))?;
        let before = entries.len();
        entries.retain(|entry| !ids.contains(entry.document.id.as_str()));
        Ok(before - entries.len())
    }

    async fn documents(&self) -> anyhow::Result<Vec<Document>> {
        Ok(self
            .entries
            .read()
            .map_err(|_| anyhow::anyhow!("in-memory vector store lock was poisoned"))?
            .iter()
            .map(|entry| entry.document.clone())
            .collect())
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        self.search_entries(embedding, &SearchOptions::new(limit))
    }

    async fn search_with_options(
        &self,
        embedding: &[f32],
        options: &SearchOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        self.search_entries(embedding, options)
    }
}

const VECTOR_STORE_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct VectorStoreSnapshot {
    version: u32,
    entries: Vec<StoredEntry>,
}

/// A durable, single-process vector store backed by atomic JSON snapshots.
///
/// Every successful mutation replaces the primary snapshot atomically and
/// writes a recovery copy. `open` restores the primary from that recovery copy
/// if the primary is missing or corrupt. Embedding dimensions are validated on
/// both load and mutation so bad writes cannot partially alter in-memory state.
pub struct FileVectorStore {
    path: PathBuf,
    entries: RwLock<Vec<StoredEntry>>,
}

impl FileVectorStore {
    pub fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = path.into();
        let entries = load_snapshot_with_recovery(&path)?;
        Ok(Self {
            path,
            entries: RwLock::new(entries),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn mutate<F>(&self, mutation: F) -> anyhow::Result<usize>
    where
        F: FnOnce(&mut Vec<StoredEntry>) -> anyhow::Result<usize>,
    {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| anyhow::anyhow!("file vector store lock was poisoned"))?;
        let mut candidate = entries.clone();
        let affected = mutation(&mut candidate)?;
        persist_snapshot(&self.path, &candidate)?;
        *entries = candidate;
        Ok(affected)
    }

    fn search_entries(
        &self,
        embedding: &[f32],
        options: &SearchOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        let entries = self
            .entries
            .read()
            .map_err(|_| anyhow::anyhow!("file vector store lock was poisoned"))?;
        validate_query_dimension(&entries, embedding)?;
        let mut results = score_entries(&entries, embedding, options.filter.as_ref());
        apply_search_options(&mut results, options);
        Ok(results)
    }
}

#[async_trait]
impl VectorStore for FileVectorStore {
    async fn add(&self, documents: Vec<Document>, embeddings: Vec<Vec<f32>>) -> anyhow::Result<()> {
        self.mutate(move |entries| {
            let incoming = make_entries(documents, embeddings, expected_dimension(entries))?;
            ensure_ids_absent(entries, &incoming)?;
            let count = incoming.len();
            entries.extend(incoming);
            Ok(count)
        })?;
        Ok(())
    }

    async fn upsert(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> anyhow::Result<()> {
        self.mutate(move |entries| {
            let incoming = make_entries(documents, embeddings, expected_dimension(entries))?;
            let ids = incoming
                .iter()
                .map(|entry| entry.document.id.as_str())
                .collect::<HashSet<_>>();
            entries.retain(|entry| !ids.contains(entry.document.id.as_str()));
            let count = incoming.len();
            entries.extend(incoming);
            Ok(count)
        })?;
        Ok(())
    }

    async fn delete(&self, ids: &[String]) -> anyhow::Result<usize> {
        let ids = ids.iter().cloned().collect::<HashSet<_>>();
        self.mutate(move |entries| {
            let before = entries.len();
            entries.retain(|entry| !ids.contains(&entry.document.id));
            Ok(before - entries.len())
        })
    }

    async fn documents(&self) -> anyhow::Result<Vec<Document>> {
        Ok(self
            .entries
            .read()
            .map_err(|_| anyhow::anyhow!("file vector store lock was poisoned"))?
            .iter()
            .map(|entry| entry.document.clone())
            .collect())
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        self.search_entries(embedding, &SearchOptions::new(limit))
    }

    async fn search_with_options(
        &self,
        embedding: &[f32],
        options: &SearchOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        self.search_entries(embedding, options)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IngestionReport {
    pub source_documents: usize,
    pub chunks_indexed: usize,
    pub chunk_ids: Vec<String>,
}

/// Validates and embeds the full input before committing it, preventing an
/// embedding failure in a later batch from leaving a partially ingested set.
pub struct IngestionPipeline {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    chunker: Arc<dyn Chunker>,
    batch_size: usize,
}

impl IngestionPipeline {
    pub fn new(
        embedder: impl Embedder + 'static,
        store: impl VectorStore + 'static,
        chunker: impl Chunker + 'static,
    ) -> Self {
        Self::from_shared(Arc::new(embedder), Arc::new(store), Arc::new(chunker))
    }

    pub fn from_shared(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        chunker: Arc<dyn Chunker>,
    ) -> Self {
        Self {
            embedder,
            store,
            chunker,
            batch_size: 64,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub async fn ingest(&self, documents: Vec<Document>) -> anyhow::Result<IngestionReport> {
        validate_document_ids(&documents)?;
        let source_documents = documents.len();
        let mut chunks = Vec::new();
        for source in documents {
            for (index, mut chunk) in self.chunker.chunk(&source).into_iter().enumerate() {
                add_lineage_metadata(&mut chunk, &source.id, index);
                chunks.push(chunk);
            }
        }
        validate_document_ids(&chunks)?;

        let mut embeddings = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(self.batch_size) {
            let texts = batch
                .iter()
                .map(|document| document.text.clone())
                .collect::<Vec<_>>();
            let mut embedded = self.embedder.embed(&texts).await?;
            if embedded.len() != batch.len() {
                anyhow::bail!(
                    "embedder returned {} vectors for {} texts",
                    embedded.len(),
                    batch.len()
                );
            }
            embeddings.append(&mut embedded);
        }

        let chunk_ids = chunks
            .iter()
            .map(|document| document.id.clone())
            .collect::<Vec<_>>();
        self.store.upsert(chunks, embeddings).await?;
        Ok(IngestionReport {
            source_documents,
            chunks_indexed: chunk_ids.len(),
            chunk_ids,
        })
    }

    pub async fn ingest_from(
        &self,
        loader: &dyn DocumentLoader,
    ) -> anyhow::Result<IngestionReport> {
        self.ingest(loader.load().await?).await
    }

    pub async fn delete_source(&self, source_id: &str) -> anyhow::Result<usize> {
        let ids = self
            .store
            .documents()
            .await?
            .into_iter()
            .filter(|document| {
                metadata_value(&document.metadata, "_rag.source_id").and_then(Value::as_str)
                    == Some(source_id)
            })
            .map(|document| document.id)
            .collect::<Vec<_>>();
        self.store.delete(&ids).await
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum RetrievalStrategy {
    #[default]
    Vector,
    Hybrid {
        vector_weight: f32,
        lexical_weight: f32,
    },
}

#[derive(Debug, Clone)]
pub struct RetrievalOptions {
    pub limit: usize,
    pub min_score: Option<f32>,
    pub filter: Option<MetadataFilter>,
    pub strategy: RetrievalStrategy,
    /// Number of candidates gathered per requested result before reranking.
    pub candidate_multiplier: usize,
}

impl RetrievalOptions {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            ..Self::default()
        }
    }
}

impl Default for RetrievalOptions {
    fn default() -> Self {
        Self {
            limit: 5,
            min_score: None,
            filter: None,
            strategy: RetrievalStrategy::Vector,
            candidate_multiplier: 4,
        }
    }
}

#[async_trait]
pub trait QueryTransformer: Send + Sync {
    /// Return one or more queries. Retrieval deduplicates results across them
    /// and always retains the original query as a safe fallback.
    async fn transform(&self, query: &str) -> anyhow::Result<Vec<String>>;
}

pub struct IdentityQueryTransformer;

#[async_trait]
impl QueryTransformer for IdentityQueryTransformer {
    async fn transform(&self, query: &str) -> anyhow::Result<Vec<String>> {
        Ok(vec![query.to_owned()])
    }
}

/// An adapter for LLM-backed or application-specific multi-query expansion.
pub struct FunctionQueryTransformer {
    function: Arc<dyn Fn(String) -> BoxFuture<'static, anyhow::Result<Vec<String>>> + Send + Sync>,
}

impl FunctionQueryTransformer {
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(String) -> BoxFuture<'static, anyhow::Result<Vec<String>>> + Send + Sync + 'static,
    {
        Self {
            function: Arc::new(function),
        }
    }
}

#[async_trait]
impl QueryTransformer for FunctionQueryTransformer {
    async fn transform(&self, query: &str) -> anyhow::Result<Vec<String>> {
        (self.function)(query.to_owned()).await
    }
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<RetrievedDocument>,
    ) -> anyhow::Result<Vec<RetrievedDocument>>;
}

/// A deterministic local reranker that blends the retriever score with token
/// coverage and an exact-phrase bonus.
pub struct LexicalReranker {
    original_score_weight: f32,
}

impl LexicalReranker {
    pub fn new(original_score_weight: f32) -> Self {
        Self {
            original_score_weight: original_score_weight.clamp(0.0, 1.0),
        }
    }
}

impl Default for LexicalReranker {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[async_trait]
impl Reranker for LexicalReranker {
    async fn rerank(
        &self,
        query: &str,
        mut documents: Vec<RetrievedDocument>,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        for result in &mut documents {
            let lexical = token_coverage(query, &result.document.text);
            result.score = self.original_score_weight * result.score
                + (1.0 - self.original_score_weight) * lexical;
        }
        sort_results(&mut documents);
        Ok(documents)
    }
}

pub struct Retriever {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    strategy: RetrievalStrategy,
    candidate_multiplier: usize,
    query_transformer: Arc<dyn QueryTransformer>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl Retriever {
    pub fn new(embedder: impl Embedder + 'static, store: impl VectorStore + 'static) -> Self {
        Self::from_shared(Arc::new(embedder), Arc::new(store))
    }

    pub fn from_shared(embedder: Arc<dyn Embedder>, store: Arc<dyn VectorStore>) -> Self {
        Self {
            embedder,
            store,
            strategy: RetrievalStrategy::Vector,
            candidate_multiplier: 4,
            query_transformer: Arc::new(IdentityQueryTransformer),
            reranker: None,
        }
    }

    pub fn hybrid(mut self, vector_weight: f32, lexical_weight: f32) -> Self {
        self.strategy = RetrievalStrategy::Hybrid {
            vector_weight,
            lexical_weight,
        };
        self
    }

    pub fn candidate_multiplier(mut self, multiplier: usize) -> Self {
        self.candidate_multiplier = multiplier.max(1);
        self
    }

    pub fn with_query_transformer(mut self, transformer: impl QueryTransformer + 'static) -> Self {
        self.query_transformer = Arc::new(transformer);
        self
    }

    pub fn with_reranker(mut self, reranker: impl Reranker + 'static) -> Self {
        self.reranker = Some(Arc::new(reranker));
        self
    }

    pub fn ingestion_pipeline(&self, chunker: impl Chunker + 'static) -> IngestionPipeline {
        IngestionPipeline::from_shared(self.embedder.clone(), self.store.clone(), Arc::new(chunker))
    }

    /// Append documents, preserving the behavior of the original API.
    pub async fn index(&self, documents: Vec<Document>) -> anyhow::Result<()> {
        let texts = documents
            .iter()
            .map(|document| document.text.clone())
            .collect::<Vec<_>>();
        self.store
            .add(documents, self.embedder.embed(&texts).await?)
            .await
    }

    pub async fn upsert(&self, documents: Vec<Document>) -> anyhow::Result<()> {
        validate_document_ids(&documents)?;
        let texts = documents
            .iter()
            .map(|document| document.text.clone())
            .collect::<Vec<_>>();
        self.store
            .upsert(documents, self.embedder.embed(&texts).await?)
            .await
    }

    pub async fn delete(&self, ids: &[String]) -> anyhow::Result<usize> {
        self.store.delete(ids).await
    }

    pub async fn retrieve(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        self.retrieve_with_options(
            query,
            RetrievalOptions {
                limit,
                strategy: self.strategy,
                candidate_multiplier: self.candidate_multiplier,
                ..RetrievalOptions::default()
            },
        )
        .await
    }

    pub async fn retrieve_with_options(
        &self,
        query: &str,
        options: RetrievalOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        if options.limit == 0 {
            return Ok(Vec::new());
        }
        validate_retrieval_options(&options)?;
        let mut queries = self.query_transformer.transform(query).await?;
        queries.push(query.to_owned());
        let mut seen_queries = HashSet::new();
        queries.retain(|candidate| {
            let candidate = candidate.trim();
            !candidate.is_empty() && seen_queries.insert(candidate.to_lowercase())
        });

        let candidate_limit = options
            .limit
            .saturating_mul(options.candidate_multiplier.max(1));
        let mut combined: HashMap<String, RetrievedDocument> = HashMap::new();
        for transformed_query in queries {
            let results = self
                .retrieve_single(&transformed_query, candidate_limit, &options)
                .await?;
            for result in results {
                combined
                    .entry(result.document.id.clone())
                    .and_modify(|current| current.score = current.score.max(result.score))
                    .or_insert(result);
            }
        }

        let mut results = combined.into_values().collect::<Vec<_>>();
        sort_results(&mut results);
        results.truncate(candidate_limit);
        if let Some(reranker) = &self.reranker {
            results = reranker.rerank(query, results).await?;
        }
        if let Some(min_score) = options.min_score {
            results.retain(|result| result.score >= min_score);
        }
        sort_results(&mut results);
        results.truncate(options.limit);
        Ok(results)
    }

    async fn retrieve_single(
        &self,
        query: &str,
        candidate_limit: usize,
        options: &RetrievalOptions,
    ) -> anyhow::Result<Vec<RetrievedDocument>> {
        let embedding = self
            .embedder
            .embed(&[query.to_owned()])
            .await?
            .pop()
            .unwrap_or_default();
        let search_options = SearchOptions {
            limit: candidate_limit,
            min_score: None,
            filter: options.filter.clone(),
        };
        let vector_results = self
            .store
            .search_with_options(&embedding, &search_options)
            .await?;
        match options.strategy {
            RetrievalStrategy::Vector => Ok(vector_results),
            RetrievalStrategy::Hybrid {
                vector_weight,
                lexical_weight,
            } => {
                let documents = self
                    .store
                    .documents()
                    .await?
                    .into_iter()
                    .filter(|document| {
                        options
                            .filter
                            .as_ref()
                            .is_none_or(|filter| filter.matches(&document.metadata))
                    })
                    .collect::<Vec<_>>();
                Ok(hybrid_results(
                    query,
                    documents,
                    vector_results,
                    vector_weight,
                    lexical_weight,
                    candidate_limit,
                ))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    pub index: usize,
    pub document_id: String,
    pub source: Option<String>,
    pub score: f32,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FormattedContext {
    pub context: String,
    pub citations: Vec<Citation>,
}

#[derive(Debug, Clone)]
pub struct ContextFormatter {
    pub max_chars: usize,
    pub include_scores: bool,
}

impl Default for ContextFormatter {
    fn default() -> Self {
        Self {
            max_chars: 12_000,
            include_scores: false,
        }
    }
}

impl ContextFormatter {
    pub fn format(&self, documents: &[RetrievedDocument]) -> FormattedContext {
        let mut context = String::new();
        let mut citations = Vec::new();
        for result in documents {
            if context.chars().count() >= self.max_chars {
                break;
            }
            let index = citations.len() + 1;
            let source = citation_source(&result.document.metadata);
            let heading = if self.include_scores {
                format!("[{index}] (score: {:.4})\n", result.score)
            } else {
                format!("[{index}]\n")
            };
            let separator = if context.is_empty() { "" } else { "\n\n" };
            let reserved = separator.chars().count() + heading.chars().count();
            let remaining = self
                .max_chars
                .saturating_sub(context.chars().count() + reserved);
            if remaining == 0 {
                break;
            }
            let excerpt = take_chars(&result.document.text, remaining);
            context.push_str(separator);
            context.push_str(&heading);
            context.push_str(&excerpt);
            citations.push(Citation {
                index,
                document_id: result.document.id.clone(),
                source,
                score: result.score,
                metadata: result.document.metadata.clone(),
            });
            if excerpt.chars().count() < result.document.text.chars().count() {
                break;
            }
        }
        FormattedContext { context, citations }
    }
}

pub struct RetrieverTool {
    retriever: Arc<Retriever>,
    limit: usize,
}

impl RetrieverTool {
    pub fn new(retriever: Arc<Retriever>, limit: usize) -> Self {
        Self { retriever, limit }
    }
}

#[async_trait]
impl Tool for RetrieverTool {
    fn name(&self) -> &str {
        "retrieve_documents"
    }

    fn description(&self) -> &str {
        "Retrieve relevant indexed documents for a query"
    }

    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})
    }

    async fn execute(&self, args: Value) -> anyhow::Result<String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("query is required"))?;
        Ok(serde_json::to_string(
            &self
                .retriever
                .retrieve(query, self.limit)
                .await?
                .into_iter()
                .map(|result| json!({"score":result.score,"document":result.document}))
                .collect::<Vec<_>>(),
        )?)
    }
}

fn make_entries(
    documents: Vec<Document>,
    embeddings: Vec<Vec<f32>>,
    expected: Option<usize>,
) -> anyhow::Result<Vec<StoredEntry>> {
    if documents.len() != embeddings.len() {
        anyhow::bail!("documents and embeddings length mismatch");
    }
    let mut dimension = expected;
    let mut ids = HashSet::new();
    let mut entries = Vec::with_capacity(documents.len());
    for (document, embedding) in documents.into_iter().zip(embeddings) {
        if document.id.trim().is_empty() {
            anyhow::bail!("document id cannot be empty");
        }
        if !ids.insert(document.id.clone()) {
            anyhow::bail!("duplicate document id '{}' in mutation", document.id);
        }
        if embedding.is_empty() {
            anyhow::bail!("embedding for '{}' cannot be empty", document.id);
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            anyhow::bail!(
                "embedding for '{}' contains a non-finite value",
                document.id
            );
        }
        match dimension {
            Some(expected) if expected != embedding.len() => anyhow::bail!(
                "embedding dimension mismatch: expected {expected}, got {} for '{}'",
                embedding.len(),
                document.id
            ),
            None => dimension = Some(embedding.len()),
            _ => {}
        }
        entries.push(StoredEntry {
            document,
            embedding,
        });
    }
    Ok(entries)
}

fn validate_document_ids(documents: &[Document]) -> anyhow::Result<()> {
    let mut ids = HashSet::new();
    for document in documents {
        if document.id.trim().is_empty() {
            anyhow::bail!("document id cannot be empty");
        }
        if !ids.insert(document.id.as_str()) {
            anyhow::bail!("duplicate document id '{}'", document.id);
        }
    }
    Ok(())
}

fn ensure_ids_absent(existing: &[StoredEntry], incoming: &[StoredEntry]) -> anyhow::Result<()> {
    let existing_ids = existing
        .iter()
        .map(|entry| entry.document.id.as_str())
        .collect::<HashSet<_>>();
    if let Some(entry) = incoming
        .iter()
        .find(|entry| existing_ids.contains(entry.document.id.as_str()))
    {
        anyhow::bail!(
            "document id '{}' already exists; use upsert to replace it",
            entry.document.id
        );
    }
    Ok(())
}

fn expected_dimension(entries: &[StoredEntry]) -> Option<usize> {
    entries.first().map(|entry| entry.embedding.len())
}

fn validate_query_dimension(entries: &[StoredEntry], embedding: &[f32]) -> anyhow::Result<()> {
    if let Some(expected) = expected_dimension(entries) {
        if embedding.len() != expected {
            anyhow::bail!(
                "query embedding dimension mismatch: expected {expected}, got {}",
                embedding.len()
            );
        }
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("query embedding contains a non-finite value");
    }
    Ok(())
}

fn score_entries(
    entries: &[StoredEntry],
    embedding: &[f32],
    filter: Option<&MetadataFilter>,
) -> Vec<RetrievedDocument> {
    entries
        .iter()
        .filter(|entry| filter.is_none_or(|filter| filter.matches(&entry.document.metadata)))
        .map(|entry| RetrievedDocument {
            document: entry.document.clone(),
            score: dot(embedding, &entry.embedding),
        })
        .collect()
}

fn apply_search_options(results: &mut Vec<RetrievedDocument>, options: &SearchOptions) {
    if let Some(filter) = &options.filter {
        results.retain(|result| filter.matches(&result.document.metadata));
    }
    if let Some(min_score) = options.min_score {
        results.retain(|result| result.score >= min_score);
    }
    sort_results(results);
    results.truncate(options.limit);
}

fn sort_results(results: &mut [RetrievedDocument]) {
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.document.id.cmp(&right.document.id))
    });
}

fn metadata_value<'a>(metadata: &'a Value, field: &str) -> Option<&'a Value> {
    if field.is_empty() {
        return Some(metadata);
    }
    field
        .split('.')
        .try_fold(metadata, |value, segment| value.get(segment))
}

fn add_lineage_metadata(document: &mut Document, source_id: &str, index: usize) {
    if !document.metadata.is_object() {
        let original = std::mem::replace(&mut document.metadata, Value::Null);
        document.metadata = Value::Object(Map::from_iter([("value".to_owned(), original)]));
    }
    document.metadata.as_object_mut().unwrap().insert(
        "_rag".to_owned(),
        json!({"source_id": source_id, "chunk_index": index}),
    );
}

fn persist_snapshot(path: &Path, entries: &[StoredEntry]) -> anyhow::Result<()> {
    let snapshot = VectorStoreSnapshot {
        version: VECTOR_STORE_VERSION,
        entries: entries.to_vec(),
    };
    let bytes = serde_json::to_vec_pretty(&snapshot)?;
    write_atomic(path, &bytes)?;
    // The primary is already durable. A recovery-copy failure should not make
    // the caller believe the committed primary mutation failed.
    let _ = write_atomic(&backup_path(path), &bytes);
    Ok(())
}

fn load_snapshot_with_recovery(path: &Path) -> anyhow::Result<Vec<StoredEntry>> {
    match fs::read(path) {
        Ok(bytes) => match decode_snapshot(path, &bytes) {
            Ok(entries) => Ok(entries),
            Err(primary_error) => recover_snapshot(path, primary_error),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let backup = backup_path(path);
            match fs::read(&backup) {
                Ok(bytes) => {
                    let entries = decode_snapshot(&backup, &bytes)?;
                    write_atomic(path, &bytes)?;
                    Ok(entries)
                }
                Err(backup_error) if backup_error.kind() == std::io::ErrorKind::NotFound => {
                    Ok(Vec::new())
                }
                Err(backup_error) => Err(backup_error)
                    .with_context(|| format!("failed to read vector store backup {backup:?}")),
            }
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to read vector store snapshot {path:?}"))
        }
    }
}

fn recover_snapshot(path: &Path, primary_error: anyhow::Error) -> anyhow::Result<Vec<StoredEntry>> {
    let backup = backup_path(path);
    let bytes = fs::read(&backup).with_context(|| {
        format!(
            "vector store primary {path:?} is invalid ({primary_error:#}) and backup {backup:?} could not be read"
        )
    })?;
    let entries = decode_snapshot(&backup, &bytes).with_context(|| {
        format!(
            "vector store primary {path:?} is invalid ({primary_error:#}) and backup is invalid"
        )
    })?;
    write_atomic(path, &bytes).context("failed to restore vector store primary from backup")?;
    Ok(entries)
}

fn decode_snapshot(path: &Path, bytes: &[u8]) -> anyhow::Result<Vec<StoredEntry>> {
    let snapshot: VectorStoreSnapshot = serde_json::from_slice(bytes)
        .with_context(|| format!("invalid vector store JSON in {path:?}"))?;
    if snapshot.version != VECTOR_STORE_VERSION {
        anyhow::bail!(
            "unsupported vector store version {} in {:?}; expected {}",
            snapshot.version,
            path,
            VECTOR_STORE_VERSION
        );
    }
    let documents = snapshot
        .entries
        .iter()
        .map(|entry| entry.document.clone())
        .collect::<Vec<_>>();
    let embeddings = snapshot
        .entries
        .iter()
        .map(|entry| entry.embedding.clone())
        .collect::<Vec<_>>();
    make_entries(documents, embeddings, None)
        .with_context(|| format!("invalid vector store entries in {path:?}"))
}

fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".bak");
    PathBuf::from(name)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    let mut temporary_name = path.as_os_str().to_os_string();
    temporary_name.push(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let temporary = PathBuf::from(temporary_name);
    let result = (|| -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        if let Some(parent) = parent {
            let _ = File::open(parent).and_then(|directory| directory.sync_all());
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.with_context(|| format!("failed to persist vector store snapshot {path:?}"))
}

fn validate_retrieval_options(options: &RetrievalOptions) -> anyhow::Result<()> {
    if let Some(score) = options.min_score {
        if !score.is_finite() {
            anyhow::bail!("minimum retrieval score must be finite");
        }
    }
    if let RetrievalStrategy::Hybrid {
        vector_weight,
        lexical_weight,
    } = options.strategy
    {
        if !vector_weight.is_finite()
            || !lexical_weight.is_finite()
            || vector_weight < 0.0
            || lexical_weight < 0.0
            || vector_weight + lexical_weight <= 0.0
        {
            anyhow::bail!("hybrid weights must be finite, non-negative, and have a positive sum");
        }
    }
    Ok(())
}

fn hybrid_results(
    query: &str,
    documents: Vec<Document>,
    vector_results: Vec<RetrievedDocument>,
    vector_weight: f32,
    lexical_weight: f32,
    limit: usize,
) -> Vec<RetrievedDocument> {
    let weight_sum = vector_weight + lexical_weight;
    let vector_weight = vector_weight / weight_sum;
    let lexical_weight = lexical_weight / weight_sum;
    let lexical = lexical_scores(query, &documents);
    let vector = vector_results
        .into_iter()
        .map(|result| (result.document.id, result.score))
        .collect::<HashMap<_, _>>();
    let mut results = documents
        .into_iter()
        .map(|document| RetrievedDocument {
            score: vector_weight * vector.get(&document.id).copied().unwrap_or_default()
                + lexical_weight * lexical.get(&document.id).copied().unwrap_or_default(),
            document,
        })
        .collect::<Vec<_>>();
    sort_results(&mut results);
    results.truncate(limit);
    results
}

fn lexical_scores(query: &str, documents: &[Document]) -> HashMap<String, f32> {
    let query_terms = tokenize(query);
    if query_terms.is_empty() || documents.is_empty() {
        return HashMap::new();
    }
    let tokenized = documents
        .iter()
        .map(|document| tokenize(&document.text))
        .collect::<Vec<_>>();
    let average_length =
        tokenized.iter().map(Vec::len).sum::<usize>().max(1) as f32 / tokenized.len() as f32;
    let document_count = documents.len() as f32;
    let unique_query_terms = query_terms.iter().collect::<HashSet<_>>();
    let mut raw = Vec::with_capacity(documents.len());
    for terms in &tokenized {
        let frequencies = term_frequencies(terms);
        let length = terms.len() as f32;
        let mut score = 0.0;
        for term in &unique_query_terms {
            let frequency = frequencies.get(term.as_str()).copied().unwrap_or_default() as f32;
            if frequency == 0.0 {
                continue;
            }
            let containing = tokenized
                .iter()
                .filter(|candidate| {
                    candidate
                        .iter()
                        .any(|value| value.as_str() == term.as_str())
                })
                .count() as f32;
            let inverse_document_frequency =
                ((document_count - containing + 0.5) / (containing + 0.5) + 1.0).ln();
            let denominator = frequency + 1.2 * (0.25 + 0.75 * length / average_length);
            score += inverse_document_frequency * (frequency * 2.2) / denominator;
        }
        raw.push(score);
    }
    let maximum = raw.iter().copied().fold(0.0_f32, f32::max);
    documents
        .iter()
        .zip(raw)
        .map(|(document, score)| {
            (
                document.id.clone(),
                if maximum > 0.0 { score / maximum } else { 0.0 },
            )
        })
        .collect()
}

fn term_frequencies(terms: &[String]) -> HashMap<&str, usize> {
    let mut frequencies = HashMap::new();
    for term in terms {
        *frequencies.entry(term.as_str()).or_default() += 1;
    }
    frequencies
}

fn token_coverage(query: &str, text: &str) -> f32 {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();
    let query_terms = tokenize(&query_lower).into_iter().collect::<HashSet<_>>();
    if query_terms.is_empty() {
        return 0.0;
    }
    let text_terms = tokenize(&text_lower).into_iter().collect::<HashSet<_>>();
    let coverage = query_terms.intersection(&text_terms).count() as f32 / query_terms.len() as f32;
    if text_lower.contains(&query_lower) {
        (coverage + 0.2).min(1.0)
    } else {
        coverage
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn citation_source(metadata: &Value) -> Option<String> {
    ["source", "url", "title"]
        .iter()
        .find_map(|field| metadata.get(field).and_then(Value::as_str))
        .map(str::to_owned)
}

fn take_chars(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
