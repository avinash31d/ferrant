//! Crash-safe local persistence primitives and typed observability stores.
//!
//! [`AtomicJsonFile`] is appropriate for current snapshots or baselines.
//! [`DurableJsonlStore`] is an append-only, checksummed record log. Appends are
//! fsynced; a torn final record is removed on the next read, while corruption
//! in the middle of a file is surfaced rather than silently skipped.

use crate::evaluation::EvaluationReport;
use crate::observability::{MetricRecord, OpenTelemetrySpan, UsageRecord};
use crate::tracing::TraceEvent;
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

/// A JSON snapshot written with write+fsync+atomic-rename semantics.
pub struct AtomicJsonFile<T> {
    path: PathBuf,
    gate: Arc<tokio::sync::Mutex<()>>,
    marker: PhantomData<fn() -> T>,
}

impl<T> Clone for AtomicJsonFile<T> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            gate: self.gate.clone(),
            marker: PhantomData,
        }
    }
}

impl<T> AtomicJsonFile<T>
where
    T: Serialize + DeserializeOwned + Send + Sync,
{
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            marker: PhantomData,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn load(&self) -> anyhow::Result<Option<T>> {
        let _guard = self.gate.lock().await;
        match tokio::fs::read(&self.path).await {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub async fn save(&self, value: &T) -> anyhow::Result<()> {
        let bytes = serde_json::to_vec_pretty(value)?;
        let _guard = self.gate.lock().await;
        let parent = parent_directory(&self.path);
        tokio::fs::create_dir_all(parent).await?;
        let temporary = self
            .path
            .with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
        let result = async {
            let mut file = tokio::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)
                .await?;
            file.write_all(&bytes).await?;
            file.flush().await?;
            file.sync_all().await?;
            drop(file);
            tokio::fs::rename(&temporary, &self.path).await?;
            sync_directory(parent).await?;
            anyhow::Ok(())
        }
        .await;
        if result.is_err() {
            let _ = tokio::fs::remove_file(&temporary).await;
        }
        result
    }

    pub async fn clear(&self) -> anyhow::Result<()> {
        let _guard = self.gate.lock().await;
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => {
                sync_directory(parent_directory(&self.path)).await?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

async fn sync_directory(path: &Path) -> anyhow::Result<()> {
    // Directory fsync makes a preceding rename durable on Unix filesystems.
    // Some platforms do not permit opening directories, so callers still get
    // atomic rename behavior there and we tolerate PermissionDenied.
    match tokio::fs::File::open(path).await {
        Ok(directory) => match directory.sync_all().await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => Ok(()),
            Err(error) => Err(error.into()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredRecord<T> {
    pub version: u8,
    pub sequence: u64,
    pub id: String,
    pub timestamp_ms: u64,
    pub checksum: String,
    pub payload: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendOutcome {
    Appended { sequence: u64 },
    AlreadyExists { sequence: u64 },
}

#[derive(Default)]
struct LogState {
    initialized: bool,
    next_sequence: u64,
    ids: BTreeMap<String, u64>,
}

/// An append-only store for a single writer (and its clones) per path.
/// `append_idempotent` makes retrying a logical write safe after ambiguous
/// process or network failures.
pub struct DurableJsonlStore<T> {
    path: PathBuf,
    state: Arc<tokio::sync::Mutex<LogState>>,
    marker: PhantomData<fn() -> T>,
}

impl<T> Clone for DurableJsonlStore<T> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            state: self.state.clone(),
            marker: PhantomData,
        }
    }
}

impl<T> DurableJsonlStore<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            state: Arc::new(tokio::sync::Mutex::new(LogState::default())),
            marker: PhantomData,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn append(&self, payload: T) -> anyhow::Result<AppendOutcome> {
        self.append_idempotent(uuid::Uuid::new_v4().to_string(), payload)
            .await
    }

    pub async fn append_idempotent(
        &self,
        id: impl Into<String>,
        payload: T,
    ) -> anyhow::Result<AppendOutcome> {
        let id = id.into();
        if id.trim().is_empty() {
            anyhow::bail!("record id must not be empty");
        }
        let mut state = self.state.lock().await;
        self.initialize(&mut state).await?;
        if let Some(sequence) = state.ids.get(&id) {
            return Ok(AppendOutcome::AlreadyExists {
                sequence: *sequence,
            });
        }

        let sequence = state.next_sequence;
        let timestamp_ms = now_ms();
        let checksum = record_checksum(sequence, &id, timestamp_ms, &payload)?;
        let record = StoredRecord {
            version: 1,
            sequence,
            id: id.clone(),
            timestamp_ms,
            checksum,
            payload,
        };
        let mut bytes = serde_json::to_vec(&record)?;
        bytes.push(b'\n');
        tokio::fs::create_dir_all(parent_directory(&self.path)).await?;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        file.write_all(&bytes).await?;
        file.flush().await?;
        file.sync_data().await?;
        state.ids.insert(id, sequence);
        state.next_sequence = sequence.saturating_add(1);
        Ok(AppendOutcome::Appended { sequence })
    }

    /// Read valid records, automatically truncating a torn final append.
    pub async fn records(&self) -> anyhow::Result<Vec<StoredRecord<T>>> {
        let mut state = self.state.lock().await;
        let records = self.read_and_recover().await?;
        rebuild_state(&mut state, &records);
        Ok(records)
    }

    pub async fn payloads(&self) -> anyhow::Result<Vec<T>> {
        Ok(self
            .records()
            .await?
            .into_iter()
            .map(|record| record.payload)
            .collect())
    }

    pub async fn clear(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock().await;
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => sync_directory(parent_directory(&self.path)).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        *state = LogState::default();
        Ok(())
    }

    async fn initialize(&self, state: &mut LogState) -> anyhow::Result<()> {
        if !state.initialized {
            let records = self.read_and_recover().await?;
            rebuild_state(state, &records);
        }
        Ok(())
    }

    async fn read_and_recover(&self) -> anyhow::Result<Vec<StoredRecord<T>>> {
        let bytes = match tokio::fs::read(&self.path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut records: Vec<StoredRecord<T>> = Vec::new();
        let mut cursor = 0usize;
        while cursor < bytes.len() {
            let newline = bytes[cursor..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map(|offset| cursor + offset);
            let line_end = newline.unwrap_or(bytes.len());
            let next = newline.map_or(bytes.len(), |position| position + 1);
            let mut line = &bytes[cursor..line_end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
            let parsed = parse_record(line, records.last().map(|record| record.sequence));
            match parsed {
                Ok(record) => records.push(record),
                Err(error) => {
                    let is_final = bytes[next..].iter().all(u8::is_ascii_whitespace);
                    if is_final {
                        let file = tokio::fs::OpenOptions::new()
                            .write(true)
                            .open(&self.path)
                            .await?;
                        file.set_len(cursor as u64).await?;
                        file.sync_all().await?;
                        break;
                    }
                    anyhow::bail!(
                        "corrupt JSONL record at byte {cursor} in '{}': {error}",
                        self.path.display()
                    );
                }
            }
            cursor = next;
        }
        Ok(records)
    }
}

fn rebuild_state<T>(state: &mut LogState, records: &[StoredRecord<T>]) {
    state.initialized = true;
    state.next_sequence = records
        .last()
        .map_or(1, |record| record.sequence.saturating_add(1));
    state.ids = records
        .iter()
        .map(|record| (record.id.clone(), record.sequence))
        .collect();
}

fn parse_record<T>(line: &[u8], previous_sequence: Option<u64>) -> anyhow::Result<StoredRecord<T>>
where
    T: Serialize + DeserializeOwned,
{
    if line.is_empty() {
        anyhow::bail!("empty record");
    }
    let record: StoredRecord<T> = serde_json::from_slice(line)?;
    if record.version != 1 {
        anyhow::bail!("unsupported record version {}", record.version);
    }
    let expected_sequence = previous_sequence.map_or(1, |sequence| sequence.saturating_add(1));
    if record.sequence != expected_sequence {
        anyhow::bail!(
            "expected sequence {expected_sequence}, found {}",
            record.sequence
        );
    }
    let checksum = record_checksum(
        record.sequence,
        &record.id,
        record.timestamp_ms,
        &record.payload,
    )?;
    if checksum != record.checksum {
        anyhow::bail!("checksum mismatch");
    }
    Ok(record)
}

#[derive(Serialize)]
struct ChecksumBody<'a, T> {
    version: u8,
    sequence: u64,
    id: &'a str,
    timestamp_ms: u64,
    payload: &'a T,
}

fn record_checksum<T: Serialize>(
    sequence: u64,
    id: &str,
    timestamp_ms: u64,
    payload: &T,
) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(&ChecksumBody {
        version: 1,
        sequence,
        id,
        timestamp_ms,
        payload,
    })?;
    // FNV-1a detects torn/corrupted local records. It is an integrity check,
    // not a cryptographic authenticity mechanism.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

#[async_trait]
pub trait PersistentRecordStore<T>: Send + Sync {
    async fn append_record(&self, record: T) -> anyhow::Result<AppendOutcome>;
    async fn load_records(&self) -> anyhow::Result<Vec<T>>;
}

#[async_trait]
impl<T> PersistentRecordStore<T> for DurableJsonlStore<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    async fn append_record(&self, record: T) -> anyhow::Result<AppendOutcome> {
        self.append(record).await
    }

    async fn load_records(&self) -> anyhow::Result<Vec<T>> {
        self.payloads().await
    }
}

pub type TraceLogStore = DurableJsonlStore<TraceEvent>;
pub type UsageLogStore = DurableJsonlStore<UsageRecord>;
pub type MetricsLogStore = DurableJsonlStore<MetricRecord>;
pub type OpenTelemetryLogStore = DurableJsonlStore<OpenTelemetrySpan>;
pub type EvaluationReportLogStore = DurableJsonlStore<EvaluationReport>;
pub type EvaluationBaselineStore = AtomicJsonFile<EvaluationReport>;
