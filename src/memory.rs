use crate::message::Message;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Pluggable storage for conversation history, keyed by session id.
/// The default `InMemoryStorage` is process-local; implement this trait
/// yourself to back it with a database, file, or Redis.
#[async_trait]
pub trait Storage: Send + Sync {
    async fn load(&self, session_id: &str) -> anyhow::Result<Vec<Message>>;
    async fn save(&self, session_id: &str, messages: &[Message]) -> anyhow::Result<()>;
    async fn clear(&self, session_id: &str) -> anyhow::Result<()>;
}

#[derive(Default)]
pub struct InMemoryStorage {
    sessions: Mutex<HashMap<String, Vec<Message>>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    async fn load(&self, session_id: &str) -> anyhow::Result<Vec<Message>> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .get(session_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn save(&self, session_id: &str, messages: &[Message]) -> anyhow::Result<()> {
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.to_string(), messages.to_vec());
        Ok(())
    }

    async fn clear(&self, session_id: &str) -> anyhow::Result<()> {
        self.sessions.lock().unwrap().remove(session_id);
        Ok(())
    }
}

/// Durable session storage backed by atomically replaced JSON files.
/// Writes are serialized per instance and flushed before commit.
pub struct FileStorage {
    directory: PathBuf,
    write_lock: tokio::sync::Mutex<()>,
}

impl FileStorage {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            write_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn path(&self, session_id: &str) -> anyhow::Result<PathBuf> {
        if session_id.is_empty()
            || !session_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        {
            anyhow::bail!("session id may contain only letters, numbers, '-' and '_'");
        }
        Ok(self.directory.join(format!("{session_id}.json")))
    }
}

#[async_trait]
impl Storage for FileStorage {
    async fn load(&self, session_id: &str) -> anyhow::Result<Vec<Message>> {
        match tokio::fs::read(self.path(session_id)?).await {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
            Err(error) => Err(error.into()),
        }
    }

    async fn save(&self, session_id: &str, messages: &[Message]) -> anyhow::Result<()> {
        let _guard = self.write_lock.lock().await;
        tokio::fs::create_dir_all(&self.directory).await?;
        let path = self.path(session_id)?;
        let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let mut file = tokio::fs::File::create(&temporary).await?;
        use tokio::io::AsyncWriteExt;
        file.write_all(&serde_json::to_vec(messages)?).await?;
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&temporary, &path).await?;
        Ok(())
    }

    async fn clear(&self, session_id: &str) -> anyhow::Result<()> {
        let _guard = self.write_lock.lock().await;
        match tokio::fs::remove_file(self.path(session_id)?).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}
