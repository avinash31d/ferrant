use std::collections::BTreeSet;
use std::sync::Mutex;

#[derive(Default)]
pub struct SessionState {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    changed_files: BTreeSet<String>,
    verified_after_change: bool,
    last_verification: Option<String>,
}

impl SessionState {
    pub fn changed(&self, path: &str) {
        let mut inner = self.inner.lock().expect("state lock poisoned");
        inner.changed_files.insert(path.to_string());
        inner.verified_after_change = false;
    }

    pub fn verified(&self, command: String) {
        let mut inner = self.inner.lock().expect("state lock poisoned");
        inner.verified_after_change = true;
        inner.last_verification = Some(command);
    }

    pub fn needs_verification(&self) -> bool {
        let inner = self.inner.lock().expect("state lock poisoned");
        !inner.changed_files.is_empty() && !inner.verified_after_change
    }

    pub fn summary(&self) -> String {
        let inner = self.inner.lock().expect("state lock poisoned");
        let files = if inner.changed_files.is_empty() {
            "none".to_string()
        } else {
            inner
                .changed_files
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        };
        let verification = inner.last_verification.as_deref().unwrap_or("none");
        format!("Changed files: {files}\nLast successful verification: {verification}")
    }
}
