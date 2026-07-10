use anyhow::{Context, Result};
use std::io::{self, Write};
use std::sync::Mutex;

#[derive(Default)]
pub struct ApprovalGate {
    input_lock: Mutex<()>,
}

impl ApprovalGate {
    pub fn request(&self, action: &str, details: &str) -> Result<bool> {
        let _guard = self.input_lock.lock().expect("approval lock poisoned");
        println!("\nPermission requested: {action}");
        println!("  {details}");
        print!("Allow? [y/N]: ");
        io::stdout().flush().context("failed to flush stdout")?;

        let mut answer = String::new();
        io::stdin()
            .read_line(&mut answer)
            .context("failed to read approval")?;
        Ok(matches!(
            answer.trim().to_ascii_lowercase().as_str(),
            "y" | "yes"
        ))
    }
}
