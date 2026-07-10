use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Component, Path, PathBuf};

const MAX_FILE_BYTES: u64 = 1_048_576;
const MAX_TREE_ENTRIES: usize = 500;

#[derive(Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn open(root: PathBuf) -> Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("cannot open workspace {}", root.display()))?;
        if !root.is_dir() {
            bail!("workspace is not a directory: {}", root.display());
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list(&self) -> Result<String> {
        let mut entries = Vec::new();
        self.walk(&self.root, 0, &mut entries)?;
        Ok(if entries.is_empty() {
            "workspace is empty".to_string()
        } else {
            entries.join("\n")
        })
    }

    pub fn read(&self, relative: &str) -> Result<String> {
        self.reject_sensitive(relative)?;
        let path = self.resolve_existing(relative)?;
        if !path.is_file() {
            bail!("not a file: {relative}");
        }
        let size = fs::metadata(&path)?.len();
        if size > MAX_FILE_BYTES {
            bail!("file is too large ({size} bytes; limit is {MAX_FILE_BYTES})");
        }
        fs::read_to_string(&path).with_context(|| format!("cannot read text file {relative}"))
    }

    pub fn write_atomic(&self, relative: &str, content: &str) -> Result<()> {
        self.reject_sensitive(relative)?;
        if content.len() as u64 > MAX_FILE_BYTES {
            bail!("content exceeds the {MAX_FILE_BYTES}-byte limit");
        }
        let path = self.resolve_for_write(relative)?;
        let parent = path.parent().context("file has no parent")?;
        fs::create_dir_all(parent)?;
        let temp = parent.join(format!(".coding-agent-{}.tmp", std::process::id()));
        fs::write(&temp, content)?;
        fs::rename(&temp, &path).with_context(|| format!("cannot replace {}", path.display()))?;
        Ok(())
    }

    pub fn old_content(&self, relative: &str) -> Result<Option<String>> {
        let candidate = self.join_validated(relative)?;
        if candidate.exists() {
            Ok(Some(self.read(relative)?))
        } else {
            Ok(None)
        }
    }

    pub fn validate_relative(&self, relative: &str) -> Result<()> {
        self.reject_sensitive(relative)?;
        self.join_validated(relative)?;
        Ok(())
    }

    pub fn executable(&self, relative: &str) -> Result<String> {
        self.reject_sensitive(relative)?;
        let path = self.resolve_existing(relative)?;
        if !path.is_file() {
            bail!("executable path is not a file: {relative}");
        }
        Ok(path.to_string_lossy().to_string())
    }

    fn resolve_existing(&self, relative: &str) -> Result<PathBuf> {
        let candidate = self.join_validated(relative)?;
        let resolved = candidate
            .canonicalize()
            .with_context(|| format!("path does not exist: {relative}"))?;
        self.ensure_inside(&resolved)?;
        Ok(resolved)
    }

    fn resolve_for_write(&self, relative: &str) -> Result<PathBuf> {
        let candidate = self.join_validated(relative)?;
        if let Ok(metadata) = fs::symlink_metadata(&candidate) {
            if metadata.file_type().is_symlink() {
                bail!("refusing to write through a symlink: {relative}");
            }
            let resolved = candidate.canonicalize()?;
            self.ensure_inside(&resolved)?;
            return Ok(resolved);
        }

        let mut ancestor = candidate.parent().context("path has no parent")?;
        while !ancestor.exists() {
            ancestor = ancestor.parent().context("path has no existing ancestor")?;
        }
        let resolved_parent = ancestor.canonicalize()?;
        self.ensure_inside(&resolved_parent)?;
        Ok(candidate)
    }

    fn join_validated(&self, relative: &str) -> Result<PathBuf> {
        if relative.is_empty() || relative.contains('\0') {
            bail!("path must be a non-empty relative path");
        }
        let path = Path::new(relative);
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            bail!("path must stay inside the workspace: {relative}");
        }
        Ok(self.root.join(path))
    }

    fn ensure_inside(&self, path: &Path) -> Result<()> {
        if !path.starts_with(&self.root) {
            bail!("resolved path escapes the workspace: {}", path.display());
        }
        Ok(())
    }

    fn reject_sensitive(&self, relative: &str) -> Result<()> {
        let lower = relative.to_ascii_lowercase();
        let first = Path::new(relative).components().next();
        if matches!(first, Some(Component::Normal(name)) if name == ".git")
            || lower == ".env"
            || lower.ends_with("/.env")
            || lower == ".code-agent-cli.config"
            || lower.ends_with("/.code-agent-cli.config")
            || lower.contains("credential")
            || lower.contains("secret")
        {
            bail!("access to Git internals and credential files is blocked");
        }
        Ok(())
    }

    fn walk(&self, directory: &Path, depth: usize, output: &mut Vec<String>) -> Result<()> {
        if depth > 5 || output.len() >= MAX_TREE_ENTRIES {
            return Ok(());
        }
        let mut children = fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
        children.sort_by_key(|entry| entry.file_name());
        for child in children {
            if output.len() >= MAX_TREE_ENTRIES {
                output.push("... output truncated ...".to_string());
                break;
            }
            let path = child.path();
            let relative = path.strip_prefix(&self.root)?.to_string_lossy().to_string();
            if relative == ".git"
                || relative == ".env"
                || relative == ".code-agent-cli.config"
                || relative.starts_with(".git/")
            {
                continue;
            }
            let file_type = child.file_type()?;
            let suffix = if file_type.is_dir() {
                "/"
            } else if file_type.is_symlink() {
                "@"
            } else {
                ""
            };
            output.push(format!("{relative}{suffix}"));
            if file_type.is_dir() {
                self.walk(&path, depth + 1, output)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Workspace;
    use std::fs;

    #[test]
    fn confines_paths_and_blocks_secrets() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("safe.txt"), "safe").unwrap();
        fs::write(directory.path().join(".env"), "SECRET=value").unwrap();
        fs::write(
            directory.path().join(".code-agent-cli.config"),
            "api_key=secret",
        )
        .unwrap();
        let workspace = Workspace::open(directory.path().to_path_buf()).unwrap();

        assert_eq!(workspace.read("safe.txt").unwrap(), "safe");
        assert!(workspace.read("../escape").is_err());
        assert!(workspace.read("/etc/passwd").is_err());
        assert!(workspace.read("dir/../../escape").is_err());
        assert!(workspace.read(".env").is_err());
        assert!(workspace.read(".code-agent-cli.config").is_err());
        assert!(!workspace.list().unwrap().contains(".code-agent-cli.config"));
        assert!(workspace.write_atomic("../escape", "bad").is_err());
        workspace.write_atomic("nested/new.txt", "new").unwrap();
        assert_eq!(workspace.read("nested/new.txt").unwrap(), "new");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_that_escape_the_workspace() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("outside.txt"), "outside").unwrap();
        symlink(outside.path(), directory.path().join("escape")).unwrap();
        let workspace = Workspace::open(directory.path().to_path_buf()).unwrap();

        assert!(workspace.read("escape/outside.txt").is_err());
        assert!(workspace.write_atomic("escape/new.txt", "bad").is_err());
        assert!(!outside.path().join("new.txt").exists());
    }
}
