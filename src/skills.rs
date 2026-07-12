use crate::tool::Tool;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use serde_yaml::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

static NEXT_GIT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub metadata: SkillMetadata,
    pub instructions: String,
    pub root: PathBuf,
    pub source: SkillSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillSource {
    Local {
        root: PathBuf,
    },
    GitHub {
        repository: String,
        git_ref: Option<String>,
        subdirectory: Option<PathBuf>,
        cache_dir: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkillLimits {
    pub max_instruction_bytes: usize,
    pub max_resource_bytes: usize,
}

impl Default for SkillLimits {
    fn default() -> Self {
        Self {
            max_instruction_bytes: 256 * 1024,
            max_resource_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill source or file error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid skill at {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("skill instructions at {path} exceed {limit} bytes")]
    InstructionTooLarge { path: PathBuf, limit: usize },
    #[error("duplicate skill name '{name}' in {first} and {second}")]
    DuplicateName {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("unknown skill '{name}'")]
    UnknownSkill { name: String },
    #[error("invalid resource path {path}")]
    InvalidResourcePath { path: PathBuf },
    #[error("resource {path} resolves outside skill root {root}")]
    ResourceOutsideRoot { path: PathBuf, root: PathBuf },
    #[error("resource at {path} is not a regular file")]
    ResourceNotFile { path: PathBuf },
    #[error("skill resource at {path} exceeds {limit} bytes")]
    ResourceTooLarge { path: PathBuf, limit: usize },
    #[error("skill resource at {path} is not UTF-8: {source}")]
    ResourceNotUtf8 {
        path: PathBuf,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("invalid Git repository URL: {repository}")]
    InvalidRepository { repository: String },
    #[error("git command failed: {command}: {message}")]
    Git { command: String, message: String },
    #[error("Git subdirectory {path} resolves outside checkout {root}")]
    GitSubdirectoryOutside { path: PathBuf, root: PathBuf },
}

#[derive(Clone, Debug, Default)]
pub struct SkillCatalog {
    skills: BTreeMap<String, Skill>,
    limits: SkillLimits,
}

impl SkillCatalog {
    pub fn load(sources: Vec<SkillSource>, limits: SkillLimits) -> Result<Self, SkillError> {
        Self::materialize(sources, limits, false)
    }

    pub fn refresh(sources: Vec<SkillSource>, limits: SkillLimits) -> Result<Self, SkillError> {
        Self::materialize(sources, limits, true)
    }

    fn materialize(
        sources: Vec<SkillSource>,
        limits: SkillLimits,
        refresh: bool,
    ) -> Result<Self, SkillError> {
        let mut files = Vec::new();
        for source in &sources {
            match source {
                SkillSource::Local { root } => discover(root, source, &mut files)?,
                SkillSource::GitHub { .. } => {
                    let root = materialize_git(source, refresh)?;
                    discover_git(&root, source, &mut files)?;
                }
            }
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));

        let mut skills: BTreeMap<String, Skill> = BTreeMap::new();
        for (path, source) in files {
            let skill = parse_skill(&path, source, limits.max_instruction_bytes)?;
            if let Some(first) = skills.get(&skill.metadata.name) {
                return Err(SkillError::DuplicateName {
                    name: skill.metadata.name.clone(),
                    first: first.root.clone(),
                    second: skill.root.clone(),
                });
            }
            skills.insert(skill.metadata.name.clone(), skill);
        }
        Ok(Self { skills, limits })
    }

    pub fn skill(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// A compact, deterministic catalog suitable for an agent system prompt.
    pub fn prompt_summary(&self) -> String {
        let mut summary = String::from(
            "<skill_usage>\nUse load_skill with a skill name to load its full Markdown instructions. Use read_skill_resource only for bounded referenced resources named by those instructions.\n</skill_usage>\n<skills>\n",
        );
        for skill in self.skills.values() {
            summary.push_str("  <skill>\n    <name>");
            summary.push_str(&escape_xml(&skill.metadata.name));
            summary.push_str("</name>\n    <description>");
            summary.push_str(&escape_xml(&skill.metadata.description));
            summary.push_str("</description>\n  </skill>\n");
        }
        summary.push_str("</skills>");
        summary
    }

    /// Reads a resource from a trusted local skill package.
    ///
    /// The package must not be concurrently mutated while this call is in progress. Path
    /// canonicalization and containment checks reject stable symlink escapes, but this API does
    /// not claim to defend against an actor replacing path components during the read.
    pub fn read_resource(&self, skill: &str, relative: &Path) -> Result<String, SkillError> {
        let skill = self
            .skills
            .get(skill)
            .ok_or_else(|| SkillError::UnknownSkill {
                name: skill.to_owned(),
            })?;
        if relative.as_os_str().is_empty()
            || relative.is_absolute()
            || relative
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(SkillError::InvalidResourcePath {
                path: relative.to_path_buf(),
            });
        }

        let joined = skill.root.join(relative);
        let target = joined.canonicalize().map_err(|source| SkillError::Io {
            path: joined.clone(),
            source,
        })?;
        if !target.starts_with(&skill.root) {
            return Err(SkillError::ResourceOutsideRoot {
                path: target,
                root: skill.root.clone(),
            });
        }
        let metadata = fs::metadata(&target).map_err(|source| SkillError::Io {
            path: target.clone(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(SkillError::ResourceNotFile { path: target });
        }
        let limit = self.limits.max_resource_bytes;
        if metadata.len() > limit as u64 {
            return Err(SkillError::ResourceTooLarge {
                path: target,
                limit,
            });
        }
        let mut bytes = Vec::new();
        File::open(&target)
            .map_err(|source| SkillError::Io {
                path: target.clone(),
                source,
            })?
            .take(limit.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|source| SkillError::Io {
                path: target.clone(),
                source,
            })?;
        if bytes.len() > limit {
            return Err(SkillError::ResourceTooLarge {
                path: target,
                limit,
            });
        }
        String::from_utf8(bytes).map_err(|source| SkillError::ResourceNotUtf8 {
            path: target,
            source,
        })
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

pub struct LoadSkillTool {
    catalog: Arc<SkillCatalog>,
}

impl LoadSkillTool {
    pub fn new(catalog: Arc<SkillCatalog>) -> Self {
        Self { catalog }
    }
}

#[async_trait]
impl Tool for LoadSkillTool {
    fn name(&self) -> &str {
        "load_skill"
    }
    fn description(&self) -> &str {
        "Load the full instructions for a skill by name"
    }
    fn parameters(&self) -> JsonValue {
        json!({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})
    }
    async fn execute(&self, args: JsonValue) -> anyhow::Result<String> {
        let name = args
            .get("name")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| anyhow::anyhow!("name must be a string"))?;
        self.catalog
            .skill(name)
            .map(|skill| skill.instructions.clone())
            .ok_or_else(|| {
                anyhow::anyhow!(SkillError::UnknownSkill {
                    name: name.to_owned()
                })
            })
    }
}

pub struct ReadSkillResourceTool {
    catalog: Arc<SkillCatalog>,
}

impl ReadSkillResourceTool {
    pub fn new(catalog: Arc<SkillCatalog>) -> Self {
        Self { catalog }
    }
}

#[async_trait]
impl Tool for ReadSkillResourceTool {
    fn name(&self) -> &str {
        "read_skill_resource"
    }
    fn description(&self) -> &str {
        "Read a UTF-8 resource file belonging to a skill"
    }
    fn parameters(&self) -> JsonValue {
        json!({"type":"object","properties":{"skill":{"type":"string"},"path":{"type":"string"}},"required":["skill","path"]})
    }
    async fn execute(&self, args: JsonValue) -> anyhow::Result<String> {
        let skill = args
            .get("skill")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| anyhow::anyhow!("skill must be a string"))?;
        let path = args
            .get("path")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| anyhow::anyhow!("path must be a string"))?;
        self.catalog
            .read_resource(skill, Path::new(path))
            .map_err(anyhow::Error::new)
    }
}

struct GitRepository;

impl GitRepository {
    fn output(cwd: Option<&Path>, args: &[&str]) -> Result<Vec<u8>, SkillError> {
        let mut command = Command::new("git");
        command.args(args);
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        let output = command.output().map_err(|source| SkillError::Io {
            path: cwd.unwrap_or_else(|| Path::new("git")).to_path_buf(),
            source,
        })?;
        if output.status.success() {
            return Ok(output.stdout);
        }
        Err(SkillError::Git {
            command: format!("git {}", args.join(" ")),
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }

    fn run(cwd: Option<&Path>, args: &[&str]) -> Result<(), SkillError> {
        Self::output(cwd, args).map(|_| ())
    }
}

struct CacheLock(PathBuf);

impl CacheLock {
    fn acquire(path: PathBuf) -> Result<Self, SkillError> {
        for _ in 0..50 {
            match File::options().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    use std::io::Write;
                    let record = format!(
                        "pid={}\ntimestamp={}\nnonce={}\n",
                        std::process::id(),
                        chrono::Utc::now().timestamp(),
                        uuid::Uuid::new_v4()
                    );
                    if let Err(source) = file.write_all(record.as_bytes()) {
                        drop(file);
                        let _ = fs::remove_file(&path);
                        return Err(SkillError::Io { path, source });
                    }
                    return Ok(Self(path));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_is_stale(&path) {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(source) => return Err(SkillError::Io { path, source }),
            }
        }
        Err(SkillError::Git {
            command: "cache lock".into(),
            message: format!("timed out waiting for {}", path.display()),
        })
    }
}

fn lock_is_stale(path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = text
        .lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|pid| pid.parse::<u32>().ok())
    else {
        return false;
    };
    if text
        .lines()
        .find_map(|line| line.strip_prefix("timestamp="))
        .and_then(|value| value.parse::<i64>().ok())
        .is_none()
        || text
            .lines()
            .find_map(|line| line.strip_prefix("nonce="))
            .filter(|value| !value.is_empty())
            .is_none()
    {
        return false;
    }
    if pid == std::process::id() {
        return false;
    }
    #[cfg(unix)]
    {
        !Path::new("/proc").join(pid.to_string()).exists()
    }
    #[cfg(windows)]
    {
        let filter = format!("PID eq {pid}");
        Command::new("tasklist")
            .args(["/FI", &filter, "/NH"])
            .output()
            .map(|output| !String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}

const SOURCE_MANIFEST: &str = ".ferrant-source.json";

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SourceManifest {
    repository: String,
    git_ref: Option<String>,
    subdirectory_native_hex: Option<String>,
    commit: String,
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct TempCheckout(PathBuf);
impl Drop for TempCheckout {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn materialize_git(source: &SkillSource, refresh: bool) -> Result<PathBuf, SkillError> {
    let SkillSource::GitHub {
        repository,
        git_ref,
        subdirectory,
        cache_dir,
    } = source
    else {
        unreachable!()
    };
    if !(repository.starts_with("https://github.com/")
        || repository.starts_with("ssh://git@github.com/")
        || repository.starts_with("git@github.com:")
        || repository.starts_with("file://"))
    {
        return Err(SkillError::InvalidRepository {
            repository: repository.clone(),
        });
    }
    fs::create_dir_all(cache_dir).map_err(|source| SkillError::Io {
        path: cache_dir.clone(),
        source,
    })?;
    let key = source_cache_key(repository, git_ref.as_deref(), subdirectory.as_deref());
    let checkout = cache_dir.join(&key);
    if !cache_is_valid(&checkout, repository, git_ref, subdirectory) || refresh {
        let _lock = CacheLock::acquire(cache_dir.join(format!("{key}.lock")))?;
        if !cache_is_valid(&checkout, repository, git_ref, subdirectory) || refresh {
            let nonce = NEXT_GIT_TEMP.fetch_add(1, Ordering::Relaxed);
            let staged_path =
                cache_dir.join(format!(".{key}.{}.{}.tmp", std::process::id(), nonce));
            let staged = TempCheckout(staged_path);
            let target = staged.0.to_string_lossy().into_owned();
            GitRepository::run(None, &["clone", "--no-checkout", repository, &target])?;
            checkout_target(&staged.0, git_ref.as_deref())?;
            selected_root(&staged.0, subdirectory)?;
            write_manifest(&staged.0, repository, git_ref, subdirectory)?;
            replace_checkout(&staged.0, &checkout)?;
        }
    }
    selected_root(&checkout, subdirectory)
}

fn selected_root(checkout: &Path, subdirectory: &Option<PathBuf>) -> Result<PathBuf, SkillError> {
    let checkout_root = checkout.canonicalize().map_err(|source| SkillError::Io {
        path: checkout.clone(),
        source,
    })?;
    let selected = match subdirectory {
        Some(relative)
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_))) =>
        {
            return Err(SkillError::GitSubdirectoryOutside {
                path: relative.clone(),
                root: checkout_root,
            });
        }
        Some(relative) => checkout_root.join(relative),
        None => checkout_root.clone(),
    };
    let selected = selected.canonicalize().map_err(|source| SkillError::Io {
        path: selected.clone(),
        source,
    })?;
    if !selected.starts_with(&checkout_root) {
        return Err(SkillError::GitSubdirectoryOutside {
            path: selected,
            root: checkout_root,
        });
    }
    Ok(selected)
}

fn replace_checkout(staged: &Path, checkout: &Path) -> Result<(), SkillError> {
    if !checkout.exists() {
        return fs::rename(staged, checkout).map_err(|source| SkillError::Io {
            path: checkout.to_path_buf(),
            source,
        });
    }
    let backup = checkout.with_extension(format!("backup-{}", std::process::id()));
    let _ = fs::remove_dir_all(&backup);
    fs::rename(checkout, &backup).map_err(|source| SkillError::Io {
        path: checkout.to_path_buf(),
        source,
    })?;
    match fs::rename(staged, checkout) {
        Ok(()) => {
            let _ = fs::remove_dir_all(backup);
            Ok(())
        }
        Err(source) => {
            let _ = fs::rename(&backup, checkout);
            Err(SkillError::Io {
                path: checkout.to_path_buf(),
                source,
            })
        }
    }
}

fn checkout_target(checkout: &Path, git_ref: Option<&str>) -> Result<(), SkillError> {
    let target = resolve_target(checkout, git_ref)?;
    GitRepository::run(
        Some(checkout),
        &["checkout", "--detach", "--force", &target],
    )?;
    GitRepository::run(Some(checkout), &["reset", "--hard", &target])
}

fn resolve_target(checkout: &Path, git_ref: Option<&str>) -> Result<String, SkillError> {
    let requested = git_ref.unwrap_or("refs/remotes/origin/HEAD");
    let mut candidates = Vec::new();
    if let Some(branch) = requested.strip_prefix("refs/heads/") {
        candidates.push(format!("refs/remotes/origin/{branch}"));
    } else if !requested.starts_with("refs/") {
        candidates.push(format!("refs/remotes/origin/{requested}"));
    }
    candidates.push(requested.to_owned());
    let mut target = None;
    for candidate in candidates {
        let expression = format!("{candidate}^{{commit}}");
        if let Ok(bytes) = GitRepository::output(
            Some(checkout),
            &["rev-parse", "--verify", "--quiet", &expression],
        ) {
            target = Some(String::from_utf8_lossy(&bytes).trim().to_owned());
            break;
        }
    }
    target.ok_or_else(|| SkillError::Git {
        command: "git rev-parse".into(),
        message: format!("unknown ref {requested}"),
    })
}

fn source_cache_key(
    repository: &str,
    git_ref: Option<&str>,
    subdirectory: Option<&Path>,
) -> String {
    let mut bytes = Vec::new();
    append_field(&mut bytes, 1, repository.as_bytes());
    append_optional_field(&mut bytes, 2, git_ref.map(str::as_bytes));
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        append_optional_field(
            &mut bytes,
            3,
            subdirectory.map(|path| path.as_os_str().as_bytes()),
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let native = subdirectory.map(|path| {
            path.as_os_str()
                .encode_wide()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
        });
        append_optional_field(&mut bytes, 3, native.as_deref());
    }
    format!("{:x}", Sha256::digest(&bytes))
}

fn append_field(identity: &mut Vec<u8>, tag: u8, value: &[u8]) {
    identity.push(tag);
    identity.extend_from_slice(&(value.len() as u64).to_le_bytes());
    identity.extend_from_slice(value);
}

fn append_optional_field(identity: &mut Vec<u8>, tag: u8, value: Option<&[u8]>) {
    identity.push(tag);
    identity.push(u8::from(value.is_some()));
    if let Some(value) = value {
        identity.extend_from_slice(&(value.len() as u64).to_le_bytes());
        identity.extend_from_slice(value);
    }
}

fn native_path_hex(path: Option<&Path>) -> Option<String> {
    path.map(|path| {
        #[cfg(unix)]
        let bytes = {
            use std::os::unix::ffi::OsStrExt;
            path.as_os_str().as_bytes().to_vec()
        };
        #[cfg(windows)]
        let bytes = {
            use std::os::windows::ffi::OsStrExt;
            path.as_os_str()
                .encode_wide()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
        };
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    })
}

fn write_manifest(
    checkout: &Path,
    repository: &str,
    git_ref: &Option<String>,
    subdirectory: &Option<PathBuf>,
) -> Result<(), SkillError> {
    let commit = String::from_utf8_lossy(&GitRepository::output(
        Some(checkout),
        &["rev-parse", "HEAD"],
    )?)
    .trim()
    .to_owned();
    let manifest = SourceManifest {
        repository: repository.to_owned(),
        git_ref: git_ref.clone(),
        subdirectory_native_hex: native_path_hex(subdirectory.as_deref()),
        commit,
    };
    let bytes = serde_json::to_vec(&manifest).expect("source manifest serializes");
    fs::write(checkout.join(SOURCE_MANIFEST), bytes).map_err(|source| SkillError::Io {
        path: checkout.join(SOURCE_MANIFEST),
        source,
    })
}

fn cache_is_valid(
    checkout: &Path,
    repository: &str,
    git_ref: &Option<String>,
    subdirectory: &Option<PathBuf>,
) -> bool {
    let Ok(bytes) = fs::read(checkout.join(SOURCE_MANIFEST)) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_slice::<SourceManifest>(&bytes) else {
        return false;
    };
    if manifest.repository != repository
        || &manifest.git_ref != git_ref
        || manifest.subdirectory_native_hex != native_path_hex(subdirectory.as_deref())
    {
        return false;
    }
    let Ok(head) = GitRepository::output(Some(checkout), &["rev-parse", "HEAD"]) else {
        return false;
    };
    if String::from_utf8_lossy(&head).trim() != manifest.commit {
        return false;
    }
    let Ok(origin) = GitRepository::output(Some(checkout), &["remote", "get-url", "origin"]) else {
        return false;
    };
    if String::from_utf8_lossy(&origin).trim() != repository {
        return false;
    }
    if resolve_target(checkout, git_ref.as_deref()).ok().as_deref()
        != Some(manifest.commit.as_str())
    {
        return false;
    }
    let Ok(status) = GitRepository::output(
        Some(checkout),
        &["status", "--porcelain", "--untracked-files=all"],
    ) else {
        return false;
    };
    String::from_utf8_lossy(&status)
        .lines()
        .all(|line| line == format!("?? {SOURCE_MANIFEST}"))
}

fn discover(
    root: &Path,
    source: &SkillSource,
    files: &mut Vec<(PathBuf, SkillSource)>,
) -> Result<(), SkillError> {
    let entries = fs::read_dir(root).map_err(|source_error| SkillError::Io {
        path: root.to_path_buf(),
        source: source_error,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source_error| SkillError::Io {
            path: root.to_path_buf(),
            source: source_error,
        })?;
        let file_type = entry.file_type().map_err(|source_error| SkillError::Io {
            path: entry.path(),
            source: source_error,
        })?;
        if file_type.is_dir() {
            discover(&entry.path(), source, files)?;
        } else if file_type.is_file() && entry.file_name() == "SKILL.md" {
            files.push((entry.path(), source.clone()));
        }
    }
    Ok(())
}

fn discover_git(
    selected_root: &Path,
    source: &SkillSource,
    files: &mut Vec<(PathBuf, SkillSource)>,
) -> Result<(), SkillError> {
    let checkout_bytes =
        GitRepository::output(Some(selected_root), &["rev-parse", "--show-toplevel"])?;
    let checkout = PathBuf::from(String::from_utf8_lossy(&checkout_bytes).trim().to_owned())
        .canonicalize()
        .map_err(|source| SkillError::Io {
            path: selected_root.to_path_buf(),
            source,
        })?;
    let mut discovered = Vec::new();
    discover(selected_root, source, &mut discovered)?;
    for (path, source) in discovered {
        let relative =
            path.strip_prefix(&checkout)
                .map_err(|_| SkillError::GitSubdirectoryOutside {
                    path: path.clone(),
                    root: checkout.clone(),
                })?;
        let status = Command::new("git")
            .current_dir(&checkout)
            .arg("ls-files")
            .arg("--error-unmatch")
            .arg("--")
            .arg(relative)
            .output()
            .map_err(|source| SkillError::Io {
                path: path.clone(),
                source,
            })?;
        if status.status.success() {
            files.push((path, source));
        }
    }
    Ok(())
}

fn parse_skill(path: &Path, source: SkillSource, limit: usize) -> Result<Skill, SkillError> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|source_error| SkillError::Io {
            path: path.to_path_buf(),
            source: source_error,
        })?
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|source_error| SkillError::Io {
            path: path.to_path_buf(),
            source: source_error,
        })?;
    if bytes.len() > limit {
        return Err(SkillError::InstructionTooLarge {
            path: path.to_path_buf(),
            limit,
        });
    }
    let text = String::from_utf8(bytes).map_err(|error| SkillError::Parse {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let (yaml, instructions) = split_frontmatter(&text).ok_or_else(|| SkillError::Parse {
        path: path.to_path_buf(),
        message: "expected leading YAML frontmatter delimited by ---".into(),
    })?;
    let mut extensions: BTreeMap<String, Value> =
        serde_yaml::from_str(yaml).map_err(|error| SkillError::Parse {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let name = required_string(&mut extensions, "name", path)?;
    let description = required_string(&mut extensions, "description", path)?;
    let root = path
        .parent()
        .expect("SKILL.md has a parent")
        .canonicalize()
        .map_err(|source_error| SkillError::Io {
            path: path.to_path_buf(),
            source: source_error,
        })?;
    Ok(Skill {
        metadata: SkillMetadata {
            name,
            description,
            extensions,
        },
        instructions: instructions.to_owned(),
        root,
        source,
    })
}

fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

fn required_string(
    map: &mut BTreeMap<String, Value>,
    key: &str,
    path: &Path,
) -> Result<String, SkillError> {
    match map.remove(key) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value),
        _ => Err(SkillError::Parse {
            path: path.to_path_buf(),
            message: format!("'{key}' must be a non-empty string"),
        }),
    }
}
