use crate::approval::ApprovalGate;
use crate::sandbox::{display_command, CommandResult, Sandbox};
use crate::state::SessionState;
use crate::workspace::Workspace;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use ferragent::tool::Tool;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Clone)]
struct ContextState {
    workspace: Workspace,
    approvals: Arc<ApprovalGate>,
    sandbox: Sandbox,
    session: Arc<SessionState>,
}

pub fn coding_tools(
    workspace: Workspace,
    approvals: Arc<ApprovalGate>,
    sandbox: Sandbox,
    session: Arc<SessionState>,
) -> Vec<Arc<dyn Tool>> {
    let context = ContextState {
        workspace,
        approvals,
        sandbox,
        session,
    };
    vec![
        Arc::new(ListFiles(context.clone())),
        Arc::new(ReadFile(context.clone())),
        Arc::new(WriteFile(context.clone())),
        Arc::new(RunCommand(context.clone())),
        Arc::new(InstallDependency(context.clone())),
        Arc::new(Git(context)),
    ]
}

struct ListFiles(ContextState);

#[async_trait]
impl Tool for ListFiles {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files in the approved workspace. Git internals and secrets are excluded."
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _args: Value) -> Result<String> {
        Ok(observe(self.0.workspace.list()))
    }
}

struct ReadFile(ContextState);

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read one UTF-8 text file using a workspace-relative path. Secret files are blocked."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let result = required_string(&args, "path").and_then(|path| self.0.workspace.read(path));
        Ok(observe(result))
    }
}

struct WriteFile(ContextState);

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or atomically replace a UTF-8 file inside the workspace after user approval."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let result = (|| {
            let path = required_string(&args, "path")?;
            let content = required_string(&args, "content")?;
            self.0.workspace.validate_relative(path)?;
            let old = self.0.workspace.old_content(path)?;
            let detail = format!(
                "Path: {}\n  Existing content:\n{}\n  New content ({} bytes):\n{}",
                path,
                old.as_deref()
                    .map(preview)
                    .unwrap_or_else(|| "<new file>".to_string()),
                content.len(),
                preview(content)
            );
            if !self.0.approvals.request("write file", &detail)? {
                bail!("permission denied; file was not written");
            }
            self.0.workspace.write_atomic(path, content)?;
            self.0.session.changed(path);
            Ok(format!("wrote {path}"))
        })();
        Ok(observe(result))
    }
}

struct RunCommand(ContextState);

#[async_trait]
impl Tool for RunCommand {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Compile, test, lint, or run workspace code in a network-disabled sandbox. Returns failures for repair."
    }

    fn parameters(&self) -> Value {
        command_schema(true)
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let result = self.execute_inner(&args).await;
        Ok(observe(result))
    }
}

impl RunCommand {
    async fn execute_inner(&self, value: &Value) -> Result<String> {
        let requested_program = required_string(value, "program")?;
        let program = if requested_program.contains('/') {
            self.0.workspace.executable(requested_program)?
        } else {
            ensure_program_allowed(requested_program, false)?;
            requested_program.to_string()
        };
        let args = string_array(value, "args")?;
        let purpose = required_string(value, "purpose")?;
        let command = display_command(&program, &args);
        let detail = format!(
            "Purpose: {purpose}\n  Workspace: {}\n  Command: {command}\n  Sandbox: {}\n  Network: denied",
            self.0.workspace.root().display(),
            self.0.sandbox.description()
        );
        if !self.0.approvals.request("run project command", &detail)? {
            bail!("permission denied; command was not run");
        }
        let output = self
            .0
            .sandbox
            .run(&program, &args, false, &self.0.approvals)
            .await?;
        if output.exit_code == Some(0) && !output.timed_out {
            self.0.session.verified(command.clone());
        }
        Ok(format_result(command, output))
    }
}

struct InstallDependency(ContextState);

#[async_trait]
impl Tool for InstallDependency {
    fn name(&self) -> &str {
        "install_dependency"
    }

    fn description(&self) -> &str {
        "Install project-local dependencies with explicit approval. Network is enabled inside the filesystem sandbox."
    }

    fn parameters(&self) -> Value {
        command_schema(false)
    }

    async fn execute(&self, value: Value) -> Result<String> {
        let result = self.execute_inner(&value).await;
        Ok(observe(result))
    }
}

impl InstallDependency {
    async fn execute_inner(&self, value: &Value) -> Result<String> {
        let program = required_string(value, "program")?;
        ensure_program_allowed(program, true)?;
        let args = string_array(value, "args")?;
        let purpose = required_string(value, "purpose")?;
        let command = display_command(program, &args);
        let detail = format!(
            "Purpose: {purpose}\n  Workspace: {}\n  Command: {command}\n  Sandbox: {}\n  Network: ENABLED for dependency download. The package manager and install scripts can transmit readable workspace content.",
            self.0.workspace.root().display(),
            self.0.sandbox.description()
        );
        if !self.0.approvals.request("install dependencies", &detail)? {
            bail!("permission denied; dependencies were not installed");
        }
        let output = self
            .0
            .sandbox
            .run(program, &args, true, &self.0.approvals)
            .await?;
        if output.exit_code == Some(0) {
            self.0.session.changed("dependency manifest/lockfile");
        }
        Ok(format_result(command, output))
    }
}

struct Git(ContextState);

#[async_trait]
impl Tool for Git {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Run a constrained Git operation: status, diff, log, init, add, or commit. Destructive and remote operations are unavailable."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string", "enum": ["status", "diff", "log", "init", "add", "commit"] },
                "paths": { "type": "array", "items": { "type": "string" } },
                "message": { "type": "string" }
            },
            "required": ["operation"]
        })
    }

    async fn execute(&self, value: Value) -> Result<String> {
        let result = self.execute_inner(&value).await;
        Ok(observe(result))
    }
}

impl Git {
    async fn execute_inner(&self, value: &Value) -> Result<String> {
        let operation = required_string(value, "operation")?;
        let args = match operation {
            "status" => vec!["status".into(), "--short".into()],
            "diff" => vec!["diff".into(), "--".into()],
            "log" => vec!["log".into(), "--oneline".into(), "-n".into(), "10".into()],
            "init" => vec!["init".into()],
            "add" => {
                let paths = string_array(value, "paths")?;
                if paths.is_empty() {
                    bail!("git add requires at least one path");
                }
                for path in &paths {
                    if path != "." {
                        self.0.workspace.validate_relative(path)?;
                    }
                }
                let mut args = vec!["add".into(), "--".into()];
                args.extend(paths);
                args
            }
            "commit" => {
                let message = required_string(value, "message")?;
                if message.is_empty() || message.len() > 500 {
                    bail!("commit message must contain 1-500 characters");
                }
                vec![
                    "-c".into(),
                    "core.hooksPath=/dev/null".into(),
                    "commit".into(),
                    "-m".into(),
                    message.into(),
                ]
            }
            _ => bail!("unsupported Git operation: {operation}"),
        };
        let command = display_command("git", &args);
        let detail = format!(
            "Operation: {operation}\n  Workspace: {}\n  Command: {command}\n  Sandbox: {}",
            self.0.workspace.root().display(),
            self.0.sandbox.description()
        );
        if !self.0.approvals.request("use Git", &detail)? {
            bail!("permission denied; Git was not run");
        }
        let output = self
            .0
            .sandbox
            .run("git", &args, false, &self.0.approvals)
            .await?;
        Ok(format_result(command, output))
    }
}

fn command_schema(run: bool) -> Value {
    let description = if run {
        "Build/test/run program such as cargo, python, npm, or make"
    } else {
        "Package manager such as cargo, npm, pip, uv, or go"
    };
    json!({
        "type": "object",
        "properties": {
            "program": { "type": "string", "description": description },
            "args": { "type": "array", "items": { "type": "string" } },
            "purpose": { "type": "string" }
        },
        "required": ["program", "args", "purpose"]
    })
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    let text = value[field]
        .as_str()
        .with_context(|| format!("{field} must be a string"))?;
    if text.len() > 1_048_576 {
        bail!("{field} is too large");
    }
    Ok(text)
}

fn string_array(value: &Value, field: &str) -> Result<Vec<String>> {
    let values = value[field]
        .as_array()
        .with_context(|| format!("{field} must be an array"))?;
    if values.len() > 100 {
        bail!("{field} contains too many values");
    }
    values
        .iter()
        .map(|item| {
            let item = item
                .as_str()
                .with_context(|| format!("every {field} value must be a string"))?;
            if item.len() > 4096 || item.contains('\0') {
                bail!("invalid {field} value");
            }
            Ok(item.to_string())
        })
        .collect()
}

fn ensure_program_allowed(program: &str, dependency: bool) -> Result<()> {
    const PROJECT_PROGRAMS: &[&str] = &[
        "cargo", "rustc", "python", "python3", "node", "npm", "pnpm", "yarn", "bun", "deno", "go",
        "make", "cmake", "ctest", "clang", "gcc", "g++", "java", "javac", "mvn", "gradle",
        "dotnet", "swift", "pytest", "ruff", "eslint", "biome",
    ];
    const DEPENDENCY_PROGRAMS: &[&str] = &[
        "cargo", "npm", "pnpm", "yarn", "bun", "pip", "pip3", "python", "python3", "uv", "go",
        "mvn", "gradle", "dotnet", "bundle", "gem",
    ];
    let allowed = if dependency {
        DEPENDENCY_PROGRAMS
    } else {
        PROJECT_PROGRAMS
    };
    if !allowed.contains(&program) {
        bail!(
            "program is not on the {:?} allowlist: {program}",
            if dependency { "dependency" } else { "project" }
        );
    }
    Ok(())
}

fn format_result(command: String, output: CommandResult) -> String {
    json!({
        "command": command,
        "exit_code": output.exit_code,
        "sandboxed": output.sandboxed,
        "timed_out": output.timed_out,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
    .to_string()
}

fn observe(result: Result<String>) -> String {
    match result {
        Ok(output) => json!({ "ok": true, "output": output }).to_string(),
        Err(error) => json!({ "ok": false, "error": error.to_string() }).to_string(),
    }
}

fn preview(value: &str) -> String {
    const LIMIT: usize = 4000;
    if value.chars().count() <= LIMIT {
        value.to_string()
    } else {
        format!(
            "{}\n... preview truncated ...",
            value.chars().take(LIMIT).collect::<String>()
        )
    }
}
