use crate::approval::ApprovalGate;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::time::timeout;

const OUTPUT_LIMIT: usize = 32 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone)]
enum Backend {
    MacOs,
    Bubblewrap(PathBuf),
    Unavailable(String),
}

#[derive(Clone)]
pub struct Sandbox {
    workspace: PathBuf,
    scratch: PathBuf,
    backend: Backend,
}

pub struct CommandResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub sandboxed: bool,
    pub timed_out: bool,
}

impl Sandbox {
    pub fn detect(workspace: &Path) -> Self {
        let backend = if cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists() {
            probe_macos().map_or_else(Backend::Unavailable, |_| Backend::MacOs)
        } else if cfg!(target_os = "linux") {
            find_on_path("bwrap").map_or_else(
                || Backend::Unavailable("bubblewrap (bwrap) is not installed".to_string()),
                Backend::Bubblewrap,
            )
        } else {
            Backend::Unavailable(format!(
                "no sandbox backend is configured for {}",
                std::env::consts::OS
            ))
        };
        Self {
            workspace: workspace.to_path_buf(),
            scratch: std::env::temp_dir()
                .join(format!("ferragent-coding-agent-{}", std::process::id())),
            backend,
        }
    }

    pub fn description(&self) -> &str {
        match &self.backend {
            Backend::MacOs => "macOS sandbox-exec (workspace-only writes)",
            Backend::Bubblewrap(_) => "Linux bubblewrap (workspace-only writes)",
            Backend::Unavailable(reason) => reason,
        }
    }

    pub async fn run(
        &self,
        program: &str,
        args: &[String],
        network: bool,
        gate: &ApprovalGate,
    ) -> Result<CommandResult> {
        std::fs::create_dir_all(self.scratch.join("home"))?;
        std::fs::create_dir_all(self.scratch.join("tmp"))?;
        std::fs::create_dir_all(self.scratch.join("cargo"))?;
        let (mut command, mut sandboxed) = match &self.backend {
            Backend::MacOs => (self.macos_command(program, args, network), true),
            Backend::Bubblewrap(path) => (self.bwrap_command(path, program, args, network), true),
            Backend::Unavailable(reason) => {
                let detail = format!(
                    "Sandbox unavailable: {reason}. Run locally instead: {}",
                    display_command(program, args)
                );
                if !gate.request("unsandboxed local command", &detail)? {
                    bail!("local fallback denied; command was not run");
                }
                (self.local_command(program, args), false)
            }
        };

        scrub_environment(&mut command, &self.scratch);
        command
            .current_dir(&self.workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        configure_process_group(&mut command);

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) if sandboxed => {
                let detail = format!(
                    "The sandbox could not start ({error}). Run this exact command locally instead: {}",
                    display_command(program, args)
                );
                if !gate.request("unsandboxed local command", &detail)? {
                    bail!("sandbox failed and local fallback was denied");
                }
                sandboxed = false;
                let mut local = self.local_command(program, args);
                scrub_environment(&mut local, &self.scratch);
                local
                    .current_dir(&self.workspace)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);
                configure_process_group(&mut local);
                local.spawn().with_context(|| {
                    format!(
                        "failed to start local fallback: {}",
                        display_command(program, args)
                    )
                })?
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to start command: {}",
                        display_command(program, args)
                    )
                })
            }
        };
        let stdout = child
            .stdout
            .take()
            .context("child stdout was not captured")?;
        let stderr = child
            .stderr
            .take()
            .context("child stderr was not captured")?;
        let stdout_task = tokio::spawn(read_limited(stdout));
        let stderr_task = tokio::spawn(read_limited(stderr));

        let (exit_code, timed_out) = match timeout(COMMAND_TIMEOUT, child.wait()).await {
            Ok(status) => (status?.code(), false),
            Err(_) => {
                kill_process_group(&child);
                let _ = child.kill().await;
                let _ = child.wait().await;
                (None, true)
            }
        };
        let stdout = collect_reader(stdout_task).await?;
        let mut stderr = collect_reader(stderr_task).await?;
        if timed_out {
            stderr.push_str(&format!(
                "\ncommand exceeded the {} second timeout",
                COMMAND_TIMEOUT.as_secs()
            ));
        }
        Ok(CommandResult {
            exit_code,
            stdout,
            stderr,
            sandboxed,
            timed_out,
        })
    }

    fn macos_command(&self, program: &str, args: &[String], network: bool) -> Command {
        let network_rule = if network {
            "(allow network-outbound)"
        } else {
            "(deny network*)"
        };
        let profile = format!(
            "(version 1) (deny default) (allow process*) (allow sysctl-read) \
             (allow mach-lookup) (allow file-read*) \
             (deny file-read* (subpath \"/Users\") (subpath \"/Volumes\") \
             (subpath \"/private/tmp\") (subpath \"/private/var/folders\")) \
             (allow file-read* (subpath (param \"WORKSPACE\")) (subpath (param \"SCRATCH\"))) \
             (allow file-write* (subpath (param \"WORKSPACE\")) (subpath (param \"SCRATCH\")) \
             (literal \"/dev/null\")) \
             (deny file-read* (literal (param \"ENV_FILE\"))) \
             (deny file-write* (literal (param \"ENV_FILE\"))) \
             (deny file-read* (literal (param \"PROVIDER_CONFIG\")) \
             (literal (param \"WORKSPACE_CONFIG\"))) \
             (deny file-write* (literal (param \"PROVIDER_CONFIG\")) \
             (literal (param \"WORKSPACE_CONFIG\"))) {network_rule}"
        );
        let provider_config =
            provider_config_path().unwrap_or_else(|| self.scratch.join("missing-provider-config"));
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command
            .arg("-D")
            .arg(format!("WORKSPACE={}", self.workspace.display()))
            .arg("-D")
            .arg(format!("SCRATCH={}", self.scratch.display()))
            .arg("-D")
            .arg(format!(
                "ENV_FILE={}",
                self.workspace.join(".env").display()
            ))
            .arg("-D")
            .arg(format!("PROVIDER_CONFIG={}", provider_config.display()))
            .arg("-D")
            .arg(format!(
                "WORKSPACE_CONFIG={}",
                self.workspace.join(".code-agent-cli.config").display()
            ))
            .arg("-p")
            .arg(profile)
            .arg(program)
            .args(args);
        command
    }

    fn bwrap_command(
        &self,
        bwrap: &Path,
        program: &str,
        args: &[String],
        network: bool,
    ) -> Command {
        let mut command = Command::new(bwrap);
        command
            .args(["--die-with-parent", "--ro-bind", "/", "/", "--bind"])
            .arg(&self.workspace)
            .arg(&self.workspace)
            .args(["--tmpfs", "/tmp", "--chdir"])
            .arg(&self.workspace);
        for config in [
            provider_config_path(),
            Some(self.workspace.join(".code-agent-cli.config")),
        ]
        .into_iter()
        .flatten()
        .filter(|path| path.exists())
        {
            command.arg("--ro-bind").arg("/dev/null").arg(config);
        }
        if !network {
            command.arg("--unshare-net");
        }
        command.arg("--").arg(program).args(args);
        command
    }

    fn local_command(&self, program: &str, args: &[String]) -> Command {
        let mut command = Command::new(program);
        command.args(args);
        command
    }
}

pub fn display_command(program: &str, args: &[String]) -> String {
    std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .map(|part| format!("{part:?}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn scrub_environment(command: &mut Command, scratch: &Path) {
    let path = std::env::var_os("PATH").unwrap_or_default();
    command.env_clear();
    command.env("PATH", path);
    command.env("HOME", scratch.join("home"));
    command.env("TMPDIR", scratch.join("tmp"));
    command.env("CARGO_HOME", scratch.join("cargo"));
    command.env("npm_config_cache", scratch.join("npm"));
    command.env("LANG", "C.UTF-8");
    command.env("LC_ALL", "C.UTF-8");
    command.env("CI", "1");
}

fn probe_macos() -> std::result::Result<(), String> {
    let output = std::process::Command::new("/usr/bin/sandbox-exec")
        .args([
            "-p",
            "(version 1) (deny default) (allow process*) (allow file-read*)",
            "/usr/bin/true",
        ])
        .output()
        .map_err(|error| format!("sandbox-exec probe could not start: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "sandbox-exec probe failed (status {}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn find_on_path(program: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|path| path.join(program))
            .find(|path| path.is_file())
    })
}

fn provider_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".code-agent-cli.config"))
}

async fn read_limited(mut stream: impl AsyncRead + Unpin) -> Result<String> {
    let mut retained = Vec::with_capacity(OUTPUT_LIMIT);
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        let remaining = OUTPUT_LIMIT.saturating_sub(retained.len());
        if remaining > 0 {
            retained.extend_from_slice(&buffer[..count.min(remaining)]);
        }
        truncated |= count > remaining;
    }
    let mut value = String::from_utf8_lossy(&retained)
        .chars()
        .filter(|character| *character == '\n' || *character == '\t' || !character.is_control())
        .collect::<String>();
    if truncated {
        value.push_str("\n... output truncated ...");
    }
    Ok(value)
}

async fn collect_reader(mut task: tokio::task::JoinHandle<Result<String>>) -> Result<String> {
    match timeout(Duration::from_secs(2), &mut task).await {
        Ok(result) => result.context("output reader failed")?,
        Err(_) => {
            task.abort();
            Ok("... output reader timed out after process exit ...".to_string())
        }
    }
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    // SAFETY: setpgid is async-signal-safe and this closure performs no
    // allocation between fork and exec.
    unsafe {
        command.as_std_mut().pre_exec(|| {
            if libc::setpgid(0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_process_group(child: &tokio::process::Child) {
    if let Some(id) = child.id() {
        // SAFETY: a negative PID targets the child-owned process group created
        // immediately before exec. Failure is harmless and followed by kill().
        unsafe {
            libc::kill(-(id as i32), libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn kill_process_group(_child: &tokio::process::Child) {}

#[cfg(test)]
mod tests {
    use super::{Backend, Sandbox};
    use crate::approval::ApprovalGate;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn sandbox_runs_safe_command_and_blocks_outside_writes() {
        // Keep the workspace under the user's home directory so the macOS
        // profile proves its broad home-directory denial is correctly
        // narrowed back to this approved workspace.
        let workspace = tempfile::Builder::new()
            .prefix(".sandbox-test-")
            .tempdir_in(std::env::current_dir().unwrap())
            .unwrap();
        let outside = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::detect(workspace.path());
        if matches!(sandbox.backend, Backend::Unavailable(_)) {
            // The interactive fallback path is exercised by the CLI. This
            // test validates confinement only where a backend can start.
            return;
        }

        let gate = ApprovalGate::default();
        let safe = sandbox
            .run("/usr/bin/true", &[], false, &gate)
            .await
            .unwrap();
        assert_eq!(safe.exit_code, Some(0));
        assert!(safe.sandboxed);

        std::fs::write(workspace.path().join(".env"), "OPENAI_API_KEY=do-not-read").unwrap();
        let secret_read = sandbox
            .run(
                "/bin/cat",
                &[workspace.path().join(".env").to_string_lossy().to_string()],
                false,
                &gate,
            )
            .await
            .unwrap();
        assert_ne!(secret_read.exit_code, Some(0));
        assert!(!secret_read.stdout.contains("do-not-read"));

        let workspace_config = workspace.path().join(".code-agent-cli.config");
        std::fs::write(&workspace_config, "api_key=do-not-read-either").unwrap();
        let config_read = sandbox
            .run(
                "/bin/cat",
                &[workspace_config.to_string_lossy().to_string()],
                false,
                &gate,
            )
            .await
            .unwrap();
        assert_ne!(config_read.exit_code, Some(0));
        assert!(!config_read.stdout.contains("do-not-read-either"));

        let environment = sandbox
            .run("/usr/bin/env", &[], false, &gate)
            .await
            .unwrap();
        assert_eq!(environment.exit_code, Some(0));
        assert!(!environment.stdout.contains("OPENAI_API_KEY"));
        assert!(!environment.stdout.contains("ANTHROPIC_API_KEY"));

        let sentinel = outside.path().join("must-not-exist");
        let blocked = sandbox
            .run(
                "/usr/bin/touch",
                &[sentinel.to_string_lossy().to_string()],
                false,
                &gate,
            )
            .await
            .unwrap();
        assert_ne!(blocked.exit_code, Some(0));
        assert!(!sentinel.exists());

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let denied = sandbox
            .run(
                "/usr/bin/curl",
                &["-sS".into(), "--max-time".into(), "1".into(), url],
                false,
                &gate,
            )
            .await
            .unwrap();
        assert_ne!(denied.exit_code, Some(0));
        drop(listener);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await
                .unwrap();
        });
        let allowed = sandbox
            .run(
                "/usr/bin/curl",
                &["-sS".into(), "--max-time".into(), "1".into(), url],
                true,
                &gate,
            )
            .await
            .unwrap();
        assert_eq!(allowed.exit_code, Some(0));
        assert_eq!(allowed.stdout, "ok");
        server.await.unwrap();
    }
}
