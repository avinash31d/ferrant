//! Command-line support for creating, running, and locally deploying Python agents.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const RUNTIME: &str = include_str!("../server/runtime/ferrant_server.py");
const DEFAULT_IMAGE: &str = "ferrant-runner:latest";

#[derive(Parser)]
#[command(
    name = "ferrant",
    about = "Create, run, and deploy Ferrant Python agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a minimal echo agent and deploy.yml manifest.
    Init {
        /// Directory to create. Defaults to the current directory.
        #[arg(default_value = ".")]
        directory: PathBuf,
    },
    /// Run the configured function locally behind Ferrant's inference server.
    Run {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        #[arg(long)]
        port: Option<u16>,
    },
    /// Start a new container from the reusable runner image and deploy the app.
    Deploy {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        #[arg(long, default_value = DEFAULT_IMAGE)]
        image: String,
    },
}

#[derive(Debug, Deserialize)]
struct DeployConfig {
    #[serde(default = "default_name")]
    name: String,
    handler: String,
    #[serde(default = "default_port")]
    port: u16,
}

fn default_name() -> String {
    "ferrant-agent".to_owned()
}
fn default_port() -> u16 {
    8000
}

pub fn run_from_env() -> Result<()> {
    execute(&std::env::args().skip(1).collect::<Vec<_>>())
}

/// Execute the CLI with arguments that exclude the executable name.
pub fn execute(args: &[String]) -> Result<()> {
    let cli =
        Cli::try_parse_from(std::iter::once("ferrant".to_owned()).chain(args.iter().cloned()))?;
    match cli.command {
        Commands::Init { directory } => init(&directory),
        Commands::Run { config, port } => run(&config, port),
        Commands::Deploy { config, image } => deploy(&config, &image),
    }
}

fn init(directory: &Path) -> Result<()> {
    let agent = directory.join("agent.py");
    let config = directory.join("deploy.yml");
    if agent.exists() || config.exists() {
        bail!(
            "{} already contains agent.py or deploy.yml",
            directory.display()
        );
    }
    fs::create_dir_all(directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;
    fs::write(&agent, "\"\"\"A minimal Ferrant function agent.\"\"\"\n\ndef reply(input: dict) -> dict:\n    return {\"echo\": input}\n")?;
    let name = directory
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("echo-agent");
    fs::write(
        &config,
        format!("name: {name}\nhandler: agent:reply\nport: 8000\n"),
    )?;
    println!("Created {} and {}", agent.display(), config.display());
    println!("Next: cd {} && ferrant run", directory.display());
    Ok(())
}

fn run(config_path: &Path, override_port: Option<u16>) -> Result<()> {
    let (config, app_dir) = load_config(config_path)?;
    let port = override_port.unwrap_or(config.port).to_string();
    let app_dir = path_string(&app_dir)?;
    println!("Running {} at http://127.0.0.1:{port}/infer", config.name);
    let status = Command::new(python_command())
        .arg(materialize_runtime()?)
        .args([
            "--app-dir",
            &app_dir,
            "--handler",
            &config.handler,
            "--port",
            &port,
        ])
        .status()
        .context("failed to start Python; install Python 3.9+ and retry")?;
    if !status.success() {
        bail!("local inference server exited with {status}");
    }
    Ok(())
}

fn deploy(config_path: &Path, image: &str) -> Result<()> {
    let (config, app_dir) = load_config(config_path)?;
    docker(["info"])?;
    docker(["image", "inspect", image]).with_context(|| {
        format!("runner image {image:?} is unavailable; build it with `docker build -t {DEFAULT_IMAGE} -f server/docker/Dockerfile server`")
    })?;
    let container = format!(
        "ferrant-{}-{}",
        slug(&config.name),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    );
    let id = docker_output([
        "run",
        "-d",
        "--rm",
        "--name",
        &container,
        "--label",
        "ferrant.managed=true",
        "--label",
        &format!("ferrant.app={}", config.name),
        "-p",
        "127.0.0.1::8000",
        image,
    ])?;
    let deploy_result = (|| -> Result<String> {
        docker([
            "cp",
            &format!("{}/.", app_dir.display()),
            &format!("{container}:/app"),
        ])?;
        docker([
            "exec",
            "-d",
            "-w",
            "/app",
            &container,
            "python",
            "/opt/ferrant/ferrant_server.py",
            "--app-dir",
            "/app",
            "--handler",
            &config.handler,
            "--port",
            "8000",
        ])?;
        let port = docker_output(["port", &container, "8000/tcp"])?;
        let endpoint = format!(
            "http://127.0.0.1:{}",
            port.trim()
                .rsplit(':')
                .next()
                .ok_or_else(|| anyhow!("could not determine Docker port"))?
        );
        wait_for_health(&endpoint)?;
        Ok(endpoint)
    })();
    if deploy_result.is_err() {
        let _ = docker(["rm", "-f", &container]);
    }
    let endpoint = deploy_result?;
    println!("Deployed {} ({})", config.name, id.trim());
    println!("Inference endpoint: {endpoint}/infer");
    println!("Container: {container}");
    Ok(())
}

fn load_config(path: &Path) -> Result<(DeployConfig, PathBuf)> {
    let source =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let config: DeployConfig = serde_yaml::from_str(&source).context("invalid deploy.yml")?;
    if !config.handler.contains(':') {
        bail!("handler must use module:function syntax, for example agent:reply");
    }
    let app_dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .with_context(|| {
            format!(
                "failed to resolve application directory for {}",
                path.display()
            )
        })?;
    Ok((config, app_dir))
}

fn materialize_runtime() -> Result<PathBuf> {
    let path = std::env::temp_dir().join("ferrant-inference-server.py");
    fs::write(&path, RUNTIME).context("failed to write bundled inference server")?;
    Ok(path)
}

fn python_command() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

fn path_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("path must be valid UTF-8"))
}

fn docker<const N: usize>(args: [&str; N]) -> Result<()> {
    let output = Command::new("docker")
        .args(args)
        .output()
        .context("Docker is not installed or not on PATH")?;
    if output.status.success() {
        return Ok(());
    }
    bail!(
        "docker command failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn docker_output<const N: usize>(args: [&str; N]) -> Result<String> {
    let output = Command::new("docker")
        .args(args)
        .output()
        .context("Docker is not installed or not on PATH")?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned());
    }
    bail!(
        "docker command failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn wait_for_health(endpoint: &str) -> Result<()> {
    let address = endpoint.trim_start_matches("http://");
    for _ in 0..30 {
        if let Ok(mut stream) = TcpStream::connect(address) {
            stream.set_read_timeout(Some(Duration::from_millis(500)))?;
            stream.write_all(
                b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )?;
            let mut response = String::new();
            stream.read_to_string(&mut response)?;
            if response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200") {
                return Ok(());
            }
        }
        thread::sleep(Duration::from_millis(500));
    }
    bail!("deployment did not become healthy; inspect the container logs with `docker logs`");
}

fn slug(value: &str) -> String {
    let value: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    value
        .trim_matches('-')
        .chars()
        .take(40)
        .collect::<String>()
        .max("app".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_safe_for_docker_names() {
        assert_eq!(slug("My agent!"), "my-agent");
        assert_eq!(slug("---"), "app");
    }
}
