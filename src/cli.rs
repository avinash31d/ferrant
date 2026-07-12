//! Command-line support for creating, running, and deploying Python agents.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::StreamExt;
use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use tar::Builder;

/// The local Python inference server embedded into the CLI binary.
///
/// Keeping this beside the CLI avoids requiring a separate `server/` checkout
/// for `ferrant run`.
const RUNTIME: &str = include_str!("ferrant_server.py");

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
    /// Package the app and send it to a Ferrant deployment server.
    Deploy {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
    },
    /// Show the current state of a deployment.
    Status {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        /// Deployment ID returned by `ferrant deploy`.
        deployment: String,
    },
    /// Show logs produced by a deployment.
    Logs {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        /// Deployment ID returned by `ferrant deploy`.
        deployment: String,
        /// Keep the connection open and print new log output as it arrives.
        #[arg(long, alias = "follow")]
        stream: bool,
    },
    /// Restart a deployment.
    Restart {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        /// Deployment ID returned by `ferrant deploy`.
        deployment: String,
    },
    /// Stop a deployment.
    Stop {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        /// Deployment ID returned by `ferrant deploy`.
        deployment: String,
    },
}

#[derive(Debug, Deserialize)]
struct DeployConfig {
    #[serde(default = "default_name")]
    name: String,
    handler: String,
    #[serde(default = "default_port")]
    port: u16,
    /// Base URL for the Ferrant deployment server.
    #[serde(default)]
    server: String,
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
        Commands::Deploy { config } => deploy(&config),
        Commands::Status { config, deployment } => deployment_status(&config, &deployment),
        Commands::Logs {
            config,
            deployment,
            stream,
        } => deployment_logs(&config, &deployment, stream),
        Commands::Restart { config, deployment } => restart_deployment(&config, &deployment),
        Commands::Stop { config, deployment } => stop_deployment(&config, &deployment),
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
        format!(
            "name: {name}\nhandler: agent:reply\nport: 8000\nserver: https://deploy.example.com\n"
        ),
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

#[derive(Deserialize)]
struct DeploymentResponse {
    id: String,
    endpoint: String,
}

fn deploy(config_path: &Path) -> Result<()> {
    let (config, app_dir) = load_config(config_path)?;
    let archive = package_application(&app_dir)?;
    let url = deployment_url(&config.server, "deployments")?;
    let (status, body) = tokio::runtime::Runtime::new()?
        .block_on(async {
            let response = reqwest::Client::new()
                .post(url)
                .header("content-type", "application/gzip")
                .header("x-ferrant-name", &config.name)
                .header("x-ferrant-handler", &config.handler)
                .body(archive)
                .send()
                .await?;
            let status = response.status();
            let body = response.text().await?;
            Ok::<_, reqwest::Error>((status, body))
        })
        .context("failed to contact deployment server")?;
    if !status.is_success() {
        bail!("deployment server returned {status}: {body}");
    }
    let deployment: DeploymentResponse =
        serde_json::from_str(&body).context("deployment server returned an invalid response")?;
    println!("Deployed {} ({})", config.name, deployment.id);
    println!("Inference endpoint: {}/infer", deployment.endpoint);
    Ok(())
}

fn deployment_status(config_path: &Path, deployment: &str) -> Result<()> {
    deployment_request(config_path, deployment, "", reqwest::Method::GET, "status")
}

fn deployment_logs(config_path: &Path, deployment: &str, stream: bool) -> Result<()> {
    let (config, _) = load_config(config_path)?;
    let deployment = validate_deployment_id(deployment)?;
    let suffix = if stream { "/logs?follow=true" } else { "/logs" };
    let url = deployment_url(&config.server, &format!("deployments/{deployment}{suffix}"))?;
    tokio::runtime::Runtime::new()?
        .block_on(async {
            let response = reqwest::Client::new().get(url).send().await?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await?;
                bail!("deployment server returned {status}: {body}");
            }

            let mut output = io::stdout().lock();
            let mut body = response.bytes_stream();
            while let Some(chunk) = body.next().await {
                output
                    .write_all(&chunk?)
                    .context("failed to write deployment logs")?;
                output.flush().context("failed to flush deployment logs")?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .with_context(|| "failed to contact deployment server for logs")
}

fn restart_deployment(config_path: &Path, deployment: &str) -> Result<()> {
    deployment_request(
        config_path,
        deployment,
        "/restart",
        reqwest::Method::POST,
        "restart",
    )
}

fn stop_deployment(config_path: &Path, deployment: &str) -> Result<()> {
    deployment_request(
        config_path,
        deployment,
        "/stop",
        reqwest::Method::POST,
        "stop",
    )
}

fn deployment_request(
    config_path: &Path,
    deployment: &str,
    action: &str,
    method: reqwest::Method,
    description: &str,
) -> Result<()> {
    let (config, _) = load_config(config_path)?;
    let deployment = validate_deployment_id(deployment)?;
    let path = format!("deployments/{deployment}{action}");
    let url = deployment_url(&config.server, &path)?;
    let (status, body) = tokio::runtime::Runtime::new()?
        .block_on(async {
            let response = reqwest::Client::new().request(method, url).send().await?;
            let status = response.status();
            let body = response.text().await?;
            Ok::<_, reqwest::Error>((status, body))
        })
        .with_context(|| format!("failed to contact deployment server for {description}"))?;
    if !status.is_success() {
        bail!("deployment server returned {status}: {body}");
    }
    print_response(&body);
    Ok(())
}

fn validate_deployment_id(deployment: &str) -> Result<&str> {
    let deployment = deployment.trim();
    if deployment.is_empty()
        || !deployment
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        bail!("deployment ID may contain only letters, numbers, hyphens, and underscores");
    }
    Ok(deployment)
}

fn deployment_url(server: &str, path: &str) -> Result<String> {
    let server = server.trim_end_matches('/');
    if server.is_empty() {
        bail!("server must be set in deploy.yml");
    }
    Ok(format!("{server}/{path}"))
}

fn print_response(body: &str) {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_else(|_| body.to_owned())
        );
    } else {
        println!("{body}");
    }
}

fn package_application(app_dir: &Path) -> Result<Vec<u8>> {
    let mut archive = Builder::new(GzEncoder::new(Vec::new(), Compression::default()));
    archive
        .append_dir_all(".", app_dir)
        .context("failed to package application directory")?;
    let encoder = archive.into_inner()?;
    encoder
        .finish()
        .context("failed to compress application package")
}

fn load_config(path: &Path) -> Result<(DeployConfig, PathBuf)> {
    let source =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let config: DeployConfig = serde_yaml::from_str(&source).context("invalid deploy.yml")?;
    if !config.handler.contains(':') {
        bail!("handler must use module:function syntax, for example agent:reply");
    }
    let app_dir = config_directory(path)?;
    Ok((config, app_dir))
}

fn config_directory(path: &Path) -> Result<PathBuf> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .with_context(|| {
            format!(
                "failed to resolve application directory for {}",
                path.display()
            )
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packages_application() {
        let directory = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("agent.py"), "pass").unwrap();
        assert!(!package_application(&directory).unwrap().is_empty());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn resolves_a_relative_config_in_the_current_directory() {
        assert_eq!(
            config_directory(Path::new("deploy.yml")).unwrap(),
            Path::new(".").canonicalize().unwrap()
        );
    }

    #[test]
    fn builds_management_urls_from_the_manifest_server() {
        assert_eq!(
            deployment_url("https://deploy.example.com/", "deployments/abc").unwrap(),
            "https://deploy.example.com/deployments/abc"
        );
    }
}
