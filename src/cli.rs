//! Command-line support for creating, running, and deploying Python agents.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tar::Builder;

const RUNTIME: &str = include_str!("../server/runtime/ferrant_server.py");

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
        /// Deployment server URL (or set FERRANT_DEPLOY_SERVER).
        #[arg(long, env = "FERRANT_DEPLOY_SERVER")]
        server: String,
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
        Commands::Deploy { config, server } => deploy(&config, &server),
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

#[derive(Deserialize)]
struct DeploymentResponse {
    id: String,
    endpoint: String,
}

fn deploy(config_path: &Path, server: &str) -> Result<()> {
    let (config, app_dir) = load_config(config_path)?;
    let archive = package_application(&app_dir)?;
    let url = format!("{}/deployments", server.trim_end_matches('/'));
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
}
