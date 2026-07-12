//! Command-line support for creating, running, and deploying Python agents.

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use tar::Builder;

/// The local Python inference server embedded into the CLI binary.
///
/// Keeping this beside the CLI avoids requiring a separate `server/` checkout
/// for `ferrant run`.
const RUNTIME: &str = include_str!("ferrant_server.py");
const ENVIRONMENT_PLACEHOLDER: &str = "replace-with-environment-id";

#[derive(Parser)]
#[command(
    name = "ferrant",
    about = "Create, run, and deploy Ferrant Python agents"
)]
struct Cli {
    /// Override the deployment server configured in deploy.yml.
    #[arg(long, global = true, env = "FERRANT_SERVER")]
    server: Option<String>,
    /// Console token for login, or a direct Bearer-token override for other commands.
    #[arg(long, global = true, env = "FERRANT_TOKEN", hide_env_values = true)]
    token: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Sign in and save a 12-hour access token for the deployment server.
    Login {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
        /// Account email associated with the console-issued token.
        #[arg(long, env = "FERRANT_EMAIL")]
        email: String,
    },
    /// Remove the saved token for the deployment server.
    Logout {
        #[arg(short, long, default_value = "deploy.yml")]
        config: PathBuf,
    },
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
        /// Create a new immutable release for an existing deployment.
        #[arg(long)]
        deployment: Option<String>,
        /// Source revision recorded with the release.
        #[arg(long)]
        git_sha: Option<String>,
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

#[derive(Debug, Deserialize, Serialize)]
struct DeployConfig {
    #[serde(default = "default_name")]
    name: String,
    handler: String,
    #[serde(default = "default_port")]
    port: u16,
    /// Base URL for the Ferrant deployment server.
    #[serde(default)]
    server: String,
    /// Target environment ID from the Ferrant control plane.
    #[serde(default)]
    environment: String,
    #[serde(default = "default_cpu")]
    cpu: f64,
    #[serde(default = "default_memory_mb")]
    memory_mb: u64,
    #[serde(default = "default_replicas")]
    replicas: u16,
    #[serde(default = "default_auth")]
    auth: String,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

fn default_name() -> String {
    "ferrant-agent".to_owned()
}
fn default_port() -> u16 {
    8000
}
fn default_cpu() -> f64 {
    0.5
}
fn default_memory_mb() -> u64 {
    512
}
fn default_replicas() -> u16 {
    1
}
fn default_auth() -> String {
    "private".to_owned()
}

pub fn run_from_env() -> Result<()> {
    execute(&std::env::args().skip(1).collect::<Vec<_>>())
}

/// Execute the CLI with arguments that exclude the executable name.
pub fn execute(args: &[String]) -> Result<()> {
    let cli =
        Cli::try_parse_from(std::iter::once("ferrant".to_owned()).chain(args.iter().cloned()))?;
    let server = cli.server.as_deref();
    let token = cli.token.as_deref();
    match cli.command {
        Commands::Login { config, email } => login(&config, server, token, &email),
        Commands::Logout { config } => logout(&config, server),
        Commands::Init { directory } => init(&directory),
        Commands::Run { config, port } => run(&config, port),
        Commands::Deploy {
            config,
            deployment,
            git_sha,
        } => deploy(
            &config,
            server,
            token,
            deployment.as_deref(),
            git_sha.as_deref(),
        ),
        Commands::Status { config, deployment } => {
            deployment_status(&config, &deployment, server, token)
        }
        Commands::Logs {
            config,
            deployment,
            stream,
        } => deployment_logs(&config, &deployment, stream, server, token),
        Commands::Restart { config, deployment } => {
            restart_deployment(&config, &deployment, server, token)
        }
        Commands::Stop { config, deployment } => {
            stop_deployment(&config, &deployment, server, token)
        }
    }
}

#[derive(Deserialize)]
struct OrganizationResponse {
    id: String,
}

#[derive(Deserialize)]
struct ProjectResponse {
    id: String,
    name: String,
}

#[derive(Deserialize)]
struct EnvironmentResponse {
    id: String,
    name: String,
}

#[derive(Serialize)]
struct ProjectRequest<'a> {
    name: &'a str,
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
    let manifest = DeployConfig {
        name: name.to_owned(),
        handler: "agent:reply".to_owned(),
        port: default_port(),
        server: "https://deploy.example.com".to_owned(),
        environment: ENVIRONMENT_PLACEHOLDER.to_owned(),
        cpu: default_cpu(),
        memory_mb: default_memory_mb(),
        replicas: default_replicas(),
        auth: default_auth(),
        env: BTreeMap::new(),
    };
    fs::write(&config, serde_yaml::to_string(&manifest)?)?;
    println!("Created {} and {}", agent.display(), config.display());
    println!("The first authenticated deploy will create the Ferrant project and development environment");
    println!("Next: cd {} && ferrant run", directory.display());
    Ok(())
}

fn create_remote_project(server: &str, token: &str, name: &str) -> Result<String> {
    let organizations_url = deployment_url(server, "organizations")?;
    tokio::runtime::Runtime::new()?.block_on(async {
        let client = reqwest::Client::new();
        let response = client
            .get(&organizations_url)
            .bearer_auth(token)
            .send()
            .await
            .with_context(|| format!("failed to contact {organizations_url}"))?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(server_response_error(status, &body));
        }
        let organizations: Vec<OrganizationResponse> = serde_json::from_str(&body)
            .context("deployment server returned invalid organizations")?;
        let organization = organizations
            .first()
            .ok_or_else(|| anyhow!("the authenticated token has no organization"))?;

        let projects_url = deployment_url(
            server,
            &format!("organizations/{}/projects", organization.id),
        )?;
        let response = client
            .get(&projects_url)
            .bearer_auth(token)
            .send()
            .await
            .with_context(|| format!("failed to contact {projects_url}"))?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(server_response_error(status, &body));
        }
        let projects: Vec<ProjectResponse> =
            serde_json::from_str(&body).context("deployment server returned invalid projects")?;
        let project = if let Some(project) =
            projects.into_iter().find(|project| project.name == name)
        {
            project
        } else {
            let response = client
                .post(&projects_url)
                .bearer_auth(token)
                .json(&ProjectRequest { name })
                .send()
                .await
                .with_context(|| format!("failed to contact {projects_url}"))?;
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                return Err(server_response_error(status, &body));
            }
            serde_json::from_str(&body).context("deployment server returned an invalid project")?
        };

        let environments_url =
            deployment_url(server, &format!("projects/{}/environments", project.id))?;
        let response = client
            .get(&environments_url)
            .bearer_auth(token)
            .send()
            .await
            .with_context(|| format!("failed to contact {environments_url}"))?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(server_response_error(status, &body));
        }
        let environments: Vec<EnvironmentResponse> = serde_json::from_str(&body)
            .context("deployment server returned invalid environments")?;
        environments
            .into_iter()
            .find(|environment| environment.name == "development")
            .map(|environment| environment.id)
            .ok_or_else(|| anyhow!("new project did not contain a development environment"))
    })
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
    endpoint: Option<String>,
    #[serde(default)]
    endpoint_key: Option<String>,
}

#[derive(Serialize)]
struct DeploymentManifest<'a> {
    cpu: f64,
    memory_mb: u64,
    replicas: u16,
    auth: &'a str,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    env: &'a BTreeMap<String, String>,
}

fn deploy(
    config_path: &Path,
    server_override: Option<&str>,
    token_override: Option<&str>,
    deployment: Option<&str>,
    git_sha: Option<&str>,
) -> Result<()> {
    let (mut config, app_dir) = load_config(config_path)?;
    let deployment = deployment.map(validate_deployment_id).transpose()?;
    let archive = package_application(&app_dir)?;
    let server = resolve_server(&config.server, server_override)?;
    let token = authentication_token(&server, token_override)?;
    if environment_requires_provisioning(&config.environment) {
        let environment = create_remote_project(&server, &token, &config.name)?;
        config.server = server.clone();
        config.environment = environment;
        fs::write(config_path, serde_yaml::to_string(&config)?).with_context(|| {
            format!(
                "created the Ferrant project but failed to save its environment ID to {}",
                config_path.display()
            )
        })?;
        println!(
            "Created/reused Ferrant project {} and saved its development environment",
            config.name
        );
    }
    let environment = config.environment.trim();
    let url = deployment_url(&server, "deployments")?;
    let manifest = serde_json::to_string(&DeploymentManifest {
        cpu: config.cpu,
        memory_mb: config.memory_mb,
        replicas: config.replicas,
        auth: &config.auth,
        env: &config.env,
    })?;
    let (status, body) = tokio::runtime::Runtime::new()?
        .block_on(async {
            let client = reqwest::Client::new();
            let mut request = client
                .post(url)
                .bearer_auth(token)
                .header("content-type", "application/gzip")
                .header("x-ferrant-name", &config.name)
                .header("x-ferrant-handler", &config.handler)
                .header("x-ferrant-environment", environment)
                .header("x-ferrant-manifest", manifest);
            if let Some(deployment) = deployment {
                request = request.header("x-ferrant-deployment", deployment);
            }
            if let Some(git_sha) = git_sha {
                request = request.header("x-ferrant-git-sha", git_sha);
            }
            let response = request.body(archive).send().await?;
            let status = response.status();
            let body = response.text().await?;
            Ok::<_, reqwest::Error>((status, body))
        })
        .context("failed to contact deployment server")?;
    if !status.is_success() {
        return Err(server_response_error(status, &body));
    }
    let deployment: DeploymentResponse =
        serde_json::from_str(&body).context("deployment server returned an invalid response")?;
    println!("Deployed {} ({})", config.name, deployment.id);
    if let Some(endpoint) = deployment.endpoint {
        println!("Inference endpoint: {endpoint}/infer");
    }
    if let Some(endpoint_key) = deployment.endpoint_key {
        println!("Endpoint key (shown once): {endpoint_key}");
    }
    Ok(())
}

fn deployment_status(
    config_path: &Path,
    deployment: &str,
    server: Option<&str>,
    token: Option<&str>,
) -> Result<()> {
    deployment_request(
        config_path,
        deployment,
        "",
        reqwest::Method::GET,
        "status",
        server,
        token,
    )
}

fn deployment_logs(
    config_path: &Path,
    deployment: &str,
    stream: bool,
    server_override: Option<&str>,
    token_override: Option<&str>,
) -> Result<()> {
    let (config, _) = load_config(config_path)?;
    let deployment = validate_deployment_id(deployment)?;
    let suffix = if stream { "/logs?follow=true" } else { "/logs" };
    let server = resolve_server(&config.server, server_override)?;
    let token = authentication_token(&server, token_override)?;
    let url = deployment_url(&server, &format!("deployments/{deployment}{suffix}"))?;
    tokio::runtime::Runtime::new()?
        .block_on(async {
            let response = reqwest::Client::new()
                .get(url)
                .bearer_auth(token)
                .send()
                .await?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await?;
                return Err(server_response_error(status, &body));
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

fn restart_deployment(
    config_path: &Path,
    deployment: &str,
    server: Option<&str>,
    token: Option<&str>,
) -> Result<()> {
    deployment_request(
        config_path,
        deployment,
        "/restart",
        reqwest::Method::POST,
        "restart",
        server,
        token,
    )
}

fn stop_deployment(
    config_path: &Path,
    deployment: &str,
    server: Option<&str>,
    token: Option<&str>,
) -> Result<()> {
    deployment_request(
        config_path,
        deployment,
        "/stop",
        reqwest::Method::POST,
        "stop",
        server,
        token,
    )
}

fn deployment_request(
    config_path: &Path,
    deployment: &str,
    action: &str,
    method: reqwest::Method,
    description: &str,
    server_override: Option<&str>,
    token_override: Option<&str>,
) -> Result<()> {
    let (config, _) = load_config(config_path)?;
    let deployment = validate_deployment_id(deployment)?;
    let path = format!("deployments/{deployment}{action}");
    let server = resolve_server(&config.server, server_override)?;
    let token = authentication_token(&server, token_override)?;
    let url = deployment_url(&server, &path)?;
    let (status, body) = tokio::runtime::Runtime::new()?
        .block_on(async {
            let response = reqwest::Client::new()
                .request(method, url)
                .bearer_auth(token)
                .send()
                .await?;
            let status = response.status();
            let body = response.text().await?;
            Ok::<_, reqwest::Error>((status, body))
        })
        .with_context(|| format!("failed to contact deployment server for {description}"))?;
    if !status.is_success() {
        return Err(server_response_error(status, &body));
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

fn environment_requires_provisioning(environment: &str) -> bool {
    let environment = environment.trim();
    environment.is_empty() || environment == ENVIRONMENT_PLACEHOLDER
}

fn deployment_url(server: &str, path: &str) -> Result<String> {
    let server = server.trim_end_matches('/');
    if server.is_empty() {
        bail!("server must be set in deploy.yml");
    }
    Ok(format!("{server}/{path}"))
}

fn resolve_server(configured: &str, server_override: Option<&str>) -> Result<String> {
    let server = server_override.unwrap_or(configured).trim_end_matches('/');
    if server.is_empty() {
        bail!("server must be set in deploy.yml or with --server");
    }
    let parsed = reqwest::Url::parse(server).context("server must be a valid URL")?;
    if !matches!(parsed.scheme(), "http" | "https") {
        bail!("server URL must use http or https");
    }
    Ok(server.to_owned())
}

fn server_response_error(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    if status == reqwest::StatusCode::UNAUTHORIZED {
        anyhow!(
            "authentication failed: {body}\nRun `ferrant login` or provide FERRANT_TOKEN and retry."
        )
    } else {
        anyhow!("deployment server returned {status}: {body}")
    }
}

#[derive(Debug, Deserialize)]
struct LoginResponse {
    access_token: String,
    expires_at: DateTime<Utc>,
}

#[derive(Default, Deserialize, Serialize)]
struct CredentialStore {
    #[serde(default)]
    servers: BTreeMap<String, StoredCredential>,
}

#[derive(Deserialize, Serialize)]
struct StoredCredential {
    token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<DateTime<Utc>>,
}

#[derive(Serialize)]
struct LoginRequest<'a> {
    email: &'a str,
    token: &'a str,
}

fn login(
    config_path: &Path,
    server_override: Option<&str>,
    supplied_token: Option<&str>,
    email: &str,
) -> Result<()> {
    let server = login_server(config_path, server_override)?;
    let email = email.trim();
    if email.is_empty() {
        bail!("email is required; pass --email or set FERRANT_EMAIL");
    }
    let supplied_token = supplied_token
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .ok_or_else(|| {
            anyhow!("a console-issued token is required; pass --token or set FERRANT_TOKEN")
        })?;
    let url = deployment_url(&server, "auth/token-login")?;
    let (status, body) = tokio::runtime::Runtime::new()?
        .block_on(async {
            let response = reqwest::Client::new()
                .post(url)
                .json(&LoginRequest {
                    email,
                    token: supplied_token,
                })
                .send()
                .await?;
            let status = response.status();
            let body = response.text().await?;
            Ok::<_, reqwest::Error>((status, body))
        })
        .context("failed to contact deployment server for token login")?;
    if !status.is_success() {
        bail!("token login failed ({status}): {body}");
    }
    let response: LoginResponse =
        serde_json::from_str(&body).context("login server returned an invalid response")?;
    let credential = StoredCredential {
        token: response.access_token,
        expires_at: Some(response.expires_at),
    };
    let expires_at = credential.expires_at;
    save_credential(&server, credential)?;
    println!(
        "Logged in to {server}; session expires at {}",
        expires_at.expect("token login always has an expiry")
    );
    Ok(())
}

fn logout(config_path: &Path, server_override: Option<&str>) -> Result<()> {
    let server = login_server(config_path, server_override)?;
    let mut credentials = load_credentials()?;
    if credentials.servers.remove(&server).is_some() {
        write_credentials(&credentials)?;
        println!("Logged out of {server}");
    } else {
        println!("No saved login for {server}");
    }
    Ok(())
}

fn login_server(config_path: &Path, server_override: Option<&str>) -> Result<String> {
    if server_override.is_some() {
        return resolve_server("", server_override);
    }
    let (config, _) = load_config(config_path).with_context(|| {
        format!(
            "could not determine server from {}; pass --server",
            config_path.display()
        )
    })?;
    resolve_server(&config.server, None)
}

fn authentication_token(server: &str, supplied: Option<&str>) -> Result<String> {
    if let Some(token) = supplied.map(str::trim).filter(|token| !token.is_empty()) {
        return Ok(token.to_owned());
    }
    let credentials = load_credentials()?;
    let credential = credentials.servers.get(server).ok_or_else(|| {
        anyhow!("not logged in to {server}; run `ferrant login` or set FERRANT_TOKEN")
    })?;
    if credential
        .expires_at
        .is_some_and(|expires_at| expires_at <= Utc::now())
    {
        bail!("login for {server} expired; run `ferrant login` again");
    }
    Ok(credential.token.clone())
}

fn credentials_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("FERRANT_CREDENTIALS_PATH") {
        return Ok(PathBuf::from(path));
    }
    if let Some(directory) = std::env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(directory)
            .join("ferrant")
            .join("credentials.json"));
    }
    if cfg!(windows) {
        if let Some(directory) = std::env::var_os("APPDATA") {
            return Ok(PathBuf::from(directory)
                .join("Ferrant")
                .join("credentials.json"));
        }
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("cannot locate the user config directory"))?;
    Ok(home
        .join(".config")
        .join("ferrant")
        .join("credentials.json"))
}

fn load_credentials() -> Result<CredentialStore> {
    let path = credentials_path()?;
    match fs::read_to_string(&path) {
        Ok(source) => serde_json::from_str(&source)
            .with_context(|| format!("invalid credential file {}", path.display())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(CredentialStore::default()),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn save_credential(server: &str, credential: StoredCredential) -> Result<()> {
    let mut credentials = load_credentials()?;
    credentials.servers.insert(server.to_owned(), credential);
    write_credentials(&credentials)
}

fn write_credentials(credentials: &CredentialStore) -> Result<()> {
    let path = credentials_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    let body = serde_json::to_vec_pretty(credentials)?;
    file.write_all(&body)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.write_all(b"\n")?;
    Ok(())
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
    if !config.cpu.is_finite() || config.cpu <= 0.0 {
        bail!("cpu must be a positive number");
    }
    if config.memory_mb < 128 {
        bail!("memory_mb must be at least 128");
    }
    if config.replicas == 0 {
        bail!("replicas must be at least 1");
    }
    if !matches!(config.auth.as_str(), "private" | "public") {
        bail!("auth must be either private or public");
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

    #[test]
    fn accepts_server_and_token_as_global_options() {
        let cli = Cli::try_parse_from([
            "ferrant",
            "deploy",
            "--server",
            "https://api.example.com",
            "--token",
            "fdt_test",
            "--deployment",
            "existing-deployment",
            "--git-sha",
            "abc123",
        ])
        .unwrap();
        assert_eq!(cli.server.as_deref(), Some("https://api.example.com"));
        assert_eq!(cli.token.as_deref(), Some("fdt_test"));
        assert!(matches!(cli.command, Commands::Deploy { .. }));
    }

    #[test]
    fn token_login_requires_email_and_sends_no_password() {
        assert!(Cli::try_parse_from([
            "ferrant",
            "login",
            "--server",
            "https://api.example.com",
            "--token",
            "fdt_test",
        ])
        .is_err());

        let request = serde_json::to_value(LoginRequest {
            email: "owner@example.com",
            token: "fdt_test",
        })
        .unwrap();
        assert_eq!(request["email"], "owner@example.com");
        assert_eq!(request["token"], "fdt_test");
        assert!(request.get("password").is_none());
    }

    #[test]
    fn serializes_the_control_plane_manifest() {
        let env = BTreeMap::from([("MODEL".to_owned(), "gpt-5".to_owned())]);
        let manifest = serde_json::to_value(DeploymentManifest {
            cpu: 1.0,
            memory_mb: 1024,
            replicas: 1,
            auth: "private",
            env: &env,
        })
        .unwrap();
        assert_eq!(manifest["cpu"], 1.0);
        assert_eq!(manifest["memory_mb"], 1024);
        assert_eq!(manifest["auth"], "private");
        assert_eq!(manifest["env"]["MODEL"], "gpt-5");
    }

    #[test]
    fn recognizes_the_generated_environment_placeholder() {
        assert!(environment_requires_provisioning(ENVIRONMENT_PLACEHOLDER));
        assert!(environment_requires_provisioning("  "));
        assert!(!environment_requires_provisioning(" env-123 "));
    }

    #[test]
    fn normalizes_and_validates_server_overrides() {
        assert_eq!(
            resolve_server("", Some("https://api.example.com/")).unwrap(),
            "https://api.example.com"
        );
        assert!(resolve_server("", Some("file:///tmp/server")).is_err());
    }
}
