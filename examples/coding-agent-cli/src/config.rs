use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use ferragent::error::Result as AgentResult;
use ferragent::llm::anthropic::AnthropicModel;
use ferragent::llm::openai::OpenAiModel;
use ferragent::llm::{Model, ModelResponse};
use ferragent::message::Message;
use ferragent::tool::ToolSpec;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const CONFIG_FILE: &str = ".code-agent-cli.config";
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

pub enum Provider {
    OpenAi(OpenAiModel),
    Anthropic(AnthropicModel),
    Compatible(OpenAiModel),
}

#[async_trait]
impl Model for Provider {
    fn id(&self) -> &str {
        match self {
            Self::OpenAi(model) | Self::Compatible(model) => model.id(),
            Self::Anthropic(model) => model.id(),
        }
    }

    async fn generate(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> AgentResult<ModelResponse> {
        match self {
            Self::OpenAi(model) | Self::Compatible(model) => model.generate(messages, tools).await,
            Self::Anthropic(model) => model.generate(messages, tools).await,
        }
    }
}

pub fn select_provider() -> Result<Provider> {
    if let Some((path, config)) = load_config()? {
        println!("Using provider configuration from {}", path.display());
        return config.into_provider();
    }

    println!("No ~/{CONFIG_FILE} found; using environment-based configuration.");
    println!("Choose a model provider:");
    println!("  1. OpenAI");
    println!("  2. Anthropic");
    println!("  3. Local/OpenAI-compatible server");

    match prompt("Provider [1-3]")?.trim() {
        "1" => {
            let key = required_env("OPENAI_API_KEY")?;
            let model = env_or("OPENAI_MODEL", "gpt-5-mini");
            Ok(Provider::OpenAi(OpenAiModel::new(model, key)))
        }
        "2" => {
            let key = required_env("ANTHROPIC_API_KEY")?;
            let model = env_or("ANTHROPIC_MODEL", "claude-sonnet-4-6");
            Ok(Provider::Anthropic(
                AnthropicModel::new(model, key).with_max_tokens(8192),
            ))
        }
        "3" => {
            let base_url = env_or("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:8080/v1");
            let model = env_or("OPENAI_COMPATIBLE_MODEL", "LiquidAI/LFM2.5-230M-GGUF:Q8_0");
            let key = env_or("OPENAI_COMPATIBLE_API_KEY", "not-needed");
            Ok(Provider::Compatible(
                OpenAiModel::new(model, key).with_base_url(base_url),
            ))
        }
        _ => bail!("provider must be 1, 2, or 3"),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct FileConfig {
    provider: String,
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
}

impl FileConfig {
    fn parse(contents: &str) -> Result<Self> {
        let mut values = HashMap::new();
        for (index, raw_line) in contents.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, raw_value) = line
                .split_once('=')
                .with_context(|| format!("config line {} must use key=value", index + 1))?;
            let key = key.trim().to_ascii_lowercase();
            if !matches!(key.as_str(), "provider" | "api_key" | "model" | "base_url") {
                bail!("unknown config key on line {}: {key}", index + 1);
            }
            let value = unquote(raw_value.trim())?;
            if values.insert(key.clone(), value).is_some() {
                bail!("duplicate config key: {key}");
            }
        }

        let provider = values
            .remove("provider")
            .context("config must define provider=openai, anthropic, or compatible")?
            .to_ascii_lowercase();
        Ok(Self {
            provider,
            api_key: values.remove("api_key"),
            model: values.remove("model"),
            base_url: values.remove("base_url"),
        })
    }

    fn into_provider(self) -> Result<Provider> {
        match self.provider.as_str() {
            "openai" => {
                self.reject_base_url()?;
                let key = required_value(self.api_key, "api_key")?;
                let model = self.model.unwrap_or_else(|| "gpt-5-mini".to_string());
                Ok(Provider::OpenAi(OpenAiModel::new(model, key)))
            }
            "anthropic" => {
                self.reject_base_url()?;
                let key = required_value(self.api_key, "api_key")?;
                let model = self
                    .model
                    .unwrap_or_else(|| "claude-sonnet-4-6".to_string());
                Ok(Provider::Anthropic(
                    AnthropicModel::new(model, key).with_max_tokens(8192),
                ))
            }
            "compatible" | "openai-compatible" | "local" => {
                let base_url = required_value(self.base_url, "base_url")?;
                if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
                    bail!("base_url must start with http:// or https://");
                }
                let model = self.model.unwrap_or_else(|| "local-model".to_string());
                let key = self.api_key.unwrap_or_else(|| "not-needed".to_string());
                Ok(Provider::Compatible(
                    OpenAiModel::new(model, key)
                        .with_base_url(base_url.trim_end_matches('/').to_string()),
                ))
            }
            other => bail!("unsupported config provider: {other}"),
        }
    }

    fn reject_base_url(&self) -> Result<()> {
        if self.base_url.is_some() {
            bail!("base_url is only valid for provider=compatible");
        }
        Ok(())
    }
}

fn load_config() -> Result<Option<(PathBuf, FileConfig)>> {
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        return Ok(None);
    };
    let path = PathBuf::from(home).join(CONFIG_FILE);
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some((path.clone(), load_config_file(&path)?)))
}

fn load_config_file(path: &std::path::Path) -> Result<FileConfig> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("cannot inspect config file {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "config must be a regular file, not a symlink: {}",
            path.display()
        );
    }
    if metadata.len() > MAX_CONFIG_BYTES {
        bail!("config file exceeds the {MAX_CONFIG_BYTES}-byte limit");
    }
    check_private_permissions(path, &metadata)?;
    let contents = fs::read_to_string(path)
        .with_context(|| format!("cannot read config file {}", path.display()))?;
    FileConfig::parse(&contents)
}

#[cfg(unix)]
fn check_private_permissions(path: &std::path::Path, metadata: &fs::Metadata) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if metadata.permissions().mode() & 0o077 != 0 {
        bail!(
            "config contains credentials and must not be accessible by group/others; run: chmod 600 {}",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_private_permissions(_path: &std::path::Path, _metadata: &fs::Metadata) -> Result<()> {
    Ok(())
}

fn required_value(value: Option<String>, name: &str) -> Result<String> {
    match value {
        Some(value) if !value.is_empty() => Ok(value),
        _ => bail!("config must define a non-empty {name}"),
    }
}

fn unquote(value: &str) -> Result<String> {
    if value.len() >= 2 {
        let first = value.as_bytes()[0];
        let last = value.as_bytes()[value.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return Ok(value[1..value.len() - 1].to_string());
        }
    }
    if value.starts_with(['"', '\'']) || value.ends_with(['"', '\'']) {
        bail!("config value has mismatched quotes");
    }
    Ok(value.to_string())
}

pub fn prompt(label: &str) -> Result<String> {
    print!("{label}: ");
    io::stdout().flush().context("failed to flush stdout")?;
    let mut value = String::new();
    io::stdin()
        .read_line(&mut value)
        .context("failed to read input")?;
    Ok(value.trim().to_string())
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("set {name} in the environment or .env"))
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::{load_config_file, FileConfig};
    use std::fs;

    #[test]
    fn parses_each_provider_shape() {
        let openai = FileConfig::parse("provider=openai\napi_key='secret'\n").unwrap();
        assert_eq!(openai.provider, "openai");
        assert_eq!(openai.api_key.as_deref(), Some("secret"));

        let anthropic = FileConfig::parse(
            "# one provider per file\nprovider = anthropic\napi_key = secret\nmodel = claude\n",
        )
        .unwrap();
        assert_eq!(anthropic.provider, "anthropic");
        assert_eq!(anthropic.model.as_deref(), Some("claude"));

        let compatible = FileConfig::parse(
            "provider=compatible\nbase_url=http://127.0.0.1:8080/v1\nmodel=local\n",
        )
        .unwrap();
        assert_eq!(compatible.provider, "compatible");
        assert!(compatible.api_key.is_none());
    }

    #[test]
    fn rejects_typos_duplicates_and_missing_provider() {
        assert!(FileConfig::parse("provder=openai").is_err());
        assert!(FileConfig::parse("provider=openai\nprovider=anthropic").is_err());
        assert!(FileConfig::parse("api_key=secret").is_err());
        assert!(FileConfig::parse("provider=\"openai").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn loads_only_private_regular_config_files() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config");
        fs::write(&path, "provider=openai\napi_key=secret\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(load_config_file(&path).unwrap().provider, "openai");

        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_config_file(&path).is_err());

        let link = directory.path().join("config-link");
        symlink(&path, &link).unwrap();
        assert!(load_config_file(&link).is_err());
    }
}
