use crate::{AgentError, ModelResponse, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Validate JSON using full JSON Schema draft auto-detection.
pub fn validate_json(schema: &Value, value: &Value) -> std::result::Result<(), Vec<String>> {
    let validator = jsonschema::validator_for(schema)
        .map_err(|error| vec![format!("invalid JSON Schema: {error}")])?;
    let errors = validator
        .iter_errors(value)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn parse_structured<T: DeserializeOwned>(
    response: &ModelResponse,
    schema: &Value,
) -> Result<T> {
    let text = response.content.as_deref().unwrap_or_default();
    let value: Value = serde_json::from_str(text).map_err(AgentError::Serde)?;
    validate_json(schema, &value)
        .map_err(|errors| AgentError::SchemaValidation(errors.join("; ")))?;
    serde_json::from_value(value).map_err(AgentError::Serde)
}
