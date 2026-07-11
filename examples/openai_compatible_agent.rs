use ferrant::llm::openai::OpenAiModel;
use ferrant::Agent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
    let model_name = std::env::var("OPENAI_COMPATIBLE_MODEL")
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-230M-GGUF:Q8_0".to_string());
    // llama-server does not require a key by default, but the shared client
    // accepts one for compatible servers that do enforce authentication.
    let api_key =
        std::env::var("OPENAI_COMPATIBLE_API_KEY").unwrap_or_else(|_| "not-needed".to_string());

    let model = OpenAiModel::new(model_name, api_key).with_base_url(base_url);
    let mut agent = Agent::builder(model)
        .instructions("You are a concise and helpful assistant.")
        .build();

    let answer = agent
        .run("In one short sentence, explain why go is memory safe.")
        .await?;
    println!("{answer}");

    Ok(())
}
