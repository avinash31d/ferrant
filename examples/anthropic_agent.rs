use ferrant::llm::anthropic::AnthropicModel;
use ferrant::tool::FunctionTool;
use ferrant::Agent;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
    let model = AnthropicModel::new("claude-sonnet-4-6", api_key).with_max_tokens(1024);

    let search_tool = FunctionTool::new(
        "search_docs",
        "Search internal documentation for a topic",
        json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        }),
        |args| {
            Box::pin(async move {
                let q = args["query"].as_str().unwrap_or_default().to_string();
                Ok(format!(
                    "Found 3 docs mentioning '{q}': Setup Guide, API Reference, FAQ."
                ))
            })
        },
    );

    let mut agent = Agent::builder(model)
        .instructions("You are a support assistant. Search docs before answering questions about the product.")
        .tool(search_tool)
        .build();

    let answer = agent.run("How do I authenticate with the API?").await?;
    println!("{answer}");

    Ok(())
}
