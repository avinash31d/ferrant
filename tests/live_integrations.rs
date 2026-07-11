//! Opt-in network/process integration tests. Run with:
//! `cargo test --test live_integrations -- --ignored --nocapture`

use ferragent::llm::anthropic::AnthropicModel;
use ferragent::llm::openai::OpenAiModel;
use ferragent::{Agent, McpClient};

#[tokio::test]
#[ignore = "requires npx and downloads the real MCP filesystem server"]
async fn real_mcp_filesystem_server_lists_tools() {
    let client = McpClient::connect(
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    .await
    .unwrap();
    assert!(!client.tools().await.unwrap().is_empty());
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and network access"]
async fn real_openai_provider_responds() {
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
    let mut agent = Agent::builder(OpenAiModel::new("gpt-4o-mini", key)).build();
    assert!(!agent
        .run("Reply with exactly: pong")
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and network access"]
async fn real_anthropic_provider_responds() {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY");
    let mut agent = Agent::builder(AnthropicModel::new("claude-3-5-haiku-latest", key)).build();
    assert!(!agent
        .run("Reply with exactly: pong")
        .await
        .unwrap()
        .is_empty());
}
