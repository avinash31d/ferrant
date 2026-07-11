//! Agent using tools discovered from an MCP filesystem server.
//! Requires Node.js/npx and an OPENAI_API_KEY.

use ferrant::llm::openai::OpenAiModel;
use ferrant::{Agent, McpClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let server = McpClient::connect(
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    .await?;

    let model = OpenAiModel::new("gpt-5-nano", std::env::var("OPENAI_API_KEY")?);
    let mut agent = Agent::builder(model)
        .instructions("Use the MCP filesystem tools to inspect the project. Never modify files unless explicitly asked.")
        .tools(server.tools().await?)
        .build();

    println!(
        "{}",
        agent
            .run("Summarize the Rust source files in this project.")
            .await?
    );
    Ok(())
}
