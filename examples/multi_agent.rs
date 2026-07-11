//! Coordinator-led team with two specialist agents.

use ferrant::llm::openai::OpenAiModel;
use ferrant::{Agent, AgentTeam};

fn model(api_key: &str) -> OpenAiModel {
    OpenAiModel::new("gpt-5-nano", api_key)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let key = std::env::var("OPENAI_API_KEY")?;

    let researcher = Agent::builder(model(&key))
        .instructions(
            "You are a careful research analyst. Identify facts, assumptions, and uncertainty.",
        )
        .build();
    let reviewer = Agent::builder(model(&key))
        .instructions(
            "You are a critical reviewer. Find risks, edge cases, and practical improvements.",
        )
        .build();

    let mut team = AgentTeam::new(model(&key))
        .member(
            "researcher",
            "Researches and analyzes the subject",
            researcher,
        )
        .member(
            "reviewer",
            "Critically reviews a proposal and finds risks",
            reviewer,
        );

    let answer = team.run(
        "Propose a safe rollout plan for moving a small API from a VM to containers. Ask both specialists and synthesize their advice."
    ).await?;
    println!("{answer}");
    Ok(())
}
