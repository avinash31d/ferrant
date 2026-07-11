//! Send text and an image to a vision-capable model.

use ferragent::llm::openai::OpenAiModel;
use ferragent::{Agent, ContentPart, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let model = OpenAiModel::new("gpt-5.4-mini", std::env::var("OPENAI_API_KEY")?);
    let mut agent = Agent::builder(model)
        .instructions("Describe visual evidence precisely and mention uncertainty.")
        .build();

    let input = Message::user_parts(vec![
        ContentPart::text("What is shown in this image?"),
        ContentPart::image_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        ),
    ]);
    let response = agent.run_message(input).await?;
    println!("{}", response.content.unwrap_or_default());
    Ok(())
}
