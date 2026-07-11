//! Request text plus generated audio and save the audio output.

use base64::prelude::*;
use ferragent::llm::openai::OpenAiModel;
use ferragent::{Agent, ContentPart, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let model = OpenAiModel::new("gpt-audio-1.5", std::env::var("OPENAI_API_KEY")?)
        .with_modalities(["text", "audio"])
        .with_audio_output("wav", "alloy");
    let mut agent = Agent::builder(model).build();

    let response = agent
        .run_message(Message::user(
            "Say a friendly two-sentence welcome to Rust.",
        ))
        .await?;
    if let Some(text) = response.content {
        println!("Transcript: {text}");
    }
    for part in response.content_parts {
        if let ContentPart::Audio { data, format, .. } = part {
            let path = format!("multimodal-output.{format}");
            std::fs::write(&path, BASE64_STANDARD.decode(data)?)?;
            println!("Wrote {path}");
        }
    }
    Ok(())
}
