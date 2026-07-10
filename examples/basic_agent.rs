use liteagent::llm::openai::OpenAiModel;
use liteagent::tool::FunctionTool;
use liteagent::Agent;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let model = OpenAiModel::new("gpt-5-nano", api_key);

    let weather_tool = FunctionTool::new(
        "get_weather",
        "Get the current weather for a given city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
        |args| {
            Box::pin(async move {
                let city = args["city"].as_str().unwrap_or("unknown").to_string();
                // Pretend this hits a real weather API.
                Ok(format!("It's 22C and sunny in {city}."))
            })
        },
    );

    let mut agent = Agent::builder(model)
        .instructions("You are a helpful weather assistant. Use tools when needed.")
        .tool(weather_tool)
        .build();

    let answer = agent.run("explain mixture of expert architecture?").await?;
    println!("{answer}");

    // Follow-up in the same in-memory conversation.
    let answer2 = agent.run("And how about Paris?").await?;
    println!("{answer2}");

    Ok(())
}
