use async_trait::async_trait;
use ferrant::llm::openai::OpenAiModel;
use ferrant::memory::InMemoryStorage;
use ferrant::{Agent, Tool};
use serde_json::{json, Value};

/// A hand-written tool (rather than a FunctionTool closure) — useful when a
/// tool needs its own state, e.g. a database connection pool.
struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate a simple arithmetic expression with + - * / on two numbers"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "op": { "type": "string", "enum": ["+", "-", "*", "/"] },
                "b": { "type": "number" }
            },
            "required": ["a", "op", "b"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<String> {
        let a = args["a"].as_f64().unwrap_or(0.0);
        let b = args["b"].as_f64().unwrap_or(0.0);
        let op = args["op"].as_str().unwrap_or("+");
        let result = match op {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            "/" => a / b,
            _ => return Err(anyhow::anyhow!("unsupported op: {op}")),
        };
        Ok(result.to_string())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let model = OpenAiModel::new("gpt-5-nano", api_key);

    let mut agent = Agent::builder(model)
        .instructions(
            "You are a precise math assistant. Always use the calculator tool for arithmetic.",
        )
        .tool(Calculator)
        .storage(InMemoryStorage::new()) // swap for your own Storage impl (Postgres, Redis, ...)
        .build();

    // Persisted, named session: history survives across separate `run_session` calls
    // and could be reloaded later (e.g. in a web server handling one request per call).
    let session = "user-42";
    let answer = agent.run_session(session, "What is 42 * 17?").await?;
    println!("{answer}");

    let answer2 = agent
        .run_session(session, "Now subtract 100 from that.")
        .await?;
    println!("{answer2}");

    Ok(())
}
