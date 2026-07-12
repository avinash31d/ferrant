use ferrant::llm::openai::OpenAiModel;
use ferrant::{Agent, SkillCatalog, SkillLimits, SkillSource};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let mut sources = vec![SkillSource::Local {
        root: PathBuf::from("./skills"),
    }];
    if let Ok(repository) = std::env::var("FERRANT_SKILLS_REPOSITORY") {
        sources.push(SkillSource::GitHub {
            repository,
            git_ref: Some("main".into()),
            subdirectory: Some(PathBuf::from("skills")),
            cache_dir: PathBuf::from(".ferrant/skills-cache"),
        });
    }

    let catalog = SkillCatalog::load(sources, SkillLimits::default())?;
    let model = OpenAiModel::new("gpt-5-nano", std::env::var("OPENAI_API_KEY")?);
    let mut agent = Agent::builder(model)
        .instructions("Use the available skills when they match the request.")
        .skills(catalog)
        .build();

    let answer = agent.run("Help me complete this task safely.").await?;
    println!("{answer}");
    Ok(())
}
