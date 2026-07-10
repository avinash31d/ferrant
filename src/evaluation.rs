//! Reproducible evaluation datasets, concurrent runners, scorers, and
//! regression gates.

use crate::agent::Agent;
use crate::llm::Usage;
use crate::message::Message;
use async_trait::async_trait;
use futures::future::{join_all, BoxFuture};
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationCase {
    pub id: String,
    pub input: Value,
    pub expected: Option<Value>,
    #[serde(default)]
    pub context: Vec<Value>,
    #[serde(default)]
    pub tags: BTreeSet<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl EvaluationCase {
    pub fn new(id: impl Into<String>, input: Value) -> Self {
        Self {
            id: id.into(),
            input,
            expected: None,
            context: Vec::new(),
            tags: BTreeSet::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn expected(mut self, expected: Value) -> Self {
        self.expected = Some(expected);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationDataset {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    pub cases: Vec<EvaluationCase>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl EvaluationDataset {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        cases: Vec<EvaluationCase>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: None,
            cases,
            metadata: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.name.trim().is_empty() {
            anyhow::bail!("dataset name must not be empty");
        }
        if self.version.trim().is_empty() {
            anyhow::bail!("dataset version must not be empty");
        }
        let mut ids = BTreeSet::new();
        for case in &self.cases {
            if case.id.trim().is_empty() {
                anyhow::bail!("evaluation case id must not be empty");
            }
            if !ids.insert(case.id.as_str()) {
                anyhow::bail!("duplicate evaluation case id '{}'", case.id);
            }
        }
        Ok(())
    }

    pub fn select_tags<'a>(
        &'a self,
        tags: &'a BTreeSet<String>,
    ) -> impl Iterator<Item = &'a EvaluationCase> {
        self.cases
            .iter()
            .filter(move |case| tags.is_empty() || !case.tags.is_disjoint(tags))
    }
}

/// Normalized output from the system under evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EvaluationOutput {
    pub value: Value,
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

impl EvaluationOutput {
    pub fn new(value: Value) -> Self {
        Self {
            value,
            ..Self::default()
        }
    }
}

#[async_trait]
pub trait EvaluationTarget: Send + Sync {
    async fn evaluate(&self, case: &EvaluationCase) -> anyhow::Result<EvaluationOutput>;
}

pub type EvaluationFn = Arc<
    dyn Fn(EvaluationCase) -> BoxFuture<'static, anyhow::Result<EvaluationOutput>> + Send + Sync,
>;

/// Evaluation target backed by an async closure. The case is cloned so the
/// closure can own it across arbitrary async boundaries.
pub struct FunctionEvaluationTarget {
    function: EvaluationFn,
}

impl FunctionEvaluationTarget {
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(EvaluationCase) -> BoxFuture<'static, anyhow::Result<EvaluationOutput>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            function: Arc::new(function),
        }
    }
}

#[async_trait]
impl EvaluationTarget for FunctionEvaluationTarget {
    async fn evaluate(&self, case: &EvaluationCase) -> anyhow::Result<EvaluationOutput> {
        (self.function)(case.clone()).await
    }
}

/// Convenience target for evaluating an existing agent. Access is serialized
/// through a Tokio mutex because `Agent` owns mutable conversation state.
/// For hermetic, parallel cases prefer [`FunctionEvaluationTarget`] with a
/// closure that constructs a fresh agent per case.
#[derive(Clone)]
pub struct AgentEvaluationTarget {
    agent: Arc<tokio::sync::Mutex<Agent>>,
    parse_json_output: bool,
}

impl AgentEvaluationTarget {
    pub fn new(agent: Agent) -> Self {
        Self {
            agent: Arc::new(tokio::sync::Mutex::new(agent)),
            parse_json_output: false,
        }
    }

    pub fn from_shared(agent: Arc<tokio::sync::Mutex<Agent>>) -> Self {
        Self {
            agent,
            parse_json_output: false,
        }
    }

    /// Decode valid JSON model text into a JSON value. Otherwise output text
    /// is represented as `Value::String`, which is the safer default.
    pub fn parse_json_output(mut self, enabled: bool) -> Self {
        self.parse_json_output = enabled;
        self
    }

    pub fn agent(&self) -> Arc<tokio::sync::Mutex<Agent>> {
        self.agent.clone()
    }
}

#[async_trait]
impl EvaluationTarget for AgentEvaluationTarget {
    async fn evaluate(&self, case: &EvaluationCase) -> anyhow::Result<EvaluationOutput> {
        let prompt = case
            .input
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| case.input.to_string());
        let started = Instant::now();
        let response = self
            .agent
            .lock()
            .await
            .run_message(Message::user(prompt))
            .await?;
        let latency_ms = elapsed_ms(started);
        let value = match response.content {
            Some(content) if self.parse_json_output => {
                serde_json::from_str(&content).unwrap_or(Value::String(content))
            }
            Some(content) => Value::String(content),
            None if !response.content_parts.is_empty() => {
                serde_json::to_value(&response.content_parts)?
            }
            None => Value::Null,
        };
        let mut attributes = BTreeMap::new();
        attributes.insert(
            "tool_call_count".into(),
            Value::from(response.tool_calls.len() as u64),
        );
        Ok(EvaluationOutput {
            value,
            latency_ms,
            usage: response.usage,
            attributes,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Score {
    pub value: f64,
    pub passed: bool,
    #[serde(default)]
    pub details: Value,
}

impl Score {
    pub fn new(value: f64, passed: bool) -> anyhow::Result<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            anyhow::bail!("score must be a finite number between 0 and 1");
        }
        Ok(Self {
            value,
            passed,
            details: Value::Null,
        })
    }

    pub fn pass() -> Self {
        Self {
            value: 1.0,
            passed: true,
            details: Value::Null,
        }
    }

    pub fn fail() -> Self {
        Self {
            value: 0.0,
            passed: false,
            details: Value::Null,
        }
    }

    pub fn details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }
}

#[async_trait]
pub trait Scorer: Send + Sync {
    fn name(&self) -> &str;

    async fn score(
        &self,
        case: &EvaluationCase,
        output: &EvaluationOutput,
    ) -> anyhow::Result<Score>;
}

pub type ScoringFn = Arc<
    dyn Fn(EvaluationCase, EvaluationOutput) -> BoxFuture<'static, anyhow::Result<Score>>
        + Send
        + Sync,
>;

pub struct FunctionScorer {
    name: String,
    function: ScoringFn,
}

impl FunctionScorer {
    pub fn new<F>(name: impl Into<String>, function: F) -> Self
    where
        F: Fn(EvaluationCase, EvaluationOutput) -> BoxFuture<'static, anyhow::Result<Score>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            function: Arc::new(function),
        }
    }
}

#[async_trait]
impl Scorer for FunctionScorer {
    fn name(&self) -> &str {
        &self.name
    }

    async fn score(
        &self,
        case: &EvaluationCase,
        output: &EvaluationOutput,
    ) -> anyhow::Result<Score> {
        (self.function)(case.clone(), output.clone()).await
    }
}

#[derive(Default)]
pub struct ExactMatchScorer;

#[async_trait]
impl Scorer for ExactMatchScorer {
    fn name(&self) -> &str {
        "exact_match"
    }

    async fn score(
        &self,
        case: &EvaluationCase,
        output: &EvaluationOutput,
    ) -> anyhow::Result<Score> {
        let expected = case
            .expected
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("case '{}' has no expected value", case.id))?;
        Ok(if expected == &output.value {
            Score::pass()
        } else {
            Score::fail().details(serde_json::json!({
                "expected": expected,
                "actual": output.value
            }))
        })
    }
}

pub struct ContainsScorer {
    case_sensitive: bool,
}

impl ContainsScorer {
    pub fn new(case_sensitive: bool) -> Self {
        Self { case_sensitive }
    }
}

#[async_trait]
impl Scorer for ContainsScorer {
    fn name(&self) -> &str {
        "contains"
    }

    async fn score(
        &self,
        case: &EvaluationCase,
        output: &EvaluationOutput,
    ) -> anyhow::Result<Score> {
        let expected = case
            .expected
            .as_ref()
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("case '{}' expected value must be a string", case.id))?;
        let actual = output
            .value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("case '{}' output must be a string", case.id))?;
        let contains = if self.case_sensitive {
            actual.contains(expected)
        } else {
            actual.to_lowercase().contains(&expected.to_lowercase())
        };
        Ok(if contains {
            Score::pass()
        } else {
            Score::fail().details(serde_json::json!({"expected_substring": expected}))
        })
    }
}

/// Scores whether the expected JSON object/array is recursively contained in
/// the actual output. Extra actual fields and array elements are allowed.
#[derive(Default)]
pub struct JsonSubsetScorer;

#[async_trait]
impl Scorer for JsonSubsetScorer {
    fn name(&self) -> &str {
        "json_subset"
    }

    async fn score(
        &self,
        case: &EvaluationCase,
        output: &EvaluationOutput,
    ) -> anyhow::Result<Score> {
        let expected = case
            .expected
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("case '{}' has no expected value", case.id))?;
        Ok(if json_contains(&output.value, expected) {
            Score::pass()
        } else {
            Score::fail().details(serde_json::json!({
                "expected_subset": expected,
                "actual": output.value
            }))
        })
    }
}

fn json_contains(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => expected.iter().all(|(key, value)| {
            actual
                .get(key)
                .is_some_and(|actual| json_contains(actual, value))
        }),
        (Value::Array(actual), Value::Array(expected)) => expected.iter().all(|expected_item| {
            actual
                .iter()
                .any(|actual_item| json_contains(actual_item, expected_item))
        }),
        _ => actual == expected,
    }
}

pub struct LatencyScorer {
    pub maximum_ms: u64,
}

#[async_trait]
impl Scorer for LatencyScorer {
    fn name(&self) -> &str {
        "latency"
    }

    async fn score(
        &self,
        _case: &EvaluationCase,
        output: &EvaluationOutput,
    ) -> anyhow::Result<Score> {
        let passed = output.latency_ms <= self.maximum_ms;
        let value = if output.latency_ms == 0 || output.latency_ms <= self.maximum_ms {
            1.0
        } else {
            self.maximum_ms as f64 / output.latency_ms as f64
        };
        Ok(Score::new(value, passed)?.details(serde_json::json!({
            "latency_ms": output.latency_ms,
            "maximum_ms": self.maximum_ms
        })))
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub concurrency: usize,
    pub target_timeout: Duration,
    pub scorer_timeout: Duration,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            concurrency: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .max(1),
            target_timeout: Duration::from_secs(60),
            scorer_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScorerResult {
    pub scorer: String,
    pub score: Option<Score>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationCaseResult {
    pub case_id: String,
    pub output: Option<EvaluationOutput>,
    pub scorers: Vec<ScorerResult>,
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

impl EvaluationCaseResult {
    pub fn passed(&self) -> bool {
        self.error.is_none()
            && !self.scorers.is_empty()
            && self.scorers.iter().all(|result| {
                result.error.is_none() && result.score.as_ref().is_some_and(|s| s.passed)
            })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ScorerAggregate {
    pub scored_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub errors: usize,
    pub mean_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationReport {
    pub id: String,
    pub created_at_ms: u64,
    pub dataset_name: String,
    pub dataset_version: String,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub target_errors: usize,
    pub scorer_errors: usize,
    pub pass_rate: f64,
    pub error_rate: f64,
    pub mean_score: f64,
    pub mean_latency_ms: f64,
    pub usage: Usage,
    pub by_scorer: BTreeMap<String, ScorerAggregate>,
    pub cases: Vec<EvaluationCaseResult>,
}

impl EvaluationReport {
    fn from_results(dataset: &EvaluationDataset, cases: Vec<EvaluationCaseResult>) -> Self {
        let total_cases = cases.len();
        let passed_cases = cases.iter().filter(|case| case.passed()).count();
        let target_errors = cases.iter().filter(|case| case.error.is_some()).count();
        let scorer_errors = cases
            .iter()
            .flat_map(|case| &case.scorers)
            .filter(|score| score.error.is_some())
            .count();
        let mut by_scorer = BTreeMap::<String, ScorerAggregate>::new();
        let mut score_sum = 0.0;
        let mut score_count = 0usize;
        let mut latency_sum = 0u64;
        let mut latency_count = 0usize;
        let mut usage = Usage::default();

        for case in &cases {
            if let Some(output) = &case.output {
                latency_sum = latency_sum.saturating_add(output.latency_ms);
                latency_count += 1;
                add_usage(&mut usage, &output.usage);
            }
            for result in &case.scorers {
                let aggregate = by_scorer.entry(result.scorer.clone()).or_default();
                match (&result.score, &result.error) {
                    (Some(score), None) => {
                        aggregate.scored_cases += 1;
                        if score.passed {
                            aggregate.passed_cases += 1;
                        } else {
                            aggregate.failed_cases += 1;
                        }
                        aggregate.mean_score += score.value;
                        score_sum += score.value;
                        score_count += 1;
                    }
                    _ => aggregate.errors += 1,
                }
            }
        }
        for aggregate in by_scorer.values_mut() {
            if aggregate.scored_cases > 0 {
                aggregate.mean_score /= aggregate.scored_cases as f64;
            }
        }

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at_ms: now_ms(),
            dataset_name: dataset.name.clone(),
            dataset_version: dataset.version.clone(),
            total_cases,
            passed_cases,
            target_errors,
            scorer_errors,
            pass_rate: divide(passed_cases, total_cases),
            error_rate: divide(target_errors, total_cases),
            mean_score: if score_count == 0 {
                0.0
            } else {
                score_sum / score_count as f64
            },
            mean_latency_ms: if latency_count == 0 {
                0.0
            } else {
                latency_sum as f64 / latency_count as f64
            },
            usage,
            by_scorer,
            cases,
        }
    }

    pub fn check_regression(
        &self,
        baseline: Option<&EvaluationReport>,
        thresholds: &RegressionThresholds,
    ) -> RegressionCheck {
        let mut violations = Vec::new();

        if let Some(minimum) = thresholds.minimum_overall_score {
            if self.mean_score < minimum {
                violations.push(RegressionViolation::new(
                    "minimum_overall_score",
                    minimum,
                    self.mean_score,
                ));
            }
        }
        if let Some(minimum) = thresholds.minimum_pass_rate {
            if self.pass_rate < minimum {
                violations.push(RegressionViolation::new(
                    "minimum_pass_rate",
                    minimum,
                    self.pass_rate,
                ));
            }
        }
        if let Some(maximum) = thresholds.maximum_error_rate {
            if self.error_rate > maximum {
                violations.push(RegressionViolation::new(
                    "maximum_error_rate",
                    maximum,
                    self.error_rate,
                ));
            }
        }
        for (name, minimum) in &thresholds.minimum_scorer_scores {
            let actual = self
                .by_scorer
                .get(name)
                .map(|score| score.mean_score)
                .unwrap_or_default();
            if actual < *minimum {
                violations.push(RegressionViolation::new(
                    format!("minimum_scorer_score:{name}"),
                    *minimum,
                    actual,
                ));
            }
        }

        if let Some(baseline) = baseline {
            if thresholds.require_same_dataset
                && (self.dataset_name != baseline.dataset_name
                    || self.dataset_version != baseline.dataset_version)
            {
                violations.push(RegressionViolation {
                    rule: "same_dataset".into(),
                    allowed: 1.0,
                    actual: 0.0,
                    message: format!(
                        "baseline is {}/{}, candidate is {}/{}",
                        baseline.dataset_name,
                        baseline.dataset_version,
                        self.dataset_name,
                        self.dataset_version
                    ),
                });
            }
            if let Some(maximum_drop) = thresholds.maximum_score_drop {
                let drop = baseline.mean_score - self.mean_score;
                if drop > maximum_drop {
                    violations.push(RegressionViolation::new(
                        "maximum_score_drop",
                        maximum_drop,
                        drop,
                    ));
                }
            }
            if let Some(maximum_drop) = thresholds.maximum_pass_rate_drop {
                let drop = baseline.pass_rate - self.pass_rate;
                if drop > maximum_drop {
                    violations.push(RegressionViolation::new(
                        "maximum_pass_rate_drop",
                        maximum_drop,
                        drop,
                    ));
                }
            }
            if let Some(maximum_increase) = thresholds.maximum_latency_increase_ratio {
                let ratio = if baseline.mean_latency_ms <= f64::EPSILON {
                    if self.mean_latency_ms <= f64::EPSILON {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                } else {
                    (self.mean_latency_ms - baseline.mean_latency_ms) / baseline.mean_latency_ms
                };
                if ratio > maximum_increase {
                    violations.push(RegressionViolation::new(
                        "maximum_latency_increase_ratio",
                        maximum_increase,
                        ratio,
                    ));
                }
            }
            for (name, maximum_drop) in &thresholds.maximum_scorer_score_drops {
                let baseline_score = baseline
                    .by_scorer
                    .get(name)
                    .map(|score| score.mean_score)
                    .unwrap_or_default();
                let current_score = self
                    .by_scorer
                    .get(name)
                    .map(|score| score.mean_score)
                    .unwrap_or_default();
                let drop = baseline_score - current_score;
                if drop > *maximum_drop {
                    violations.push(RegressionViolation::new(
                        format!("maximum_scorer_score_drop:{name}"),
                        *maximum_drop,
                        drop,
                    ));
                }
            }
        }

        RegressionCheck {
            passed: violations.is_empty(),
            violations,
        }
    }
}

fn add_usage(total: &mut Usage, usage: &Usage) {
    total.input_tokens = total.input_tokens.saturating_add(usage.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(usage.output_tokens);
    total.total_tokens = total.total_tokens.saturating_add(usage.total_tokens);
    total.cached_input_tokens = total
        .cached_input_tokens
        .saturating_add(usage.cached_input_tokens);
    total.reasoning_tokens = total
        .reasoning_tokens
        .saturating_add(usage.reasoning_tokens);
}

fn divide(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub struct EvaluationRunner {
    target: Arc<dyn EvaluationTarget>,
    scorers: Vec<Arc<dyn Scorer>>,
    config: EvaluationConfig,
}

impl EvaluationRunner {
    pub fn new(target: impl EvaluationTarget + 'static) -> Self {
        Self {
            target: Arc::new(target),
            scorers: Vec::new(),
            config: EvaluationConfig::default(),
        }
    }

    pub fn from_shared(target: Arc<dyn EvaluationTarget>) -> Self {
        Self {
            target,
            scorers: Vec::new(),
            config: EvaluationConfig::default(),
        }
    }

    pub fn scorer(mut self, scorer: impl Scorer + 'static) -> Self {
        self.scorers.push(Arc::new(scorer));
        self
    }

    pub fn shared_scorer(mut self, scorer: Arc<dyn Scorer>) -> Self {
        self.scorers.push(scorer);
        self
    }

    pub fn config(mut self, config: EvaluationConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn run(&self, dataset: &EvaluationDataset) -> anyhow::Result<EvaluationReport> {
        dataset.validate()?;
        if self.scorers.is_empty() {
            anyhow::bail!("at least one scorer is required");
        }
        if self.config.concurrency == 0 {
            anyhow::bail!("evaluation concurrency must be at least one");
        }

        let target = self.target.clone();
        let scorers = self.scorers.clone();
        let config = self.config.clone();
        let mut results = stream::iter(dataset.cases.iter().cloned().enumerate())
            .map(move |(index, case)| {
                let target = target.clone();
                let scorers = scorers.clone();
                let config = config.clone();
                async move { (index, evaluate_case(target, scorers, config, case).await) }
            })
            .buffer_unordered(self.config.concurrency)
            .collect::<Vec<_>>()
            .await;
        results.sort_by_key(|(index, _)| *index);
        Ok(EvaluationReport::from_results(
            dataset,
            results.into_iter().map(|(_, result)| result).collect(),
        ))
    }
}

async fn evaluate_case(
    target: Arc<dyn EvaluationTarget>,
    scorers: Vec<Arc<dyn Scorer>>,
    config: EvaluationConfig,
    case: EvaluationCase,
) -> EvaluationCaseResult {
    let started = Instant::now();
    let target_result = tokio::time::timeout(config.target_timeout, target.evaluate(&case)).await;
    let mut output = match target_result {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return EvaluationCaseResult {
                case_id: case.id,
                output: None,
                scorers: Vec::new(),
                error: Some(error.to_string()),
                elapsed_ms: elapsed_ms(started),
            }
        }
        Err(_) => {
            return EvaluationCaseResult {
                case_id: case.id,
                output: None,
                scorers: Vec::new(),
                error: Some(format!(
                    "target timed out after {} ms",
                    config.target_timeout.as_millis()
                )),
                elapsed_ms: elapsed_ms(started),
            }
        }
    };
    if output.latency_ms == 0 {
        output.latency_ms = elapsed_ms(started);
    }

    let scores = join_all(scorers.into_iter().map(|scorer| {
        let case = case.clone();
        let output = output.clone();
        async move {
            let name = scorer.name().to_owned();
            match tokio::time::timeout(config.scorer_timeout, scorer.score(&case, &output)).await {
                Ok(Ok(score)) => ScorerResult {
                    scorer: name,
                    score: Some(score),
                    error: None,
                },
                Ok(Err(error)) => ScorerResult {
                    scorer: name,
                    score: None,
                    error: Some(error.to_string()),
                },
                Err(_) => ScorerResult {
                    scorer: name,
                    score: None,
                    error: Some(format!(
                        "scorer timed out after {} ms",
                        config.scorer_timeout.as_millis()
                    )),
                },
            }
        }
    }))
    .await;

    EvaluationCaseResult {
        case_id: case.id,
        output: Some(output),
        scorers: scores,
        error: None,
        elapsed_ms: elapsed_ms(started),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RegressionThresholds {
    pub minimum_overall_score: Option<f64>,
    pub minimum_pass_rate: Option<f64>,
    pub maximum_error_rate: Option<f64>,
    pub maximum_score_drop: Option<f64>,
    pub maximum_pass_rate_drop: Option<f64>,
    /// `0.2` allows a 20% increase over baseline latency.
    pub maximum_latency_increase_ratio: Option<f64>,
    #[serde(default)]
    pub minimum_scorer_scores: BTreeMap<String, f64>,
    #[serde(default)]
    pub maximum_scorer_score_drops: BTreeMap<String, f64>,
    pub require_same_dataset: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionViolation {
    pub rule: String,
    pub allowed: f64,
    pub actual: f64,
    pub message: String,
}

impl RegressionViolation {
    fn new(rule: impl Into<String>, allowed: f64, actual: f64) -> Self {
        let rule = rule.into();
        Self {
            message: format!("{rule}: allowed {allowed:.6}, observed {actual:.6}"),
            rule,
            allowed,
            actual,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionCheck {
    pub passed: bool,
    pub violations: Vec<RegressionViolation>,
}

impl RegressionCheck {
    /// Convenient for CI: turns any failed gate into a descriptive error.
    pub fn enforce(&self) -> anyhow::Result<()> {
        if self.passed {
            Ok(())
        } else {
            anyhow::bail!(
                "evaluation regression: {}",
                self.violations
                    .iter()
                    .map(|violation| violation.message.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        }
    }
}
