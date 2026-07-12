use crate::error::{AgentError, Result};
use crate::llm::{Model, ModelResponse};
use crate::memory::{InMemoryStorage, Storage};
use crate::message::Message;
use crate::observability::{
    MetricRecord, MetricsRecorder, OperationKind, OperationOutcome, UsageRecord, UsageRecorder,
};
use crate::runtime::{ExecutionPolicy, StreamEvent};
use crate::skills::{LoadSkillTool, ReadSkillResourceTool, SkillCatalog};
use crate::structured::parse_structured;
use crate::tool::{Tool, ToolSpec};
use crate::tracing::{NoopTracer, TraceEvent, Tracer};
use futures::future::join_all;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// A single reasoning agent: a model, a set of tools it may call, an
/// instruction/system prompt, and (optionally) persistent session memory.
pub struct Agent {
    model: Box<dyn Model>,
    tools: Vec<Arc<dyn Tool>>,
    instructions: Option<String>,
    storage: Arc<dyn Storage>,
    max_steps: usize,
    policy: ExecutionPolicy,
    tracer: Arc<dyn Tracer>,
    usage_recorder: Option<Arc<dyn UsageRecorder>>,
    metrics_recorder: Option<Arc<dyn MetricsRecorder>>,
    /// In-process history used when no session id is given to `run`.
    history: Vec<Message>,
}

/// Fluent builder for [`Agent`].
pub struct AgentBuilder {
    model: Box<dyn Model>,
    tools: Vec<Arc<dyn Tool>>,
    instructions: Option<String>,
    storage: Option<Arc<dyn Storage>>,
    max_steps: usize,
    policy: ExecutionPolicy,
    tracer: Option<Arc<dyn Tracer>>,
    usage_recorder: Option<Arc<dyn UsageRecorder>>,
    metrics_recorder: Option<Arc<dyn MetricsRecorder>>,
    skills_enabled: bool,
}

impl AgentBuilder {
    pub fn new(model: impl Model + 'static) -> Self {
        Self {
            model: Box::new(model),
            tools: vec![],
            instructions: None,
            storage: None,
            max_steps: 10,
            policy: ExecutionPolicy::default(),
            tracer: None,
            usage_recorder: None,
            metrics_recorder: None,
            skills_enabled: false,
        }
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        if !self.skills_enabled || !is_reserved_skill_tool(tool.name()) {
            self.tools.push(Arc::new(tool));
        }
        self
    }

    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(
            tools
                .into_iter()
                .filter(|tool| !self.skills_enabled || !is_reserved_skill_tool(tool.name())),
        );
        self
    }

    pub fn skills(mut self, catalog: SkillCatalog) -> Self {
        let summary = catalog.prompt_summary();
        self.instructions = Some(match self.instructions.take() {
            Some(instructions) => format!("{instructions}\n\n{summary}"),
            None => summary,
        });
        let catalog = Arc::new(catalog);
        self.skills_enabled = true;
        self.tools
            .retain(|tool| !is_reserved_skill_tool(tool.name()));
        self.tools
            .push(Arc::new(LoadSkillTool::new(catalog.clone())));
        self.tools
            .push(Arc::new(ReadSkillResourceTool::new(catalog)));
        self
    }

    pub fn storage(mut self, storage: impl Storage + 'static) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    /// Maximum number of model<->tool round-trips per `run()` call before
    /// giving up (guards against infinite tool-call loops). Default: 10.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn tracer(mut self, tracer: impl Tracer + 'static) -> Self {
        self.tracer = Some(Arc::new(tracer));
        self
    }

    pub fn usage_recorder(mut self, recorder: impl UsageRecorder + 'static) -> Self {
        self.usage_recorder = Some(Arc::new(recorder));
        self
    }

    pub fn metrics_recorder(mut self, recorder: impl MetricsRecorder + 'static) -> Self {
        self.metrics_recorder = Some(Arc::new(recorder));
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            model: self.model,
            tools: self.tools,
            instructions: self.instructions,
            storage: self
                .storage
                .unwrap_or_else(|| Arc::new(InMemoryStorage::new())),
            max_steps: self.max_steps,
            policy: self.policy,
            tracer: self.tracer.unwrap_or_else(|| Arc::new(NoopTracer)),
            usage_recorder: self.usage_recorder,
            metrics_recorder: self.metrics_recorder,
            history: vec![],
        }
    }
}

fn is_reserved_skill_tool(name: &str) -> bool {
    matches!(name, "load_skill" | "read_skill_resource")
}

impl Agent {
    pub fn builder(model: impl Model + 'static) -> AgentBuilder {
        AgentBuilder::new(model)
    }

    pub(crate) fn add_shared_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    fn find_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name)
    }

    /// Run the agent on a single input with no persisted session — history
    /// only lives for the lifetime of this `Agent` value.
    pub async fn run(&mut self, input: impl Into<String>) -> Result<String> {
        let mut messages = std::mem::take(&mut self.history);
        if messages.is_empty() {
            if let Some(instructions) = &self.instructions {
                messages.push(Message::system(instructions.clone()));
            }
        }
        messages.push(Message::user(input.into()));

        let result = self
            .step_loop(&mut messages)
            .await
            .map(|r| r.content.unwrap_or_default());
        self.history = messages;
        result
    }

    /// Run with a pre-built message and return all output modalities.
    pub async fn run_message(&mut self, input: Message) -> Result<ModelResponse> {
        let mut messages = std::mem::take(&mut self.history);
        if messages.is_empty() {
            if let Some(instructions) = &self.instructions {
                messages.push(Message::system(instructions.clone()));
            }
        }
        messages.push(input);
        let result = self.step_loop(&mut messages).await;
        self.history = messages;
        result
    }

    /// Stream content and tool-call events through a bounded channel while
    /// retaining the same final response and history semantics as `run`.
    pub async fn run_stream(
        &mut self,
        input: impl Into<String>,
        events: mpsc::Sender<Result<StreamEvent>>,
    ) -> Result<ModelResponse> {
        let mut messages = std::mem::take(&mut self.history);
        if messages.is_empty() {
            if let Some(instructions) = &self.instructions {
                messages.push(Message::system(instructions.clone()));
            }
        }
        messages.push(Message::user(input.into()));
        let result = self
            .step_loop_stream(&mut messages, Some(events.clone()), None)
            .await;
        if result.is_ok() {
            let _ = events.send(Ok(StreamEvent::Done)).await;
        }
        self.history = messages;
        result
    }

    /// Run and deserialize a JSON response after validating it against the
    /// supplied schema.
    pub async fn run_structured<T: DeserializeOwned>(
        &mut self,
        input: impl Into<String>,
        schema: &Value,
    ) -> Result<T> {
        let mut messages = std::mem::take(&mut self.history);
        if messages.is_empty() {
            if let Some(instructions) = &self.instructions {
                messages.push(Message::system(instructions.clone()));
            }
        }
        messages.push(Message::user(input));
        let response = self.step_loop_schema(&mut messages, schema).await;
        self.history = messages;
        let response = response?;
        parse_structured(&response, schema)
    }

    /// Run the agent within a named, persisted session: history is loaded
    /// from and saved back to `self`'s storage backend before/after the call.
    pub async fn run_session(
        &mut self,
        session_id: &str,
        input: impl Into<String>,
    ) -> Result<String> {
        let mut messages = self
            .storage
            .load(session_id)
            .await
            .map_err(AgentError::Other)?;
        if messages.is_empty() {
            if let Some(instructions) = &self.instructions {
                messages.push(Message::system(instructions.clone()));
            }
        }
        messages.push(Message::user(input.into()));

        let result = self
            .step_loop(&mut messages)
            .await
            .map(|r| r.content.unwrap_or_default());
        self.storage
            .save(session_id, &messages)
            .await
            .map_err(AgentError::Other)?;
        result
    }

    /// Clear a persisted session's history.
    pub async fn clear_session(&self, session_id: &str) -> Result<()> {
        self.storage
            .clear(session_id)
            .await
            .map_err(AgentError::Other)
    }

    /// Core reasoning loop: call the model, execute any requested tool
    /// calls, feed results back, and repeat until the model returns plain
    /// content or `max_steps` is exceeded.
    async fn step_loop(&self, messages: &mut Vec<Message>) -> Result<ModelResponse> {
        self.step_loop_stream(messages, None, None).await
    }

    async fn step_loop_schema(
        &self,
        messages: &mut Vec<Message>,
        schema: &Value,
    ) -> Result<ModelResponse> {
        self.step_loop_stream(messages, None, Some(schema)).await
    }

    async fn step_loop_stream(
        &self,
        messages: &mut Vec<Message>,
        stream: Option<mpsc::Sender<Result<StreamEvent>>>,
        schema: Option<&Value>,
    ) -> Result<ModelResponse> {
        let specs = self.tool_specs();
        let run_id = uuid::Uuid::new_v4().to_string();
        let run_started = Instant::now();
        self.trace(&run_id, "run_started", json!({"model":self.model.id()}));
        let mut total_usage = crate::llm::Usage::default();

        for step in 0..self.max_steps {
            self.trace(&run_id, "model_started", json!({"step":step}));
            let model_started = Instant::now();
            let mut response = match self
                .call_model(messages, &specs, stream.clone(), schema)
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    self.record_metric(
                        &run_id,
                        OperationKind::Model,
                        self.model.id(),
                        model_started,
                        outcome_for_error(&error),
                    );
                    self.record_failed_usage(&run_id, model_started, &error);
                    self.trace(
                        &run_id,
                        "model_failed",
                        json!({"step":step,"error":error.to_string()}),
                    );
                    self.record_metric(
                        &run_id,
                        OperationKind::Run,
                        "agent",
                        run_started,
                        outcome_for_error(&error),
                    );
                    return Err(error);
                }
            };
            self.record_metric(
                &run_id,
                OperationKind::Model,
                self.model.id(),
                model_started,
                OperationOutcome::Success,
            );
            self.record_usage(&run_id, model_started, &response.usage);
            total_usage.add_assign(&response.usage);
            self.trace(
                &run_id,
                "model_completed",
                json!({"step":step,"tool_calls":response.tool_calls.len()}),
            );

            if response.tool_calls.is_empty() {
                response.usage = total_usage;
                if response.content_parts.is_empty() {
                    messages.push(Message::assistant(
                        response.content.clone().unwrap_or_default(),
                    ));
                } else {
                    messages.push(Message::assistant_parts(response.content_parts.clone()));
                }
                self.trace(&run_id, "run_completed", json!({"step":step}));
                self.record_metric(
                    &run_id,
                    OperationKind::Run,
                    "agent",
                    run_started,
                    OperationOutcome::Success,
                );
                return Ok(response);
            }

            messages.push(Message::assistant_tool_calls(response.tool_calls.clone()));

            let calls = response.tool_calls.clone();
            let execute = |call: crate::message::ToolCall| {
                let run_id = run_id.clone();
                async move {
                    let tool_started = Instant::now();
                    let tool = self
                        .find_tool(&call.name)
                        .cloned()
                        .ok_or_else(|| AgentError::ToolNotFound(call.name.clone()))?;
                    self.trace(
                        &run_id,
                        "tool_started",
                        json!({"name":call.name,"id":call.id}),
                    );
                    let result = tokio::time::timeout(
                        self.policy.tool_timeout,
                        tool.execute(call.arguments.clone()),
                    )
                    .await
                    .map_err(|_| AgentError::Timeout(format!("tool '{}'", call.name)))?
                    .map_err(|source| AgentError::ToolExecution {
                        name: call.name.clone(),
                        source,
                    });
                    self.trace(
                        &run_id,
                        "tool_completed",
                        json!({"name":call.name,"id":call.id,"ok":result.is_ok()}),
                    );
                    self.record_metric(
                        &run_id,
                        OperationKind::Tool,
                        &call.name,
                        tool_started,
                        match &result {
                            Ok(_) => OperationOutcome::Success,
                            Err(error) => outcome_for_error(error),
                        },
                    );
                    result.map(|output| (call, output))
                }
            };
            let results = if self.policy.parallel_tool_calls {
                join_all(calls.into_iter().map(execute)).await
            } else {
                let mut out = Vec::new();
                for call in calls {
                    out.push(execute(call).await);
                }
                out
            };
            for result in results {
                let (call, output) = match result {
                    Ok(value) => value,
                    Err(error) => {
                        self.trace(&run_id, "run_failed", json!({"error":error.to_string()}));
                        self.record_metric(
                            &run_id,
                            OperationKind::Run,
                            "agent",
                            run_started,
                            outcome_for_error(&error),
                        );
                        return Err(error);
                    }
                };
                messages.push(Message::tool_result(
                    call.id.clone(),
                    call.name.clone(),
                    output,
                ));
            }
        }

        let error = AgentError::MaxStepsExceeded(self.max_steps);
        self.trace(&run_id, "run_failed", json!({"error":error.to_string()}));
        self.record_metric(
            &run_id,
            OperationKind::Run,
            "agent",
            run_started,
            OperationOutcome::Error,
        );
        Err(error)
    }

    async fn call_model(
        &self,
        messages: &[Message],
        specs: &[ToolSpec],
        stream: Option<mpsc::Sender<Result<StreamEvent>>>,
        schema: Option<&Value>,
    ) -> Result<ModelResponse> {
        let attempts = self.policy.max_attempts.max(1);
        let mut delay = self.policy.initial_backoff;
        for attempt in 1..=attempts {
            let request = async {
                if let Some(events) = stream.clone() {
                    self.model.generate_stream(messages, specs, events).await
                } else if let Some(schema) = schema {
                    self.model
                        .generate_structured(messages, specs, schema)
                        .await
                } else {
                    self.model.generate(messages, specs).await
                }
            };
            match tokio::time::timeout(self.policy.request_timeout, request).await {
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(error)) if attempt == attempts => return Err(error),
                Err(_) if attempt == attempts => {
                    return Err(AgentError::Timeout("model request".into()))
                }
                _ => {
                    tokio::time::sleep(delay).await;
                    delay = delay.saturating_mul(2);
                }
            }
        }
        unreachable!()
    }

    fn trace(&self, run_id: &str, kind: &str, fields: Value) {
        self.tracer.record(TraceEvent::new(run_id, kind, fields));
    }

    fn record_metric(
        &self,
        run_id: &str,
        kind: OperationKind,
        name: &str,
        started: Instant,
        outcome: OperationOutcome,
    ) {
        if let Some(recorder) = &self.metrics_recorder {
            let mut record = MetricRecord::success(
                run_id,
                kind,
                name,
                started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            );
            record.outcome = outcome;
            recorder.record_metric(record);
        }
    }

    fn record_usage(&self, run_id: &str, started: Instant, usage: &crate::llm::Usage) {
        if let Some(recorder) = &self.usage_recorder {
            let mut record = UsageRecord::new(
                run_id,
                self.model.provider(),
                self.model.id(),
                usage.clone(),
            );
            record.latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            recorder.record_usage(record);
        }
    }

    fn record_failed_usage(&self, run_id: &str, started: Instant, error: &AgentError) {
        if let Some(recorder) = &self.usage_recorder {
            let mut record = UsageRecord::new(
                run_id,
                self.model.provider(),
                self.model.id(),
                crate::llm::Usage::default(),
            );
            record.success = false;
            record.error_kind = Some(error.to_string());
            record.latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            recorder.record_usage(record);
        }
    }
}

fn outcome_for_error(error: &AgentError) -> OperationOutcome {
    if matches!(error, AgentError::Timeout(_)) {
        OperationOutcome::Timeout
    } else {
        OperationOutcome::Error
    }
}
