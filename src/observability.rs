//! Provider-neutral usage accounting, operational metrics, and tracing adapters.
//!
//! The OpenTelemetry adapter intentionally depends on an exporter trait instead of
//! the OpenTelemetry SDK. Applications can bridge [`OpenTelemetryExporter`] to the
//! SDK/OTLP version they already use without forcing that dependency on all users.

use crate::llm::Usage;
use crate::tracing::{TraceEvent, Tracer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

/// Token and cost data for one provider request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageRecord {
    pub timestamp_ms: u64,
    pub run_id: String,
    pub request_id: Option<String>,
    pub provider: String,
    pub model: String,
    pub usage: Usage,
    /// Estimated cost in millionths of a US dollar. `None` means that no
    /// pricing table was configured, rather than that the request was free.
    pub estimated_cost_microusd: Option<u64>,
    pub latency_ms: u64,
    pub success: bool,
    pub error_kind: Option<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

impl UsageRecord {
    pub fn new(
        run_id: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        usage: Usage,
    ) -> Self {
        Self {
            timestamp_ms: now_ms(),
            run_id: run_id.into(),
            request_id: None,
            provider: provider.into(),
            model: model.into(),
            usage,
            estimated_cost_microusd: None,
            latency_ms: 0,
            success: true,
            error_kind: None,
            attributes: BTreeMap::new(),
        }
    }
}

/// Receives usage records. Implementations should return quickly; durable or
/// remote sinks can buffer records before exporting them.
pub trait UsageRecorder: Send + Sync {
    fn record_usage(&self, record: UsageRecord);
}

/// User-supplied price rates. Values are millionths of a US dollar per one
/// million tokens, so `$5 / 1M tokens` is represented as `5_000_000`.
/// No volatile provider prices are compiled into the framework.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelPricing {
    pub input_microusd_per_million_tokens: u64,
    pub cached_input_microusd_per_million_tokens: Option<u64>,
    pub output_microusd_per_million_tokens: u64,
    /// When set, reasoning tokens are removed from ordinary output tokens and
    /// charged at this rate. When unset, output pricing includes reasoning.
    pub reasoning_microusd_per_million_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PricingTable {
    prices: BTreeMap<String, ModelPricing>,
}

impl PricingTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
        pricing: ModelPricing,
    ) -> Option<ModelPricing> {
        self.prices
            .insert(pricing_key(provider.as_ref(), model.as_ref()), pricing)
    }

    pub fn with_model(
        mut self,
        provider: impl AsRef<str>,
        model: impl AsRef<str>,
        pricing: ModelPricing,
    ) -> Self {
        self.insert(provider, model, pricing);
        self
    }

    pub fn pricing(&self, provider: &str, model: &str) -> Option<&ModelPricing> {
        self.prices
            .get(&pricing_key(provider, model))
            .or_else(|| self.prices.get(&pricing_key(provider, "*")))
            .or_else(|| self.prices.get(&pricing_key("*", "*")))
    }

    pub fn estimate_microusd(&self, provider: &str, model: &str, usage: &Usage) -> Option<u64> {
        let pricing = self.pricing(provider, model)?;
        let uncached_input = usage.input_tokens.saturating_sub(usage.cached_input_tokens);
        let cached_rate = pricing
            .cached_input_microusd_per_million_tokens
            .unwrap_or(pricing.input_microusd_per_million_tokens);
        let (ordinary_output, reasoning_rate) = match pricing.reasoning_microusd_per_million_tokens
        {
            Some(rate) => (
                usage.output_tokens.saturating_sub(usage.reasoning_tokens),
                rate,
            ),
            None => (usage.output_tokens, 0),
        };
        let mut estimate = price_tokens(uncached_input, pricing.input_microusd_per_million_tokens);
        estimate = estimate.saturating_add(price_tokens(usage.cached_input_tokens, cached_rate));
        estimate = estimate.saturating_add(price_tokens(
            ordinary_output,
            pricing.output_microusd_per_million_tokens,
        ));
        if pricing.reasoning_microusd_per_million_tokens.is_some() {
            estimate =
                estimate.saturating_add(price_tokens(usage.reasoning_tokens, reasoning_rate));
        }
        Some(estimate)
    }
}

fn pricing_key(provider: &str, model: &str) -> String {
    format!(
        "{}/{}",
        provider.trim().to_ascii_lowercase(),
        model.trim().to_ascii_lowercase()
    )
}

fn price_tokens(tokens: u64, rate: u64) -> u64 {
    let numerator = u128::from(tokens).saturating_mul(u128::from(rate));
    // Nearest microdollar, with saturation for adversarial counters/rates.
    ((numerator.saturating_add(500_000) / 1_000_000).min(u128::from(u64::MAX))) as u64
}

/// Decorates any usage recorder with cost estimation. An explicit cost set by
/// the caller is preserved; otherwise the configured pricing table is used.
#[derive(Clone)]
pub struct PricingUsageRecorder<R> {
    inner: R,
    pricing: Arc<PricingTable>,
}

impl<R> PricingUsageRecorder<R> {
    pub fn new(inner: R, pricing: PricingTable) -> Self {
        Self {
            inner,
            pricing: Arc::new(pricing),
        }
    }

    pub fn inner(&self) -> &R {
        &self.inner
    }

    pub fn pricing(&self) -> &PricingTable {
        &self.pricing
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: UsageRecorder> UsageRecorder for PricingUsageRecorder<R> {
    fn record_usage(&self, mut record: UsageRecord) {
        if record.estimated_cost_microusd.is_none() {
            record.estimated_cost_microusd =
                self.pricing
                    .estimate_microusd(&record.provider, &record.model, &record.usage);
        }
        self.inner.record_usage(record);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelUsageSummary {
    pub requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub usage: Usage,
    pub estimated_cost_microusd: u64,
    pub total_latency_ms: u64,
}

impl ModelUsageSummary {
    pub fn average_latency_ms(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.requests as f64
        }
    }

    fn add(&mut self, record: &UsageRecord) {
        self.requests = self.requests.saturating_add(1);
        if record.success {
            self.successful_requests = self.successful_requests.saturating_add(1);
        } else {
            self.failed_requests = self.failed_requests.saturating_add(1);
        }
        add_usage(&mut self.usage, &record.usage);
        self.estimated_cost_microusd = self
            .estimated_cost_microusd
            .saturating_add(record.estimated_cost_microusd.unwrap_or_default());
        self.total_latency_ms = self.total_latency_ms.saturating_add(record.latency_ms);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageSummary {
    pub totals: ModelUsageSummary,
    /// Keyed as `provider/model` for stable serialization and display.
    pub by_model: BTreeMap<String, ModelUsageSummary>,
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

/// Thread-safe collector suitable for embedding or for periodic export.
#[derive(Clone, Default)]
pub struct InMemoryUsageCollector {
    records: Arc<Mutex<Vec<UsageRecord>>>,
}

impl InMemoryUsageCollector {
    pub fn records(&self) -> Vec<UsageRecord> {
        lock(&self.records).clone()
    }

    pub fn drain(&self) -> Vec<UsageRecord> {
        std::mem::take(&mut *lock(&self.records))
    }

    pub fn summary(&self) -> UsageSummary {
        let mut summary = UsageSummary::default();
        for record in lock(&self.records).iter() {
            summary.totals.add(record);
            summary
                .by_model
                .entry(format!("{}/{}", record.provider, record.model))
                .or_default()
                .add(record);
        }
        summary
    }
}

impl UsageRecorder for InMemoryUsageCollector {
    fn record_usage(&self, record: UsageRecord) {
        lock(&self.records).push(record);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Run,
    Model,
    Tool,
    Retrieval,
    Workflow,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationOutcome {
    Success,
    Error,
    Timeout,
    Cancelled,
}

/// One completed operation. This is intentionally useful independently of
/// tracing, for components that can report exact timings directly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricRecord {
    pub timestamp_ms: u64,
    pub run_id: String,
    pub kind: OperationKind,
    pub name: String,
    pub duration_ms: u64,
    pub outcome: OperationOutcome,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

impl MetricRecord {
    pub fn success(
        run_id: impl Into<String>,
        kind: OperationKind,
        name: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            timestamp_ms: now_ms(),
            run_id: run_id.into(),
            kind,
            name: name.into(),
            duration_ms,
            outcome: OperationOutcome::Success,
            attributes: BTreeMap::new(),
        }
    }
}

pub trait MetricsRecorder: Send + Sync {
    fn record_metric(&self, record: MetricRecord);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LatencySummary {
    pub count: u64,
    pub errors: u64,
    pub timeouts: u64,
    pub cancelled: u64,
    pub total_ms: u64,
    pub min_ms: Option<u64>,
    pub max_ms: Option<u64>,
}

impl LatencySummary {
    pub fn average_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ms as f64 / self.count as f64
        }
    }

    pub fn error_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.errors + self.timeouts + self.cancelled) as f64 / self.count as f64
        }
    }

    fn add(&mut self, record: &MetricRecord) {
        self.count = self.count.saturating_add(1);
        self.total_ms = self.total_ms.saturating_add(record.duration_ms);
        self.min_ms = Some(
            self.min_ms
                .map_or(record.duration_ms, |value| value.min(record.duration_ms)),
        );
        self.max_ms = Some(
            self.max_ms
                .map_or(record.duration_ms, |value| value.max(record.duration_ms)),
        );
        match record.outcome {
            OperationOutcome::Success => {}
            OperationOutcome::Error => self.errors = self.errors.saturating_add(1),
            OperationOutcome::Timeout => self.timeouts = self.timeouts.saturating_add(1),
            OperationOutcome::Cancelled => self.cancelled = self.cancelled.saturating_add(1),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MetricsSnapshot {
    pub overall: LatencySummary,
    pub by_kind: BTreeMap<OperationKind, LatencySummary>,
    /// Keyed as `<kind>:<name>`.
    pub by_operation: BTreeMap<String, LatencySummary>,
}

#[derive(Clone, Default)]
pub struct InMemoryMetricsCollector {
    records: Arc<Mutex<Vec<MetricRecord>>>,
}

impl InMemoryMetricsCollector {
    pub fn records(&self) -> Vec<MetricRecord> {
        lock(&self.records).clone()
    }

    pub fn drain(&self) -> Vec<MetricRecord> {
        std::mem::take(&mut *lock(&self.records))
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let mut snapshot = MetricsSnapshot::default();
        for record in lock(&self.records).iter() {
            snapshot.overall.add(record);
            snapshot.by_kind.entry(record.kind).or_default().add(record);
            snapshot
                .by_operation
                .entry(format!("{:?}:{}", record.kind, record.name).to_ascii_lowercase())
                .or_default()
                .add(record);
        }
        snapshot
    }
}

impl MetricsRecorder for InMemoryMetricsCollector {
    fn record_metric(&self, record: MetricRecord) {
        lock(&self.records).push(record);
    }
}

#[derive(Debug)]
struct PendingMetric {
    timestamp_ms: u128,
    kind: OperationKind,
    name: String,
}

/// Derives run/model/tool latency and failure metrics from existing trace
/// events. Direct calls to [`MetricsRecorder`] remain preferable when exact
/// sub-millisecond timings or explicit timeout classification are available.
#[derive(Clone)]
pub struct TraceMetricsAdapter {
    collector: InMemoryMetricsCollector,
    pending: Arc<Mutex<HashMap<String, PendingMetric>>>,
}

impl Default for TraceMetricsAdapter {
    fn default() -> Self {
        Self::new(InMemoryMetricsCollector::default())
    }
}

impl TraceMetricsAdapter {
    pub fn new(collector: InMemoryMetricsCollector) -> Self {
        Self {
            collector,
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn collector(&self) -> InMemoryMetricsCollector {
        self.collector.clone()
    }
}

impl Tracer for TraceMetricsAdapter {
    fn record(&self, event: TraceEvent) {
        let Some(operation) = traced_operation(&event) else {
            return;
        };
        let key = operation_key(&event, operation);
        if event.kind.ends_with("_started") {
            lock(&self.pending).insert(
                key,
                PendingMetric {
                    timestamp_ms: event.timestamp_ms,
                    kind: operation,
                    name: operation_name(&event, operation),
                },
            );
            return;
        }

        if event.kind.ends_with("_completed") || event.kind.ends_with("_failed") {
            if let Some(pending) = lock(&self.pending).remove(&key) {
                let outcome = if event.kind.ends_with("_failed")
                    || event.fields.get("ok").and_then(Value::as_bool) == Some(false)
                {
                    OperationOutcome::Error
                } else {
                    OperationOutcome::Success
                };
                self.collector.record_metric(MetricRecord {
                    timestamp_ms: event.timestamp_ms.min(u64::MAX as u128) as u64,
                    run_id: event.run_id,
                    kind: pending.kind,
                    name: pending.name,
                    duration_ms: event
                        .timestamp_ms
                        .saturating_sub(pending.timestamp_ms)
                        .min(u64::MAX as u128) as u64,
                    outcome,
                    attributes: json_object_to_map(event.fields),
                });
            }
        }
    }
}

fn traced_operation(event: &TraceEvent) -> Option<OperationKind> {
    let prefix = event.kind.split('_').next()?;
    match prefix {
        "run" => Some(OperationKind::Run),
        "model" => Some(OperationKind::Model),
        "tool" => Some(OperationKind::Tool),
        "retrieval" => Some(OperationKind::Retrieval),
        "workflow" => Some(OperationKind::Workflow),
        _ => None,
    }
}

fn operation_key(event: &TraceEvent, operation: OperationKind) -> String {
    let discriminator = match operation {
        OperationKind::Model => event.fields.get("step"),
        OperationKind::Tool => event.fields.get("id"),
        _ => None,
    }
    .map(Value::to_string)
    .unwrap_or_default();
    format!("{}:{operation:?}:{discriminator}", event.run_id)
}

fn operation_name(event: &TraceEvent, operation: OperationKind) -> String {
    match operation {
        OperationKind::Tool => event
            .fields
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_owned(),
        OperationKind::Model => event
            .fields
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("model")
            .to_owned(),
        _ => format!("{operation:?}").to_ascii_lowercase(),
    }
}

fn json_object_to_map(value: Value) -> BTreeMap<String, Value> {
    value
        .as_object()
        .map(|object| object.clone().into_iter().collect())
        .unwrap_or_default()
}

/// A tracer that fans events out in registration order.
#[derive(Clone, Default)]
pub struct CompositeTracer {
    tracers: Vec<Arc<dyn Tracer>>,
}

impl CompositeTracer {
    pub fn new(tracers: Vec<Arc<dyn Tracer>>) -> Self {
        Self { tracers }
    }

    pub fn push(mut self, tracer: impl Tracer + 'static) -> Self {
        self.tracers.push(Arc::new(tracer));
        self
    }
}

impl Tracer for CompositeTracer {
    fn record(&self, event: TraceEvent) {
        for tracer in &self.tracers {
            tracer.record(event.clone());
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Internal,
    Client,
    Server,
    Producer,
    Consumer,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    Unset,
    Ok,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenTelemetrySpanEvent {
    pub name: String,
    pub timestamp_unix_nanos: u64,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

/// Dependency-neutral representation of OpenTelemetry span data. IDs follow
/// the OTel 32-hex trace / 16-hex span conventions and timestamps are Unix ns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenTelemetrySpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub kind: SpanKind,
    pub start_time_unix_nanos: u64,
    pub end_time_unix_nanos: u64,
    pub status: SpanStatus,
    pub status_description: Option<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
    #[serde(default)]
    pub events: Vec<OpenTelemetrySpanEvent>,
}

/// Implement this trait with the OpenTelemetry SDK, OTLP, a collector agent,
/// or a durable local buffer. Export is synchronous because [`Tracer`] is;
/// network exporters should enqueue batches and flush them asynchronously.
pub trait OpenTelemetryExporter: Send + Sync {
    fn export(&self, spans: &[OpenTelemetrySpan]) -> anyhow::Result<()>;

    fn force_flush(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn shutdown(&self) -> anyhow::Result<()> {
        self.force_flush()
    }
}

#[derive(Clone, Default)]
pub struct InMemorySpanExporter {
    spans: Arc<Mutex<Vec<OpenTelemetrySpan>>>,
}

impl InMemorySpanExporter {
    pub fn spans(&self) -> Vec<OpenTelemetrySpan> {
        lock(&self.spans).clone()
    }
}

impl OpenTelemetryExporter for InMemorySpanExporter {
    fn export(&self, spans: &[OpenTelemetrySpan]) -> anyhow::Result<()> {
        lock(&self.spans).extend_from_slice(spans);
        Ok(())
    }
}

#[derive(Debug)]
struct PendingSpan {
    trace_id: String,
    span_id: String,
    parent_span_id: Option<String>,
    name: String,
    kind: SpanKind,
    start_time_unix_nanos: u64,
    attributes: BTreeMap<String, Value>,
}

#[derive(Default)]
struct OpenTelemetryState {
    pending: HashMap<String, PendingSpan>,
    run_spans: HashMap<String, String>,
    trace_ids: HashMap<String, String>,
}

/// Converts liteagent lifecycle traces to OpenTelemetry-compatible spans.
pub struct OpenTelemetryAdapter {
    exporter: Arc<dyn OpenTelemetryExporter>,
    state: Mutex<OpenTelemetryState>,
    export_errors: Mutex<Vec<String>>,
}

impl OpenTelemetryAdapter {
    pub fn new(exporter: impl OpenTelemetryExporter + 'static) -> Self {
        Self {
            exporter: Arc::new(exporter),
            state: Mutex::new(OpenTelemetryState::default()),
            export_errors: Mutex::new(Vec::new()),
        }
    }

    pub fn from_shared(exporter: Arc<dyn OpenTelemetryExporter>) -> Self {
        Self {
            exporter,
            state: Mutex::new(OpenTelemetryState::default()),
            export_errors: Mutex::new(Vec::new()),
        }
    }

    /// Export failures cannot be returned through [`Tracer::record`], so they
    /// remain observable here instead of being silently discarded.
    pub fn export_errors(&self) -> Vec<String> {
        lock(&self.export_errors).clone()
    }

    pub fn force_flush(&self) -> anyhow::Result<()> {
        self.exporter.force_flush()
    }
}

impl Tracer for OpenTelemetryAdapter {
    fn record(&self, event: TraceEvent) {
        let Some(operation) = traced_operation(&event) else {
            return;
        };
        let key = operation_key(&event, operation);
        if event.kind.ends_with("_started") {
            let mut state = lock(&self.state);
            let trace_id = state
                .trace_ids
                .entry(event.run_id.clone())
                .or_insert_with(new_trace_id)
                .clone();
            let span_id = new_span_id();
            let parent_span_id = if operation == OperationKind::Run {
                None
            } else {
                state.run_spans.get(&event.run_id).cloned()
            };
            if operation == OperationKind::Run {
                state
                    .run_spans
                    .insert(event.run_id.clone(), span_id.clone());
            }
            state.pending.insert(
                key,
                PendingSpan {
                    trace_id,
                    span_id,
                    parent_span_id,
                    name: operation_name(&event, operation),
                    kind: match operation {
                        OperationKind::Model | OperationKind::Tool | OperationKind::Retrieval => {
                            SpanKind::Client
                        }
                        _ => SpanKind::Internal,
                    },
                    start_time_unix_nanos: event
                        .timestamp_ms
                        .saturating_mul(1_000_000)
                        .min(u64::MAX as u128) as u64,
                    attributes: json_object_to_map(event.fields),
                },
            );
            return;
        }

        if !event.kind.ends_with("_completed") && !event.kind.ends_with("_failed") {
            return;
        }

        let pending = {
            let mut state = lock(&self.state);
            let pending = state.pending.remove(&key);
            if operation == OperationKind::Run {
                state.run_spans.remove(&event.run_id);
                state.trace_ids.remove(&event.run_id);
            }
            pending
        };
        let Some(pending) = pending else {
            return;
        };
        let failed = event.kind.ends_with("_failed")
            || event.fields.get("ok").and_then(Value::as_bool) == Some(false);
        let status_description = failed.then(|| {
            event
                .fields
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("operation failed")
                .to_owned()
        });
        let span = OpenTelemetrySpan {
            trace_id: pending.trace_id,
            span_id: pending.span_id,
            parent_span_id: pending.parent_span_id,
            name: pending.name,
            kind: pending.kind,
            start_time_unix_nanos: pending.start_time_unix_nanos,
            end_time_unix_nanos: event
                .timestamp_ms
                .saturating_mul(1_000_000)
                .min(u64::MAX as u128) as u64,
            status: if failed {
                SpanStatus::Error
            } else {
                SpanStatus::Ok
            },
            status_description,
            attributes: pending.attributes,
            events: vec![OpenTelemetrySpanEvent {
                name: event.kind,
                timestamp_unix_nanos: now_ns(),
                attributes: json_object_to_map(event.fields),
            }],
        };
        if let Err(error) = self.exporter.export(&[span]) {
            lock(&self.export_errors).push(error.to_string());
        }
    }
}

fn new_trace_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn new_span_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..16].to_owned()
}
