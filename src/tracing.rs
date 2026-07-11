use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub timestamp_ms: u128,
    pub run_id: String,
    pub kind: String,
    pub fields: Value,
}

impl TraceEvent {
    pub fn new(run_id: impl Into<String>, kind: impl Into<String>, fields: Value) -> Self {
        Self {
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            run_id: run_id.into(),
            kind: kind.into(),
            fields,
        }
    }
}

pub trait Tracer: Send + Sync {
    fn record(&self, event: TraceEvent);
}

#[derive(Default)]
pub struct NoopTracer;
impl Tracer for NoopTracer {
    fn record(&self, _event: TraceEvent) {}
}

/// Thread-safe tracer useful in tests and small applications.
#[derive(Default, Clone)]
pub struct InMemoryTracer(pub Arc<Mutex<Vec<TraceEvent>>>);
impl InMemoryTracer {
    pub fn events(&self) -> Vec<TraceEvent> {
        self.0.lock().unwrap().clone()
    }
}
impl Tracer for InMemoryTracer {
    fn record(&self, event: TraceEvent) {
        self.0.lock().unwrap().push(event);
    }
}

/// Bridge trace events into the application's configured OpenTelemetry
/// global tracer. Enable the `opentelemetry` crate feature and install any
/// OpenTelemetry SDK/exporter appropriate for the deployment.
#[cfg(feature = "opentelemetry")]
#[derive(Default)]
pub struct OpenTelemetryTracer;

#[cfg(feature = "opentelemetry")]
impl Tracer for OpenTelemetryTracer {
    fn record(&self, event: TraceEvent) {
        use opentelemetry::trace::{Span, Tracer as _};
        use opentelemetry::KeyValue;
        let tracer = opentelemetry::global::tracer("ferragent");
        let mut span = tracer.start(event.kind);
        span.set_attribute(KeyValue::new("ferragent.run_id", event.run_id));
        span.set_attribute(KeyValue::new(
            "ferragent.timestamp_ms",
            event.timestamp_ms.to_string(),
        ));
        span.set_attribute(KeyValue::new("ferragent.fields", event.fields.to_string()));
        span.end();
    }
}
