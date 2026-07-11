use anyhow::Context;
use async_trait::async_trait;
use ferrant::graph::{FileGraphStore, Graph, NodeContext, NodeOutput, NodeRetryPolicy};
use ferrant::llm::anthropic::AnthropicModel;
use ferrant::llm::openai::OpenAiModel;
use ferrant::rag::{
    Document, Embedder, FileVectorStore, HashEmbedder, InMemoryVectorStore, LexicalReranker,
    Retriever as RustRetriever, VectorStore,
};
use ferrant::{
    Agent as RustAgent, ContentPart, FileStorage, McpClient, Message, ModelResponse, StreamEvent,
    Tool as RustTool,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn py_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let encoded: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&encoded).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let encoded = serde_json::to_string(value).map_err(runtime_error)?;
    Ok(py
        .import("json")?
        .call_method1("loads", (encoded,))?
        .unbind())
}

fn response_value(response: &ModelResponse) -> Value {
    json!({
        "content": response.content,
        "content_parts": response.content_parts,
        "tool_calls": response.tool_calls,
        "usage": response.usage,
    })
}

#[pyfunction]
fn run_cli(args: Vec<String>) -> PyResult<i32> {
    ferrant::cli::execute(&args).map_err(runtime_error)?;
    Ok(0)
}

struct PythonFunctionTool {
    name: String,
    description: String,
    schema: Value,
    callback: Py<PyAny>,
}

#[async_trait]
impl RustTool for PythonFunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, args: Value) -> anyhow::Result<String> {
        Python::attach(|py| {
            let input = value_to_py(py, &args)?;
            let output = self.callback.call1(py, (input,))?;
            if let Ok(text) = output.extract::<String>(py) {
                return Ok::<String, PyErr>(text);
            }
            let bound = output.bind(py);
            let encoded: String = py
                .import("json")?
                .call_method1("dumps", (bound,))?
                .extract()?;
            Ok::<String, PyErr>(encoded)
        })
        .map_err(anyhow::Error::from)
    }
}

#[pyclass(name = "Tool")]
struct PyTool {
    inner: Arc<dyn RustTool>,
}

#[pymethods]
impl PyTool {
    #[new]
    fn new(
        name: String,
        description: String,
        schema: &Bound<'_, PyAny>,
        callback: Py<PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(PythonFunctionTool {
                name,
                description,
                schema: py_to_value(schema)?,
                callback,
            }),
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_owned()
    }
}

#[pyclass(name = "McpTools")]
struct PyMcpTools {
    tools: Vec<Arc<dyn RustTool>>,
}

#[pymethods]
impl PyMcpTools {
    #[staticmethod]
    #[pyo3(signature = (command, args = Vec::new()))]
    fn connect<'py>(
        py: Python<'py>,
        command: String,
        args: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = McpClient::connect(command, args)
                .await
                .map_err(runtime_error)?;
            let tools = client.tools().await.map_err(runtime_error)?;
            Ok(PyMcpTools { tools })
        })
    }

    fn names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.name().to_owned())
            .collect()
    }
}

#[pyclass(name = "Agent")]
struct PyAgent {
    inner: Arc<Mutex<RustAgent>>,
}

fn collect_tools(
    py: Python<'_>,
    tools: Option<Vec<Py<PyTool>>>,
    mcp: Option<Py<PyMcpTools>>,
) -> Vec<Arc<dyn RustTool>> {
    let mut collected = tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| tool.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    if let Some(mcp) = mcp {
        collected.extend(mcp.borrow(py).tools.iter().cloned());
    }
    collected
}

#[pymethods]
impl PyAgent {
    #[staticmethod]
    #[pyo3(signature = (model, api_key, instructions=None, base_url=None, tools=None, mcp=None, storage_path=None, temperature=None, modalities=None, audio_format=None, audio_voice=None))]
    #[allow(clippy::too_many_arguments)]
    fn openai(
        py: Python<'_>,
        model: String,
        api_key: String,
        instructions: Option<String>,
        base_url: Option<String>,
        tools: Option<Vec<Py<PyTool>>>,
        mcp: Option<Py<PyMcpTools>>,
        storage_path: Option<String>,
        temperature: Option<f32>,
        modalities: Option<Vec<String>>,
        audio_format: Option<String>,
        audio_voice: Option<String>,
    ) -> Self {
        let mut model = OpenAiModel::new(model, api_key);
        if let Some(base_url) = base_url {
            model = model.with_base_url(base_url);
        }
        if let Some(temperature) = temperature {
            model = model.with_temperature(temperature);
        }
        if let Some(modalities) = modalities {
            model = model.with_modalities(modalities);
        }
        if let (Some(format), Some(voice)) = (audio_format, audio_voice) {
            model = model.with_audio_output(format, voice);
        }
        let mut builder = RustAgent::builder(model).tools(collect_tools(py, tools, mcp));
        if let Some(instructions) = instructions {
            builder = builder.instructions(instructions);
        }
        if let Some(path) = storage_path {
            builder = builder.storage(FileStorage::new(path));
        }
        Self {
            inner: Arc::new(Mutex::new(builder.build())),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (model, api_key, instructions=None, tools=None, mcp=None, storage_path=None, max_tokens=2048))]
    #[allow(clippy::too_many_arguments)]
    fn anthropic(
        py: Python<'_>,
        model: String,
        api_key: String,
        instructions: Option<String>,
        tools: Option<Vec<Py<PyTool>>>,
        mcp: Option<Py<PyMcpTools>>,
        storage_path: Option<String>,
        max_tokens: u32,
    ) -> Self {
        let model = AnthropicModel::new(model, api_key).with_max_tokens(max_tokens);
        let mut builder = RustAgent::builder(model).tools(collect_tools(py, tools, mcp));
        if let Some(instructions) = instructions {
            builder = builder.instructions(instructions);
        }
        if let Some(path) = storage_path {
            builder = builder.storage(FileStorage::new(path));
        }
        Self {
            inner: Arc::new(Mutex::new(builder.build())),
        }
    }

    fn run<'py>(&self, py: Python<'py>, input: String) -> PyResult<Bound<'py, PyAny>> {
        let agent = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            agent.lock().await.run(input).await.map_err(runtime_error)
        })
    }

    fn run_session<'py>(
        &self,
        py: Python<'py>,
        session_id: String,
        input: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let agent = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            agent
                .lock()
                .await
                .run_session(&session_id, input)
                .await
                .map_err(runtime_error)
        })
    }

    fn run_structured<'py>(
        &self,
        py: Python<'py>,
        input: String,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let schema = py_to_value(schema)?;
        let agent = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let value = agent
                .lock()
                .await
                .run_structured::<Value>(input, &schema)
                .await
                .map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &value))
        })
    }

    fn run_multimodal<'py>(
        &self,
        py: Python<'py>,
        parts: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parts = py_to_value(parts)?;
        let parts: Vec<ContentPart> = serde_json::from_value(parts)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let agent = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = agent
                .lock()
                .await
                .run_message(Message::user_parts(parts))
                .await
                .map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &response_value(&response)))
        })
    }

    fn run_stream<'py>(
        &self,
        py: Python<'py>,
        input: String,
        callback: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let agent = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (sender, mut receiver) =
                tokio::sync::mpsc::channel::<ferrant::Result<StreamEvent>>(64);
            let mut locked = agent.lock().await;
            let consume = async {
                while let Some(event) = receiver.recv().await {
                    let event = event.map_err(runtime_error)?;
                    let value = serde_json::to_value(event).map_err(runtime_error)?;
                    Python::attach(|py| {
                        let argument = value_to_py(py, &value)?;
                        callback.call1(py, (argument,))?;
                        Ok::<_, PyErr>(())
                    })?;
                }
                Ok::<_, PyErr>(())
            };
            let (response, consumed) = tokio::join!(locked.run_stream(input, sender), consume);
            consumed?;
            let response = response.map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &response_value(&response)))
        })
    }
}

struct SharedAgentTool {
    name: String,
    description: String,
    agent: Arc<Mutex<RustAgent>>,
}

#[async_trait]
impl RustTool for SharedAgentTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"task":{"type":"string"}},"required":["task"]})
    }
    async fn execute(&self, args: Value) -> anyhow::Result<String> {
        let task = args
            .get("task")
            .and_then(Value::as_str)
            .context("task is required")?;
        Ok(self.agent.lock().await.run(task).await?)
    }
}

#[pyclass(name = "Team")]
struct PyTeam {
    coordinator: Arc<Mutex<RustAgent>>,
}

#[pymethods]
impl PyTeam {
    #[staticmethod]
    fn openai(
        py: Python<'_>,
        model: String,
        api_key: String,
        members: Vec<(String, String, Py<PyAgent>)>,
    ) -> Self {
        let tools = members
            .into_iter()
            .map(|(name, description, agent)| {
                Arc::new(SharedAgentTool {
                    name,
                    description,
                    agent: agent.borrow(py).inner.clone(),
                }) as Arc<dyn RustTool>
            })
            .collect();
        let coordinator = RustAgent::builder(OpenAiModel::new(model, api_key))
            .instructions("Coordinate specialist agents, delegate when useful, and synthesize the final answer.")
            .tools(tools)
            .build();
        Self {
            coordinator: Arc::new(Mutex::new(coordinator)),
        }
    }

    fn run<'py>(&self, py: Python<'py>, input: String) -> PyResult<Bound<'py, PyAny>> {
        let coordinator = self.coordinator.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            coordinator
                .lock()
                .await
                .run(input)
                .await
                .map_err(runtime_error)
        })
    }
}

#[pyclass(name = "Retriever")]
struct PyRetriever {
    inner: Arc<RustRetriever>,
}

#[pymethods]
impl PyRetriever {
    #[new]
    #[pyo3(signature = (dimensions=256, path=None, hybrid=true))]
    fn new(dimensions: usize, path: Option<String>, hybrid: bool) -> PyResult<Self> {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(dimensions));
        let store: Arc<dyn VectorStore> = if let Some(path) = path {
            Arc::new(FileVectorStore::open(path).map_err(runtime_error)?)
        } else {
            Arc::new(InMemoryVectorStore::default())
        };
        let retriever = RustRetriever::from_shared(embedder, store);
        let retriever = if hybrid {
            retriever
                .hybrid(0.7, 0.3)
                .with_reranker(LexicalReranker::default())
        } else {
            retriever
        };
        Ok(Self {
            inner: Arc::new(retriever),
        })
    }

    fn upsert<'py>(
        &self,
        py: Python<'py>,
        documents: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let documents: Vec<Document> = serde_json::from_value(py_to_value(documents)?)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let retriever = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            retriever.upsert(documents).await.map_err(runtime_error)
        })
    }

    #[pyo3(signature = (query, limit=5))]
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let retriever = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = retriever
                .retrieve(&query, limit)
                .await
                .map_err(runtime_error)?;
            let value = serde_json::to_value(results).map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &value))
        })
    }
}

struct PythonNode {
    name: String,
    callback: Py<PyAny>,
    max_attempts: u32,
    timeout_seconds: Option<f64>,
}

enum PythonEdge {
    Edge(String, String),
    Route(String, String, String),
    Join(Vec<String>, String),
}

#[pyclass(name = "WorkflowBuilder")]
struct PyWorkflowBuilder {
    name: String,
    directory: String,
    version: String,
    max_steps: u64,
    entry: Option<String>,
    nodes: Vec<PythonNode>,
    edges: Vec<PythonEdge>,
    interrupts: Vec<String>,
}

#[pymethods]
impl PyWorkflowBuilder {
    #[new]
    #[pyo3(signature = (name, checkpoint_directory, version="1".to_owned(), max_steps=1000))]
    fn new(name: String, checkpoint_directory: String, version: String, max_steps: u64) -> Self {
        Self {
            name,
            directory: checkpoint_directory,
            version,
            max_steps,
            entry: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            interrupts: Vec::new(),
        }
    }

    fn entry(&mut self, node: String) {
        self.entry = Some(node);
    }

    #[pyo3(signature = (name, callback, max_attempts=1, timeout_seconds=None))]
    fn node(
        &mut self,
        name: String,
        callback: Py<PyAny>,
        max_attempts: u32,
        timeout_seconds: Option<f64>,
    ) {
        self.nodes.push(PythonNode {
            name,
            callback,
            max_attempts,
            timeout_seconds,
        });
    }

    fn edge(&mut self, source: String, target: String) {
        self.edges.push(PythonEdge::Edge(source, target));
    }

    fn route(&mut self, source: String, label: String, target: String) {
        self.edges.push(PythonEdge::Route(source, label, target));
    }

    fn join(&mut self, predecessors: Vec<String>, target: String) {
        self.edges.push(PythonEdge::Join(predecessors, target));
    }

    fn interrupt_before(&mut self, node: String) {
        self.interrupts.push(node);
    }

    fn build(&self, py: Python<'_>) -> PyResult<PyWorkflowGraph> {
        let entry = self
            .entry
            .clone()
            .ok_or_else(|| PyValueError::new_err("workflow entry is required"))?;
        let mut builder = Graph::builder(&self.name, FileGraphStore::new(&self.directory))
            .version(&self.version)
            .entry(entry)
            .max_steps(self.max_steps);
        for node in &self.nodes {
            let callback = node.callback.clone_ref(py);
            let mut retry = NodeRetryPolicy::attempts(node.max_attempts);
            if let Some(seconds) = node.timeout_seconds {
                if !seconds.is_finite() || seconds <= 0.0 {
                    return Err(PyValueError::new_err("timeout_seconds must be positive"));
                }
                retry = retry.with_timeout(Duration::from_secs_f64(seconds));
            }
            builder = builder.node_with_retry(&node.name, retry, move |context| {
                let callback = Python::attach(|py| callback.clone_ref(py));
                async move { call_python_node(callback, context) }
            });
        }
        for edge in &self.edges {
            builder = match edge {
                PythonEdge::Edge(source, target) => builder.edge(source, target),
                PythonEdge::Route(source, label, target) => builder.route(source, label, target),
                PythonEdge::Join(predecessors, target) => {
                    builder.join(predecessors.clone(), target)
                }
            };
        }
        for node in &self.interrupts {
            builder = builder.interrupt_before(node);
        }
        Ok(PyWorkflowGraph {
            inner: Arc::new(builder.build().map_err(runtime_error)?),
        })
    }
}

fn call_python_node(callback: Py<PyAny>, context: NodeContext) -> anyhow::Result<NodeOutput> {
    Python::attach(|py| {
        let context = json!({
            "execution_id": context.execution_id,
            "node_name": context.node_name,
            "idempotency_key": context.idempotency_key,
            "attempt": context.attempt,
            "state": context.state,
        });
        let argument = value_to_py(py, &context)?;
        let output = callback.call1(py, (argument,))?;
        let value = py_to_value(output.bind(py))?;
        let update = value
            .get("update")
            .cloned()
            .unwrap_or_else(|| value.clone());
        let mut node_output = NodeOutput::merge(update);
        if let Some(routes) = value.get("routes").and_then(Value::as_array) {
            node_output = node_output.routes(routes.iter().filter_map(Value::as_str));
        }
        if let Some(reason) = value.get("pause_reason").and_then(Value::as_str) {
            node_output = node_output.and_pause(reason);
        }
        Ok::<NodeOutput, PyErr>(node_output)
    })
    .map_err(anyhow::Error::from)
}

#[pyclass(name = "WorkflowGraph")]
struct PyWorkflowGraph {
    inner: Arc<Graph>,
}

#[pymethods]
impl PyWorkflowGraph {
    fn run<'py>(
        &self,
        py: Python<'py>,
        execution_id: String,
        state: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = py_to_value(state)?;
        let graph = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let checkpoint = graph
                .run(&execution_id, state)
                .await
                .map_err(runtime_error)?;
            let value = serde_json::to_value(checkpoint).map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &value))
        })
    }

    #[pyo3(signature = (execution_id, state_update=None))]
    fn resume<'py>(
        &self,
        py: Python<'py>,
        execution_id: String,
        state_update: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let update = state_update.map(py_to_value).transpose()?;
        let graph = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let checkpoint = graph
                .resume_with(&execution_id, update)
                .await
                .map_err(runtime_error)?;
            let value = serde_json::to_value(checkpoint).map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &value))
        })
    }

    fn recover<'py>(&self, py: Python<'py>, execution_id: String) -> PyResult<Bound<'py, PyAny>> {
        let graph = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let checkpoint = graph.recover(&execution_id).await.map_err(runtime_error)?;
            let value = serde_json::to_value(checkpoint).map_err(runtime_error)?;
            Python::attach(|py| value_to_py(py, &value))
        })
    }
}

#[pymodule]
#[pyo3(name = "_native")]
fn ferrant_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cli, module)?)?;
    module.add_class::<PyTool>()?;
    module.add_class::<PyMcpTools>()?;
    module.add_class::<PyAgent>()?;
    module.add_class::<PyTeam>()?;
    module.add_class::<PyRetriever>()?;
    module.add_class::<PyWorkflowBuilder>()?;
    module.add_class::<PyWorkflowGraph>()?;
    Ok(())
}
