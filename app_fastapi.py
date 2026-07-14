# FastAPI Web Application v2 - LLMOps Edition

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os

from eval_framework_minimal import MinimalEvaluator
from guardrail_minimal import MinimalGuardrails
from fine_tuning_config_minimal import MinimalFineTuningPipeline

from vector_store.chroma_rag_pipeline import ChromaRAGPipeline
from registry.prompt_registry_db import PromptRegistryDB, PromptStatus
from tracking.mlflow_tracking import MLflowTracker

app = FastAPI(
    title="ACME Employee HR Assistant - LLMOps Edition",
    description="RAG + Prompt Registry + Vector Store + MLflow Tracking",
    version="2.0",
)

# ------------------------------------------------------------------
# Initialize components
# ------------------------------------------------------------------
tracker = MLflowTracker(experiment_name="acme-hr-rag-assistant")

rag = ChromaRAGPipeline(use_huggingface=True)
tracker.log_vector_store_event(rag.get_vector_store_stats(), event="startup")

registry = PromptRegistryDB()
if not registry.get_all_versions("hr_assistant"):
    seed = registry.create_prompt(
        "hr_assistant", "You are a helpful HR assistant.", tags=["baseline"], notes="Initial seed prompt"
    )
    registry.promote(seed.name, seed.version, {"faithfulness": 1.0, "answer_relevance": 1.0, "latency_ms": 1})
    tracker.log_prompt_version(seed, event="created")

guardrails = MinimalGuardrails()
evaluator = MinimalEvaluator()


# ------------------------------------------------------------------
# Request/response models
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    response: str
    quality_score: float
    passed_quality_gate: bool
    context_count: int
    latency_ms: float
    violations: List[str] = []


class CreatePromptRequest(BaseModel):
    name: str
    content: str
    tags: Optional[List[str]] = None
    notes: Optional[str] = ""


class PromoteRequest(BaseModel):
    version: int
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    latency_ms: Optional[float] = None
    force: bool = False


# ------------------------------------------------------------------
# Home page
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


# ------------------------------------------------------------------
# Original-compatible endpoints (same contract as app_fastapi.py)
# ------------------------------------------------------------------
@app.post("/api/query")
async def query(request: QueryRequest):
    # TODO [10 Marks]: API Integration - End-to-End Query Orchestration
    # ------------------------------------------------------------------
    # Implement the full request pipeline for the main chat endpoint,
    # wiring together every other component built in this assessment.
    #
    # Requirements (in order):
    # 1. Validate `request.question` with `guardrails.validate_input(...)`.
    #    If the result is not valid, raise `HTTPException(status_code=400,
    #    detail=guard_result.message)`.
    # 2. Run the RAG pipeline: `response, metadata = rag.query(request.question)`.
    #    Coerce `response` to a string, falling back to
    #    "Unable to generate response" if it is falsy.
    # 3. Sanitize the response with `guardrails.sanitize_output(response)`.
    # 4. Evaluate quality with `evaluator.evaluate(query=..., response=...,
    #    context=metadata.matched_documents if available else [])`.
    # 5. Log the interaction via `tracker.log_query(query=..., response=...,
    #    metadata=metadata, eval_result=eval_result,
    #    extra_params={"embedding_model": rag.embedding_model_name})`.
    #    (This must never be allowed to break the response — see the
    #    MLflowTracker.log_query() TODO, which already fails safe.)
    # 6. Return a `QueryResponse` populating `response`, `quality_score`
    #    (`eval_result.overall_score`), `passed_quality_gate`
    #    (`eval_result.passed`), `context_count`
    #    (`metadata.context_count`), `latency_ms`
    #    (`metadata.total_latency_ms`), and `violations` (stringified list
    #    from `guard_result.violations`, or empty list).
    # 7. Wrap steps 2-6 in try/except: re-raise `HTTPException` as-is (from
    #    step 1), but catch any other `Exception`, print it, and return a
    #    `QueryResponse` with `quality_score=0.0`, `passed_quality_gate=False`,
    #    `context_count=0`, `latency_ms=0`, and the error message in both
    #    `response` and `violations`.
    #
    # Inputs: `request: QueryRequest` (has `.question: str`).
    # Outputs: `QueryResponse` (see the Pydantic model above) or an
    # `HTTPException` for guardrail failures.
    # Dependencies: `guardrails`, `rag`, `evaluator`, `tracker` (module-level
    # instances constructed earlier in this file).
    # Acceptance criteria: a normal HR question returns HTTP 200 with a
    # populated response and quality metrics; a question containing PII or
    # an injection attempt returns HTTP 400; an internal error never
    # produces an unhandled 500 — it degrades to a QueryResponse describing
    # the failure.
    raise NotImplementedError("Student Implementation Required: /api/query orchestration")


@app.post("/api/validate")
async def validate(text: dict):
    result = guardrails.validate_input(text["text"])
    return {"valid": result.valid, "message": result.message, "violations": result.violations}


@app.get("/api/retrieve")
async def retrieve(query: str):
    docs, _latency = rag.retrieve(query, k=5)
    return {"documents": docs}


@app.get("/api/retrieve/detailed")
async def retrieve_detailed(query: str, k: int = 5):
    """Vector search results with similarity scores + metadata."""
    return {"results": rag.retrieve_with_scores(query, k=k)}


@app.post("/api/evaluate")
async def evaluate(data: dict):
    query_text = data["query"]
    response_text = data["response"]

    _, metadata = rag.query(query_text)
    context = metadata.matched_documents if hasattr(metadata, "matched_documents") else []

    result = evaluator.evaluate(query=query_text, response=response_text, context=context)
    return {"faithfulness": result.faithfulness, "relevance": result.answer_relevance, "overall": result.overall_score}


@app.get("/api/finetuning")
async def finetuning():
    pipeline = MinimalFineTuningPipeline()
    return {"lora": pipeline.get_lora_config(), "training": pipeline.get_training_config()}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "components": 7,
        "vector_store": rag.get_vector_store_stats(),
        "mlflow": tracker.get_info(),
    }


# ------------------------------------------------------------------
# NEW: Vector store management endpoints
# ------------------------------------------------------------------
@app.get("/api/vectorstore/stats")
async def vectorstore_stats():
    return rag.get_vector_store_stats()


@app.post("/api/vectorstore/reingest")
async def vectorstore_reingest():
    stats = rag.reingest()
    tracker.log_vector_store_event(stats, event="reingest")
    return stats


# ------------------------------------------------------------------
# NEW: Persistent prompt registry endpoints
# ------------------------------------------------------------------
@app.get("/api/prompts")
async def list_prompts():
    return {"names": registry.list_prompt_names()}


@app.post("/api/prompts")
async def create_prompt(req: CreatePromptRequest):
    version = registry.create_prompt(req.name, req.content, tags=req.tags, notes=req.notes or "")
    tracker.log_prompt_version(version, event="created")
    return version.__dict__


@app.get("/api/prompts/{name}/versions")
async def get_versions(name: str):
    versions = registry.get_all_versions(name)
    return {"name": name, "versions": [v.__dict__ for v in versions]}


@app.get("/api/prompts/{name}/production")
async def get_production(name: str):
    version = registry.get_current_production(name)
    if not version:
        raise HTTPException(status_code=404, detail=f"No production version found for '{name}'")
    return version.__dict__


@app.post("/api/prompts/{name}/promote")
async def promote_prompt(name: str, req: PromoteRequest):
    metrics: Dict = {}
    if req.faithfulness is not None:
        metrics["faithfulness"] = req.faithfulness
    if req.answer_relevance is not None:
        metrics["answer_relevance"] = req.answer_relevance
    if req.latency_ms is not None:
        metrics["latency_ms"] = req.latency_ms

    result = registry.promote(name, req.version, metrics, force=req.force)
    tracker.log_promotion_attempt(name, req.version, metrics, result["promoted"], result.get("failures"))
    if result["promoted"]:
        version = registry.get_version(name, req.version)
        tracker.log_prompt_version(version, event="promoted")
    return result


@app.post("/api/prompts/{name}/rollback")
async def rollback_prompt(name: str):
    version = registry.rollback(name)
    if not version:
        raise HTTPException(status_code=400, detail="No archived production version available to roll back to")
    tracker.log_prompt_version(version, event="rollback")
    return version.__dict__


@app.get("/api/prompts/{name}/diff")
async def diff_prompt(name: str, v1: int, v2: int):
    lines = registry.diff(name, v1, v2)
    return {"name": name, "v1": v1, "v2": v2, "diff": lines}


# ------------------------------------------------------------------
# NEW: MLflow info endpoint
# ------------------------------------------------------------------
@app.get("/api/mlflow/info")
async def mlflow_info():
    return tracker.get_info()


# ------------------------------------------------------------------
# HTML UI (original 6 tabs + 3 new LLMOps tabs)
# ------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ACME Employees Assistant Portal - LLMOps Edition</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px;
        }
        .container { background: white; border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); max-width: 900px; width: 100%; padding: 40px; }
        h1 { color: #333; margin-bottom: 10px; text-align: center; }
        .subtitle { color: #666; text-align: center; margin-bottom: 30px; font-size: 14px; }
        .tabs { display: flex; gap: 8px; margin-bottom: 30px; border-bottom: 2px solid #f0f0f0; flex-wrap: wrap; }
        .tab-button { padding: 10px 16px; border: none; background: none; cursor: pointer; font-size: 13px; font-weight: 600; color: #666; border-bottom: 3px solid transparent; transition: all 0.3s; }
        .tab-button.active { color: #667eea; border-bottom-color: #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .input-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 600; }
        input, textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-family: inherit; font-size: 14px; }
        textarea { resize: vertical; min-height: 80px; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px; transition: transform 0.2s; margin-right: 8px; }
        button:hover { transform: translateY(-2px); }
        .result { background: #f5f5f5; border-left: 4px solid #667eea; padding: 20px; border-radius: 8px; margin-top: 20px; display: none; white-space: pre-wrap; }
        .result.success { display: block; border-left-color: #28a745; }
        .result.error { display: block; border-left-color: #dc3545; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px; }
        .metric { background: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd; text-align: center; }
        .metric-value { font-size: 22px; font-weight: bold; color: #667eea; }
        .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
        .status { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .status-item { flex: 1; min-width: 120px; }
        .status-item strong { display: block; color: #333; }
        .status-item span { color: #28a745; font-size: 12px; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 12px; border-top: 1px solid #f0f0f0; padding-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; font-size: 13px; }
        code { background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ACME Employees Assistant Portal</h1>
        <p class="subtitle">LLMOps Edition &middot; ChromaDB Vector Store &middot; Persistent Prompt Registry &middot; MLflow Tracking</p>

        <div class="status">
            <div class="status-item"><strong>Status</strong><span>Active</span></div>
            <div class="status-item"><strong>Components</strong><span>7/7</span></div>
            <div class="status-item"><strong>Mode</strong><span>Local Only (HuggingFace)</span></div>
        </div>

        <div class="tabs">
            <button class="tab-button active" onclick="showTab('chat')">Chat</button>
            <button class="tab-button" onclick="showTab('guardrails')">Guardrails</button>
            <button class="tab-button" onclick="showTab('rag')">RAG</button>
            <button class="tab-button" onclick="showTab('vectorstore')">Vector Store</button>
            <button class="tab-button" onclick="showTab('registry')">Prompt Registry</button>
            <button class="tab-button" onclick="showTab('eval')">Evaluation</button>
            <button class="tab-button" onclick="showTab('ft')">Fine-tuning</button>
            <button class="tab-button" onclick="showTab('mlflow')">MLflow</button>
        </div>

        <div id="chat" class="tab-content active">
            <h2>Chat with HR Assistant</h2>
            <div class="input-group">
                <label>Your Question</label>
                <textarea id="chatInput" placeholder="e.g., What is the leave policy?"></textarea>
            </div>
            <button onclick="sendQuery()">Send Query</button>
            <div id="chatResult" class="result"></div>
        </div>

        <div id="guardrails" class="tab-content">
            <h2>Guardrails</h2>
            <div class="input-group">
                <label>Test Input</label>
                <textarea id="guardrailInput" placeholder="e.g., What is the company policy?"></textarea>
            </div>
            <button onclick="validateInput()">Validate</button>
            <div id="guardrailResult" class="result"></div>
        </div>

        <div id="rag" class="tab-content">
            <h2>RAG Pipeline (ChromaDB vector search)</h2>
            <div class="input-group">
                <label>Query to Retrieve</label>
                <input type="text" id="ragInput" placeholder="e.g., leave policy" value="leave policy">
            </div>
            <button onclick="getDocuments()">Retrieve (simple)</button>
            <button onclick="getDocumentsDetailed()">Retrieve with similarity scores</button>
            <div id="ragResult" class="result"></div>
        </div>

        <div id="vectorstore" class="tab-content">
            <h2>Vector Store (ChromaDB, embedded &amp; persistent)</h2>
            <button onclick="getVectorStoreStats()">Refresh Stats</button>
            <button onclick="reingestVectorStore()">Re-ingest documents</button>
            <div id="vectorstoreResult" class="result"></div>
        </div>

        <div id="registry" class="tab-content">
            <h2>Prompt Registry (persistent, SQLite-backed)</h2>
            <div class="input-group">
                <label>Prompt Name</label>
                <input type="text" id="promptName" value="hr_assistant">
            </div>
            <button onclick="listVersions()">List Versions</button>
            <button onclick="getProduction()">Get Production Version</button>
            <hr style="margin:15px 0;border:none;border-top:1px solid #eee;">
            <div class="input-group">
                <label>New Prompt Content (creates next version)</label>
                <textarea id="promptContent" placeholder="You are a helpful HR assistant..."></textarea>
            </div>
            <button onclick="createPromptVersion()">Create Version</button>
            <hr style="margin:15px 0;border:none;border-top:1px solid #eee;">
            <div class="input-group">
                <label>Version to Promote</label>
                <input type="number" id="promoteVersion" value="1" min="1">
                <label style="margin-top:10px;">Faithfulness</label>
                <input type="range" id="faithfulness" min="0" max="1" step="0.05" value="0.9" oninput="updateRange(this)">
                <span id="faithValue">0.9</span>
                <label style="margin-top:10px;">Answer Relevance</label>
                <input type="range" id="relevance" min="0" max="1" step="0.05" value="0.85" oninput="updateRange(this)">
                <span id="relevValue">0.85</span>
            </div>
            <button onclick="testPromotion()">Attempt Promotion</button>
            <button onclick="rollbackPrompt()">Rollback</button>
            <div id="registryResult" class="result"></div>
        </div>

        <div id="eval" class="tab-content">
            <h2>Evaluation Framework</h2>
            <div class="input-group">
                <label>Query</label>
                <input type="text" id="evalQuery" value="What is leave policy?">
            </div>
            <div class="input-group">
                <label>Response</label>
                <input type="text" id="evalResponse" value="20 days paid leave">
            </div>
            <button onclick="evaluateResponse()">Evaluate</button>
            <div id="evalResult" class="result"></div>
        </div>

        <div id="ft" class="tab-content">
            <h2>Fine-tuning Configuration</h2>
            <button onclick="getFineTuningConfig()">Load Configuration</button>
            <div id="ftResult" class="result"></div>
        </div>

        <div id="mlflow" class="tab-content">
            <h2>MLflow Tracking</h2>
            <p>All queries, prompt-registry events and vector-store ingests are logged as MLflow runs in a local, file-based tracking store - no external account required.</p>
            <button onclick="getMlflowInfo()">Show Tracking Info</button>
            <div id="mlflowResult" class="result"></div>
        </div>

        <div class="footer">
            <p><strong>ACME Employee HR Assistant - LLMOps Edition</strong></p>
            <p>Development rights @ RoadRunner &amp; Coyote</p>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        function updateRange(element) {
            let id = element.id === 'faithfulness' ? 'faithValue' : 'relevValue';
            document.getElementById(id).textContent = element.value;
        }

        async function sendQuery() {
            const question = document.getElementById('chatInput').value;
            if (!question) { alert('Please enter a question'); return; }
            try {
                const response = await fetch('/api/query', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });
                const data = await response.json();
                let html = `<h3>Response</h3><p>${data.response}</p>
                    <div class="metrics">
                        <div class="metric"><div class="metric-value">${(data.quality_score * 100).toFixed(0)}%</div><div class="metric-label">Quality Score</div></div>
                        <div class="metric"><div class="metric-value">${data.context_count}</div><div class="metric-label">Context Retrieved</div></div>
                        <div class="metric"><div class="metric-value">${data.latency_ms.toFixed(0)}ms</div><div class="metric-label">Latency</div></div>
                    </div>`;
                document.getElementById('chatResult').innerHTML = html;
                document.getElementById('chatResult').classList.add('success');
            } catch (error) {
                document.getElementById('chatResult').innerHTML = `<p>Error: ${error.message}</p>`;
                document.getElementById('chatResult').classList.add('error');
            }
        }

        async function validateInput() {
            const text = document.getElementById('guardrailInput').value;
            try {
                const response = await fetch('/api/validate', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})
                });
                const data = await response.json();
                let html = data.valid ? `<h3>Valid Input</h3><p>${data.message}</p>` :
                    `<h3>Invalid Input</h3><p>${data.message}</p><p>Violations: ${data.violations.join(', ')}</p>`;
                document.getElementById('guardrailResult').innerHTML = html;
                document.getElementById('guardrailResult').classList.add(data.valid ? 'success' : 'error');
            } catch (error) {
                document.getElementById('guardrailResult').innerHTML = `<p>Error: ${error.message}</p>`;
                document.getElementById('guardrailResult').classList.add('error');
            }
        }

        async function getDocuments() {
            const query = document.getElementById('ragInput').value;
            try {
                const response = await fetch(`/api/retrieve?query=${encodeURIComponent(query)}`);
                const data = await response.json();
                let html = '<h3>Retrieved Documents</h3>';
                data.documents.forEach((doc, i) => { html += `<p>${i+1}. ${doc.substring(0,200)}...</p>`; });
                document.getElementById('ragResult').innerHTML = html;
                document.getElementById('ragResult').classList.add('success');
            } catch (error) {
                document.getElementById('ragResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function getDocumentsDetailed() {
            const query = document.getElementById('ragInput').value;
            try {
                const response = await fetch(`/api/retrieve/detailed?query=${encodeURIComponent(query)}`);
                const data = await response.json();
                let html = '<h3>Retrieved Documents (with similarity)</h3>';
                data.results.forEach((r, i) => {
                    html += `<p><b>${i+1}. [similarity ${r.similarity}] ${r.metadata.category}</b><br>${r.content.substring(0,200)}...</p>`;
                });
                document.getElementById('ragResult').innerHTML = html;
                document.getElementById('ragResult').classList.add('success');
            } catch (error) {
                document.getElementById('ragResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function getVectorStoreStats() {
            try {
                const response = await fetch('/api/vectorstore/stats');
                const data = await response.json();
                document.getElementById('vectorstoreResult').innerHTML =
                    '<h3>Vector Store Stats</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                document.getElementById('vectorstoreResult').classList.add('success');
            } catch (error) {
                document.getElementById('vectorstoreResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function reingestVectorStore() {
            try {
                const response = await fetch('/api/vectorstore/reingest', {method: 'POST'});
                const data = await response.json();
                document.getElementById('vectorstoreResult').innerHTML =
                    '<h3>Re-ingest Complete</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                document.getElementById('vectorstoreResult').classList.add('success');
            } catch (error) {
                document.getElementById('vectorstoreResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function listVersions() {
            const name = document.getElementById('promptName').value;
            try {
                const response = await fetch(`/api/prompts/${name}/versions`);
                const data = await response.json();
                let html = '<h3>Versions for ' + name + '</h3><table><tr><th>Version</th><th>Status</th><th>Created</th><th>Metrics</th></tr>';
                data.versions.forEach(v => {
                    html += `<tr><td>${v.version}</td><td>${v.status}</td><td>${v.created_at.substring(0,19)}</td><td>${JSON.stringify(v.metrics)}</td></tr>`;
                });
                html += '</table>';
                document.getElementById('registryResult').innerHTML = html;
                document.getElementById('registryResult').classList.add('success');
            } catch (error) {
                document.getElementById('registryResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function getProduction() {
            const name = document.getElementById('promptName').value;
            try {
                const response = await fetch(`/api/prompts/${name}/production`);
                if (!response.ok) { throw new Error('No production version yet'); }
                const data = await response.json();
                document.getElementById('registryResult').innerHTML = '<h3>Current Production</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                document.getElementById('registryResult').classList.add('success');
            } catch (error) {
                document.getElementById('registryResult').innerHTML = `<p>${error.message}</p>`;
                document.getElementById('registryResult').classList.add('error');
            }
        }

        async function createPromptVersion() {
            const name = document.getElementById('promptName').value;
            const content = document.getElementById('promptContent').value;
            if (!content) { alert('Enter prompt content'); return; }
            try {
                const response = await fetch('/api/prompts', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, content})
                });
                const data = await response.json();
                document.getElementById('registryResult').innerHTML = '<h3>Version Created</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                document.getElementById('registryResult').classList.add('success');
            } catch (error) {
                document.getElementById('registryResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function testPromotion() {
            const name = document.getElementById('promptName').value;
            const version = parseInt(document.getElementById('promoteVersion').value);
            const faithfulness = parseFloat(document.getElementById('faithfulness').value);
            const answer_relevance = parseFloat(document.getElementById('relevance').value);
            try {
                const response = await fetch(`/api/prompts/${name}/promote`, {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({version, faithfulness, answer_relevance})
                });
                const data = await response.json();
                let html = data.promoted ? '<h3>Promotion Successful!</h3>' :
                    '<h3>Promotion Failed</h3><p>' + (data.failures || []).join('<br>') + '</p>';
                document.getElementById('registryResult').innerHTML = html;
                document.getElementById('registryResult').classList.add(data.promoted ? 'success' : 'error');
            } catch (error) {
                document.getElementById('registryResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function rollbackPrompt() {
            const name = document.getElementById('promptName').value;
            try {
                const response = await fetch(`/api/prompts/${name}/rollback`, {method: 'POST'});
                const data = await response.json();
                document.getElementById('registryResult').innerHTML = '<h3>Rolled back</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                document.getElementById('registryResult').classList.add('success');
            } catch (error) {
                document.getElementById('registryResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function evaluateResponse() {
            const query = document.getElementById('evalQuery').value;
            const response_ = document.getElementById('evalResponse').value;
            try {
                const res = await fetch('/api/evaluate', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, response: response_})
                });
                const data = await res.json();
                let html = `<h3>Evaluation Results</h3>
                    <div class="metrics">
                        <div class="metric"><div class="metric-value">${(data.faithfulness * 100).toFixed(0)}%</div><div class="metric-label">Faithfulness</div></div>
                        <div class="metric"><div class="metric-value">${(data.relevance * 100).toFixed(0)}%</div><div class="metric-label">Relevance</div></div>
                        <div class="metric"><div class="metric-value">${(data.overall * 100).toFixed(0)}%</div><div class="metric-label">Overall</div></div>
                    </div>`;
                document.getElementById('evalResult').innerHTML = html;
                document.getElementById('evalResult').classList.add('success');
            } catch (error) {
                document.getElementById('evalResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function getFineTuningConfig() {
            try {
                const response = await fetch('/api/finetuning');
                const data = await response.json();
                let html = '<h3>LoRA Configuration</h3>';
                html += `<p>Rank: ${data.lora.r}</p><p>Alpha: ${data.lora.lora_alpha}</p><p>Dropout: ${data.lora.lora_dropout}</p>`;
                html += '<h3>Training Configuration</h3>';
                html += `<p>Batch Size: ${data.training.batch_size}</p><p>Learning Rate: ${data.training.learning_rate}</p><p>Epochs: ${data.training.num_epochs}</p>`;
                document.getElementById('ftResult').innerHTML = html;
                document.getElementById('ftResult').classList.add('success');
            } catch (error) {
                document.getElementById('ftResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }

        async function getMlflowInfo() {
            try {
                const response = await fetch('/api/mlflow/info');
                const data = await response.json();
                document.getElementById('mlflowResult').innerHTML = '<h3>MLflow Tracking Info</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                document.getElementById('mlflowResult').classList.add('success');
            } catch (error) {
                document.getElementById('mlflowResult').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
