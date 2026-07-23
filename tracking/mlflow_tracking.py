# MLflow LLM Tracking Extension
# ------------------------------------------------------------------
# Adds experiment tracking / observability on top of the existing app
# WITHOUT touching any existing file. Uses MLflow's local, file-based
# tracking store (./mlruns) - fully open-source, no account, no API key,
# no network calls. This is the requested "Weights & Biases / MLflow LLM
# extension" piece.
#
# Every call is wrapped in try/except and gated behind `self.enabled` so
# that if mlflow is not installed, or logging fails for any reason, the
# rest of the application keeps working exactly as before.
# ------------------------------------------------------------------

import os
from typing import Dict, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

TRACKING_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
TRACKING_DB = os.path.join(TRACKING_ROOT, "mlflow.db")


class MLflowTracker:
    """Thin, defensive wrapper around MLflow so the rest of the app never
    breaks even if MLflow is unavailable or a logging call fails.

    Uses a local SQLite-backed tracking store (mlruns/mlflow.db). Modern
    MLflow (2.x+) has deprecated the plain filesystem store in favor of a
    database backend, so SQLite is used here - it is still 100% local,
    file-based, and requires no external service or account.
    """

    def __init__(self, experiment_name: str = "acme-hr-rag-assistant", enabled: bool = True):
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.tracking_dir = TRACKING_ROOT
        self.tracking_uri = f"sqlite:///{TRACKING_DB}"
        self.experiment_name = experiment_name

        if self.enabled:
            try:
                os.makedirs(self.tracking_dir, exist_ok=True)
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(experiment_name)
                print(
                    f"[MLflow] Tracking enabled -> {self.tracking_uri}\n"
                    f"[MLflow] View the UI with:\n"
                    f"         mlflow ui --backend-store-uri {self.tracking_uri}"
                )
            except Exception as e:
                print(f"[MLflow] Failed to initialize, disabling tracking: {e}")
                self.enabled = False
        else:
            reason = "mlflow not installed" if not MLFLOW_AVAILABLE else "disabled"
            print(f"[MLflow] Tracking disabled ({reason})")

    # ------------------------------------------------------------------
    def log_query(self, query: str, response: str, metadata, eval_result, extra_params: Optional[Dict] = None):
        """Log one RAG query/response cycle as an MLflow run."""
        # TODO : Observability, Error Handling & Logging - log_query()
        # ------------------------------------------------------------------
        # Implement fail-safe MLflow logging for a single RAG query/response
        # cycle. This is the most important tracking call in the app (every
        # `/api/query` request goes through it), so it must NEVER raise and
        # break the request — logging failures must be swallowed and printed.
        #
        # Requirements:
        # - If `self.enabled` is False, return immediately (no-op).
        # - Wrap everything else in try/except; on any exception, print a
        #   non-fatal warning (e.g. "[MLflow] log_query failed (non-fatal): {e}")
        #   and DO NOT re-raise.
        # - Inside a `with mlflow.start_run(run_name="rag_query"):` block:
        #   - Tag the run with `component="rag_pipeline"`.
        #   - Log the query text as a param (truncate to 250 chars to avoid
        #     MLflow param length limits).
        #   - Log every key/value in `extra_params` (if provided) as params.
        #   - Log these metrics (cast to float): `metadata.total_latency_ms`,
        #     `metadata.retrieval_latency_ms`, `metadata.generation_latency_ms`,
        #     `metadata.context_count`, `eval_result.faithfulness`,
        #     `eval_result.answer_relevance`, `eval_result.overall_score`,
        #     and `passed_quality_gate` as `int(eval_result.passed)`.
        #   - Log the full `response` text as an artifact named "response.txt"
        #     via `mlflow.log_text`.
        #
        # Inputs: `query` (str), `response` (str), `metadata` (RAGMetadata-like
        # object), `eval_result` (EvaluationResult-like object),
        # `extra_params` (Optional[Dict]).
        # Outputs: None (side effect: one MLflow run logged, or a printed
        # warning on failure — the caller must never see an exception).
        # Dependencies: `mlflow.start_run`, `mlflow.set_tag`, `mlflow.log_param`,
        # `mlflow.log_metric`, `mlflow.log_text`.
        # Acceptance criteria: calling `log_query(...)` with a well-formed
        # metadata/eval_result never raises; disabling MLflow (uninstalling
        # the package or `enabled=False`) still allows `/api/query` to work.
        pass  # Student Implementation Required: MLflow query logging

    def log_prompt_version(self, prompt_version, event: str = "created"):
        """Log a prompt registry create/promote/rollback event."""
        if not self.enabled:
            return
        try:
            with mlflow.start_run(run_name=f"prompt_{event}_{prompt_version.name}_v{prompt_version.version}"):
                mlflow.set_tag("component", "prompt_registry")
                mlflow.set_tag("event", event)
                mlflow.log_param("prompt_name", prompt_version.name)
                mlflow.log_param("version", prompt_version.version)
                mlflow.log_param("status", prompt_version.status)
                mlflow.log_text(prompt_version.content, f"prompt_v{prompt_version.version}.txt")
                for k, v in (prompt_version.metrics or {}).items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            print(f"[MLflow] log_prompt_version failed (non-fatal): {e}")

    def log_promotion_attempt(self, name: str, version: int, metrics: Dict, promoted: bool, failures=None):
        if not self.enabled:
            return
        try:
            with mlflow.start_run(run_name=f"promotion_{name}_v{version}"):
                mlflow.set_tag("component", "prompt_registry")
                mlflow.set_tag("event", "promotion_attempt")
                mlflow.log_param("prompt_name", name)
                mlflow.log_param("version", version)
                mlflow.log_param("promoted", promoted)
                if failures:
                    mlflow.log_param("failures", "; ".join(failures))
                for k, v in (metrics or {}).items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            print(f"[MLflow] log_promotion_attempt failed (non-fatal): {e}")

    def log_vector_store_event(self, stats: Dict, event: str = "ingest"):
        if not self.enabled:
            return
        try:
            with mlflow.start_run(run_name=f"vector_store_{event}"):
                mlflow.set_tag("component", "vector_store")
                mlflow.set_tag("event", event)
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                    else:
                        mlflow.log_param(k, str(v))
        except Exception as e:
            print(f"[MLflow] log_vector_store_event failed (non-fatal): {e}")

    def get_info(self) -> Dict:
        return {
            "enabled": self.enabled,
            "mlflow_installed": MLFLOW_AVAILABLE,
            "tracking_uri": self.tracking_uri if self.enabled else None,
            "experiment_name": self.experiment_name if self.enabled else None,
            "launch_ui_command": f"mlflow ui --backend-store-uri {self.tracking_uri}",
        }
