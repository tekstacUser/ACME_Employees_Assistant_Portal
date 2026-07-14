# Persistent Prompt Registry (SQLite-backed)
# ------------------------------------------------------------------
# Upgrades prompt_registry_minimal.MinimalPromptRegistry (in-memory only,
# lost on restart) into a durable registry with:
#   - Full version history per prompt name
#   - Status lifecycle: draft -> test -> staging -> production -> archived
#   - Promotion gates (same semantics as the original minimal registry)
#   - Rollback to the previous production version
#   - Unified diff between any two versions
#   - Tags + free-text notes per version
#
# Storage: a single local SQLite file (prompt_registry.db). No external
# service, account, or network access required. The original
# prompt_registry_minimal.py is left completely untouched - this is an
# additive module.
# ------------------------------------------------------------------

import sqlite3
import json
import os
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum


class PromptStatus(str, Enum):
    DRAFT = "draft"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompt_registry.db"
)


@dataclass
class PromptVersion:
    id: int
    name: str
    version: int
    content: str
    status: str
    created_at: str
    metrics: Dict[str, float]
    tags: List[str]
    notes: str = ""


class PromptRegistryDB:
    """SQLite-backed, persistent prompt registry with full version history."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, gates: Optional[Dict[str, float]] = None):
        self.db_path = db_path
        # Lower-is-better metrics vs higher-is-better metrics
        self.lower_is_better = {"latency_ms"}
        self.gates = gates or {
            "faithfulness": 0.85,
            "answer_relevance": 0.80,
            "latency_ms": 1000,
        }
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL,
                metrics TEXT NOT NULL DEFAULT '{}',
                tags TEXT NOT NULL DEFAULT '[]',
                notes TEXT DEFAULT '',
                UNIQUE(name, version)
            )
            """
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # CRUD / versioning
    # ------------------------------------------------------------------
    def create_prompt(
        self, name: str, content: str, tags: Optional[List[str]] = None, notes: str = ""
    ) -> PromptVersion:
        conn = self._connect()
        row = conn.execute("SELECT MAX(version) as v FROM prompts WHERE name=?", (name,)).fetchone()
        version = (row["v"] or 0) + 1
        created_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO prompts (name, version, content, status, created_at, metrics, tags, notes) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, version, content, PromptStatus.DRAFT.value, created_at, "{}", json.dumps(tags or []), notes),
        )
        conn.commit()
        conn.close()
        return self.get_version(name, version)

    def get_version(self, name: str, version: int) -> Optional[PromptVersion]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM prompts WHERE name=? AND version=?", (name, version)).fetchone()
        conn.close()
        return self._row_to_version(row) if row else None

    def get_all_versions(self, name: str) -> List[PromptVersion]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM prompts WHERE name=? ORDER BY version ASC", (name,)).fetchall()
        conn.close()
        return [self._row_to_version(r) for r in rows]

    def list_prompt_names(self) -> List[str]:
        conn = self._connect()
        rows = conn.execute("SELECT DISTINCT name FROM prompts ORDER BY name").fetchall()
        conn.close()
        return [r["name"] for r in rows]

    def get_current_production(self, name: str) -> Optional[PromptVersion]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM prompts WHERE name=? AND status=? ORDER BY version DESC LIMIT 1",
            (name, PromptStatus.PRODUCTION.value),
        ).fetchone()
        conn.close()
        return self._row_to_version(row) if row else None

    def set_status(self, name: str, version: int, status: PromptStatus):
        conn = self._connect()
        conn.execute("UPDATE prompts SET status=? WHERE name=? AND version=?", (status.value, name, version))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Promotion gates (A/B style quality gates)
    # ------------------------------------------------------------------
    def promote(self, name: str, version: int, metrics: Dict[str, float], force: bool = False) -> Dict:
        # TODO [10 Marks]: Prompt Registry & Lifecycle Management - promote()
        # ------------------------------------------------------------------
        # Implement quality-gated promotion of a prompt version to
        # "production" status.
        #
        # Requirements:
        # - Unless `force` is True, check every gate in `self.gates`
        #   (e.g. {"faithfulness": 0.85, "answer_relevance": 0.80,
        #   "latency_ms": 1000}) against the supplied `metrics` dict:
        #     - Skip a gate if its name is not present in `metrics`.
        #     - For gate names in `self.lower_is_better` (e.g. "latency_ms"),
        #       the check passes if `metrics[gate_name] <= threshold`.
        #     - For all other gates, the check passes if
        #       `metrics[gate_name] >= threshold`.
        #     - Collect a human-readable failure string for every gate that
        #       does not pass, e.g. f"{gate_name}={metrics[gate_name]} (threshold {threshold})".
        # - If there are failures and `force` is False, return
        #   `{"promoted": False, "failures": failures}` WITHOUT touching the DB.
        # - Otherwise (gates passed, or `force=True`):
        #   - Demote any existing "production" version of this prompt name to
        #     "archived" (there must only ever be one production version).
        #   - Set the target `version`'s status to "production" and persist
        #     the given `metrics` (as JSON) on that row.
        #   - Return `{"promoted": True, "failures": []}`.
        #
        # Inputs: `name` (str), `version` (int), `metrics` (Dict[str, float]),
        # `force` (bool, default False).
        # Outputs: `Dict` with keys `promoted` (bool) and `failures` (List[str]).
        # Dependencies: `self._connect()`, `self.gates`, `self.lower_is_better`,
        # `PromptStatus`.
        # Acceptance criteria: a version with metrics below any gate is
        # rejected (no DB changes) unless `force=True`; a passing version
        # becomes the sole "production" row for that prompt name, and any
        # prior production version becomes "archived".
        raise NotImplementedError("Student Implementation Required: prompt promotion with quality gates")

    def rollback(self, name: str) -> Optional[PromptVersion]:
        """Roll production back to the most recent previously-archived production version."""
        # TODO [Part of 10 Marks - Prompt Registry & Lifecycle Management]: rollback()
        # ------------------------------------------------------------------
        # Implement rollback of the current production version to the most
        # recently archived version of the same prompt.
        #
        # Requirements:
        # - Fetch all versions for `name` via `self.get_all_versions(name)`.
        # - Filter to versions whose status is "archived"; if none exist,
        #   return `None` (nothing to roll back to).
        # - Pick the last (highest-version) archived entry as the rollback
        #   target.
        # - Demote the current "production" version (if any) to "archived".
        # - Set the target version's status to "production".
        # - Return the updated `PromptVersion` for the target (re-fetch it
        #   after the status change so `status` reflects "production").
        #
        # Inputs: `name` (str).
        # Outputs: `Optional[PromptVersion]` — the newly-restored production
        # version, or `None` if no archived version was available.
        # Dependencies: `self.get_all_versions`, `self._connect()`,
        # `self.get_version`, `PromptStatus`.
        # Acceptance criteria: after promoting v1 then v2, calling
        # `rollback()` restores v1 to "production" and demotes v2 to
        # "archived".
        raise NotImplementedError("Student Implementation Required: prompt rollback")

    def diff(self, name: str, v1: int, v2: int) -> List[str]:
        a = self.get_version(name, v1)
        b = self.get_version(name, v2)
        if not a or not b:
            return []
        return list(
            difflib.unified_diff(
                a.content.splitlines(),
                b.content.splitlines(),
                fromfile=f"{name} v{v1}",
                tofile=f"{name} v{v2}",
                lineterm="",
            )
        )

    def _row_to_version(self, row) -> PromptVersion:
        return PromptVersion(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            content=row["content"],
            status=row["status"],
            created_at=row["created_at"],
            metrics=json.loads(row["metrics"]),
            tags=json.loads(row["tags"]),
            notes=row["notes"] or "",
        )


if __name__ == "__main__":
    reg = PromptRegistryDB(db_path=":memory:".replace(":memory:", DEFAULT_DB_PATH))
    v1 = reg.create_prompt("hr_assistant", "You are a helpful HR assistant.", tags=["baseline"])
    print("Created", v1)
    v2 = reg.create_prompt("hr_assistant", "You are an expert, empathetic HR assistant.", tags=["improved"])
    print("Created", v2)
    result = reg.promote("hr_assistant", v2.version, {"faithfulness": 0.9, "answer_relevance": 0.85, "latency_ms": 700})
    print("Promotion:", result)
    print("Production:", reg.get_current_production("hr_assistant"))
    print("Diff v1 vs v2:")
    print("\n".join(reg.diff("hr_assistant", v1.version, v2.version)))
