# ChromaDB Vector Store RAG Pipeline
# ------------------------------------------------------------------
# Extends the ORIGINAL rag_pipeline.EnhancedRAGPipeline with a real
# vector store (embedded/persistent ChromaDB) + HuggingFace sentence-transformer
# embeddings, replacing the in-memory keyword scoring `retrieve()` method.
#
# Design goal: DO NOT modify rag_pipeline.py. Everything else
# (HuggingFace text-generation, smart fact extraction, response formatting,
# retrieval-quality scoring) is inherited unchanged from EnhancedRAGPipeline,
# so behaviour for the rest of the app stays identical.
#
# No external APIs / API keys are used anywhere - only local HuggingFace
# models (embedding model + generation model), both downloaded once from the
# HuggingFace Hub and then cached locally.
# ------------------------------------------------------------------

import os
import sys
import time
from typing import List, Dict, Tuple, Optional

# Make sure the parent directory (where rag_pipeline.py lives) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions

from rag_pipeline_minimal import EnhancedRAGPipeline  # noqa: E402  (import after sys.path tweak)

DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_store"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATION_MODEL = "EleutherAI/gpt-neo-125m"


class ChromaRAGPipeline(EnhancedRAGPipeline):
    """
    Drop-in, backwards-compatible replacement for EnhancedRAGPipeline.

    - retrieve() now performs real cosine-similarity vector search against an
      embedded, on-disk ChromaDB collection instead of keyword scoring.
    - generate_response(), _extract_key_facts(), evaluate_retrieval_quality()
      and every other method are inherited unmodified from the parent class,
      so downstream behaviour (guardrails, evaluation, fine-tuning tabs) is
      unaffected.
    - Embeddings are computed locally via a HuggingFace sentence-transformers
      model (no external API calls, no API keys).
    """

    def __init__(
        self,
        use_huggingface: bool = True,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = "hr_policies",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        force_reingest: bool = False,
    ):
        # --- Replica of the bookkeeping fields the parent class normally sets
        # in __init__. We intentionally do NOT call super().__init__() because
        # the parent constructor builds an in-memory keyword index we no
        # longer need (it's fully replaced by ChromaDB below). ---
        self.chunk_size = 800
        self.overlap_percent = 20
        self.atomic_chunks = True

        raw_documents = self._load_comprehensive_documents()
        # Kept so get_sample_documents() / any legacy caller still works
        self.documents = self._create_atomic_chunks(raw_documents)
        self.document_metadata = self._create_metadata()

        # --- Vector store (ChromaDB embedded, persistent on local disk) ---
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # TODO : Vector Database Implementation - Client & Collection Setup
        # ------------------------------------------------------------------
        # Implement ChromaDB embedded/persistent client initialization and
        # collection creation, then trigger ingestion when needed.
        #
        # Requirements:
        # - Instantiate `chromadb.PersistentClient(path=self.persist_dir)`
        #   and assign it to `self.client`.
        # - Build a `embedding_functions.SentenceTransformerEmbeddingFunction`
        #   using `model_name=embedding_model` and assign it to `self.embed_fn`.
        # - Call `self.client.get_or_create_collection(...)` passing
        #   `name=collection_name`, `embedding_function=self.embed_fn`, and
        #   `metadata={"hnsw:space": "cosine"}`; assign the result to
        #   `self.collection`.
        # - If `force_reingest` is True OR `self.collection.count() == 0`,
        #   call `self._ingest_into_chroma(raw_documents)`. Otherwise, print
        #   a message that the existing collection (with its current vector
        #   count) is being reused and skip re-ingesting.
        #
        # Inputs available in scope: `raw_documents` (Dict[str, str], loaded
        # above), `collection_name`, `embedding_model`, `force_reingest`.
        # Outputs/side effects required: `self.client`, `self.embed_fn`, and
        # `self.collection` must all be set before `__init__` returns.
        # Dependencies: chromadb.PersistentClient,
        # chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction.
        # Acceptance criteria: after construction with an empty persist_dir,
        # `pipeline.get_vector_store_stats()["vector_count"]` equals the
        # number of documents in `hr_policies.json` (15); restarting with the
        # same persist_dir does NOT re-embed (vector count stays the same
        # and no re-ingest log line is printed).
        raise NotImplementedError("Student Implementation Required: ChromaDB client/collection initialization")

        # --- HuggingFace generation model (same model/approach as the
        # original pipeline, loaded independently here since we skip
        # super().__init__()) ---
        try:
            from transformers import pipeline as hf_pipeline
            hf_available = True
        except ImportError:
            hf_available = False

        self.use_huggingface = use_huggingface and hf_available
        self.llm_pipeline = None
        if self.use_huggingface:
            try:
                print(f"[HuggingFace] Loading generation model: {generation_model}")
                from transformers import pipeline as hf_pipeline
                self.llm_pipeline = hf_pipeline(
                    "text-generation", model=generation_model, device=-1
                )
                print("[HuggingFace] Generation model loaded successfully.")
            except Exception as e:
                print(f"[HuggingFace] Warning: could not load generation model: {e}")
                self.use_huggingface = False

    # ------------------------------------------------------------------
    # Vector store operations
    # ------------------------------------------------------------------
    def _ingest_into_chroma(self, raw_documents: Dict[str, str]):
        # TODO [Part - Vector Database Implementation]: Document Ingestion
        # ------------------------------------------------------------------
        # Embed and store every HR policy document as one atomic vector.
        #
        # Requirements:
        # - Build three parallel lists from `raw_documents` (a dict of
        #   `{doc_id: text}`): `ids` (the doc_id), `docs` (the stripped
        #   text), and `metadatas` (a dict per document containing
        #   `source_id` and `category`, using `self._infer_category(doc_id)`).
        # - Call `self.collection.add(ids=ids, documents=docs,
        #   metadatas=metadatas)` to embed (via the collection's configured
        #   embedding function) and persist every document in one batch call.
        # - Log how many documents were ingested and how long it took.
        #
        # Inputs: `raw_documents: Dict[str, str]` (one entry per HR policy
        # document, e.g. 15 entries).
        # Outputs/side effects: `self.collection` contains one vector per
        # input document after this call returns.
        # Dependencies: `self.collection.add`, `self._infer_category`.
        # Acceptance criteria: `self.collection.count()` equals
        # `len(raw_documents)` immediately after this method runs.
        raise NotImplementedError("Student Implementation Required: ChromaDB document ingestion")

    @staticmethod
    def _infer_category(source_id: str) -> str:
        source_id = source_id.lower()
        for key in [
            "leave", "benefits", "compensation", "work_hours", "remote",
            "performance", "conduct", "safety", "termination",
        ]:
            if key in source_id:
                return key
        return "general"

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], float]:
        """Vector similarity retrieval via ChromaDB (replaces keyword scoring)."""
        # TODO : Embedding & Retrieval Logic - retrieve()
        # ------------------------------------------------------------------
        # Implement real vector similarity search using the ChromaDB
        # collection (overrides the parent class's keyword-scoring retrieve()).
        #
        # Requirements:
        # - Record a start timestamp so retrieval latency can be measured.
        # - Compute `n = min(k, max(self.collection.count(), 1))` so we never
        #   request more results than vectors exist.
        # - Call `self.collection.query(query_texts=[query], n_results=n)`.
        # - Compute `retrieval_latency` in milliseconds (elapsed time * 1000).
        # - Extract the matched document texts from
        #   `results["documents"][0]` (empty list if no results / no query).
        #
        # Inputs: `query` (str), `k` (int, default 5).
        # Outputs: `Tuple[List[str], float]` -> (matched document texts,
        # retrieval latency in ms).
        # Dependencies: `self.collection.query`, `time.time()`.
        # Acceptance criteria: querying "maternity leave" returns the
        # maternity leave policy document as the top (or near-top) result,
        # and latency is a positive float.
        raise NotImplementedError("Student Implementation Required: ChromaDB vector retrieval")

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve chunks along with similarity scores + metadata (for UI/debugging)."""
        # TODO [Part - Embedding & Retrieval Logic]: retrieve_with_scores()
        # ------------------------------------------------------------------
        # Same vector search as retrieve(), but return similarity score and
        # metadata for each hit (used by the "Vector Store" tab / the
        # `/api/retrieve/detailed` endpoint).
        #
        # Requirements:
        # - Compute `n` the same way as in `retrieve()`.
        # - Call `self.collection.query(query_texts=[query], n_results=n)`.
        # - For each returned document, build a dict with keys: `id`,
        #   `content`, `metadata`, `distance`, and `similarity` (computed as
        #   `round(1 - distance, 4)` — ChromaDB cosine distance -> similarity).
        # - Return the list of these dicts, one per matched document, in the
        #   order returned by ChromaDB (best match first).
        #
        # Inputs: `query` (str), `k` (int, default 5).
        # Outputs: `List[Dict]`, each with keys `id`, `content`, `metadata`,
        # `distance`, `similarity`.
        # Dependencies: `self.collection.query`.
        # Acceptance criteria: results are sorted best-match-first and
        # `similarity` values are in the [0, 1] range for typical queries.
        raise NotImplementedError("Student Implementation Required: ChromaDB retrieval with similarity scores")

    def get_vector_store_stats(self) -> Dict:
        return {
            "backend": "ChromaDB (embedded, persistent, local disk)",
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "vector_count": self.collection.count(),
        }

    def reingest(self) -> Dict:
        """Drop and rebuild the collection from hr_policies.json."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        raw_documents = self._load_comprehensive_documents()
        self._ingest_into_chroma(raw_documents)
        return self.get_vector_store_stats()


# Compatibility alias so this module can be swapped in for rag_pipeline_minimal
# without touching call sites: `from vector_store.chroma_rag_pipeline import MinimalRAGPipeline`
MinimalRAGPipeline = ChromaRAGPipeline


if __name__ == "__main__":
    print("=" * 80)
    print("CHROMADB VECTOR STORE RAG PIPELINE TEST")
    print("=" * 80)

    pipeline = ChromaRAGPipeline()
    print("\nStats:", pipeline.get_vector_store_stats())

    test_queries = [
        "What is the maternity leave policy?",
        "Can I work from home?",
        "What are the health insurance benefits?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        response, metadata = pipeline.query(q, k=3)
        print(f"Response:\n{response[:300]}")
        print(f"Latency: total={metadata.total_latency_ms:.1f}ms "
              f"retrieval={metadata.retrieval_latency_ms:.1f}ms "
              f"generation={metadata.generation_latency_ms:.1f}ms")

        scored = pipeline.retrieve_with_scores(q, k=3)
        for r in scored:
            print(f"  - [{r['similarity']}] {r['id']} ({r['metadata']['category']})")
