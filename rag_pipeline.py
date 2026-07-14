# Enhanced RAG Pipeline with Comprehensive HR Data
# Now with Hugging Face model for response generation!

from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
from enum import Enum

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    content: str
    source_id: str
    chunk_index: int
    total_chunks: int
    overlap_start: int  # Character position where overlap starts

@dataclass
class RAGMetadata:
    context_count: int
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    matched_documents: List[str]

class DocumentCategory(Enum):
    LEAVE = "leave_policy"
    BENEFITS = "benefits"
    COMPENSATION = "compensation"
    WORK_HOURS = "work_hours"
    REMOTE = "remote_work"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    CODE_OF_CONDUCT = "code_of_conduct"
    TRAINING = "training"
    TERMINATION = "termination"

class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with comprehensive HR documentation + Hugging Face response generation"""

    def __init__(self, use_huggingface=True, chunk_size=800, overlap_percent=20, atomic_chunks=True):
        # Store chunking parameters first (needed by _create_metadata)
        self.chunk_size = chunk_size
        self.overlap_percent = overlap_percent
        self.atomic_chunks = atomic_chunks

        # Load documents
        raw_documents = self._load_comprehensive_documents()

        # If atomic_chunks=True, treat each JSON document as a single chunk (no sub-chunking)
        # This improves faithfulness by preserving document boundaries
        if atomic_chunks:
            self.documents = self._create_atomic_chunks(raw_documents)
        else:
            self.documents = self._chunk_documents(raw_documents, chunk_size, overlap_percent)

        self.document_metadata = self._create_metadata()

        # Initialize Hugging Face model for response generation
        self.use_huggingface = use_huggingface and HF_AVAILABLE
        self.llm_pipeline = None

        if self.use_huggingface:
            try:
                print("Loading Hugging Face model for response generation...")
                # Using EleutherAI/gpt-neo-125m: lightweight, fast, good quality
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model="EleutherAI/gpt-neo-125m",
                    device=-1  # -1 for CPU, 0+ for GPU
                )
                print("Hugging Face model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load Hugging Face model: {e}")
                print("Falling back to template-based responses")
                self.use_huggingface = False

    def ingest_documents(self, documents: List[str]):
        """Compatibility method for app_fastapi.py and app_streamlit.py
        Documents are already loaded in __init__, so this method does nothing."""
        pass

    def _create_atomic_chunks(self, documents: Dict[str, str]) -> List[DocumentChunk]:
        """
        Create atomic chunks where each JSON document becomes a single chunk.
        This preserves document boundaries and improves answer faithfulness by avoiding
        fragmented context mixing unrelated policy sections.

        Returns:
            List of DocumentChunk objects (one per JSON document)
        """
        chunks = []

        for doc_id, doc_text in documents.items():
            chunk = DocumentChunk(
                content=doc_text.strip(),
                source_id=doc_id,
                chunk_index=0,
                total_chunks=1,
                overlap_start=0
            )
            chunks.append(chunk)

        print(f"[Chunking] Created {len(chunks)} atomic chunks from {len(documents)} documents")
        print(f"[Chunking] Mode: ATOMIC (no sub-chunking, document boundaries preserved)")

        return chunks

    def _chunk_documents(self, documents: Dict[str, str], chunk_size: int = 800, overlap_percent: int = 20) -> List[DocumentChunk]:
        """
        Split documents into chunks with overlap for better retrieval.

        This preserves section headers by detecting numbered sections (1., 2., etc.)
        and ensuring section headers stay with their content.

        Args:
            documents: Dictionary of document_id -> document_text
            chunk_size: Target size of each chunk in characters
            overlap_percent: Percentage of overlap between chunks (0-100)

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        overlap_chars = max(1, int(chunk_size * overlap_percent / 100))

        for doc_id, doc_text in documents.items():
            # Clean up the text
            doc_text = doc_text.strip()

            if len(doc_text) <= chunk_size:
                # Document fits in one chunk
                chunk = DocumentChunk(
                    content=doc_text,
                    source_id=doc_id,
                    chunk_index=0,
                    total_chunks=1,
                    overlap_start=0
                )
                chunks.append(chunk)
            else:
                # Split document into chunks, trying to keep sections together
                chunk_index = 0
                start_pos = 0

                while start_pos < len(doc_text):
                    # Calculate end position
                    end_pos = min(start_pos + chunk_size, len(doc_text))

                    # Try to break at a good boundary
                    if end_pos < len(doc_text):
                        # Preference order: double newline > section boundary > single newline > period
                        break_pos = -1

                        # Try double newline first (section breaks)
                        test_pos = doc_text.rfind('\n\n', start_pos + 200, end_pos)
                        if test_pos > start_pos + 200:
                            break_pos = test_pos + 2

                        # Try section markers (numbered sections like "\n2. ")
                        if break_pos == -1:
                            import re
                            section_match = None
                            for match in re.finditer(r'\n\d+\. ', doc_text[start_pos + 200:end_pos]):
                                section_match = match
                            if section_match:
                                break_pos = start_pos + 200 + section_match.start() + 1

                        # Try single newline
                        if break_pos == -1:
                            test_pos = doc_text.rfind('\n', start_pos + 200, end_pos)
                            if test_pos > start_pos + 200:
                                break_pos = test_pos + 1

                        # Try period
                        if break_pos == -1:
                            test_pos = doc_text.rfind('. ', start_pos + 200, end_pos)
                            if test_pos > start_pos + 200:
                                break_pos = test_pos + 2

                        if break_pos > start_pos + 200:
                            end_pos = break_pos

                    # Extract chunk
                    chunk_text = doc_text[start_pos:end_pos].strip()

                    if chunk_text and len(chunk_text) > 50:  # Only add meaningful chunks
                        chunk = DocumentChunk(
                            content=chunk_text,
                            source_id=doc_id,
                            chunk_index=chunk_index,
                            total_chunks=-1,  # Will be updated after all chunks created
                            overlap_start=max(0, start_pos)
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                    # Move to next chunk position with overlap
                    if end_pos >= len(doc_text):
                        break  # Reached end of document

                    start_pos = end_pos - overlap_chars

        print(f"[Chunking] Created {len(chunks)} chunks from {len(documents)} documents")
        print(f"[Chunking] Chunk size: {chunk_size}, Overlap: {overlap_percent}%")

        return chunks

    def _load_comprehensive_documents(self) -> Dict[str, str]:
        """Load comprehensive HR documentation from JSON file"""
        import json
        import os

        json_path = os.path.join(os.path.dirname(__file__), "hr_policies.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"ERROR: hr_policies.json not found at {json_path}\n"
                f"Please ensure the hr_policies.json file is in the same directory as rag_pipeline_minimal.py"
            )

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                print(f"[Load] Loading documents from {json_path}")
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"ERROR: Invalid JSON in {json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"ERROR: Could not load {json_path}: {e}")

    def _create_metadata(self) -> Dict[str, Dict]:
        """Create metadata for chunks"""
        metadata = {}

        # Group chunks by source document
        sources = {}
        for chunk in self.documents:
            if chunk.source_id not in sources:
                sources[chunk.source_id] = []
            sources[chunk.source_id].append(chunk)

        for source_id, chunks in sources.items():
            category = "general"
            if "leave" in source_id:
                category = "leave"
            elif "benefits" in source_id:
                category = "benefits"
            elif "compensation" in source_id:
                category = "compensation"
            elif "work_hours" in source_id:
                category = "work_hours"
            elif "remote" in source_id:
                category = "remote"
            elif "performance" in source_id:
                category = "performance"
            elif "conduct" in source_id:
                category = "conduct"
            elif "safety" in source_id:
                category = "safety"
            elif "termination" in source_id:
                category = "termination"

            # Get title from first chunk
            first_chunk_lines = chunks[0].content.strip().split('\n')
            title = next((line.strip() for line in first_chunk_lines if line.strip() and not line.startswith('-')), "Untitled")

            total_length = sum(len(chunk.content.split()) for chunk in chunks)

            metadata[source_id] = {
                "category": category,
                "title": title,
                "length": total_length,
                "chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "overlap_percent": self.overlap_percent
            }

        return metadata

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], float]:
        """Retrieve relevant document chunks using improved semantic matching with better tie-breaking"""
        start_time = time.time()

        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 2)  # Filter small words

        scored_chunks = []

        # Score each chunk
        for chunk in self.documents:
            content_lower = chunk.content.lower()
            chunk_header = content_lower[:500]  # First 500 chars usually contains headers

            # 1. EXACT PHRASE MATCHING - highest priority
            phrase_score = 0.0
            phrase_matches = [
                ("health insurance", "health insurance"),
                ("work from home", "work from home"),
                ("work from home", "wfh"),
                ("hybrid", "hybrid"),  # Match HYBRID & INTERNATIONAL WFH
                ("international", "international"),  # Match HYBRID & INTERNATIONAL WFH
                ("maternity leave", "maternity leave"),
                ("maternity", "maternity"),
                ("sick leave", "sick leave"),
                ("sick leave", "sick"),
                ("annual leave", "annual leave"),
                ("annual leave", "annual"),
                ("paternity leave", "paternity"),
                ("bereavement leave", "bereavement"),
                ("performance appraisal", "performance"),
                ("work from home", "wfh eligible"),
                ("resignation", "resignation"),
                ("dress code", "dress code"),
            ]

            for phrase_query, phrase_content in phrase_matches:
                if phrase_query in query_lower and phrase_content in content_lower:
                    phrase_score = 2.0
                    # Additional boost for specific detailed content
                    if ("health" in query_lower or "insurance" in query_lower) and any(term in content_lower for term in ["dental", "vision", "outpatient", "inpatient"]):
                        phrase_score = 3.0
                    # Boost for HYBRID/INTERNATIONAL section match in header
                    elif ("hybrid" in query_lower or "international" in query_lower) and (phrase_content in chunk_header):
                        phrase_score = 3.5  # Extra boost for header match
                    break

            # 2. SECTION HEADER MATCHING - look in first 500 chars (headers area)
            section_score = 0.0

            # Specific section matching with special attention to subheadings
            if "sick" in query_lower and "sick" in chunk_header:
                section_score = 1.5
            elif "maternity" in query_lower and "maternity" in chunk_header:
                section_score = 1.5
            elif "paternity" in query_lower and "paternity" in chunk_header:
                section_score = 1.5
            elif "bereavement" in query_lower and "bereavement" in chunk_header:
                section_score = 1.5
            elif "annual" in query_lower and "annual leave" in chunk_header:
                section_score = 1.5
            elif "health" in query_lower and ("health insurance" in chunk_header or "health" in chunk_header):
                section_score = 1.5
            elif "insurance" in query_lower and "insurance" in chunk_header:
                section_score = 1.5
            elif "hybrid" in query_lower and "hybrid" in chunk_header:
                section_score = 2.0  # High boost for HYBRID section
            elif "international" in query_lower and "international" in chunk_header:
                section_score = 2.0  # High boost for INTERNATIONAL section
            elif "work from home" in query_lower and ("wfh" in chunk_header or "work from home" in chunk_header):
                section_score = 1.5
            elif "performance" in query_lower and "performance" in chunk_header:
                section_score = 1.5

            # 3. KEYWORD MATCHING - general word matching
            keyword_score = 0.0
            for word in query_words:
                if word in content_lower:
                    keyword_score += 1
            if query_words:
                keyword_score = (keyword_score / len(query_words)) * 0.5  # Lower weight

            # 4. CATEGORY BONUS - chunk source document match
            category_bonus = 0.0
            if "leave" in query_lower and "leave" in chunk.source_id:
                category_bonus = 0.5
            elif ("health" in query_lower or "insurance" in query_lower) and "benefits" in chunk.source_id:
                category_bonus = 0.5
            elif ("salary" in query_lower or "compensation" in query_lower) and "compensation" in chunk.source_id:
                category_bonus = 0.5
            elif ("work from home" in query_lower or "remote" in query_lower or "wfh" in query_lower or "hybrid" in query_lower or "international" in query_lower) and "remote" in chunk.source_id:
                category_bonus = 0.5
            elif "performance" in query_lower and "performance" in chunk.source_id:
                category_bonus = 0.5
            elif ("work hour" in query_lower or "attendance" in query_lower) and "work_hours" in chunk.source_id:
                category_bonus = 0.5
            elif ("resign" in query_lower or "termination" in query_lower) and "termination" in chunk.source_id:
                category_bonus = 0.5
            elif "dress code" in query_lower and "conduct" in chunk.source_id:
                category_bonus = 0.5

            # Combine scores
            total_score = (
                phrase_score * 3.0 +        # Exact phrase: 3x
                section_score * 2.5 +       # Section header: 2.5x
                category_bonus * 1.5 +      # Category match: 1.5x
                keyword_score * 1.0         # Keywords: 1x
            )

            if total_score > 0:
                scored_chunks.append((chunk.source_id, chunk.chunk_index, total_score, chunk.content))

        # Sort by score with improved tie-breaking
        def chunk_priority_key(item):
            source_id, chunk_idx, score, content = item

            # Check if chunk header contains query keywords (exact section match)
            content_lower = content.lower()
            header = content_lower[:500]

            # Bonus points for exact section match in header
            exact_section_match = 0
            if "hybrid" in query_lower and "hybrid" in header:
                exact_section_match += 5
            if "international" in query_lower and "international" in header:
                exact_section_match += 5
            if "maternity" in query_lower and "maternity" in header:
                exact_section_match += 5
            if "sick" in query_lower and "sick" in header:
                exact_section_match += 5

            content_specificity = content.count('-') + content.count('•') + content.count(':')

            # Return tuple: score (primary), exact match (secondary), specificity (tertiary)
            return (score + exact_section_match, content_specificity)

        scored_chunks.sort(key=chunk_priority_key, reverse=True)
        retrieved = [chunk[3] for chunk in scored_chunks[:k]]

        retrieval_latency = (time.time() - start_time) * 1000
        return retrieved, retrieval_latency

    def generate_response(self, query: str, context: List[str]) -> Tuple[str, Dict]:
        """Generate response: Smart extraction of policy facts + HuggingFace formatting"""
        start_time = time.time()

        # Step 1: Extract key policy facts (this is accurate and reliable)
        extracted_lines = self._extract_key_facts(query, context)

        # Step 2: Format with HuggingFace if enabled (add natural language wrapper)
        if self.use_huggingface and self.llm_pipeline and extracted_lines:
            try:
                # Format extracted facts for better readability
                facts_text = "\n".join(extracted_lines[:3])  # Top 3 facts

                # Have HF create a professional intro/formatting
                format_prompt = f"""Create a professional HR response using these facts:

Facts:
{facts_text}

Professional Response (2-3 sentences):"""

                output = self.llm_pipeline(
                    format_prompt,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.3,
                    num_return_sequences=1
                )

                if output and len(output) > 0 and 'generated_text' in output[0]:
                    full_text = output[0]['generated_text']
                    formatted = full_text[len(format_prompt):].strip()

                    # Extract first meaningful part
                    for marker in ["\n\n", "Facts:", "Question:"]:
                        if marker in formatted:
                            formatted = formatted.split(marker)[0].strip()

                    # Combine formatted intro with facts
                    if formatted and len(formatted) > 15:
                        response = f"{formatted}\n\n"
                        response += "\n".join(f"• {line}" for line in extracted_lines[:3])
                        generation_latency = (time.time() - start_time) * 1000
                        return response, {
                            "latency_ms": generation_latency,
                            "context_used": len(context),
                            "model_used": "HuggingFace + Extracted Facts"
                        }

            except Exception as e:
                pass  # Fall through to standard format

        # Standard format: Just the extracted facts (quick and accurate)
        if extracted_lines:
            response = "As per company HR policy:\n"
            response += "\n".join(f"- {line}" for line in extracted_lines[:4])
            generation_latency = (time.time() - start_time) * 1000
            return response, {
                "latency_ms": generation_latency,
                "context_used": len(context),
                "model_used": "Smart Fact Extraction"
            }

        # Final fallback
        response = "Based on company HR policies, please contact HR for specific policy details."
        generation_latency = (time.time() - start_time) * 1000
        return response, {
            "latency_ms": generation_latency,
            "context_used": len(context),
            "model_used": "Fallback"
        }

    def _extract_key_facts(self, query: str, context: List[str]) -> List[str]:
        """Extract key factual lines from policy documents, prioritizing query-relevant lines"""
        query_lower = query.lower()
        priority_facts = []
        secondary_facts = []

        # Identify query domain with more specific patterns
        if "health" in query_lower or "insurance" in query_lower:
            relevant_terms = ["health insurance", "premium", "coverage", "dental", "vision", "medical", "hospital", "outpatient", "inpatient", "waiting", "mental health", "wellness"]
            skip_terms = ["enrollment", "effective date", "beneficiary", "life insurance", "salary", "termination"]
        elif "sick leave" in query_lower or ("sick" in query_lower and "leave" in query_lower):
            relevant_terms = ["sick leave", "10 days", "medical certificate", "illness", "family"]
            skip_terms = ["maternity", "paternity", "bereavement", "annual", "personal", "termination"]
        elif "maternity" in query_lower:
            relevant_terms = ["maternity", "16 weeks", "weeks paid", "unpaid", "birthing"]
            skip_terms = ["paternity", "partner", "bereavement", "annual", "termination"]
        elif "paternity" in query_lower:
            relevant_terms = ["paternity", "2 weeks", "partner leave", "weeks paid"]
            skip_terms = ["maternity", "birthing", "bereavement"]
        elif "annual leave" in query_lower or ("leave" in query_lower and ("annual" in query_lower or "how much" in query_lower or "how many" in query_lower)):
            relevant_terms = ["annual leave", "20 days", "calendar year", "full-time", "part-time", "carried forward"]
            skip_terms = ["sick", "maternity", "paternity", "bereavement", "termination"]
        elif "hybrid" in query_lower or "international" in query_lower:
            relevant_terms = ["hybrid", "international", "arrangement", "days", "approval", "timezone", "equipment"]
            skip_terms = ["office", "core hours", "biometric", "attendance"]
        elif "leave" in query_lower:
            relevant_terms = ["leave", "days", "weeks", "policy", "annual", "sick", "paid"]
            skip_terms = ["termination", "salary"]
        elif "salary" in query_lower or "compensation" in query_lower:
            relevant_terms = ["salary", "basic", "allowance", "bonus", "increment", "compensation"]
            skip_terms = ["leave", "insurance", "termination"]
        elif "work from home" in query_lower or "wfh" in query_lower or "remote" in query_lower:
            relevant_terms = ["work from home", "wfh", "remote", "hybrid", "2 days", "core hours", "approval", "international"]
            skip_terms = ["office", "attendance", "leave", "termination", "biometric"]
        else:
            relevant_terms = [t for t in query_lower.split() if len(t) > 3]
            skip_terms = []

        # Process context chunks in order (first chunk is most relevant)
        for chunk_idx, ctx in enumerate(context):
            chunk_priority_boost = 10 if chunk_idx == 0 else (5 if chunk_idx == 1 else 1)  # Boost first chunk

            for line in ctx.split('\n'):
                line = line.strip()
                if not line or len(line) < 15:
                    continue

                line_lower = line.lower()

                # HARD SKIP: Lines that are clearly from wrong section
                if any(skip_term in line_lower for skip_term in skip_terms):
                    # But don't skip if it's a very specific match to query
                    if not any(rterm in line_lower for rterm in relevant_terms[:2]):
                        continue

                # Skip section headers (all caps headers)
                if line.isupper() and len(line) > 30:
                    continue

                # Skip numbered section markers like "1. HEALTH INSURANCE"
                if line and line[0].isdigit() and line[1:3] == '. ':
                    continue

                # Check if line contains relevant terms
                has_relevant_term = any(term in line_lower for term in relevant_terms)

                # Look for factual lines (starting with - or •)
                is_factual = line.startswith('-') or line.startswith('•')

                if is_factual and has_relevant_term:
                    clean_line = line[1:].strip() if line.startswith('-') or line.startswith('•') else line.strip()

                    # Count how many relevant terms this line matches
                    match_score = sum(1 for term in relevant_terms if term in line_lower)

                    # Boost score if from first (most relevant) chunk
                    match_score = match_score * chunk_priority_boost

                    priority_facts.append((match_score, clean_line))

                elif is_factual and not has_relevant_term:
                    # Secondary facts with lower priority
                    secondary_facts.append(line[1:].strip() if line.startswith('-') or line.startswith('•') else line.strip())

        # Sort priority facts by score (descending)
        priority_facts.sort(key=lambda x: x[0], reverse=True)

        # Return only the line text, removing tuple - limit to top facts
        result = [line for _, line in priority_facts[:4]]

        # Add secondary facts if we don't have enough
        if len(result) < 2:
            result.extend(secondary_facts[:2])

        return result

    def _extract_policy_response(self, query: str, context: List[str]) -> str:
        """Fallback: Extract key policy lines when HuggingFace is unavailable"""
        query_lower = query.lower()
        relevant_lines = []

        # Extract keywords from query
        query_words = set(w.lower() for w in query_lower.split() if len(w) > 3)

        # Detect if it's a leave query
        leave_keywords = ['leave', 'sick', 'maternity', 'paternity', 'annual', 'bereavement', 'parental']
        is_leave_query = any(kw in query_lower for kw in leave_keywords)

        # Process each context document
        for ctx in context:
            lines = ctx.split('\n')
            ctx_lower = ctx.lower()
            is_leave_doc = 'leave' in ctx_lower and 'policy' in ctx_lower

            for line in lines:
                line_stripped = line.strip()

                # Skip empty lines
                if not line_stripped:
                    continue

                line_lower = line_stripped.lower()
                word_matches = sum(1 for qw in query_words if qw in line_lower)

                # Prioritize lines with specific information
                doc_boost = 5 if (is_leave_query and is_leave_doc) else 0

                # Priority 1: Lines with numbers and relevant terms
                if line_stripped.startswith('-') and any(char.isdigit() for char in line_stripped):
                    if any(term in line_lower for term in ['day', 'week', 'month', 'year']):
                        score = 8 + doc_boost
                        if any(leave_type in line_lower for leave_type in leave_keywords):
                            score += 2
                        relevant_lines.append((score, line_stripped))

                # Priority 2: Section headers matching query
                elif any(term in line_lower for term in leave_keywords):
                    if line_lower.startswith(tuple('0123456789')):
                        relevant_lines.append((6 + doc_boost, line_stripped))

        # Sort by score and extract unique lines
        relevant_lines = sorted(relevant_lines, key=lambda x: (-x[0], len(x[1])))

        unique_lines = []
        seen = set()
        for score, line in relevant_lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)

        # Build response with appropriate length
        if unique_lines:
            if any(q in query_lower for q in ['how many', 'how much']):
                response = "As per company HR policy:\n" + "\n".join(unique_lines[:2])
            elif any(q in query_lower for q in ['what is']):
                response = "According to company HR policies:\n" + "\n".join(unique_lines[:3])
            else:
                response = "Based on company HR policies:\n" + "\n".join(unique_lines[:4])
        else:
            response = "Based on company HR policies, please refer to the relevant policy section or contact HR."

        return response

    def query(self, query: str, k: int = 5) -> Tuple[str, RAGMetadata]:
        """Execute full RAG pipeline: retrieve → generate → evaluate"""
        total_start = time.time()

        retrieved_docs, retrieval_latency = self.retrieve(query, k)

        response, gen_metrics = self.generate_response(query, retrieved_docs)

        total_latency = (time.time() - total_start) * 1000

        metadata = RAGMetadata(
            context_count=len(retrieved_docs),
            total_latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=gen_metrics["latency_ms"],
            matched_documents=retrieved_docs[:2]
        )

        return response, metadata

    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[str]) -> Dict:
        """Evaluate quality of retrieved documents"""
        query_words = set(query.lower().split())

        quality_metrics = {
            "relevance_score": 0.0,
            "coverage_score": 0.0,
            "diversity_score": 0.0,
            "doc_count": len(retrieved_docs)
        }

        query_covered = 0
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            for word in query_words:
                if word in doc_lower:
                    query_covered += 1

        quality_metrics["relevance_score"] = query_covered / len(query_words) if query_words else 0

        total_words = sum(len(doc.split()) for doc in retrieved_docs)
        quality_metrics["coverage_score"] = min(total_words / 500, 1.0)

        quality_metrics["diversity_score"] = min(len(retrieved_docs) / 5, 1.0)

        quality_metrics["overall_quality"] = (
            quality_metrics["relevance_score"] * 0.5 +
            quality_metrics["coverage_score"] * 0.3 +
            quality_metrics["diversity_score"] * 0.2
        )

        return quality_metrics


def get_sample_documents() -> List[str]:
    """Get sample documents for demo (returns chunk contents)"""
    pipeline = EnhancedRAGPipeline()
    # Return chunk contents (documents are now chunks after chunking)
    return [chunk.content for chunk in pipeline.documents]


if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED RAG PIPELINE TEST")
    print("=" * 80)

    pipeline = EnhancedRAGPipeline()

    test_queries = [
        "What is the leave policy?",
        "How much annual leave do I get?",
        "What are the health insurance benefits?",
        "Can I work from home?",
        "How does the performance appraisal work?",
        "What is the dress code?",
        "Tell me about resignation process"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        response, metadata = pipeline.query(query, k=3)

        print(f"Response:\n{response[:300]}...\n")
        print(f"Metadata:")
        print(f"  - Documents retrieved: {metadata.context_count}")
        print(f"  - Total latency: {metadata.total_latency_ms:.2f}ms")
        print(f"  - Retrieval latency: {metadata.retrieval_latency_ms:.2f}ms")
        print(f"  - Generation latency: {metadata.generation_latency_ms:.2f}ms")

        retrieved, _ = pipeline.retrieve(query, k=3)
        quality = pipeline.evaluate_retrieval_quality(query, retrieved)
        print(f"\nRetrieval Quality:")
        print(f"  - Relevance: {quality['relevance_score']:.2%}")
        print(f"  - Coverage: {quality['coverage_score']:.2%}")
        print(f"  - Diversity: {quality['diversity_score']:.2%}")
        print(f"  - Overall: {quality['overall_quality']:.2%}")


# Compatibility alias for app_fastapi.py and app_streamlit.py
MinimalRAGPipeline = EnhancedRAGPipeline
