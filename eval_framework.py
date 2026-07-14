# 2.1 - LLM Evaluation Framework
# Uses simple string matching for faithfulness + relevance
# No external LLM calls needed

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevance: float
    overall_score: float
    passed: bool

class MinimalEvaluator:
    """Minimal RAGAS-like evaluator"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(self, query: str, response: str, context: List[str]) -> EvaluationResult:
        """Evaluate response quality"""
        faithfulness = self._calculate_faithfulness(response, context)
        relevance = self._calculate_relevance(query, response)

        # Overall score is primarily based on faithfulness (how grounded in context)
        # Relevance provides a secondary boost if the response matches the query intent
        overall = (faithfulness * 0.7) + (relevance * 0.3)

        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            overall_score=overall,
            passed=overall >= self.threshold
        )

    def _calculate_faithfulness(self, response: str, context: List[str]) -> float:
        """
        Calculate faithfulness: % of response facts grounded in context.

        Faithfulness measures whether the response is supported by retrieved context.
        Higher score = more facts from response appear in context.
        """
        # TODO [10 Marks]: Business Logic Completion - Faithfulness Scoring
        # ------------------------------------------------------------------
        # Implement a minimal, RAGAS-style faithfulness metric using word
        # overlap between the response and the retrieved context.
        #
        # Requirements:
        # - Return 0.0 immediately if `response` or `context` is empty.
        # - Join all context strings into one lowercase string
        #   (`" ".join(...)`, joined with spaces to avoid accidental word
        #   merges) to search against.
        # - Use this stop-word set (case-insensitive) to exclude
        #   non-informative words: {"is", "are", "the", "a", "an", "and",
        #   "or", "but", "in", "on", "at", "to", "as", "per", "company",
        #   "hr", "policy"}.
        # - Tokenize the response by whitespace, strip surrounding
        #   punctuation (`.,;:-()`) from each token, lowercase it, and keep
        #   only tokens with length > 2 that are not stop words and not
        #   empty after stripping -> these are the "meaningful words".
        # - Return 0.0 if there are no meaningful words.
        # - Count how many meaningful words appear as a substring of the
        #   joined context text, and divide by the total meaningful word
        #   count to get the faithfulness score.
        # - Clamp the final score to a maximum of 1.0.
        #
        # Inputs: `response` (str), `context` (List[str]).
        # Outputs: `float` in [0.0, 1.0].
        # Dependencies: none beyond the standard library.
        # Acceptance criteria: matches the reference implementation's output
        # (within floating-point tolerance) for the example in `__main__`
        # below (query "What is the leave policy?" scores 1.0 faithfulness).
        raise NotImplementedError("Student Implementation Required: faithfulness scoring")

    def _calculate_relevance(self, query: str, response: str) -> float:
        """% of query words in response"""
        # TODO [Part of 10 Marks - Business Logic Completion]: Relevance Scoring
        # ------------------------------------------------------------------
        # Implement a minimal relevance metric: what fraction of the
        # response's words also appear in the query.
        #
        # Requirements:
        # - Return 0.0 immediately if `query` or `response` is empty.
        # - Build a lowercase set of query words (`set(query.lower().split())`).
        # - Split the response into lowercase words (`response.lower().split()`).
        # - Return 0.0 if the response has no words.
        # - Count how many response words are present in the query word set.
        # - Return `matches / len(response_words)`, clamped to a maximum of 1.0.
        #
        # Inputs: `query` (str), `response` (str).
        # Outputs: `float` in [0.0, 1.0].
        # Dependencies: none beyond the standard library.
        # Acceptance criteria: matches the reference implementation's output
        # for the `__main__` example below.
        raise NotImplementedError("Student Implementation Required: relevance scoring")


# Example usage
if __name__ == "__main__":
    evaluator = MinimalEvaluator()
    result = evaluator.evaluate(
        query="What is the leave policy?",
        response="The company policy provides 20 days of leave annually",
        context=["Leave policy: 20 days paid leave per year"]
    )
    print(f"Faithfulness: {result.faithfulness:.2f}")
    print(f"Relevance: {result.answer_relevance:.2f}")
    print(f"Overall: {result.overall_score:.2f}")
    print(f"Passed: {result.passed}")
