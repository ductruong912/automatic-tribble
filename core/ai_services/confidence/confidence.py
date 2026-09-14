"""
Confidence scoring module for LLM responses.
Assesses response quality and reliability based on multiple factors including context alignment, coherence, and uncertainty indicators.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score with detailed breakdown of quality factors."""

    overall_score: float  # 0.0 to 1.0 - overall confidence rating
    context_alignment: float  # How well response aligns with provided context
    response_length_appropriateness: float  # Response length vs query complexity
    semantic_coherence: float  # Internal consistency and logical flow
    source_citation: float  # Whether response cites sources from context
    uncertainty_indicators: float  # Presence of uncertainty markers
    factors: Dict[str, float]  # Detailed factor breakdown for analysis
    reasoning: str  # Human-readable explanation of the score


class ConfidenceScorer:
    """
    Multi-factor confidence scoring system for LLM responses.
    Evaluates response quality based on context alignment, coherence, and linguistic indicators.
    """

    def __init__(self):
        # Confidence must be grounded in the retrieved evidence. Do not use
        # language-specific filler or confidence phrases as evidence.
        self._token_pattern = re.compile(r"\b\w{2,}\b", re.UNICODE)

    def calculate_confidence(
        self,
        response: str,
        query: str,
        context: str = "",
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score for an LLM response.

        Args:
            response: LLM response text to evaluate
            query: Original user query for context
            context: Retrieved document context used for response
            search_results: Search results with similarity scores

        Returns:
            ConfidenceScore object with detailed quality breakdown
        """
        try:
            factors = {}

            # 1. Context alignment - how well response uses provided context
            factors["context_alignment"] = self._calculate_context_alignment(
                response, context, search_results
            )

            # 2. Response length appropriateness - matches query complexity
            factors["response_length_appropriateness"] = (
                self._calculate_length_appropriateness(response, query)
            )

            # 3. Semantic coherence - internal consistency and logical flow
            factors["semantic_coherence"] = self._calculate_semantic_coherence(response)

            # 4. Source citation - whether response cites sources from context
            factors["source_citation"] = self._calculate_source_citation(
                response, context, search_results
            )

            # 5. Uncertainty indicators - presence of uncertainty markers
            factors["uncertainty_indicators"] = self._calculate_uncertainty_score(
                response, context
            )

            # Calculate overall score using weighted average of all factors
            weights = {
                "context_alignment": 0.35,  # Most important - response should use context
                "response_length_appropriateness": 0.20,  # Length should match query
                "semantic_coherence": 0.25,  # Response should be internally consistent
                "source_citation": 0.10,  # Should cite sources when appropriate
                "uncertainty_indicators": 0.10,  # Should avoid uncertainty markers
            }

            overall_score = sum(
                factors[factor] * weights[factor] for factor in weights.keys()
            )

            # Generate human-readable reasoning for the score
            reasoning = self._generate_reasoning(factors, overall_score)

            return ConfidenceScore(
                overall_score=overall_score,
                context_alignment=factors["context_alignment"],
                response_length_appropriateness=factors[
                    "response_length_appropriateness"
                ],
                semantic_coherence=factors["semantic_coherence"],
                source_citation=factors["source_citation"],
                uncertainty_indicators=factors["uncertainty_indicators"],
                factors=factors,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning(f" Confidence calculation failed: {e}")
            # Return neutral score on error to avoid breaking the system
            return ConfidenceScore(
                overall_score=0.5,  # Neutral score on error
                context_alignment=0.5,
                response_length_appropriateness=0.5,
                semantic_coherence=0.5,
                source_citation=0.5,
                uncertainty_indicators=0.5,
                factors={},
                reasoning=f"Confidence calculation failed: {e}",
            )

    def _calculate_context_alignment(
        self,
        response: str,
        context: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Calculate how well the response aligns with the provided context.
        Evaluates word overlap, length appropriateness, and search result quality.
        """
        if not context or not context.strip():
            return 0.3  # Low score when no context available

        context_words = self._content_tokens(context)
        response_words = self._content_tokens(response)

        if not context_words:
            return 0.5

        # Measure how much of the answer's substantive vocabulary is supported
        # by the retrieved context. This works for Vietnamese and HTML alike.
        overlap = len(context_words.intersection(response_words))
        word_alignment = overlap / len(response_words) if response_words else 0.0

        # Check if response length is appropriate for context length
        context_length = len(context)
        response_length = len(response)

        if context_length > 0:
            length_ratio = response_length / context_length
            # Ideal ratio is between 0.1 and 0.5 (response should be shorter than context)
            if 0.1 <= length_ratio <= 0.5:
                length_score = 1.0
            elif length_ratio < 0.1:
                length_score = length_ratio / 0.1
            else:
                length_score = max(0.0, 1.0 - (length_ratio - 0.5) / 0.5)
        else:
            length_score = 0.5

        # Retrieval quality is supporting evidence, not a language heuristic.
        search_score = 0.0
        if search_results and len(search_results) > 0:
            scores = [float(r.get("combined_score", 0.0)) for r in search_results]
            search_score = sum(max(0.0, min(1.0, score)) for score in scores) / len(scores)

        # Combine all alignment factors with appropriate weights
        alignment_score = word_alignment * 0.4 + length_score * 0.3 + search_score * 0.3

        return min(1.0, max(0.0, alignment_score))

    def _calculate_length_appropriateness(self, response: str, query: str) -> float:
        """
        Calculate if response length is appropriate for query complexity.
        Uses heuristics based on query word count to determine expected response length.
        """
        query_words = len(query.split())
        response_words = len(response.split())

        # Simple heuristics for appropriate length based on query complexity
        if query_words <= 5:  # Simple question - expect concise response
            if 10 <= response_words <= 50:
                return 1.0
            elif response_words < 10:
                return response_words / 10
            else:
                return max(0.0, 1.0 - (response_words - 50) / 50)

        elif query_words <= 15:  # Medium question
            if 20 <= response_words <= 100:
                return 1.0
            elif response_words < 20:
                return response_words / 20
            else:
                return max(0.0, 1.0 - (response_words - 100) / 100)

        else:  # Complex question
            if 30 <= response_words <= 200:
                return 1.0
            elif response_words < 30:
                return response_words / 30
            else:
                return max(0.0, 1.0 - (response_words - 200) / 200)

    def _calculate_semantic_coherence(self, response: str) -> float:
        """Estimate structural coherence without language-specific connectors."""
        sentences = re.split(r"[.!?\n]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0
        if len(sentences) == 1:
            return 1.0

        sentence_lengths = [len(self._content_tokens(sentence)) for sentence in sentences]
        non_empty_ratio = sum(length > 0 for length in sentence_lengths) / len(sentences)
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variation = (
            sum(abs(length - mean_length) for length in sentence_lengths)
            / (len(sentence_lengths) * mean_length)
            if mean_length
            else 1.0
        )
        repetition_score = self._repetition_score(response)
        structure_score = max(0.0, 1.0 - min(1.0, length_variation))
        return min(1.0, max(0.0, non_empty_ratio * (0.45 * repetition_score + 0.55 * structure_score)))

    def _calculate_source_citation(
        self,
        response: str,
        context: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Score explicit chunk references and evidence overlap, not wording."""
        if not context.strip():
            return 0.0

        chunk_references = set(re.findall(r"\[chunk\s+(\d+)\]", response, re.IGNORECASE))
        available_chunks = set(re.findall(r"\[chunk\s+(\d+)\]", context, re.IGNORECASE))
        citation_score = (
            len(chunk_references.intersection(available_chunks)) / len(chunk_references)
            if chunk_references
            else 0.0
        )
        overlap_score = self._token_overlap(response, context)
        retrieval_score = self._retrieval_score(search_results)
        return min(1.0, 0.5 * citation_score + 0.35 * overlap_score + 0.15 * retrieval_score)

    def _calculate_uncertainty_score(self, response: str, context: str) -> float:
        """Estimate evidence sufficiency; the value is retained for API compatibility."""
        if not response.strip():
            return 0.0
        if not context.strip():
            return 0.25
        return self._token_overlap(response, context)

    def _content_tokens(self, text: str) -> Set[str]:
        """Return normalized content tokens while ignoring HTML tags and chunk labels."""
        without_markup = re.sub(r"<[^>]+>|\[/?chunk\s+\d+\]", " ", text, flags=re.IGNORECASE)
        return {token.lower() for token in self._token_pattern.findall(without_markup)}

    def _token_overlap(self, response: str, context: str) -> float:
        response_tokens = self._content_tokens(response)
        context_tokens = self._content_tokens(context)
        if not response_tokens or not context_tokens:
            return 0.0
        return len(response_tokens & context_tokens) / len(response_tokens)

    def _retrieval_score(self, search_results: Optional[List[Dict[str, Any]]]) -> float:
        if not search_results:
            return 0.0
        scores = [float(result.get("combined_score", 0.0)) for result in search_results]
        return sum(max(0.0, min(1.0, score)) for score in scores) / len(scores)

    def _repetition_score(self, response: str) -> float:
        tokens = self._content_tokens(response)
        all_tokens = self._token_pattern.findall(response.lower())
        if not all_tokens:
            return 0.0
        return len(tokens) / len(all_tokens)

    def _generate_reasoning(
        self, factors: Dict[str, float], overall_score: float
    ) -> str:
        """Generate human-readable reasoning for the confidence score."""
        reasoning_parts = []

        if factors.get("context_alignment", 0) > 0.7:
            reasoning_parts.append("Response well-aligned with provided context")
        elif factors.get("context_alignment", 0) < 0.4:
            reasoning_parts.append("Response shows limited use of provided context")

        if factors.get("response_length_appropriateness", 0) > 0.8:
            reasoning_parts.append("Response length appropriate for query complexity")
        elif factors.get("response_length_appropriateness", 0) < 0.5:
            reasoning_parts.append("Response length may not match query expectations")

        if factors.get("semantic_coherence", 0) > 0.8:
            reasoning_parts.append("Response shows good internal coherence")
        elif factors.get("semantic_coherence", 0) < 0.6:
            reasoning_parts.append("Response may lack logical flow")

        if factors.get("source_citation", 0) > 0.6:
            reasoning_parts.append("Response includes source references")

        if factors.get("uncertainty_indicators", 0) < 0.6:
            reasoning_parts.append("Response contains uncertainty indicators")

        if overall_score > 0.8:
            reasoning_parts.append("High confidence in response quality")
        elif overall_score > 0.6:
            reasoning_parts.append("Moderate confidence in response quality")
        elif overall_score > 0.4:
            reasoning_parts.append("Lower confidence in response quality")
        else:
            reasoning_parts.append("Low confidence in response quality")

        return (
            ". ".join(reasoning_parts)
            if reasoning_parts
            else "Confidence assessment completed"
        )

    def get_confidence_level(self, score: float) -> str:
        """Convert numerical score to confidence level."""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Moderate"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
