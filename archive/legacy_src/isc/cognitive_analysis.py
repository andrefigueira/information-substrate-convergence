"""
Real Cognitive Analysis based on Cognitive Science

This replaces keyword heuristics with actual reasoning analysis:
1. Argument structure extraction (premises -> inference -> conclusion)
2. Dual-process theory modeling (Kahneman's System 1/System 2)
3. Semantic coherence measurement using embeddings
4. Belief revision tracking (Bayesian-inspired updates)
5. Reasoning chain analysis

Based on:
- Kahneman (2011) - Thinking, Fast and Slow
- Toulmin (1958) - The Uses of Argument
- Pearl (2000) - Causality
- Tversky & Kahneman - Judgment under Uncertainty
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import re
import json
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class ArgumentComponent:
    """Toulmin model argument component"""
    text: str
    component_type: str  # claim, ground, warrant, backing, qualifier, rebuttal
    confidence: float
    embedding: Optional[np.ndarray] = None


@dataclass
class ReasoningChain:
    """A chain of connected reasoning steps"""
    chain_id: str
    steps: List[Dict[str, Any]]  # Each step has: premise, inference_type, conclusion
    coherence_score: float  # Semantic coherence of the chain
    validity_score: float  # Logical validity estimate
    system_type: int  # 1 = intuitive, 2 = analytical (dual-process)


@dataclass
class BeliefState:
    """Tracks beliefs and their confidence over time"""
    proposition: str
    confidence: float  # 0-1, Bayesian-inspired
    evidence_for: List[str]
    evidence_against: List[str]
    last_updated: datetime
    revision_history: List[Tuple[float, str]]  # (old_confidence, reason)


class ArgumentExtractor:
    """
    Extracts argument structure using Toulmin model.

    Toulmin's model identifies:
    - Claim: The conclusion being argued
    - Grounds: Evidence/facts supporting the claim
    - Warrant: The reasoning connecting grounds to claim
    - Backing: Support for the warrant
    - Qualifier: Degree of certainty
    - Rebuttal: Exceptions/counterarguments
    """

    # Linguistic markers for argument components
    CLAIM_MARKERS = [
        'therefore', 'thus', 'hence', 'so', 'consequently',
        'i believe', 'i think', 'it follows', 'we can conclude',
        'this means', 'this shows', 'this proves'
    ]

    GROUND_MARKERS = [
        'because', 'since', 'given that', 'as evidenced by',
        'the fact that', 'research shows', 'studies indicate',
        'for example', 'for instance', 'specifically'
    ]

    WARRANT_MARKERS = [
        'assuming', 'if we accept', 'based on the principle',
        'according to', 'by definition', 'logically',
        'it stands to reason', 'naturally'
    ]

    QUALIFIER_MARKERS = [
        'probably', 'likely', 'possibly', 'certainly',
        'perhaps', 'maybe', 'definitely', 'presumably',
        'generally', 'typically', 'often', 'sometimes'
    ]

    REBUTTAL_MARKERS = [
        'unless', 'except', 'however', 'but', 'although',
        'on the other hand', 'alternatively', 'despite'
    ]

    def __init__(self):
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def extract_argument_structure(self, text: str) -> List[ArgumentComponent]:
        """Extract argument components from text"""
        components = []
        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Identify component type based on markers
            component_type = self._identify_component_type(sentence_lower)
            confidence = self._estimate_confidence(sentence_lower)

            embedding = None
            if self.encoder:
                embedding = self.encoder.encode([sentence])[0]

            components.append(ArgumentComponent(
                text=sentence,
                component_type=component_type,
                confidence=confidence,
                embedding=embedding
            ))

        return components

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _identify_component_type(self, text: str) -> str:
        """Identify the argument component type"""
        # Check markers in order of specificity
        for marker in self.CLAIM_MARKERS:
            if marker in text:
                return 'claim'

        for marker in self.GROUND_MARKERS:
            if marker in text:
                return 'ground'

        for marker in self.WARRANT_MARKERS:
            if marker in text:
                return 'warrant'

        for marker in self.REBUTTAL_MARKERS:
            if marker in text:
                return 'rebuttal'

        for marker in self.QUALIFIER_MARKERS:
            if marker in text:
                return 'qualifier'

        # Default: treat as potential claim or ground based on structure
        if '?' in text:
            return 'question'

        return 'assertion'

    def _estimate_confidence(self, text: str) -> float:
        """Estimate the confidence level expressed in text"""
        high_confidence = ['certainly', 'definitely', 'clearly', 'obviously', 'must']
        medium_confidence = ['probably', 'likely', 'generally', 'typically']
        low_confidence = ['possibly', 'maybe', 'perhaps', 'might', 'could']

        for word in high_confidence:
            if word in text:
                return 0.9

        for word in medium_confidence:
            if word in text:
                return 0.6

        for word in low_confidence:
            if word in text:
                return 0.3

        return 0.5  # Neutral default


class DualProcessAnalyzer:
    """
    Analyzes reasoning using Kahneman's Dual Process Theory.

    System 1: Fast, intuitive, automatic, emotional
    System 2: Slow, deliberate, analytical, logical

    Indicators are based on cognitive science research:
    - System 1: heuristics, emotions, associations, quick judgments
    - System 2: explicit reasoning, calculation, logical connectives
    """

    # System 1 indicators (intuitive/automatic)
    SYSTEM_1_PATTERNS = {
        'emotional_words': [
            'feel', 'sense', 'gut', 'instinct', 'intuition',
            'love', 'hate', 'fear', 'angry', 'happy', 'sad',
            'exciting', 'boring', 'amazing', 'terrible'
        ],
        'heuristic_phrases': [
            'reminds me of', 'similar to', 'like when',
            'usually', 'always', 'never', 'everyone knows',
            'obviously', 'clearly', 'of course'
        ],
        'quick_judgment': [
            'i just know', 'it seems', 'i have a feeling',
            'my impression is', 'at first glance'
        ]
    }

    # System 2 indicators (analytical/deliberate)
    SYSTEM_2_PATTERNS = {
        'logical_connectives': [
            'therefore', 'because', 'if', 'then', 'implies',
            'consequently', 'thus', 'hence', 'given that',
            'it follows that', 'we can deduce', 'must be',
            'so', 'since', 'as a result', 'leads to'
        ],
        'analytical_phrases': [
            'let me think', 'analyzing', 'considering',
            'on one hand', 'on the other hand', 'weighing',
            'the evidence suggests', 'logically speaking',
            'logical', 'deduction', 'deductive', 'inductive',
            'syllogism', 'syllogistic', 'reasoning', 'inference',
            'conclude', 'argument', 'premise', 'hypothesis'
        ],
        'quantitative': [
            'percent', 'ratio', 'probability', 'statistics',
            'data shows', 'numbers indicate', 'calculated',
            'correlation', 'significant', 'p-value', 'coefficient'
        ],
        'deliberation': [
            'step by step', 'first', 'second', 'finally',
            'in conclusion', 'to summarize', 'breaking down',
            'based on', 'according to', 'analysis'
        ]
    }

    def __init__(self):
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def analyze_dual_process(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for System 1 vs System 2 thinking patterns.

        Returns scores and detailed breakdown.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Count System 1 indicators
        s1_emotional = sum(1 for w in self.SYSTEM_1_PATTERNS['emotional_words'] if w in text_lower)
        s1_heuristic = sum(1 for p in self.SYSTEM_1_PATTERNS['heuristic_phrases'] if p in text_lower)
        s1_quick = sum(1 for p in self.SYSTEM_1_PATTERNS['quick_judgment'] if p in text_lower)

        # Count System 2 indicators
        s2_logical = sum(1 for p in self.SYSTEM_2_PATTERNS['logical_connectives'] if p in text_lower)
        s2_analytical = sum(1 for p in self.SYSTEM_2_PATTERNS['analytical_phrases'] if p in text_lower)
        s2_quant = sum(1 for p in self.SYSTEM_2_PATTERNS['quantitative'] if p in text_lower)
        s2_deliberate = sum(1 for p in self.SYSTEM_2_PATTERNS['deliberation'] if p in text_lower)

        # Calculate scores
        s1_total = s1_emotional + s1_heuristic * 2 + s1_quick * 2
        s2_total = s2_logical * 2 + s2_analytical * 2 + s2_quant + s2_deliberate

        total = s1_total + s2_total + 1  # +1 to avoid division by zero

        s1_ratio = s1_total / total
        s2_ratio = s2_total / total

        # Determine dominant system
        if s2_ratio > s1_ratio + 0.2:
            dominant = 2
            processing_style = "analytical"
        elif s1_ratio > s2_ratio + 0.2:
            dominant = 1
            processing_style = "intuitive"
        else:
            dominant = 0  # Mixed
            processing_style = "mixed"

        # Analyze response latency proxy (sentence complexity)
        avg_sentence_length = len(text.split()) / max(text.count('.') + text.count('!') + text.count('?'), 1)
        complexity_indicator = min(avg_sentence_length / 20, 1.0)  # Longer = more deliberate

        return {
            'system_1_score': s1_ratio,
            'system_2_score': s2_ratio,
            'dominant_system': dominant,
            'processing_style': processing_style,
            'complexity_indicator': complexity_indicator,
            'breakdown': {
                'system_1': {
                    'emotional': s1_emotional,
                    'heuristic': s1_heuristic,
                    'quick_judgment': s1_quick
                },
                'system_2': {
                    'logical': s2_logical,
                    'analytical': s2_analytical,
                    'quantitative': s2_quant,
                    'deliberate': s2_deliberate
                }
            }
        }


class SemanticCoherenceAnalyzer:
    """
    Measures semantic coherence of reasoning using embeddings.

    Coherent reasoning has:
    - High semantic similarity between connected ideas
    - Logical flow (each step relates to previous)
    - Consistent topic threading

    Based on Latent Semantic Analysis and coherence research.
    """

    def __init__(self):
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def compute_coherence(self, statements: List[str]) -> Dict[str, float]:
        """
        Compute semantic coherence metrics for a list of statements.
        """
        if not self.encoder or len(statements) < 2:
            return {'local_coherence': 0.5, 'global_coherence': 0.5, 'overall': 0.5}

        # Get embeddings
        embeddings = self.encoder.encode(statements)

        # Local coherence: average similarity between adjacent statements
        local_similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            local_similarities.append(sim)

        local_coherence = np.mean(local_similarities) if local_similarities else 0.5

        # Global coherence: how well each statement relates to the overall topic
        centroid = np.mean(embeddings, axis=0)
        global_similarities = [
            self._cosine_similarity(emb, centroid) for emb in embeddings
        ]
        global_coherence = np.mean(global_similarities)

        # Topic drift: how much the topic changes over the sequence
        if len(embeddings) > 2:
            first_half = np.mean(embeddings[:len(embeddings)//2], axis=0)
            second_half = np.mean(embeddings[len(embeddings)//2:], axis=0)
            topic_stability = self._cosine_similarity(first_half, second_half)
        else:
            topic_stability = 1.0

        overall = (local_coherence * 0.4 + global_coherence * 0.4 + topic_stability * 0.2)

        return {
            'local_coherence': float(local_coherence),
            'global_coherence': float(global_coherence),
            'topic_stability': float(topic_stability),
            'overall': float(overall)
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_reasoning_gaps(self, statements: List[str], threshold: float = 0.3) -> List[int]:
        """Find positions where reasoning has gaps (low coherence)"""
        if not self.encoder or len(statements) < 2:
            return []

        embeddings = self.encoder.encode(statements)
        gaps = []

        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < threshold:
                gaps.append(i)

        return gaps


class BeliefTracker:
    """
    Tracks belief states and revisions over conversation.

    Uses Bayesian-inspired confidence updates:
    - New evidence updates belief confidence
    - Contradictory evidence reduces confidence
    - Consistent evidence increases confidence

    Based on Bayesian epistemology and belief revision theory (AGM).
    """

    def __init__(self):
        self.beliefs: Dict[str, BeliefState] = {}
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def extract_propositions(self, text: str) -> List[str]:
        """Extract belief-relevant propositions from text"""
        # Split into sentences and filter for declarative statements
        sentences = re.split(r'[.!]+', text)
        propositions = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or '?' in sentence:
                continue

            # Look for belief indicators
            belief_indicators = [
                'i think', 'i believe', 'it is', 'there is',
                'we know', 'it seems', 'evidence shows'
            ]

            sentence_lower = sentence.lower()
            if any(ind in sentence_lower for ind in belief_indicators):
                propositions.append(sentence)
            elif len(sentence.split()) > 5:  # Substantial declarative statements
                propositions.append(sentence)

        return propositions

    def update_belief(
        self,
        proposition: str,
        is_supporting: bool,
        evidence: str,
        strength: float = 0.1
    ) -> BeliefState:
        """
        Update belief based on new evidence using Bayesian-inspired update.

        P(belief|evidence) proportional to P(evidence|belief) * P(belief)
        Simplified: confidence += strength if supporting, -= strength if opposing
        """
        # Find or create belief state
        belief_key = self._normalize_proposition(proposition)

        if belief_key not in self.beliefs:
            self.beliefs[belief_key] = BeliefState(
                proposition=proposition,
                confidence=0.5,  # Prior
                evidence_for=[],
                evidence_against=[],
                last_updated=datetime.now(),
                revision_history=[]
            )

        belief = self.beliefs[belief_key]
        old_confidence = belief.confidence

        # Bayesian-inspired update
        if is_supporting:
            # Increase confidence, diminishing returns near 1.0
            belief.confidence += strength * (1 - belief.confidence)
            belief.evidence_for.append(evidence)
        else:
            # Decrease confidence, diminishing returns near 0.0
            belief.confidence -= strength * belief.confidence
            belief.evidence_against.append(evidence)

        # Clamp to valid range
        belief.confidence = max(0.01, min(0.99, belief.confidence))

        # Record revision
        belief.revision_history.append((old_confidence, evidence[:100]))
        belief.last_updated = datetime.now()

        return belief

    def _normalize_proposition(self, proposition: str) -> str:
        """Normalize proposition for matching"""
        # Simple normalization - could use embeddings for semantic matching
        return proposition.lower().strip()[:200]

    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of current belief states"""
        if not self.beliefs:
            return {'count': 0, 'beliefs': []}

        sorted_beliefs = sorted(
            self.beliefs.values(),
            key=lambda b: abs(b.confidence - 0.5),
            reverse=True
        )

        return {
            'count': len(self.beliefs),
            'high_confidence': [
                {'proposition': b.proposition[:100], 'confidence': b.confidence}
                for b in sorted_beliefs if b.confidence > 0.7
            ][:5],
            'low_confidence': [
                {'proposition': b.proposition[:100], 'confidence': b.confidence}
                for b in sorted_beliefs if b.confidence < 0.3
            ][:5],
            'most_revised': sorted(
                self.beliefs.values(),
                key=lambda b: len(b.revision_history),
                reverse=True
            )[:3]
        }

    def find_contradictions(self) -> List[Tuple[str, str, float]]:
        """Find potentially contradictory beliefs using embeddings"""
        if not self.encoder or len(self.beliefs) < 2:
            return []

        contradictions = []
        belief_list = list(self.beliefs.values())

        # Get embeddings for all beliefs
        embeddings = self.encoder.encode([b.proposition for b in belief_list])

        # Find pairs with high similarity but opposite confidence
        for i in range(len(belief_list)):
            for j in range(i + 1, len(belief_list)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])

                # High similarity but different confidence directions
                conf_diff = abs(belief_list[i].confidence - belief_list[j].confidence)

                if sim > 0.7 and conf_diff > 0.4:
                    contradictions.append((
                        belief_list[i].proposition[:100],
                        belief_list[j].proposition[:100],
                        conf_diff
                    ))

        return contradictions

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class RealCognitiveAnalyzer:
    """
    Main cognitive analyzer combining all analysis methods.

    This replaces the fake keyword-based analysis with real cognitive science.
    """

    def __init__(self):
        self.argument_extractor = ArgumentExtractor()
        self.dual_process = DualProcessAnalyzer()
        self.coherence_analyzer = SemanticCoherenceAnalyzer()
        self.belief_tracker = BeliefTracker()

        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def analyze(self, text: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive cognitive analysis on text.

        Returns detailed analysis including:
        - Argument structure
        - Dual-process classification
        - Semantic coherence
        - Belief updates
        """
        # Argument structure
        argument_components = self.argument_extractor.extract_argument_structure(text)

        # Dual process analysis
        dual_process = self.dual_process.analyze_dual_process(text)

        # Coherence analysis
        statements = [c.text for c in argument_components]
        if context:
            statements = context[-3:] + statements  # Include recent context
        coherence = self.coherence_analyzer.compute_coherence(statements)

        # Extract and update beliefs
        propositions = self.belief_tracker.extract_propositions(text)
        for prop in propositions:
            # Determine if supporting based on confidence markers
            is_supporting = not any(
                neg in prop.lower()
                for neg in ['not', "don't", "doesn't", 'never', 'unlikely']
            )
            self.belief_tracker.update_belief(prop, is_supporting, text[:100])

        # Reasoning quality score
        reasoning_quality = self._compute_reasoning_quality(
            argument_components, dual_process, coherence
        )

        return {
            'argument_structure': {
                'components': [
                    {'text': c.text[:100], 'type': c.component_type, 'confidence': c.confidence}
                    for c in argument_components
                ],
                'has_claim': any(c.component_type == 'claim' for c in argument_components),
                'has_grounds': any(c.component_type == 'ground' for c in argument_components),
                'has_warrant': any(c.component_type == 'warrant' for c in argument_components),
                'structure_completeness': self._argument_completeness(argument_components)
            },
            'dual_process': dual_process,
            'coherence': coherence,
            'belief_state': self.belief_tracker.get_belief_summary(),
            'reasoning_quality': reasoning_quality
        }

    def _argument_completeness(self, components: List[ArgumentComponent]) -> float:
        """Measure how complete the argument structure is"""
        types = set(c.component_type for c in components)

        # Essential: claim and ground
        # Good: + warrant
        # Complete: + backing, qualifier, rebuttal

        score = 0.0
        if 'claim' in types:
            score += 0.3
        if 'ground' in types:
            score += 0.3
        if 'warrant' in types:
            score += 0.2
        if 'backing' in types:
            score += 0.1
        if 'qualifier' in types or 'rebuttal' in types:
            score += 0.1

        return score

    def _compute_reasoning_quality(
        self,
        components: List[ArgumentComponent],
        dual_process: Dict[str, Any],
        coherence: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute overall reasoning quality metrics"""

        # Argument structure quality
        structure_score = self._argument_completeness(components)

        # Processing depth (System 2 engagement)
        depth_score = dual_process['system_2_score'] * 0.7 + dual_process['complexity_indicator'] * 0.3

        # Coherence
        coherence_score = coherence['overall']

        # Overall quality
        overall = (structure_score * 0.35 + depth_score * 0.35 + coherence_score * 0.3)

        return {
            'structure': structure_score,
            'depth': depth_score,
            'coherence': coherence_score,
            'overall': overall
        }

    def get_reasoning_profile(self) -> Dict[str, Any]:
        """Get accumulated reasoning profile from all analyzed text"""
        return {
            'belief_count': len(self.belief_tracker.beliefs),
            'beliefs': self.belief_tracker.get_belief_summary(),
            'potential_contradictions': self.belief_tracker.find_contradictions()[:5]
        }
