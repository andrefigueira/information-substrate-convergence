"""
Cognitive Architecture Modeling for ISC

This module implements user-specific cognitive modeling, enabling the ISC system
to learn HOW a user thinks, not just WHAT they say. This is the foundation
for the Synthetic Cognition Platform.

Key components:
- CognitiveProfile: Captures individual reasoning patterns, heuristics, and preferences
- ReasoningPatternAnalyzer: Identifies patterns in how users approach problems
- CognitiveProfiler: Builds and updates profiles from interactions
- CognitiveComposer: Synthesizes multiple cognitive profiles for novel reasoning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import json
import hashlib
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class ReasoningPattern:
    """Represents a single reasoning pattern"""
    pattern_id: str
    pattern_type: str  # analytical, intuitive, systematic, creative, etc.
    trigger_concepts: Set[str]
    typical_transitions: List[Tuple[str, str]]  # concept transitions
    confidence: float
    frequency: int
    examples: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'trigger_concepts': list(self.trigger_concepts),
            'typical_transitions': self.typical_transitions,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'examples': self.examples[:5]  # Keep only 5 examples
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReasoningPattern':
        data['trigger_concepts'] = set(data.get('trigger_concepts', []))
        return cls(**data)


@dataclass
class CognitiveProfile:
    """
    Captures an individual's cognitive architecture.

    This goes beyond storing what someone knows to modeling HOW they think:
    - Their preferred reasoning approaches
    - Their conceptual organization patterns
    - Their heuristics and biases
    - Their learning style
    """
    profile_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Reasoning patterns
    reasoning_patterns: Dict[str, ReasoningPattern] = field(default_factory=dict)

    # Cognitive style metrics (0-1 scale)
    analytical_tendency: float = 0.5
    intuitive_tendency: float = 0.5
    systematic_tendency: float = 0.5
    creative_tendency: float = 0.5
    detail_orientation: float = 0.5
    big_picture_orientation: float = 0.5

    # Conceptual preferences
    preferred_domains: Dict[str, float] = field(default_factory=dict)  # domain -> affinity
    concept_connections: Dict[str, Set[str]] = field(default_factory=dict)  # concept -> related concepts

    # Learning characteristics
    question_asking_style: str = "balanced"  # exploratory, focused, probing, etc.
    explanation_preference: str = "balanced"  # examples, analogies, formal, intuitive

    # Embedding for quick similarity computation
    profile_embedding: Optional[np.ndarray] = None

    # Interaction history summary
    total_interactions: int = 0
    topic_distribution: Dict[str, float] = field(default_factory=dict)

    def compute_embedding(self, encoder: Optional[Any] = None) -> np.ndarray:
        """Compute a dense embedding representing this cognitive profile"""
        # Create a text summary of the profile
        summary_parts = [
            f"analytical {self.analytical_tendency:.2f}",
            f"intuitive {self.intuitive_tendency:.2f}",
            f"systematic {self.systematic_tendency:.2f}",
            f"creative {self.creative_tendency:.2f}",
            f"detail oriented {self.detail_orientation:.2f}",
            f"big picture {self.big_picture_orientation:.2f}",
            f"question style {self.question_asking_style}",
            f"explanation preference {self.explanation_preference}",
        ]

        # Add top domains
        top_domains = sorted(self.preferred_domains.items(), key=lambda x: -x[1])[:5]
        for domain, score in top_domains:
            summary_parts.append(f"interested in {domain}")

        # Add top reasoning patterns
        top_patterns = sorted(
            self.reasoning_patterns.values(),
            key=lambda p: -p.frequency
        )[:3]
        for pattern in top_patterns:
            summary_parts.append(f"tends to reason {pattern.pattern_type}")

        summary = " ".join(summary_parts)

        if encoder is not None:
            self.profile_embedding = encoder.encode([summary])[0]
        else:
            # Fallback: create embedding from numeric features
            features = [
                self.analytical_tendency,
                self.intuitive_tendency,
                self.systematic_tendency,
                self.creative_tendency,
                self.detail_orientation,
                self.big_picture_orientation,
                self.total_interactions / 1000,  # Normalized
            ]
            self.profile_embedding = np.array(features)

        return self.profile_embedding

    def similarity_to(self, other: 'CognitiveProfile') -> float:
        """Compute cognitive similarity to another profile"""
        if self.profile_embedding is None or other.profile_embedding is None:
            # Fallback to basic similarity
            style_diff = abs(self.analytical_tendency - other.analytical_tendency)
            style_diff += abs(self.intuitive_tendency - other.intuitive_tendency)
            style_diff += abs(self.systematic_tendency - other.systematic_tendency)
            style_diff += abs(self.creative_tendency - other.creative_tendency)
            return 1.0 - (style_diff / 4.0)

        # Cosine similarity of embeddings
        norm1 = np.linalg.norm(self.profile_embedding)
        norm2 = np.linalg.norm(other.profile_embedding)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(self.profile_embedding, other.profile_embedding) / (norm1 * norm2))

    def to_dict(self) -> Dict:
        return {
            'profile_id': self.profile_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'reasoning_patterns': {k: v.to_dict() for k, v in self.reasoning_patterns.items()},
            'analytical_tendency': self.analytical_tendency,
            'intuitive_tendency': self.intuitive_tendency,
            'systematic_tendency': self.systematic_tendency,
            'creative_tendency': self.creative_tendency,
            'detail_orientation': self.detail_orientation,
            'big_picture_orientation': self.big_picture_orientation,
            'preferred_domains': self.preferred_domains,
            'concept_connections': {k: list(v) for k, v in self.concept_connections.items()},
            'question_asking_style': self.question_asking_style,
            'explanation_preference': self.explanation_preference,
            'total_interactions': self.total_interactions,
            'topic_distribution': self.topic_distribution,
            'profile_embedding': self.profile_embedding.tolist() if self.profile_embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveProfile':
        profile = cls(profile_id=data['profile_id'])
        profile.created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        profile.updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        profile.reasoning_patterns = {
            k: ReasoningPattern.from_dict(v)
            for k, v in data.get('reasoning_patterns', {}).items()
        }
        profile.analytical_tendency = data.get('analytical_tendency', 0.5)
        profile.intuitive_tendency = data.get('intuitive_tendency', 0.5)
        profile.systematic_tendency = data.get('systematic_tendency', 0.5)
        profile.creative_tendency = data.get('creative_tendency', 0.5)
        profile.detail_orientation = data.get('detail_orientation', 0.5)
        profile.big_picture_orientation = data.get('big_picture_orientation', 0.5)
        profile.preferred_domains = data.get('preferred_domains', {})
        profile.concept_connections = {
            k: set(v) for k, v in data.get('concept_connections', {}).items()
        }
        profile.question_asking_style = data.get('question_asking_style', 'balanced')
        profile.explanation_preference = data.get('explanation_preference', 'balanced')
        profile.total_interactions = data.get('total_interactions', 0)
        profile.topic_distribution = data.get('topic_distribution', {})
        if data.get('profile_embedding') is not None:
            profile.profile_embedding = np.array(data['profile_embedding'])
        return profile


class ReasoningPatternAnalyzer:
    """
    Analyzes user interactions to identify reasoning patterns.

    UPGRADED to use real cognitive science:
    - Dual Process Theory (Kahneman): System 1 (intuitive) vs System 2 (analytical)
    - Toulmin Model: Argument structure analysis
    - Semantic coherence: Embedding-based reasoning quality

    Falls back to keyword heuristics only if real analysis is unavailable.
    """

    # Fallback pattern indicators (used only if real analyzer unavailable)
    ANALYTICAL_INDICATORS = {
        'why', 'because', 'therefore', 'logic', 'reason', 'evidence',
        'analyze', 'breakdown', 'component', 'structure', 'systematic'
    }

    INTUITIVE_INDICATORS = {
        'feel', 'sense', 'seems', 'might', 'maybe', 'probably',
        'guess', 'hunch', 'impression', 'vibe'
    }

    SYSTEMATIC_INDICATORS = {
        'step', 'first', 'then', 'next', 'process', 'method',
        'procedure', 'sequence', 'order', 'plan'
    }

    CREATIVE_INDICATORS = {
        'what if', 'imagine', 'alternative', 'different', 'new',
        'novel', 'unusual', 'combine', 'connect', 'possibility'
    }

    DETAIL_INDICATORS = {
        'specific', 'exactly', 'precisely', 'detail', 'particular',
        'example', 'instance', 'case'
    }

    BIG_PICTURE_INDICATORS = {
        'overall', 'general', 'broad', 'main', 'key', 'essential',
        'fundamental', 'core', 'basic', 'principle'
    }

    def __init__(self):
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

        # Try to use real cognitive analyzer
        self.real_analyzer = None
        try:
            from .cognitive_analysis import RealCognitiveAnalyzer
            self.real_analyzer = RealCognitiveAnalyzer()
        except ImportError:
            pass

    def analyze_interaction(
        self,
        user_input: str,
        response: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze a single interaction for reasoning patterns using real cognitive science"""

        # Try real cognitive analysis first
        if self.real_analyzer is not None:
            try:
                return self._real_analysis(user_input, response, context)
            except Exception:
                pass  # Fall back to heuristics

        # Fallback to keyword heuristics
        return self._heuristic_analysis(user_input, response, context)

    def _real_analysis(
        self,
        user_input: str,
        response: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Use real cognitive science for analysis"""
        combined_text = f"{user_input} {response}"

        # Get real cognitive analysis
        real_result = self.real_analyzer.analyze(combined_text)

        # Extract dual process scores
        dual_process = real_result.get('dual_process', {})
        system_1 = dual_process.get('system_1_score', 0.5)
        system_2 = dual_process.get('system_2_score', 0.5)
        processing_style = dual_process.get('processing_style', 'mixed')

        # Map to cognitive profile scores
        # System 2 correlates with analytical and systematic
        # System 1 correlates with intuitive and creative
        analytical_score = system_2
        intuitive_score = system_1
        systematic_score = system_2 * 0.8 + dual_process.get('complexity_indicator', 0.5) * 0.2
        creative_score = system_1 * 0.5 + 0.5  # Creative needs more nuance

        # Get argument structure for detail/big picture
        arg_structure = real_result.get('argument_structure', {})
        structure_completeness = arg_structure.get('structure_completeness', 0.5)

        # More complete arguments suggest detail orientation
        detail_score = structure_completeness
        big_picture_score = 1 - structure_completeness  # Inverse

        # Get reasoning quality
        reasoning_quality = real_result.get('reasoning_quality', {})

        return {
            'analytical_score': analytical_score,
            'intuitive_score': intuitive_score,
            'systematic_score': systematic_score,
            'creative_score': creative_score,
            'detail_score': detail_score,
            'big_picture_score': big_picture_score,
            'question_type': self._classify_question(user_input),
            'concepts_mentioned': self._extract_concepts(user_input),
            'complexity': reasoning_quality.get('depth', 0.5),
            # Additional real analysis data
            'processing_style': processing_style,
            'argument_has_claim': arg_structure.get('has_claim', False),
            'argument_has_grounds': arg_structure.get('has_grounds', False),
            'coherence_score': real_result.get('coherence', {}).get('overall', 0.5),
            'analysis_method': 'real_cognitive_science'
        }

    def _heuristic_analysis(
        self,
        user_input: str,
        response: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Fallback keyword-based analysis"""
        words = set(user_input.lower().split())

        return {
            'analytical_score': self._count_indicators(words, self.ANALYTICAL_INDICATORS),
            'intuitive_score': self._count_indicators(words, self.INTUITIVE_INDICATORS),
            'systematic_score': self._count_indicators(words, self.SYSTEMATIC_INDICATORS),
            'creative_score': self._count_indicators(words, self.CREATIVE_INDICATORS),
            'detail_score': self._count_indicators(words, self.DETAIL_INDICATORS),
            'big_picture_score': self._count_indicators(words, self.BIG_PICTURE_INDICATORS),
            'question_type': self._classify_question(user_input),
            'concepts_mentioned': self._extract_concepts(user_input),
            'complexity': self._assess_complexity(user_input),
            'analysis_method': 'keyword_heuristics'
        }

    def _count_indicators(self, words: Set[str], indicators: Set[str]) -> float:
        """Count how many indicator words are present"""
        count = len(words.intersection(indicators))
        return min(count / 3, 1.0)  # Normalize to 0-1

    def _classify_question(self, text: str) -> str:
        """Classify the type of question/statement"""
        text_lower = text.lower()

        if '?' not in text:
            return 'statement'

        if text_lower.startswith(('what is', 'what are', 'what does')):
            return 'definitional'
        elif text_lower.startswith(('how do', 'how does', 'how can', 'how to')):
            return 'procedural'
        elif text_lower.startswith(('why do', 'why does', 'why is')):
            return 'causal'
        elif text_lower.startswith(('what if', 'could', 'would')):
            return 'hypothetical'
        elif text_lower.startswith(('do you', 'can you', 'are you')):
            return 'personal'
        else:
            return 'exploratory'

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple extraction based on significant words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                      'as', 'until', 'while', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'what', 'which', 'who', 'this', 'that'}

        words = text.lower().split()
        concepts = [w.strip('.,!?;:') for w in words if len(w) > 3 and w.lower() not in stop_words]
        return list(set(concepts))[:10]  # Top 10 unique concepts

    def _assess_complexity(self, text: str) -> float:
        """Assess the complexity of the input (0-1)"""
        # Simple heuristics
        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        question_count = text.count('?')

        complexity = 0.0
        complexity += min(word_count / 50, 0.4)  # Length factor
        complexity += min(avg_word_length / 10, 0.3)  # Word complexity factor
        complexity += min(question_count * 0.1, 0.3)  # Question depth factor

        return min(complexity, 1.0)


class CognitiveProfiler:
    """
    Builds and updates cognitive profiles from user interactions.

    This is the core of the Cognitive Architecture Modeling system.
    It learns HOW a user thinks by analyzing their interaction patterns.
    """

    def __init__(self, save_path: str = "models/cognitive_profiles"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.profiles: Dict[str, CognitiveProfile] = {}
        self.analyzer = ReasoningPatternAnalyzer()
        self.interaction_buffer: Dict[str, List[Dict]] = defaultdict(list)

        # Load existing profiles
        self._load_profiles()

    def get_or_create_profile(self, user_id: str) -> CognitiveProfile:
        """Get existing profile or create new one"""
        if user_id not in self.profiles:
            self.profiles[user_id] = CognitiveProfile(profile_id=user_id)
        return self.profiles[user_id]

    def update_profile(
        self,
        user_id: str,
        user_input: str,
        response: str,
        context: Optional[Dict] = None
    ) -> CognitiveProfile:
        """Update a cognitive profile based on a new interaction"""
        profile = self.get_or_create_profile(user_id)

        # Analyze the interaction
        analysis = self.analyzer.analyze_interaction(user_input, response, context)

        # Update cognitive style metrics with exponential moving average
        alpha = 0.1  # Learning rate
        profile.analytical_tendency = (1 - alpha) * profile.analytical_tendency + alpha * analysis['analytical_score']
        profile.intuitive_tendency = (1 - alpha) * profile.intuitive_tendency + alpha * analysis['intuitive_score']
        profile.systematic_tendency = (1 - alpha) * profile.systematic_tendency + alpha * analysis['systematic_score']
        profile.creative_tendency = (1 - alpha) * profile.creative_tendency + alpha * analysis['creative_score']
        profile.detail_orientation = (1 - alpha) * profile.detail_orientation + alpha * analysis['detail_score']
        profile.big_picture_orientation = (1 - alpha) * profile.big_picture_orientation + alpha * analysis['big_picture_score']

        # Update question style
        question_type = analysis['question_type']
        if question_type in ['definitional', 'causal']:
            profile.question_asking_style = 'analytical'
        elif question_type in ['hypothetical', 'exploratory']:
            profile.question_asking_style = 'exploratory'
        elif question_type == 'procedural':
            profile.question_asking_style = 'systematic'

        # Update concept connections
        concepts = analysis['concepts_mentioned']
        for i, concept in enumerate(concepts):
            if concept not in profile.concept_connections:
                profile.concept_connections[concept] = set()
            # Connect to nearby concepts
            for j, other_concept in enumerate(concepts):
                if i != j:
                    profile.concept_connections[concept].add(other_concept)

        # Update topic distribution
        for concept in concepts:
            profile.topic_distribution[concept] = profile.topic_distribution.get(concept, 0) + 1

        # Normalize topic distribution
        total = sum(profile.topic_distribution.values())
        if total > 0:
            profile.topic_distribution = {
                k: v / total for k, v in profile.topic_distribution.items()
            }

        # Update metadata
        profile.total_interactions += 1
        profile.updated_at = datetime.now()

        # Store in buffer for pattern detection
        self.interaction_buffer[user_id].append({
            'input': user_input,
            'response': response,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })

        # Detect reasoning patterns periodically
        if len(self.interaction_buffer[user_id]) >= 10:
            self._detect_reasoning_patterns(profile, self.interaction_buffer[user_id])
            self.interaction_buffer[user_id] = self.interaction_buffer[user_id][-5:]  # Keep recent

        # Recompute embedding
        profile.compute_embedding(self.analyzer.encoder)

        return profile

    def _detect_reasoning_patterns(
        self,
        profile: CognitiveProfile,
        interactions: List[Dict]
    ):
        """Detect and update reasoning patterns from interaction history"""
        # Analyze concept transition patterns
        concept_transitions: Dict[Tuple[str, str], int] = defaultdict(int)

        for i in range(len(interactions) - 1):
            curr_concepts = interactions[i]['analysis']['concepts_mentioned']
            next_concepts = interactions[i + 1]['analysis']['concepts_mentioned']

            for c1 in curr_concepts:
                for c2 in next_concepts:
                    if c1 != c2:
                        concept_transitions[(c1, c2)] += 1

        # Find dominant transition patterns
        if concept_transitions:
            top_transitions = sorted(
                concept_transitions.items(),
                key=lambda x: -x[1]
            )[:5]

            # Create or update reasoning pattern
            avg_analytical = np.mean([i['analysis']['analytical_score'] for i in interactions])
            avg_creative = np.mean([i['analysis']['creative_score'] for i in interactions])

            if avg_analytical > 0.5:
                pattern_type = "analytical"
            elif avg_creative > 0.5:
                pattern_type = "creative"
            else:
                pattern_type = "balanced"

            pattern_id = hashlib.md5(
                str(top_transitions).encode()
            ).hexdigest()[:8]

            trigger_concepts = set()
            for (c1, c2), _ in top_transitions:
                trigger_concepts.add(c1)

            pattern = ReasoningPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                trigger_concepts=trigger_concepts,
                typical_transitions=[t[0] for t in top_transitions],
                confidence=min(len(interactions) / 20, 1.0),
                frequency=len(interactions),
                examples=[
                    {'input': i['input'], 'response': i['response']}
                    for i in interactions[:3]
                ]
            )

            profile.reasoning_patterns[pattern_id] = pattern

    def get_cognitive_summary(self, user_id: str) -> str:
        """Generate a human-readable summary of a user's cognitive profile"""
        profile = self.get_or_create_profile(user_id)

        summary_parts = [f"Cognitive Profile for {user_id}:"]
        summary_parts.append(f"  Total interactions: {profile.total_interactions}")

        # Cognitive style
        style_parts = []
        if profile.analytical_tendency > 0.6:
            style_parts.append("analytical")
        if profile.intuitive_tendency > 0.6:
            style_parts.append("intuitive")
        if profile.systematic_tendency > 0.6:
            style_parts.append("systematic")
        if profile.creative_tendency > 0.6:
            style_parts.append("creative")

        if style_parts:
            summary_parts.append(f"  Cognitive style: {', '.join(style_parts)}")

        # Top domains
        top_domains = sorted(
            profile.topic_distribution.items(),
            key=lambda x: -x[1]
        )[:5]
        if top_domains:
            domains = [d[0] for d in top_domains]
            summary_parts.append(f"  Top interests: {', '.join(domains)}")

        # Reasoning patterns
        if profile.reasoning_patterns:
            patterns = list(profile.reasoning_patterns.values())
            pattern_types = [p.pattern_type for p in patterns]
            summary_parts.append(f"  Reasoning patterns: {', '.join(set(pattern_types))}")

        return "\n".join(summary_parts)

    def save_profiles(self):
        """Save all profiles to disk"""
        for user_id, profile in self.profiles.items():
            profile_path = self.save_path / f"{user_id}.json"
            with open(profile_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)

    def _load_profiles(self):
        """Load profiles from disk"""
        for profile_path in self.save_path.glob("*.json"):
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                profile = CognitiveProfile.from_dict(data)
                self.profiles[profile.profile_id] = profile
            except Exception as e:
                print(f"Warning: Could not load profile {profile_path}: {e}")


class CognitiveComposer:
    """
    Synthesizes multiple cognitive profiles to create novel reasoning approaches.

    This enables:
    - Combining different cognitive styles to solve problems
    - Generating reasoning strategies no single person has developed
    - Translating insights between different cognitive frameworks
    """

    def __init__(self, profiler: CognitiveProfiler):
        self.profiler = profiler
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass

    def compute_compatibility(
        self,
        profile1: CognitiveProfile,
        profile2: CognitiveProfile
    ) -> float:
        """
        Compute cognitive compatibility (phi-like metric) between profiles.

        Higher compatibility means the profiles could integrate well together.
        Lower compatibility means they might conflict or not mesh.
        """
        # Style compatibility
        style_diff = 0.0
        style_diff += abs(profile1.analytical_tendency - profile2.analytical_tendency)
        style_diff += abs(profile1.intuitive_tendency - profile2.intuitive_tendency)
        style_diff += abs(profile1.systematic_tendency - profile2.systematic_tendency)
        style_diff += abs(profile1.creative_tendency - profile2.creative_tendency)
        style_compatibility = 1.0 - (style_diff / 4.0)

        # Concept overlap
        concepts1 = set(profile1.topic_distribution.keys())
        concepts2 = set(profile2.topic_distribution.keys())
        if concepts1 or concepts2:
            overlap = len(concepts1.intersection(concepts2))
            union = len(concepts1.union(concepts2))
            concept_compatibility = overlap / max(union, 1)
        else:
            concept_compatibility = 0.5

        # Embedding similarity
        embedding_similarity = profile1.similarity_to(profile2)

        # Weighted combination
        compatibility = (
            0.3 * style_compatibility +
            0.3 * concept_compatibility +
            0.4 * embedding_similarity
        )

        return compatibility

    def compose_profiles(
        self,
        profiles: List[CognitiveProfile],
        weights: Optional[List[float]] = None
    ) -> CognitiveProfile:
        """
        Create a synthetic cognitive profile by combining multiple profiles.

        This is the core of the "cognitive composability" feature.
        """
        if not profiles:
            return CognitiveProfile(profile_id="synthetic_empty")

        if weights is None:
            weights = [1.0 / len(profiles)] * len(profiles)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Create synthetic profile
        synthetic = CognitiveProfile(
            profile_id=f"synthetic_{'_'.join(p.profile_id[:4] for p in profiles)}"
        )

        # Weighted average of cognitive styles
        synthetic.analytical_tendency = sum(
            p.analytical_tendency * w for p, w in zip(profiles, weights)
        )
        synthetic.intuitive_tendency = sum(
            p.intuitive_tendency * w for p, w in zip(profiles, weights)
        )
        synthetic.systematic_tendency = sum(
            p.systematic_tendency * w for p, w in zip(profiles, weights)
        )
        synthetic.creative_tendency = sum(
            p.creative_tendency * w for p, w in zip(profiles, weights)
        )
        synthetic.detail_orientation = sum(
            p.detail_orientation * w for p, w in zip(profiles, weights)
        )
        synthetic.big_picture_orientation = sum(
            p.big_picture_orientation * w for p, w in zip(profiles, weights)
        )

        # Merge concept connections
        for profile, weight in zip(profiles, weights):
            for concept, connections in profile.concept_connections.items():
                if concept not in synthetic.concept_connections:
                    synthetic.concept_connections[concept] = set()
                synthetic.concept_connections[concept].update(connections)

        # Merge topic distributions (weighted)
        for profile, weight in zip(profiles, weights):
            for topic, score in profile.topic_distribution.items():
                synthetic.topic_distribution[topic] = (
                    synthetic.topic_distribution.get(topic, 0) + score * weight
                )

        # Merge reasoning patterns (take most frequent from each)
        for profile in profiles:
            for pattern_id, pattern in profile.reasoning_patterns.items():
                if pattern_id not in synthetic.reasoning_patterns:
                    synthetic.reasoning_patterns[pattern_id] = pattern
                elif pattern.frequency > synthetic.reasoning_patterns[pattern_id].frequency:
                    synthetic.reasoning_patterns[pattern_id] = pattern

        # Compute embedding for the synthetic profile
        synthetic.compute_embedding(self.encoder)

        return synthetic

    def generate_novel_reasoning_approach(
        self,
        problem_context: str,
        available_profiles: List[CognitiveProfile]
    ) -> Dict[str, Any]:
        """
        Generate a novel reasoning approach by analyzing which profile combinations
        might be most effective for a given problem.

        This is the foundation for "novel reasoning generation".
        """
        if not available_profiles:
            return {'approach': 'default', 'profiles': [], 'strategy': 'generic'}

        # Analyze the problem to determine what cognitive styles might help
        analyzer = ReasoningPatternAnalyzer()
        problem_analysis = analyzer.analyze_interaction(problem_context, "", None)

        # Score each profile for relevance
        profile_scores = []
        for profile in available_profiles:
            score = 0.0

            # Match cognitive style to problem needs
            if problem_analysis['analytical_score'] > 0.5:
                score += profile.analytical_tendency * 0.3
            if problem_analysis['creative_score'] > 0.5:
                score += profile.creative_tendency * 0.3
            if problem_analysis['systematic_score'] > 0.5:
                score += profile.systematic_tendency * 0.3

            # Match concepts
            problem_concepts = set(problem_analysis['concepts_mentioned'])
            profile_concepts = set(profile.topic_distribution.keys())
            concept_overlap = len(problem_concepts.intersection(profile_concepts))
            score += concept_overlap * 0.1

            profile_scores.append((profile, score))

        # Sort by score and take top profiles
        profile_scores.sort(key=lambda x: -x[1])
        selected_profiles = [ps[0] for ps in profile_scores[:3]]

        # Create weights based on scores
        total_score = sum(ps[1] for ps in profile_scores[:3])
        if total_score > 0:
            weights = [ps[1] / total_score for ps in profile_scores[:3]]
        else:
            weights = [1/3, 1/3, 1/3]

        # Compose the selected profiles
        synthetic = self.compose_profiles(selected_profiles, weights)

        # Generate strategy description
        strategy_parts = []
        if synthetic.analytical_tendency > 0.6:
            strategy_parts.append("systematic analysis")
        if synthetic.creative_tendency > 0.6:
            strategy_parts.append("creative exploration")
        if synthetic.intuitive_tendency > 0.6:
            strategy_parts.append("intuitive connections")

        strategy = " combined with ".join(strategy_parts) if strategy_parts else "balanced approach"

        return {
            'approach': 'synthesized',
            'profiles': [p.profile_id for p in selected_profiles],
            'weights': weights,
            'synthetic_profile': synthetic,
            'strategy': strategy,
            'compatibility_scores': [
                self.compute_compatibility(selected_profiles[0], p)
                for p in selected_profiles[1:]
            ] if len(selected_profiles) > 1 else []
        }


class InsightTranslator:
    """
    Translates insights and explanations to match a user's cognitive profile.

    This enables personalized explanations that resonate with how each user thinks.
    """

    def __init__(self, profiler: CognitiveProfiler):
        self.profiler = profiler

    def translate_insight(
        self,
        insight: str,
        target_profile: CognitiveProfile
    ) -> str:
        """
        Translate an insight to match the target user's cognitive style.

        This adapts:
        - Level of detail vs big picture
        - Use of examples vs formal definitions
        - Analytical vs intuitive framing
        """
        # Analyze the insight
        analyzer = ReasoningPatternAnalyzer()
        insight_analysis = analyzer.analyze_interaction(insight, "", None)

        translated = insight

        # Add detail if user prefers it
        if target_profile.detail_orientation > 0.6:
            if not any(word in insight.lower() for word in ['specifically', 'exactly', 'for example']):
                translated = f"Specifically, {insight.lower()}"

        # Add big picture framing if user prefers it
        elif target_profile.big_picture_orientation > 0.6:
            if not any(word in insight.lower() for word in ['overall', 'fundamentally', 'essentially']):
                translated = f"Fundamentally, {insight.lower()}"

        # Add analytical framing for analytical users
        if target_profile.analytical_tendency > 0.6:
            if not any(word in insight.lower() for word in ['because', 'therefore', 'this means']):
                translated = f"{translated} This follows logically from the core principles."

        # Add intuitive framing for intuitive users
        elif target_profile.intuitive_tendency > 0.6:
            if not any(word in insight.lower() for word in ['feels', 'sense', 'seems']):
                translated = f"You might sense that {translated.lower()}"

        return translated

    def generate_personalized_explanation(
        self,
        topic: str,
        user_id: str,
        base_explanation: str
    ) -> str:
        """
        Generate a personalized explanation based on user's cognitive profile.
        """
        profile = self.profiler.get_or_create_profile(user_id)

        # Start with the base explanation
        explanation = base_explanation

        # Adapt based on question asking style
        if profile.question_asking_style == 'analytical':
            explanation = f"{explanation}\n\nThe key reasoning here is that "
        elif profile.question_asking_style == 'exploratory':
            explanation = f"{explanation}\n\nThis opens up interesting questions like "
        elif profile.question_asking_style == 'systematic':
            explanation = f"{explanation}\n\nTo break this down step by step: "

        # Reference familiar concepts from their profile
        familiar_concepts = list(profile.topic_distribution.keys())[:5]
        if familiar_concepts:
            concept = familiar_concepts[0]
            explanation = f"{explanation}\n\nYou might relate this to your interest in {concept}."

        return explanation
