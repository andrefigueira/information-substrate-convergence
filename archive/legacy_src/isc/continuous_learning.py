"""
Continuous Cross-Domain Learning for ISC

Implements never-forgetting continuous learning that accumulates knowledge
across all domains without catastrophic forgetting.

Based on concepts from:
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks
- Nested Learning approaches

Key features:
- Domain-aware memory consolidation
- Importance-weighted parameter updates
- Cross-domain knowledge transfer
- Semantic memory persistence
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import json
from pathlib import Path
import hashlib


@dataclass
class DomainKnowledge:
    """Represents accumulated knowledge in a specific domain"""
    domain_id: str
    concepts: Set[str] = field(default_factory=set)
    relationships: Dict[Tuple[str, str], float] = field(default_factory=dict)
    exemplars: List[Dict[str, Any]] = field(default_factory=list)
    importance_weights: Dict[str, float] = field(default_factory=dict)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    consolidation_level: float = 0.0  # 0-1, higher = more consolidated

    def update_access(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Consolidation increases with access
        self.consolidation_level = min(
            1.0,
            self.consolidation_level + 0.1 * (1 - self.consolidation_level)
        )

    def to_dict(self) -> Dict:
        return {
            'domain_id': self.domain_id,
            'concepts': list(self.concepts),
            'relationships': {f"{k[0]}|{k[1]}": v for k, v in self.relationships.items()},
            'exemplars': self.exemplars[-20:],  # Keep last 20
            'importance_weights': self.importance_weights,
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'consolidation_level': self.consolidation_level
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DomainKnowledge':
        dk = cls(domain_id=data['domain_id'])
        dk.concepts = set(data.get('concepts', []))
        dk.relationships = {
            tuple(k.split('|')): v
            for k, v in data.get('relationships', {}).items()
        }
        dk.exemplars = data.get('exemplars', [])
        dk.importance_weights = data.get('importance_weights', {})
        dk.last_accessed = datetime.fromisoformat(
            data.get('last_accessed', datetime.now().isoformat())
        )
        dk.access_count = data.get('access_count', 0)
        dk.consolidation_level = data.get('consolidation_level', 0.0)
        return dk


class ImportanceWeightedMemory:
    """
    Memory system that weights information by importance.

    Prevents catastrophic forgetting by:
    1. Computing importance of each piece of knowledge
    2. Protecting important knowledge during updates
    3. Allowing less important knowledge to be updated/replaced
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memories: Dict[str, Dict[str, Any]] = {}  # memory_id -> memory
        self.importance_scores: Dict[str, float] = {}  # memory_id -> importance

    def add_memory(
        self,
        memory_id: str,
        content: Dict[str, Any],
        initial_importance: float = 0.5
    ):
        """Add a new memory with importance tracking"""
        if len(self.memories) >= self.capacity:
            # Remove least important memory
            least_important = min(
                self.importance_scores.items(),
                key=lambda x: x[1]
            )[0]
            del self.memories[least_important]
            del self.importance_scores[least_important]

        self.memories[memory_id] = content
        self.importance_scores[memory_id] = initial_importance

    def access_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Access a memory, increasing its importance"""
        if memory_id in self.memories:
            # Increase importance on access
            self.importance_scores[memory_id] = min(
                1.0,
                self.importance_scores[memory_id] + 0.1
            )
            return self.memories[memory_id]
        return None

    def update_memory(
        self,
        memory_id: str,
        new_content: Dict[str, Any],
        blend_ratio: Optional[float] = None
    ):
        """Update a memory with importance-weighted blending"""
        if memory_id not in self.memories:
            self.add_memory(memory_id, new_content)
            return

        importance = self.importance_scores[memory_id]

        # High importance = preserve more of old content
        if blend_ratio is None:
            blend_ratio = 1.0 - importance

        old_content = self.memories[memory_id]
        blended = {}

        # Blend numeric values
        for key in set(old_content.keys()).union(new_content.keys()):
            old_val = old_content.get(key)
            new_val = new_content.get(key)

            if old_val is None:
                blended[key] = new_val
            elif new_val is None:
                blended[key] = old_val
            elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                blended[key] = (1 - blend_ratio) * old_val + blend_ratio * new_val
            else:
                # For non-numeric, prefer new if blend_ratio > 0.5
                blended[key] = new_val if blend_ratio > 0.5 else old_val

        self.memories[memory_id] = blended

    def decay_importance(self, decay_rate: float = 0.01):
        """Apply decay to all importance scores"""
        for memory_id in self.importance_scores:
            self.importance_scores[memory_id] = max(
                0.1,  # Minimum importance
                self.importance_scores[memory_id] * (1 - decay_rate)
            )


class CrossDomainKnowledgeGraph:
    """
    Graph structure that maintains knowledge across domains with cross-links.

    Enables:
    - Knowledge transfer between domains
    - Discovery of analogies and connections
    - Domain-agnostic reasoning patterns
    """

    def __init__(self):
        self.domains: Dict[str, DomainKnowledge] = {}
        self.cross_domain_links: Dict[Tuple[str, str, str, str], float] = {}
        # (domain1, concept1, domain2, concept2) -> strength

    def get_or_create_domain(self, domain_id: str) -> DomainKnowledge:
        """Get or create a domain knowledge container"""
        if domain_id not in self.domains:
            self.domains[domain_id] = DomainKnowledge(domain_id=domain_id)
        return self.domains[domain_id]

    def add_concept_to_domain(
        self,
        domain_id: str,
        concept: str,
        importance: float = 0.5
    ):
        """Add a concept to a domain"""
        domain = self.get_or_create_domain(domain_id)
        domain.concepts.add(concept)
        domain.importance_weights[concept] = importance
        domain.update_access()

    def add_relationship(
        self,
        domain_id: str,
        concept1: str,
        concept2: str,
        strength: float
    ):
        """Add a relationship between concepts in a domain"""
        domain = self.get_or_create_domain(domain_id)
        domain.relationships[(concept1, concept2)] = strength
        domain.concepts.add(concept1)
        domain.concepts.add(concept2)

    def add_cross_domain_link(
        self,
        domain1: str,
        concept1: str,
        domain2: str,
        concept2: str,
        strength: float
    ):
        """Add a link between concepts in different domains"""
        self.cross_domain_links[(domain1, concept1, domain2, concept2)] = strength
        # Also add reverse link
        self.cross_domain_links[(domain2, concept2, domain1, concept1)] = strength

    def find_analogies(
        self,
        source_domain: str,
        source_concept: str,
        target_domain: str
    ) -> List[Tuple[str, float]]:
        """Find analogous concepts in a target domain"""
        analogies = []

        for (d1, c1, d2, c2), strength in self.cross_domain_links.items():
            if d1 == source_domain and c1 == source_concept and d2 == target_domain:
                analogies.append((c2, strength))

        return sorted(analogies, key=lambda x: -x[1])

    def get_related_across_domains(
        self,
        concept: str,
        max_results: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Find related concepts across all domains"""
        related = []

        for (d1, c1, d2, c2), strength in self.cross_domain_links.items():
            if c1 == concept:
                related.append((d2, c2, strength))
            elif c2 == concept:
                related.append((d1, c1, strength))

        return sorted(related, key=lambda x: -x[2])[:max_results]

    def consolidate(self):
        """Consolidate knowledge by strengthening frequently accessed connections"""
        for domain in self.domains.values():
            # Strengthen relationships based on access
            for rel, strength in list(domain.relationships.items()):
                domain.relationships[rel] = min(
                    1.0,
                    strength * (1 + 0.1 * domain.consolidation_level)
                )

    def to_dict(self) -> Dict:
        return {
            'domains': {k: v.to_dict() for k, v in self.domains.items()},
            'cross_domain_links': {
                f"{d1}|{c1}|{d2}|{c2}": s
                for (d1, c1, d2, c2), s in self.cross_domain_links.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CrossDomainKnowledgeGraph':
        graph = cls()
        graph.domains = {
            k: DomainKnowledge.from_dict(v)
            for k, v in data.get('domains', {}).items()
        }
        graph.cross_domain_links = {
            tuple(k.split('|')): v
            for k, v in data.get('cross_domain_links', {}).items()
        }
        return graph


class ContinuousLearningEngine:
    """
    Main engine for continuous cross-domain learning.

    Implements:
    - Never-forgetting learning through importance weighting
    - Cross-domain knowledge transfer
    - Progressive knowledge accumulation
    - Semantic memory consolidation
    """

    def __init__(self, save_path: str = "models/continuous_learning"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.knowledge_graph = CrossDomainKnowledgeGraph()
        self.memory = ImportanceWeightedMemory(capacity=50000)
        self.learning_rate = 0.1
        self.consolidation_interval = 100  # Consolidate every N updates
        self.update_count = 0

        # Domain detection patterns
        self.domain_keywords: Dict[str, Set[str]] = {
            'science': {'physics', 'biology', 'chemistry', 'science', 'experiment', 'theory', 'hypothesis'},
            'technology': {'computer', 'software', 'code', 'programming', 'ai', 'algorithm', 'data'},
            'philosophy': {'consciousness', 'existence', 'meaning', 'ethics', 'truth', 'reality', 'mind'},
            'psychology': {'emotion', 'feeling', 'thought', 'behavior', 'cognition', 'personality', 'memory'},
            'art': {'creative', 'art', 'music', 'design', 'aesthetic', 'expression', 'beauty'},
            'social': {'relationship', 'society', 'culture', 'communication', 'community', 'interaction'},
            'practical': {'how', 'help', 'solve', 'problem', 'issue', 'need', 'want'}
        }

        # Load existing state
        self._load_state()

    def detect_domain(self, text: str) -> str:
        """Detect the primary domain of a piece of text"""
        text_words = set(text.lower().split())

        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            overlap = len(text_words.intersection(keywords))
            domain_scores[domain] = overlap

        if max(domain_scores.values()) == 0:
            return 'general'

        return max(domain_scores.items(), key=lambda x: x[1])[0]

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'to',
                      'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                      'into', 'through', 'during', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'what', 'which', 'who', 'this', 'that', 'and',
                      'but', 'or', 'if', 'because', 'so', 'than', 'too', 'very'}

        words = text.lower().split()
        concepts = [
            w.strip('.,!?;:"\'-')
            for w in words
            if len(w) > 3 and w.lower() not in stop_words
        ]
        return list(set(concepts))[:15]

    def learn(
        self,
        input_text: str,
        response_text: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Learn from an interaction without forgetting previous knowledge.

        Returns learning statistics.
        """
        # Detect domain
        domain = self.detect_domain(input_text)

        # Extract concepts
        input_concepts = self.extract_concepts(input_text)
        response_concepts = self.extract_concepts(response_text)
        all_concepts = list(set(input_concepts + response_concepts))

        # Update domain knowledge
        domain_knowledge = self.knowledge_graph.get_or_create_domain(domain)

        for concept in all_concepts:
            current_importance = domain_knowledge.importance_weights.get(concept, 0.5)
            # Increase importance for concepts that appear frequently
            new_importance = min(1.0, current_importance + self.learning_rate * 0.5)
            self.knowledge_graph.add_concept_to_domain(domain, concept, new_importance)

        # Add relationships between concepts that appear together
        for i, c1 in enumerate(all_concepts):
            for c2 in all_concepts[i+1:]:
                existing = domain_knowledge.relationships.get((c1, c2), 0.0)
                new_strength = min(1.0, existing + self.learning_rate)
                self.knowledge_graph.add_relationship(domain, c1, c2, new_strength)

        # Store exemplar
        exemplar = {
            'input': input_text[:200],
            'response': response_text[:200],
            'concepts': all_concepts,
            'timestamp': datetime.now().isoformat()
        }
        domain_knowledge.exemplars.append(exemplar)
        if len(domain_knowledge.exemplars) > 50:
            domain_knowledge.exemplars = domain_knowledge.exemplars[-50:]

        # Create cross-domain links
        self._create_cross_domain_links(domain, all_concepts)

        # Update memory with importance weighting
        memory_id = hashlib.md5(input_text.encode()).hexdigest()[:12]
        self.memory.add_memory(
            memory_id,
            {'input': input_text, 'response': response_text, 'domain': domain},
            initial_importance=0.5
        )

        # Periodic consolidation
        self.update_count += 1
        if self.update_count % self.consolidation_interval == 0:
            self.consolidate()

        return {
            'domain': domain,
            'concepts_learned': len(all_concepts),
            'relationships_updated': len(all_concepts) * (len(all_concepts) - 1) // 2,
            'total_domain_concepts': len(domain_knowledge.concepts),
            'consolidation_level': domain_knowledge.consolidation_level
        }

    def _create_cross_domain_links(self, current_domain: str, concepts: List[str]):
        """Create links between concepts across domains based on semantic similarity"""
        for other_domain_id, other_domain in self.knowledge_graph.domains.items():
            if other_domain_id == current_domain:
                continue

            for concept in concepts:
                # Check if concept exists in other domain
                if concept in other_domain.concepts:
                    # Same concept in multiple domains = strong link
                    self.knowledge_graph.add_cross_domain_link(
                        current_domain, concept,
                        other_domain_id, concept,
                        strength=0.8
                    )

                # Check for partial matches (simple substring matching)
                for other_concept in other_domain.concepts:
                    if concept in other_concept or other_concept in concept:
                        self.knowledge_graph.add_cross_domain_link(
                            current_domain, concept,
                            other_domain_id, other_concept,
                            strength=0.4
                        )

    def recall(
        self,
        query: str,
        target_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recall relevant knowledge for a query.

        Uses cross-domain connections to find related information.
        """
        query_concepts = self.extract_concepts(query)
        query_domain = target_domain or self.detect_domain(query)

        results = {
            'primary_domain': query_domain,
            'related_concepts': [],
            'cross_domain_connections': [],
            'exemplars': []
        }

        # Get concepts from query domain
        if query_domain in self.knowledge_graph.domains:
            domain = self.knowledge_graph.domains[query_domain]
            domain.update_access()

            # Find related concepts
            for concept in query_concepts:
                if concept in domain.concepts:
                    # Get relationships
                    for (c1, c2), strength in domain.relationships.items():
                        if c1 == concept or c2 == concept:
                            related = c2 if c1 == concept else c1
                            results['related_concepts'].append((related, strength))

            # Get exemplars
            results['exemplars'] = domain.exemplars[-5:]

        # Get cross-domain connections
        for concept in query_concepts:
            cross_domain = self.knowledge_graph.get_related_across_domains(concept, max_results=5)
            for domain_id, related_concept, strength in cross_domain:
                results['cross_domain_connections'].append({
                    'domain': domain_id,
                    'concept': related_concept,
                    'strength': strength
                })

        # Sort and deduplicate
        results['related_concepts'] = sorted(
            list(set(results['related_concepts'])),
            key=lambda x: -x[1]
        )[:10]

        return results

    def consolidate(self):
        """Consolidate knowledge across all domains"""
        # Consolidate knowledge graph
        self.knowledge_graph.consolidate()

        # Apply decay to memory importance
        self.memory.decay_importance(decay_rate=0.01)

        # Save state
        self._save_state()

    def get_domain_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all domains"""
        stats = {}
        for domain_id, domain in self.knowledge_graph.domains.items():
            stats[domain_id] = {
                'concept_count': len(domain.concepts),
                'relationship_count': len(domain.relationships),
                'exemplar_count': len(domain.exemplars),
                'consolidation_level': domain.consolidation_level,
                'access_count': domain.access_count
            }
        return stats

    def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str,
        concept: str
    ) -> Optional[str]:
        """
        Transfer a concept's understanding from one domain to another.

        Returns an analogy or related concept in the target domain.
        """
        analogies = self.knowledge_graph.find_analogies(
            source_domain, concept, target_domain
        )

        if analogies:
            return analogies[0][0]  # Return best analogy
        return None

    def _save_state(self):
        """Save learning state to disk"""
        state = {
            'knowledge_graph': self.knowledge_graph.to_dict(),
            'update_count': self.update_count
        }

        state_path = self.save_path / "continuous_learning_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load learning state from disk"""
        state_path = self.save_path / "continuous_learning_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.knowledge_graph = CrossDomainKnowledgeGraph.from_dict(
                    state.get('knowledge_graph', {})
                )
                self.update_count = state.get('update_count', 0)
            except Exception as e:
                print(f"Warning: Could not load continuous learning state: {e}")
