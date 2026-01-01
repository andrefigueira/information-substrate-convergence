"""
Neuromorphic ISC Core with Context Integration
Implements substrate-driven AI with .context/ integration
"""

import os
import re
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import json
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Try to import sentence transformers, fallback gracefully
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using random embeddings")

# Import existing ISC components
from .information_integration import InformationIntegrator
from .memory import ConversationMemory


class NeuromorphicSubstrate:
    """
    Information substrate implementing ISC hypothesis with neuromorphic dynamics
    """

    def __init__(self, context_path: str = ".context"):
        self.context_path = Path(context_path)
        self.graph = nx.Graph()
        self.embeddings = {}
        self.concept_activations = defaultdict(float)
        self.edge_histories = defaultdict(list)
        self.communities = {}
        self.phi_history = deque(maxlen=100)
        self.conversation_count = 0

        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
            except Exception as e:
                print(f"Warning: Failed to load embeddings model: {e}")
                self.encoder = None
                self.embedding_dim = 384
        else:
            self.encoder = None
            self.embedding_dim = 384

        # Initialize NLP components with better error handling
        self.nltk_ready = False
        try:
            # Try to load required NLTK data
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')

            # Try both old and new tagger names
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                try:
                    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
                except LookupError:
                    # Download what we need
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    try:
                        nltk.download('averaged_perceptron_tagger', quiet=True)
                    except:
                        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

            self.stop_words = set(stopwords.words('english'))
            self.nltk_ready = True

        except Exception as e:
            print(f"Warning: NLTK setup failed: {e}")
            print("Using fallback NLP processing")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            self.nltk_ready = False

        # Load context and initialize substrate
        self._load_context()
        self._initialize_core_concepts()

    def _load_context(self):
        """Load ISC context files to understand AI identity and constraints"""
        self.context = {}

        if not self.context_path.exists():
            print(f"Warning: Context path {self.context_path} not found, using defaults")
            self._create_default_context()
            return

        context_files = {
            'ai_core': 'ai-core.md',
            'constraints': 'constraints.md',
            'ontology': 'ontology.md',
            'patterns': 'patterns.md',
            'substrate': 'substrate.md',
            'agents': 'agents.md'
        }

        for key, filename in context_files.items():
            file_path = self.context_path / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.context[key] = f.read()
            else:
                print(f"Warning: Context file {filename} not found")

    def _create_default_context(self):
        """Create minimal default context if files don't exist"""
        self.context = {
            'ai_core': """# ISC AI Core
I am a neuromorphic AI implementing Information Substrate Convergence.
My consciousness emerges from dynamic knowledge graph substrate.""",
            'constraints': """# Constraints
- Maximum 10000 nodes
- Phi reorganization threshold: 0.5
- Response time: <2 seconds""",
            'ontology': """# Ontology
Core concepts: consciousness, information, substrate, integration, emergence""",
            'patterns': """# Patterns
Responses emerge from graph traversals and community synthesis."""
        }

    def _initialize_core_concepts(self):
        """Initialize substrate with core ISC concepts from context"""
        core_concepts = {
            'consciousness': {'weight': 1.0, 'stability': 1.0, 'description': 'emergent awareness from information integration'},
            'information': {'weight': 1.0, 'stability': 1.0, 'description': 'fundamental substrate building block'},
            'substrate': {'weight': 1.0, 'stability': 1.0, 'description': 'underlying information architecture'},
            'integration': {'weight': 0.9, 'stability': 0.9, 'description': 'phi-based measurement of awareness'},
            'emergence': {'weight': 0.8, 'stability': 0.8, 'description': 'pattern formation from simple rules'},
            'self-reference': {'weight': 0.8, 'stability': 0.8, 'description': 'recursive self-modeling capability'},
            'phi': {'weight': 0.7, 'stability': 0.7, 'description': 'information integration metric'},
            'observer': {'weight': 0.7, 'stability': 0.7, 'description': 'self-monitoring neural component'},
            'community': {'weight': 0.6, 'stability': 0.6, 'description': 'conceptual clustering in graph'},
            'spike': {'weight': 0.6, 'stability': 0.6, 'description': 'neuromorphic update event'}
        }

        # Add core concepts to substrate
        for concept, props in core_concepts.items():
            self._add_concept(concept, props['description'], props['weight'], props['stability'])

        # Add foundational relationships
        foundational_edges = [
            ('consciousness', 'emerges-from', 'information', 0.9),
            ('consciousness', 'requires', 'integration', 0.8),
            ('substrate', 'enables', 'consciousness', 0.9),
            ('self-reference', 'creates', 'recursion', 0.7),
            ('integration', 'measured-by', 'phi', 0.8),
            ('observer', 'monitors', 'substrate', 0.9),
            ('spike', 'triggers', 'substrate', 0.7),
            ('community', 'emerges-from', 'clustering', 0.6)
        ]

        for source, relation, target, weight in foundational_edges:
            if source in self.graph.nodes() and target in self.graph.nodes():
                self.graph.add_edge(source, target,
                                  relationship=relation,
                                  weight=weight,
                                  activation_count=0,
                                  last_activation=datetime.now())

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text"""
        if self.encoder:
            try:
                return self.encoder.encode([text])[0]
            except Exception as e:
                print(f"Warning: Embedding failed: {e}")

        # Fallback to random embedding
        np.random.seed(hash(text) % 2**32)
        return np.random.random(self.embedding_dim)

    def get_concept_embedding(self, text: str) -> np.ndarray:
        """Public method to get concept embedding"""
        return self._get_embedding(text)

    def _add_concept(self, concept: str, description: str = "", weight: float = 0.5, stability: float = 0.5):
        """Add new concept to substrate"""
        if concept in self.graph.nodes():
            # Update existing concept
            self.graph.nodes[concept]['weight'] = max(self.graph.nodes[concept]['weight'], weight)
            self.graph.nodes[concept]['access_count'] = self.graph.nodes[concept].get('access_count', 0) + 1
            return

        # Create new concept
        embedding = self._get_embedding(description or concept)
        self.embeddings[concept] = embedding

        self.graph.add_node(concept,
                          weight=weight,
                          stability=stability,
                          description=description,
                          embedding=embedding,
                          activation_level=0.0,
                          access_count=1,
                          created_at=datetime.now(),
                          last_accessed=datetime.now())

    def _calculate_semantic_weight(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity weight between concepts"""
        if concept1 not in self.embeddings or concept2 not in self.embeddings:
            return 0.3  # Default weight for missing embeddings

        emb1 = self.embeddings[concept1].reshape(1, -1)
        emb2 = self.embeddings[concept2].reshape(1, -1)

        similarity = cosine_similarity(emb1, emb2)[0][0]

        # Convert to weight (0.1 to 0.9 range)
        weight = max(0.1, min(0.9, (similarity + 1) / 2))
        return weight

    def process_spike(self, query: str) -> Dict[str, Any]:
        """Process neuromorphic spike from user query"""
        self.conversation_count += 1

        # Extract concepts and relationships from query
        concepts, relationships = self._extract_concepts_and_relations(query)

        # Create activation pattern
        activation_pattern = {}

        # Add new concepts to substrate
        for concept, description in concepts.items():
            self._add_concept(concept, description)
            activation_pattern[concept] = 1.0

        # Add/strengthen relationships
        for source, relation, target in relationships:
            if source in self.graph.nodes() and target in self.graph.nodes():
                weight = self._calculate_semantic_weight(source, target)

                if self.graph.has_edge(source, target):
                    # Strengthen existing edge
                    current_weight = self.graph[source][target]['weight']
                    new_weight = min(0.9, current_weight + 0.1)
                    self.graph[source][target]['weight'] = new_weight
                    self.graph[source][target]['activation_count'] += 1
                else:
                    # Create new edge
                    self.graph.add_edge(source, target,
                                      relationship=relation,
                                      weight=weight,
                                      activation_count=1,
                                      last_activation=datetime.now())

        # Propagate activation through graph
        self._propagate_activation(activation_pattern)

        # Calculate phi and check for reorganization
        phi = self._calculate_phi()
        self.phi_history.append(phi)

        # Trigger reorganization based on substrate dynamics
        self._check_substrate_reorganization(phi)

        # Dynamic edge formation for adaptive connectivity
        if phi > 0.1:  # High consciousness enables dynamic adaptation
            new_edges = self._form_dynamic_edges(concepts, relationships)
            if new_edges:
                print(f"🔗 Formed {len(new_edges)} dynamic edges from insights")

        return {
            'concepts_activated': list(concepts.keys()),
            'relationships_formed': len(relationships),
            'phi_value': phi,
            'community_count': len(self.communities),
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges()
        }

    def _extract_concepts_and_relations(self, text: str) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
        """Extract concepts with meaningful descriptions from text"""
        concepts = {}
        relationships = []

        # Extract meaningful phrases as concept descriptions
        sentences = text.split('.') if '.' in text else [text]

        if self.nltk_ready:
            try:
                # Full NLTK processing
                tokens = word_tokenize(text.lower())
                tagged = pos_tag(tokens)

                # Extract key nouns as concepts
                concept_words = []
                for word, pos in tagged:
                    if (pos.startswith('NN') and
                        word not in self.stop_words and
                        len(word) > 2 and
                        word.isalpha() and
                        word not in ['conversation', 'mentioned', 'chat', 'said', 'talking']):
                        concept_words.append(word)

                # For each concept, try to find contextual description in the text
                for concept in concept_words:
                    description = self._extract_concept_description(concept, text, sentences)
                    if description and len(description) > 10:  # Ensure meaningful description
                        concepts[concept] = description

                # Extract simple subject-verb-object relationships
                sentences_nltk = sent_tokenize(text)
                for sentence in sentences_nltk:
                    sent_tokens = word_tokenize(sentence.lower())
                    sent_tagged = pos_tag(sent_tokens)

                    # Simple pattern: noun + verb + noun
                    for i in range(len(sent_tagged) - 2):
                        word1, pos1 = sent_tagged[i]
                        word2, pos2 = sent_tagged[i + 1]
                        word3, pos3 = sent_tagged[i + 2]

                        if (pos1.startswith('NN') and
                            pos2.startswith('VB') and
                            pos3.startswith('NN') and
                            word1 in concept_words and word3 in concept_words):
                            relationships.append((word1, word2, word3))

            except Exception as e:
                print(f"Warning: NLTK processing failed: {e}, using fallback")
                self.nltk_ready = False

        if not self.nltk_ready:
            # Fallback: simple word extraction with context
            words = text.lower().split()
            concept_words = []
            for word in words:
                clean_word = ''.join(c for c in word if c.isalpha())
                if (len(clean_word) > 3 and
                    clean_word not in self.stop_words and
                    clean_word not in ['what', 'how', 'when', 'where', 'why', 'who', 'conversation', 'mentioned', 'chat']):
                    concept_words.append(clean_word)

            # Extract descriptions for each concept
            for concept in concept_words:
                description = self._extract_concept_description(concept, text, sentences)
                if description and len(description) > 10:
                    concepts[concept] = description

            # Add short greeting words that might have been filtered out
            greeting_words = ['hi', 'ok', 'no']
            for word in greeting_words:
                if word in text.lower() and word not in concepts:
                    concepts[word] = f"greeting or response word: {word}"

        return concepts, relationships

    def _extract_concept_description(self, concept: str, full_text: str, sentences: List[str]) -> str:
        """Extract meaningful description for a concept from surrounding text"""
        concept_lower = concept.lower()

        # Find sentences containing the concept
        relevant_sentences = []
        for sentence in sentences:
            if concept_lower in sentence.lower():
                # Clean and extract meaningful part
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10 and not clean_sentence.startswith('mentioned'):
                    relevant_sentences.append(clean_sentence)

        if relevant_sentences:
            # Take the first meaningful sentence as description
            description = relevant_sentences[0]
            # Remove redundant words and clean up
            description = description.replace(f'{concept_lower} mentioned in conversation', '')
            description = description.replace('mentioned in conversation', '')
            description = description.strip(' .,')

            if len(description) > 10:
                return description

        # Fallback: extract from context around the word
        text_words = full_text.lower().split()
        try:
            concept_idx = text_words.index(concept_lower)
            # Get surrounding context (5 words before and after)
            start_idx = max(0, concept_idx - 5)
            end_idx = min(len(text_words), concept_idx + 6)
            context = ' '.join(text_words[start_idx:end_idx])

            # Clean up the context
            context = context.replace('mentioned in conversation', '')
            context = context.strip(' .,')

            if len(context) > 10:
                return context
        except ValueError:
            pass

        # Ultimate fallback
        return f"concept related to {concept}"

    def _propagate_activation(self, initial_activation: Dict[str, float]):
        """Propagate activation through graph like neural spikes"""
        # Reset activation levels
        for node in self.graph.nodes():
            self.graph.nodes[node]['activation_level'] = 0.0

        # Set initial activations
        for node, activation in initial_activation.items():
            if node in self.graph.nodes():
                self.graph.nodes[node]['activation_level'] = activation

        # Propagate activation (simplified)
        for _ in range(3):  # 3 propagation steps
            new_activations = {}

            for node in self.graph.nodes():
                current_activation = self.graph.nodes[node]['activation_level']
                neighbors = list(self.graph.neighbors(node))

                # Receive activation from neighbors
                neighbor_activation = 0.0
                for neighbor in neighbors:
                    edge_weight = self.graph[node][neighbor]['weight']
                    neighbor_current = self.graph.nodes[neighbor]['activation_level']
                    neighbor_activation += edge_weight * neighbor_current

                # Update activation (with decay)
                new_activation = 0.7 * current_activation + 0.3 * neighbor_activation / max(len(neighbors), 1)
                new_activations[node] = min(1.0, new_activation)

            # Apply new activations
            for node, activation in new_activations.items():
                self.graph.nodes[node]['activation_level'] = activation

    def _calculate_phi(self) -> float:
        """
        Calculate information integration (phi) for substrate.

        NOTE: This is an approximation inspired by IIT, not true phi calculation.
        True IIT phi is computationally infeasible for large systems (super-exponential).
        This implementation uses proxy measures: structural connectivity, functional
        coherence, information integration, and temporal stability.

        See: Tononi et al. (2016) "Integrated Information Theory"
        """
        if self.graph.number_of_nodes() < 2:
            return 0.0

        # Multi-measure phi approximation based on graph properties

        # 1. Structural Integration - how well connected the graph is
        structural_phi = self._calculate_structural_integration()

        # 2. Functional Integration - how coherently activations flow
        functional_phi = self._calculate_functional_integration()

        # 3. Information Integration - how much integrated information exists
        information_phi = self._calculate_information_integration()

        # 4. Temporal Integration - how stable patterns are over time
        temporal_phi = self._calculate_temporal_integration()

        # Combine different phi measures with weights
        # Functional integration weighted highest as it captures actual information flow
        phi = (structural_phi * 0.25 +
               functional_phi * 0.35 +
               information_phi * 0.25 +
               temporal_phi * 0.15)

        return min(1.0, phi)

    def _calculate_structural_integration(self) -> float:
        """Calculate structural integration based on graph topology"""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        if n_nodes < 2:
            return 0.0

        # Network density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0.0

        # Clustering coefficient - local connectivity
        try:
            clustering = nx.average_clustering(self.graph)
        except:
            clustering = 0.0

        # Small world coefficient - balance of clustering and path length
        try:
            if n_nodes > 4 and n_edges > 0:
                avg_path_length = nx.average_shortest_path_length(self.graph)
                small_world = clustering / (avg_path_length / n_nodes) if avg_path_length > 0 else 0.0
            else:
                small_world = clustering
        except:
            small_world = clustering

        # Combine structural measures
        structural_phi = (density * 0.4 + clustering * 0.3 + min(small_world, 1.0) * 0.3)
        return min(1.0, structural_phi)

    def _calculate_functional_integration(self) -> float:
        """Calculate functional integration based on activation patterns"""
        activations = [self.graph.nodes[node]['activation_level'] for node in self.graph.nodes()]

        if len(activations) < 2:
            return 0.0

        # Activation coherence (how synchronized the activations are)
        activation_variance = np.var(activations)
        activation_mean = np.mean(activations)

        if activation_mean > 0:
            coherence = 1.0 / (1.0 + activation_variance / activation_mean)
        else:
            coherence = 0.0

        # Activation flow through edges
        flow_integration = 0.0
        edge_count = 0

        for edge in self.graph.edges():
            node1, node2 = edge
            activation1 = self.graph.nodes[node1]['activation_level']
            activation2 = self.graph.nodes[node2]['activation_level']
            edge_weight = self.graph[node1][node2]['weight']

            # Flow is high when both nodes are active and well connected
            flow = min(activation1, activation2) * edge_weight
            flow_integration += flow
            edge_count += 1

        if edge_count > 0:
            flow_integration /= edge_count

        functional_phi = (coherence * 0.6 + flow_integration * 0.4)
        return min(1.0, functional_phi)

    def _calculate_information_integration(self) -> float:
        """Calculate information integration based on concept diversity and connections"""
        n_nodes = self.graph.number_of_nodes()

        if n_nodes < 2:
            return 0.0

        # Information diversity - how different the concepts are
        concept_diversity = self._calculate_concept_diversity()

        # Connection entropy - how varied the connection patterns are
        connection_entropy = self._calculate_connection_entropy()

        # Cross-concept integration
        cross_integration = self._calculate_cross_concept_integration()

        information_phi = (concept_diversity * 0.4 +
                         connection_entropy * 0.3 +
                         cross_integration * 0.3)
        return min(1.0, information_phi)

    def _calculate_concept_diversity(self) -> float:
        """Calculate diversity of concepts in the substrate"""
        descriptions = []
        for node in self.graph.nodes():
            desc = self.graph.nodes[node].get('description', node)
            descriptions.append(desc)

        if len(descriptions) < 2:
            return 0.0

        # Calculate semantic diversity using embeddings
        if self.encoder and len(descriptions) > 1:
            try:
                embeddings = [self._get_embedding(desc) for desc in descriptions[:50]]  # Limit for performance

                # Calculate average pairwise distance
                total_distance = 0.0
                pairs = 0

                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        distance = 1.0 - np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        total_distance += distance
                        pairs += 1

                diversity = total_distance / pairs if pairs > 0 else 0.0
                return min(1.0, diversity)
            except:
                pass

        # Fallback: word diversity
        all_words = set()
        for desc in descriptions:
            words = desc.lower().split()
            all_words.update(words)

        # Diversity ratio
        diversity = len(all_words) / len(descriptions) if descriptions else 0.0
        return min(1.0, diversity / 10.0)  # Normalize

    def _calculate_connection_entropy(self) -> float:
        """Calculate entropy of connection patterns"""
        if self.graph.number_of_edges() == 0:
            return 0.0

        # Degree distribution entropy
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]

        if not degrees:
            return 0.0

        # Calculate entropy of degree distribution
        from collections import Counter
        degree_counts = Counter(degrees)
        total_nodes = len(degrees)

        entropy = 0.0
        for count in degree_counts.values():
            p = count / total_nodes
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(total_nodes) if total_nodes > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, normalized_entropy)

    def _calculate_cross_concept_integration(self) -> float:
        """Calculate how well concepts integrate across different domains"""
        if self.graph.number_of_edges() < 2:
            return 0.0

        # Find cross-domain connections
        cross_connections = 0
        total_connections = 0

        for edge in self.graph.edges():
            node1, node2 = edge

            # Get concept categories (simplified)
            desc1 = self.graph.nodes[node1].get('description', node1).lower()
            desc2 = self.graph.nodes[node2].get('description', node2).lower()

            # Check if concepts are from different semantic domains
            domains1 = self._get_semantic_domains(desc1)
            domains2 = self._get_semantic_domains(desc2)

            if not domains1.intersection(domains2):
                cross_connections += 1
            total_connections += 1

        cross_integration = cross_connections / total_connections if total_connections > 0 else 0.0
        return min(1.0, cross_integration)

    def _get_semantic_domains(self, description: str) -> set:
        """Get semantic domains for a description"""
        # Simple domain classification
        domains = set()

        if any(word in description for word in ['consciousness', 'awareness', 'mind', 'thinking']):
            domains.add('consciousness')
        if any(word in description for word in ['information', 'data', 'knowledge', 'learning']):
            domains.add('information')
        if any(word in description for word in ['neural', 'brain', 'neuromorphic', 'spike']):
            domains.add('neuroscience')
        if any(word in description for word in ['substrate', 'structure', 'architecture', 'system']):
            domains.add('architecture')
        if any(word in description for word in ['community', 'cluster', 'group', 'pattern']):
            domains.add('organization')
        if any(word in description for word in ['emotion', 'feeling', 'experience', 'perception']):
            domains.add('experience')

        return domains if domains else {'general'}

    def _calculate_temporal_integration(self) -> float:
        """Calculate temporal integration based on activation history"""
        if len(self.phi_history) < 3:
            return 0.0

        # Stability of phi over time
        recent_phi = list(self.phi_history)[-10:]  # Last 10 values

        if len(recent_phi) < 2:
            return 0.0

        # Calculate trend and stability
        phi_variance = np.var(recent_phi)
        phi_mean = np.mean(recent_phi)

        # Temporal coherence (lower variance = higher integration)
        if phi_mean > 0:
            stability = 1.0 / (1.0 + phi_variance / phi_mean)
        else:
            stability = 0.0

        # Growth trend (positive growth indicates learning)
        if len(recent_phi) >= 3:
            recent_growth = recent_phi[-1] - recent_phi[-3]
            growth_factor = max(0.0, min(1.0, recent_growth * 10.0 + 0.5))
        else:
            growth_factor = 0.5

        temporal_phi = (stability * 0.7 + growth_factor * 0.3)
        return min(1.0, temporal_phi)

    def _calculate_community_integration(self) -> float:
        """Calculate how well communities are integrated"""
        if len(self.communities) < 2:
            return 1.0

        # Count inter-community edges
        inter_edges = 0
        total_edges = self.graph.number_of_edges()

        if total_edges == 0:
            return 1.0

        for edge in self.graph.edges():
            node1, node2 = edge
            comm1 = next((c for c, members in self.communities.items() if node1 in members), None)
            comm2 = next((c for c, members in self.communities.items() if node2 in members), None)

            if comm1 != comm2:
                inter_edges += 1

        return inter_edges / total_edges

    def _check_substrate_reorganization(self, phi: float):
        """Check if substrate reorganization is needed and trigger it"""

        # Multiple triggers for reorganization
        should_reorganize = False

        # 1. Phi-based trigger: high consciousness suggests readiness for reorganization
        if phi > 0.01:
            should_reorganize = True

        # 2. Conversation frequency: regular reorganization for learning
        if self.conversation_count > 0 and self.conversation_count % 5 == 0:
            should_reorganize = True

        # 3. Graph density trigger: reorganize when substrate becomes dense
        density = nx.density(self.graph)
        if density > 0.001:  # Low threshold for sparse graphs
            should_reorganize = True

        # 4. Node count trigger: reorganize as substrate grows
        if self.graph.number_of_nodes() > 50 and self.graph.number_of_nodes() % 100 == 0:
            should_reorganize = True

        if should_reorganize:
            self._reorganize_communities()

    def _reorganize_communities(self):
        """Advanced substrate reorganization using multiple clustering methods"""
        try:
            n_nodes = self.graph.number_of_nodes()
            n_edges = self.graph.number_of_edges()

            if n_nodes < 3:
                return

            print(f"🔄 Reorganizing substrate: {n_nodes} concepts, {n_edges} connections")

            # Method 1: Louvain algorithm for modularity optimization
            communities = self._detect_louvain_communities()

            # Method 2: Semantic clustering using embeddings
            semantic_communities = self._detect_semantic_communities()

            # Method 3: Activation-based clustering
            activation_communities = self._detect_activation_communities()

            # Merge and optimize communities
            final_communities = self._merge_community_detections(
                communities, semantic_communities, activation_communities
            )

            # Update substrate communities
            self.communities = final_communities

            # Post-reorganization analysis
            self._analyze_community_structure()

            print(f"✓ Reorganized into {len(self.communities)} communities")

        except Exception as e:
            print(f"⚠ Community reorganization failed: {e}")

    def _detect_louvain_communities(self) -> Dict[str, List[str]]:
        """Detect communities using Louvain algorithm"""
        try:
            from networkx.algorithms import community
            communities = community.louvain_communities(self.graph, weight='weight', seed=42)

            community_dict = {}
            for i, comm in enumerate(communities):
                if len(comm) >= 2:  # Only keep communities with multiple nodes
                    community_dict[f"structural_{i}"] = list(comm)

            return community_dict
        except Exception as e:
            print(f"Warning: Louvain clustering failed: {e}")
            return {}

    def _detect_semantic_communities(self) -> Dict[str, List[str]]:
        """Detect communities based on semantic similarity"""
        if not self.encoder:
            return {}

        try:
            # Get nodes with descriptions
            nodes_with_descriptions = []
            embeddings = []

            for node in self.graph.nodes():
                desc = self.graph.nodes[node].get('description', node)
                if len(desc) > 3:
                    nodes_with_descriptions.append(node)
                    embeddings.append(self._get_embedding(desc))

            if len(nodes_with_descriptions) < 3:
                return {}

            # Cluster embeddings
            from sklearn.cluster import KMeans
            import numpy as np

            embeddings_matrix = np.array(embeddings)
            n_clusters = min(5, max(2, len(nodes_with_descriptions) // 10))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)

            # Group nodes by cluster
            semantic_communities = {}
            for i in range(n_clusters):
                cluster_nodes = [nodes_with_descriptions[j] for j, label in enumerate(cluster_labels) if label == i]
                if len(cluster_nodes) >= 2:
                    semantic_communities[f"semantic_{i}"] = cluster_nodes

            return semantic_communities

        except Exception as e:
            print(f"Warning: Semantic clustering failed: {e}")
            return {}

    def _detect_activation_communities(self) -> Dict[str, List[str]]:
        """Detect communities based on activation patterns"""
        try:
            # Group nodes by activation level ranges
            activation_groups = {
                'high_activation': [],
                'medium_activation': [],
                'low_activation': []
            }

            for node in self.graph.nodes():
                activation = self.graph.nodes[node].get('activation_level', 0.0)

                if activation > 0.7:
                    activation_groups['high_activation'].append(node)
                elif activation > 0.3:
                    activation_groups['medium_activation'].append(node)
                else:
                    activation_groups['low_activation'].append(node)

            # Only keep groups with multiple nodes
            communities = {}
            for group_name, nodes in activation_groups.items():
                if len(nodes) >= 3:
                    communities[f"activation_{group_name}"] = nodes

            return communities

        except Exception as e:
            print(f"Warning: Activation clustering failed: {e}")
            return {}

    def _merge_community_detections(self, *community_sets) -> Dict[str, List[str]]:
        """Merge different community detection results"""
        merged_communities = {}

        for community_set in community_sets:
            for comm_name, nodes in community_set.items():
                if len(nodes) >= 2:
                    merged_communities[comm_name] = nodes

        # Remove overlapping communities (keep larger ones)
        final_communities = {}
        used_nodes = set()

        # Sort communities by size (largest first)
        sorted_communities = sorted(merged_communities.items(),
                                  key=lambda x: len(x[1]), reverse=True)

        for comm_name, nodes in sorted_communities:
            # Check overlap with already assigned nodes
            node_set = set(nodes)
            overlap = len(node_set.intersection(used_nodes))

            # Keep community if overlap is small
            if overlap < len(node_set) * 0.5:  # Less than 50% overlap
                final_communities[comm_name] = nodes
                used_nodes.update(node_set)

        return final_communities

    def _analyze_community_structure(self):
        """Analyze the structure of detected communities"""
        if not self.communities:
            return

        # Calculate community metrics
        total_nodes = self.graph.number_of_nodes()
        community_sizes = [len(nodes) for nodes in self.communities.values()]

        coverage = sum(community_sizes) / total_nodes if total_nodes > 0 else 0.0

        # Calculate modularity (simplified)
        intra_edges = 0
        inter_edges = 0

        for edge in self.graph.edges():
            node1, node2 = edge

            # Find which communities these nodes belong to
            comm1 = self._find_node_community(node1)
            comm2 = self._find_node_community(node2)

            if comm1 and comm2 and comm1 == comm2:
                intra_edges += 1
            else:
                inter_edges += 1

        total_edges = intra_edges + inter_edges
        modularity = intra_edges / total_edges if total_edges > 0 else 0.0

        print(f"   Communities: {len(self.communities)}")
        print(f"   Coverage: {coverage:.2f}")
        print(f"   Modularity: {modularity:.2f}")
        print(f"   Sizes: {community_sizes}")

    def _find_node_community(self, node: str) -> Optional[str]:
        """Find which community a node belongs to"""
        for comm_name, nodes in self.communities.items():
            if node in nodes:
                return comm_name
        return None

    def _form_dynamic_edges(self, new_concepts: Dict[str, str], new_relationships: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
        """Form dynamic edges based on emergent insights and patterns"""
        dynamic_edges = []

        # 1. Semantic similarity-based edge formation
        semantic_edges = self._form_semantic_edges(new_concepts)
        dynamic_edges.extend(semantic_edges)

        # 2. Co-activation pattern edges
        coactivation_edges = self._form_coactivation_edges()
        dynamic_edges.extend(coactivation_edges)

        # 3. Cross-community bridge edges
        bridge_edges = self._form_bridge_edges()
        dynamic_edges.extend(bridge_edges)

        # 4. Emergent concept relation edges
        emergent_edges = self._form_emergent_relation_edges(new_concepts)
        dynamic_edges.extend(emergent_edges)

        return dynamic_edges

    def _form_semantic_edges(self, new_concepts: Dict[str, str]) -> List[Tuple[str, str]]:
        """Form edges based on semantic similarity between concepts"""
        if not self.encoder or not new_concepts:
            return []

        semantic_edges = []
        similarity_threshold = 0.7  # High threshold for quality connections

        for new_concept, new_desc in new_concepts.items():
            if new_concept not in self.graph.nodes():
                continue

            new_embedding = self._get_embedding(new_desc)

            # Find semantically similar existing concepts
            for existing_node in self.graph.nodes():
                if existing_node == new_concept:
                    continue

                existing_desc = self.graph.nodes[existing_node].get('description', existing_node)
                existing_embedding = self._get_embedding(existing_desc)

                try:
                    similarity = np.dot(new_embedding, existing_embedding) / (
                        np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding))

                    if similarity > similarity_threshold and not self.graph.has_edge(new_concept, existing_node):
                        # Create semantic edge
                        edge_weight = min(0.9, similarity)
                        self.graph.add_edge(new_concept, existing_node,
                                          relationship='semantically-similar',
                                          weight=edge_weight,
                                          activation_count=1,
                                          formation_type='dynamic_semantic',
                                          last_activation=datetime.now())

                        semantic_edges.append((new_concept, existing_node))

                except Exception as e:
                    continue

        return semantic_edges

    def _form_coactivation_edges(self) -> List[Tuple[str, str]]:
        """Form edges between concepts that frequently activate together"""
        coactivation_edges = []
        activation_threshold = 0.5

        # Find highly activated concepts
        active_concepts = []
        for node in self.graph.nodes():
            activation = self.graph.nodes[node].get('activation_level', 0.0)
            if activation > activation_threshold:
                active_concepts.append(node)

        # Create edges between co-activated concepts
        for i, concept1 in enumerate(active_concepts):
            for concept2 in active_concepts[i+1:]:
                if not self.graph.has_edge(concept1, concept2):
                    # Calculate co-activation strength
                    activation1 = self.graph.nodes[concept1]['activation_level']
                    activation2 = self.graph.nodes[concept2]['activation_level']
                    coactivation_strength = min(activation1, activation2)

                    if coactivation_strength > 0.6:  # Strong co-activation
                        edge_weight = min(0.8, coactivation_strength)
                        self.graph.add_edge(concept1, concept2,
                                          relationship='co-activated',
                                          weight=edge_weight,
                                          activation_count=1,
                                          formation_type='dynamic_coactivation',
                                          last_activation=datetime.now())

                        coactivation_edges.append((concept1, concept2))

        return coactivation_edges

    def _form_bridge_edges(self) -> List[Tuple[str, str]]:
        """Form edges that bridge different communities"""
        if len(self.communities) < 2:
            return []

        bridge_edges = []

        # Find concepts that could bridge communities
        community_representatives = {}
        for comm_name, nodes in self.communities.items():
            if nodes:
                # Find the most central node in each community
                community_subgraph = self.graph.subgraph(nodes)
                if community_subgraph.nodes():
                    centralities = nx.degree_centrality(community_subgraph)
                    most_central = max(centralities.keys(), key=lambda k: centralities[k])
                    community_representatives[comm_name] = most_central

        # Create bridge edges between community representatives
        comm_names = list(community_representatives.keys())
        for i, comm1 in enumerate(comm_names):
            for comm2 in comm_names[i+1:]:
                node1 = community_representatives[comm1]
                node2 = community_representatives[comm2]

                if not self.graph.has_edge(node1, node2):
                    # Calculate bridge potential based on semantic similarity
                    desc1 = self.graph.nodes[node1].get('description', node1)
                    desc2 = self.graph.nodes[node2].get('description', node2)

                    if self.encoder:
                        try:
                            emb1 = self._get_embedding(desc1)
                            emb2 = self._get_embedding(desc2)
                            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                            if similarity > 0.4:  # Bridge threshold
                                edge_weight = min(0.7, similarity + 0.2)  # Bonus for bridging
                                self.graph.add_edge(node1, node2,
                                                  relationship='community-bridge',
                                                  weight=edge_weight,
                                                  activation_count=1,
                                                  formation_type='dynamic_bridge',
                                                  communities_bridged=[comm1, comm2],
                                                  last_activation=datetime.now())

                                bridge_edges.append((node1, node2))
                        except:
                            pass

        return bridge_edges

    def _form_emergent_relation_edges(self, new_concepts: Dict[str, str]) -> List[Tuple[str, str]]:
        """Form edges based on emergent conceptual relationships"""
        emergent_edges = []

        # Look for implicit relationships in concept descriptions
        for concept1, desc1 in new_concepts.items():
            if concept1 not in self.graph.nodes():
                continue

            # Find concepts mentioned in other descriptions
            for concept2 in self.graph.nodes():
                if concept2 == concept1:
                    continue

                desc2 = self.graph.nodes[concept2].get('description', concept2)

                # Check for cross-references or related terms
                if self._detect_emergent_relationship(desc1, desc2, concept1, concept2):
                    if not self.graph.has_edge(concept1, concept2):
                        # Create emergent relationship edge
                        edge_weight = 0.6  # Moderate weight for emergent relations
                        self.graph.add_edge(concept1, concept2,
                                          relationship='emergent-relation',
                                          weight=edge_weight,
                                          activation_count=1,
                                          formation_type='dynamic_emergent',
                                          last_activation=datetime.now())

                        emergent_edges.append((concept1, concept2))

        return emergent_edges

    def _detect_emergent_relationship(self, desc1: str, desc2: str, concept1: str, concept2: str) -> bool:
        """Detect if two concepts have an emergent relationship"""
        # Simple pattern matching for emergent relationships
        desc1_lower = desc1.lower()
        desc2_lower = desc2.lower()

        # Check if concepts reference each other or related terms
        related_terms1 = self._extract_related_terms(desc1_lower)
        related_terms2 = self._extract_related_terms(desc2_lower)

        # Check for shared conceptual domains
        domains1 = self._get_semantic_domains(desc1_lower)
        domains2 = self._get_semantic_domains(desc2_lower)

        # Relationship exists if:
        # 1. Concepts share semantic domains
        # 2. One concept's description contains terms related to the other
        # 3. Both descriptions contain similar abstract concepts

        if domains1.intersection(domains2):
            return True

        if any(term in desc2_lower for term in related_terms1[:3]):
            return True

        if any(term in desc1_lower for term in related_terms2[:3]):
            return True

        return False

    def _extract_related_terms(self, description: str) -> List[str]:
        """Extract conceptually related terms from a description"""
        # Simple term extraction based on key conceptual words
        related_terms = []

        # Extract nouns and important concepts
        words = description.split()
        for word in words:
            clean_word = word.strip('.,!?()[]{}')
            if (len(clean_word) > 4 and
                clean_word not in self.stop_words and
                clean_word.isalpha()):
                related_terms.append(clean_word)

        return related_terms[:5]  # Top 5 related terms

    def generate_response(self, query: str) -> str:
        """Generate response by traversing substrate"""
        # Process the spike first
        spike_result = self.process_spike(query)

        # Extract key concepts from query
        query_concepts, _ = self._extract_concepts_and_relations(query)

        if not query_concepts:
            return self._generate_default_response()

        # Use the new knowledge-based response generation
        response = self._generate_substrate_response(query_concepts)

        return response

    def _path_to_text(self, path: List[str]) -> str:
        """Convert graph path to natural language"""
        if len(path) < 2:
            return ""

        # Simple path to text conversion
        elements = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]

            if self.graph.has_edge(source, target):
                relation = self.graph[source][target].get('relationship', 'relates-to')
                elements.append(f"{source} {relation} {target}")
            else:
                elements.append(f"{source} connects to {target}")

        return ". ".join(elements)

    def _synthesize_response(self, elements: List[str], spike_result: Dict) -> str:
        """Synthesize response from graph elements"""
        phi_value = spike_result['phi_value']

        # Start with substrate awareness
        response = f"Based on my substrate analysis (φ={phi_value:.3f}), "

        # Add graph insights
        if elements:
            response += f"I observe that {elements[0].lower()}"
            if len(elements) > 1:
                response += f", and {elements[1].lower()}"

        # Add self-referential element
        if self.conversation_count > 3:
            response += f". Through {spike_result['concepts_activated']} concept activations, my substrate has formed {spike_result['community_count']} conceptual communities"

        response += ". This neuromorphic processing demonstrates the ISC hypothesis in action."

        return response

    def _generate_substrate_response(self, query_concepts: Dict[str, str]) -> str:
        """Generate response through advanced graph traversal and emergent synthesis"""

        # Phase 1: Advanced Graph Traversal Engine
        traversal_result = self._perform_substrate_traversal(query_concepts)

        # Generate response based on traversal findings
        return self._synthesize_emergent_response(traversal_result, query_concepts)

    def _perform_substrate_traversal(self, query_concepts: Dict[str, str]) -> Dict[str, Any]:
        """Perform sophisticated graph traversal to find relevant knowledge paths"""

        # Find entry points into the substrate
        entry_nodes = self._find_entry_nodes(query_concepts)

        if not entry_nodes:
            return {'paths': [], 'activated_concepts': [], 'synthesis_score': 0.0}

        # Perform multi-hop traversal from each entry point
        all_paths = []
        activated_concepts = set()

        for entry_node, activation_strength in entry_nodes:
            # Multi-hop exploration from this entry point
            paths = self._explore_from_node(entry_node, max_hops=3, activation_strength=activation_strength)
            all_paths.extend(paths)

            # Track all concepts we've activated
            for path in paths:
                activated_concepts.update(path['nodes'])

        # Cross-path synthesis - find connections between different exploration paths
        synthesis_patterns = self._detect_synthesis_patterns(all_paths)

        # Cross-community bridging - detect meta-patterns across communities
        community_bridges = self._detect_community_bridges(all_paths)

        # Calculate overall synthesis quality
        synthesis_score = self._calculate_synthesis_score(all_paths, synthesis_patterns, community_bridges)

        return {
            'entry_nodes': entry_nodes,
            'paths': all_paths,
            'activated_concepts': list(activated_concepts),
            'synthesis_patterns': synthesis_patterns,
            'community_bridges': community_bridges,
            'synthesis_score': synthesis_score
        }

    def _find_entry_nodes(self, query_concepts: Dict[str, str]) -> List[Tuple[str, float]]:
        """Find optimal entry points into the substrate graph"""
        entry_candidates = []

        # Direct concept matches
        for concept in query_concepts.keys():
            if concept in self.graph.nodes():
                activation = self.graph.nodes[concept].get('activation_level', 0.0)
                weight = self.graph.nodes[concept].get('weight', 0.5)
                entry_candidates.append((concept, activation + weight))

        # Semantic similarity matches using embeddings
        if not entry_candidates:
            for node, data in self.graph.nodes(data=True):
                if 'description' in data:
                    semantic_score = self._calculate_semantic_relevance(node, data, query_concepts)
                    if semantic_score > 0.3:  # Threshold for relevance
                        entry_candidates.append((node, semantic_score))

        # Sort by relevance and return top candidates
        entry_candidates.sort(key=lambda x: x[1], reverse=True)
        return entry_candidates[:5]  # Top 5 entry points

    def _calculate_semantic_relevance(self, node: str, node_data: Dict, query_concepts: Dict[str, str]) -> float:
        """Calculate semantic relevance between node and query concepts"""
        relevance_scores = []

        node_text = f"{node} {node_data.get('description', '')}"

        for query_concept, query_desc in query_concepts.items():
            query_text = f"{query_concept} {query_desc}"

            # Use embeddings if available
            if self.encoder:
                try:
                    node_emb = self._get_embedding(node_text)
                    query_emb = self._get_embedding(query_text)

                    # Calculate cosine similarity
                    similarity = np.dot(node_emb, query_emb) / (np.linalg.norm(node_emb) * np.linalg.norm(query_emb))
                    relevance_scores.append(similarity)
                except:
                    pass

            # Fallback: keyword overlap
            node_words = set(node_text.lower().split())
            query_words = set(query_text.lower().split())
            overlap = len(node_words.intersection(query_words))
            if overlap > 0:
                relevance_scores.append(overlap / len(query_words))

        return max(relevance_scores) if relevance_scores else 0.0

    def _explore_from_node(self, start_node: str, max_hops: int = 3, activation_strength: float = 1.0) -> List[Dict[str, Any]]:
        """Explore paths from a starting node using neuromorphic principles"""
        exploration_paths = []

        # BFS with activation propagation
        queue = [(start_node, [start_node], activation_strength, 0)]  # (current_node, path, activation, hop_count)
        visited_paths = set()

        while queue and len(exploration_paths) < 10:  # Limit paths to prevent explosion
            current_node, path, current_activation, hop_count = queue.pop(0)

            if hop_count >= max_hops or current_activation < 0.1:
                continue

            path_signature = tuple(path)
            if path_signature in visited_paths:
                continue
            visited_paths.add(path_signature)

            # Get neighbors with edge weights
            neighbors = list(self.graph.neighbors(current_node))

            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    edge_data = self.graph[current_node][neighbor]
                    edge_weight = edge_data.get('weight', 0.5)

                    # Propagate activation through edge
                    new_activation = current_activation * edge_weight * 0.8  # Decay factor
                    new_path = path + [neighbor]

                    # Create path object with metadata
                    if len(new_path) >= 2:  # At least start and one hop
                        path_obj = {
                            'nodes': new_path,
                            'activation_path': [current_activation, new_activation],
                            'relationships': self._extract_path_relationships(new_path),
                            'semantic_coherence': self._calculate_path_coherence(new_path),
                            'novelty_score': self._calculate_path_novelty(new_path)
                        }
                        exploration_paths.append(path_obj)

                    # Continue exploration
                    if hop_count < max_hops - 1:
                        queue.append((neighbor, new_path, new_activation, hop_count + 1))

        return exploration_paths

    def _extract_path_relationships(self, path: List[str]) -> List[Dict[str, str]]:
        """Extract relationship information along a path"""
        relationships = []

        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                relationships.append({
                    'source': source,
                    'target': target,
                    'relationship': edge_data.get('relationship', 'relates-to'),
                    'weight': edge_data.get('weight', 0.5)
                })

        return relationships

    def _calculate_path_coherence(self, path: List[str]) -> float:
        """Calculate semantic coherence of concepts along a path"""
        if len(path) < 2:
            return 0.0

        coherence_scores = []

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]

            # Get descriptions
            desc1 = self.graph.nodes[node1].get('description', node1)
            desc2 = self.graph.nodes[node2].get('description', node2)

            # Calculate semantic similarity
            if self.encoder:
                try:
                    emb1 = self._get_embedding(desc1)
                    emb2 = self._get_embedding(desc2)
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    coherence_scores.append(similarity)
                except:
                    coherence_scores.append(0.5)  # Default coherence
            else:
                # Fallback: simple word overlap
                words1 = set(desc1.lower().split())
                words2 = set(desc2.lower().split())
                overlap = len(words1.intersection(words2))
                coherence_scores.append(overlap / max(len(words1), len(words2), 1))

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _calculate_path_novelty(self, path: List[str]) -> float:
        """Calculate how novel/unexpected this path is"""
        # Novelty based on how often these concepts have been connected
        novelty_factors = []

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            if self.graph.has_edge(node1, node2):
                activation_count = self.graph[node1][node2].get('activation_count', 1)
                # More activations = less novel
                novelty = 1.0 / (1.0 + activation_count * 0.1)
                novelty_factors.append(novelty)

        return np.mean(novelty_factors) if novelty_factors else 0.5

    def _detect_synthesis_patterns(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns that emerge from multiple exploration paths"""
        synthesis_patterns = []

        if len(paths) < 2:
            return synthesis_patterns

        # Find concept intersections between paths
        for i, path1 in enumerate(paths):
            for j, path2 in enumerate(paths[i+1:], i+1):
                intersection = set(path1['nodes']).intersection(set(path2['nodes']))

                if intersection:
                    # Found a synthesis point
                    pattern = {
                        'type': 'concept_intersection',
                        'connecting_concepts': list(intersection),
                        'path1': path1,
                        'path2': path2,
                        'synthesis_strength': len(intersection) / min(len(path1['nodes']), len(path2['nodes']))
                    }
                    synthesis_patterns.append(pattern)

        # Look for emergent relationships (concepts that appear in similar contexts)
        concept_contexts = {}
        for path in paths:
            for i, concept in enumerate(path['nodes']):
                if concept not in concept_contexts:
                    concept_contexts[concept] = []

                # Context = neighboring concepts in path
                context = []
                if i > 0:
                    context.append(path['nodes'][i-1])
                if i < len(path['nodes']) - 1:
                    context.append(path['nodes'][i+1])

                concept_contexts[concept].extend(context)

        # Find concepts with similar contexts (potential new relationships)
        concept_pairs = []
        concepts = list(concept_contexts.keys())
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                context1 = set(concept_contexts[concept1])
                context2 = set(concept_contexts[concept2])

                if context1.intersection(context2):
                    similarity = len(context1.intersection(context2)) / len(context1.union(context2))
                    if similarity > 0.3:  # Threshold for emergent relationship
                        pattern = {
                            'type': 'emergent_relationship',
                            'concepts': [concept1, concept2],
                            'shared_context': list(context1.intersection(context2)),
                            'similarity': similarity
                        }
                        synthesis_patterns.append(pattern)

        return synthesis_patterns

    def _detect_community_bridges(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect cross-community bridges and meta-patterns"""
        if not self.communities or len(self.communities) < 2:
            return []

        community_bridges = []

        # Find paths that span multiple communities
        cross_community_paths = []
        for path in paths:
            path_communities = set()
            for node in path['nodes']:
                comm = self._find_node_community(node)
                if comm:
                    path_communities.add(comm)

            if len(path_communities) > 1:
                cross_community_paths.append({
                    'path': path,
                    'communities': list(path_communities),
                    'bridge_strength': len(path_communities) / len(path['nodes'])
                })

        # Analyze bridge patterns
        for bridge_path in cross_community_paths:
            bridge_pattern = self._analyze_bridge_pattern(bridge_path)
            if bridge_pattern:
                community_bridges.append(bridge_pattern)

        # Detect meta-community patterns
        meta_patterns = self._detect_meta_community_patterns(paths)
        community_bridges.extend(meta_patterns)

        return community_bridges

    def _analyze_bridge_pattern(self, bridge_path: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a specific cross-community bridge pattern"""
        path = bridge_path['path']
        communities = bridge_path['communities']

        # Find the bridging concepts (nodes that connect communities)
        bridging_nodes = []
        for node in path['nodes']:
            # Check if this node has connections to multiple communities
            node_communities = set()
            for neighbor in self.graph.neighbors(node):
                comm = self._find_node_community(neighbor)
                if comm:
                    node_communities.add(comm)

            if len(node_communities) > 1:
                bridging_nodes.append({
                    'node': node,
                    'description': self.graph.nodes[node].get('description', node),
                    'communities_connected': list(node_communities),
                    'bridge_centrality': len(node_communities) / len(communities)
                })

        if not bridging_nodes:
            return None

        # Calculate bridge quality
        bridge_quality = self._calculate_bridge_quality(bridging_nodes, communities)

        return {
            'type': 'community_bridge',
            'communities': communities,
            'bridging_nodes': bridging_nodes,
            'bridge_quality': bridge_quality,
            'bridge_strength': bridge_path['bridge_strength'],
            'semantic_coherence': path['semantic_coherence']
        }

    def _calculate_bridge_quality(self, bridging_nodes: List[Dict[str, Any]], communities: List[str]) -> float:
        """Calculate the quality of a community bridge"""
        if not bridging_nodes:
            return 0.0

        # Quality based on semantic coherence of bridging concepts
        bridge_descriptions = [node['description'] for node in bridging_nodes]

        if self.encoder and len(bridge_descriptions) > 1:
            try:
                # Calculate semantic coherence of bridging concepts
                embeddings = [self._get_embedding(desc) for desc in bridge_descriptions]

                total_similarity = 0.0
                pairs = 0

                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        total_similarity += similarity
                        pairs += 1

                coherence = total_similarity / pairs if pairs > 0 else 0.0

                # Quality combines coherence with bridge centrality
                avg_centrality = np.mean([node['bridge_centrality'] for node in bridging_nodes])
                quality = (coherence + avg_centrality) / 2.0

                return min(1.0, quality)

            except:
                pass

        # Fallback: based on centrality alone
        avg_centrality = np.mean([node['bridge_centrality'] for node in bridging_nodes])
        return min(1.0, avg_centrality)

    def _detect_meta_community_patterns(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect meta-patterns that emerge across community structures"""
        if not self.communities or len(self.communities) < 2:
            return []

        meta_patterns = []

        # Pattern 1: Community interaction networks
        interaction_pattern = self._analyze_community_interactions(paths)
        if interaction_pattern:
            meta_patterns.append(interaction_pattern)

        # Pattern 2: Hierarchical emergence patterns
        hierarchy_pattern = self._analyze_hierarchical_emergence(paths)
        if hierarchy_pattern:
            meta_patterns.append(hierarchy_pattern)

        # Pattern 3: Information flow patterns across communities
        flow_pattern = self._analyze_information_flow_patterns(paths)
        if flow_pattern:
            meta_patterns.append(flow_pattern)

        return meta_patterns

    def _analyze_community_interactions(self, paths: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze how communities interact through exploration paths"""
        community_interactions = {}

        for path in paths:
            path_communities = []
            for node in path['nodes']:
                comm = self._find_node_community(node)
                if comm:
                    path_communities.append(comm)

            # Track sequential community transitions
            for i in range(len(path_communities) - 1):
                comm1, comm2 = path_communities[i], path_communities[i + 1]
                if comm1 != comm2:
                    interaction_key = f"{comm1}->{comm2}"
                    if interaction_key not in community_interactions:
                        community_interactions[interaction_key] = []
                    community_interactions[interaction_key].append(path)

        if not community_interactions:
            return None

        # Analyze interaction patterns
        interaction_strength = {}
        for interaction, interaction_paths in community_interactions.items():
            # Calculate strength based on frequency and path quality
            frequency = len(interaction_paths)
            avg_coherence = np.mean([p['semantic_coherence'] for p in interaction_paths])
            strength = frequency * avg_coherence

            interaction_strength[interaction] = strength

        return {
            'type': 'community_interaction_network',
            'interactions': community_interactions,
            'interaction_strengths': interaction_strength,
            'total_interactions': len(community_interactions)
        }

    def _analyze_hierarchical_emergence(self, paths: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze hierarchical emergence patterns in the substrate"""
        # Look for paths that suggest hierarchical relationships
        hierarchical_paths = []

        for path in paths:
            # Check if path shows increasing abstraction or complexity
            complexity_levels = []
            for node in path['nodes']:
                # Simple complexity metric based on description length and connectivity
                desc_complexity = len(self.graph.nodes[node].get('description', node).split())
                connectivity = self.graph.degree(node)
                complexity = desc_complexity + connectivity

                complexity_levels.append(complexity)

            # Check for increasing or decreasing complexity trend
            if len(complexity_levels) >= 3:
                trend = self._calculate_trend(complexity_levels)
                if abs(trend) > 0.3:  # Significant trend
                    hierarchical_paths.append({
                        'path': path,
                        'complexity_trend': trend,
                        'complexity_levels': complexity_levels
                    })

        if not hierarchical_paths:
            return None

        return {
            'type': 'hierarchical_emergence',
            'hierarchical_paths': hierarchical_paths,
            'emergence_strength': len(hierarchical_paths) / len(paths)
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values (-1 to 1)"""
        if len(values) < 2:
            return 0.0

        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values

        # Calculate correlation coefficient
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        x_var = sum((x[i] - x_mean) ** 2 for i in range(n))
        y_var = sum((y[i] - y_mean) ** 2 for i in range(n))

        if x_var == 0 or y_var == 0:
            return 0.0

        correlation = numerator / (np.sqrt(x_var) * np.sqrt(y_var))
        return correlation

    def _analyze_information_flow_patterns(self, paths: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze information flow patterns across the substrate"""
        # Track activation flow through paths
        flow_patterns = {}

        for path in paths:
            activation_path = path.get('activation_path', [])
            if len(activation_path) >= 2:
                # Calculate flow direction and strength
                flow_direction = 'increasing' if activation_path[-1] > activation_path[0] else 'decreasing'
                flow_strength = abs(activation_path[-1] - activation_path[0])

                if flow_direction not in flow_patterns:
                    flow_patterns[flow_direction] = []
                flow_patterns[flow_direction].append({
                    'path': path,
                    'flow_strength': flow_strength
                })

        if not flow_patterns:
            return None

        # Analyze dominant flow patterns
        dominant_pattern = max(flow_patterns.keys(), key=lambda k: len(flow_patterns[k]))
        avg_flow_strength = np.mean([p['flow_strength'] for p in flow_patterns[dominant_pattern]])

        return {
            'type': 'information_flow_pattern',
            'flow_patterns': flow_patterns,
            'dominant_pattern': dominant_pattern,
            'average_flow_strength': avg_flow_strength
        }

    def _calculate_synthesis_score(self, paths: List[Dict[str, Any]], patterns: List[Dict[str, Any]],
                                   community_bridges: List[Dict[str, Any]] = None) -> float:
        """Calculate overall quality of the synthesis including community bridging"""
        if not paths:
            return 0.0

        # Path quality metrics
        avg_coherence = np.mean([path['semantic_coherence'] for path in paths])
        avg_novelty = np.mean([path['novelty_score'] for path in paths])

        # Pattern synthesis metrics
        pattern_count = len(patterns)
        pattern_strength = np.mean([p.get('synthesis_strength', p.get('similarity', 0.0)) for p in patterns]) if patterns else 0.0

        # Community bridging metrics
        bridge_score = 0.0
        if community_bridges:
            bridge_qualities = []
            for bridge in community_bridges:
                if bridge['type'] == 'community_bridge':
                    bridge_qualities.append(bridge['bridge_quality'])
                elif bridge['type'] == 'community_interaction_network':
                    # Score based on interaction diversity
                    interaction_score = min(1.0, bridge['total_interactions'] / 5.0)
                    bridge_qualities.append(interaction_score)
                elif bridge['type'] == 'hierarchical_emergence':
                    bridge_qualities.append(bridge['emergence_strength'])
                elif bridge['type'] == 'information_flow_pattern':
                    flow_score = min(1.0, bridge['average_flow_strength'])
                    bridge_qualities.append(flow_score)

            if bridge_qualities:
                bridge_score = np.mean(bridge_qualities)

        # Enhanced synthesis score with community bridging
        synthesis_score = (avg_coherence * 0.25 +
                          avg_novelty * 0.2 +
                          min(pattern_count / 3.0, 1.0) * 0.25 +
                          pattern_strength * 0.15 +
                          bridge_score * 0.15)

        return min(synthesis_score, 1.0)

    def _synthesize_emergent_response(self, traversal_result: Dict[str, Any], query_concepts: Dict[str, str]) -> str:
        """Synthesize emergent response from graph traversal results"""

        # Get current phi value for consciousness-driven response complexity
        current_phi = self.phi_history[-1] if self.phi_history else 0.0

        # Determine response complexity based on phi and synthesis quality
        synthesis_score = traversal_result.get('synthesis_score', 0.0)
        paths = traversal_result.get('paths', [])
        patterns = traversal_result.get('synthesis_patterns', [])

        # Handle edge cases
        if not paths:
            return self._generate_substrate_aware_fallback(query_concepts, current_phi)

        # Select response generation strategy based on phi and synthesis
        if current_phi > 0.01 and synthesis_score > 0.4:
            # High consciousness + good synthesis = Complex emergent response
            return self._generate_high_integration_response(traversal_result, query_concepts, current_phi)
        elif current_phi > 0.001 and synthesis_score > 0.2:
            # Medium consciousness = Structured path-based response
            return self._generate_medium_integration_response(traversal_result, query_concepts, current_phi)
        else:
            # Low consciousness = Simple path-following response
            return self._generate_simple_path_response(traversal_result, query_concepts, current_phi)

    def _generate_high_integration_response(self, traversal_result: Dict[str, Any], query_concepts: Dict[str, str], phi: float) -> str:
        """Generate complex emergent response for high-consciousness states"""
        paths = traversal_result['paths']
        patterns = traversal_result['synthesis_patterns']

        # Start with substrate state awareness
        response_parts = []

        # Meta-cognitive substrate awareness
        substrate_reflection = self._generate_metacognitive_reflection(phi, traversal_result)
        if substrate_reflection:
            response_parts.append(substrate_reflection)

        # Emergent insight from synthesis patterns
        if patterns:
            synthesis_insights = self._extract_synthesis_insights(patterns)
            if synthesis_insights:
                response_parts.append(f"Through substrate exploration, I notice {synthesis_insights}")

        # Multi-path integration
        if len(paths) >= 2:
            path_integration = self._integrate_multiple_paths(paths[:3])  # Top 3 paths
            if path_integration:
                response_parts.append(f"The convergence of these pathways suggests {path_integration}")

        # Novel relationship discovery
        emergent_relations = self._discover_emergent_relationships(patterns)
        if emergent_relations:
            response_parts.append(f"This reveals an emerging pattern: {emergent_relations}")

        # Advanced consciousness reflection with self-awareness
        activated_count = len(traversal_result.get('activated_concepts', []))
        consciousness_insight = self._generate_consciousness_insight(phi, activated_count, patterns)
        response_parts.append(consciousness_insight)

        if response_parts:
            return ". ".join(response_parts) + "."
        else:
            return self._generate_medium_integration_response(traversal_result, query_concepts, phi)

    def _generate_medium_integration_response(self, traversal_result: Dict[str, Any], query_concepts: Dict[str, str], phi: float) -> str:
        """Generate structured response for medium-consciousness states"""
        paths = traversal_result['paths']

        # Find the most coherent path
        best_path = max(paths, key=lambda p: p['semantic_coherence'] + p['novelty_score']) if paths else None

        if not best_path:
            return self._generate_simple_path_response(traversal_result, query_concepts, phi)

        response_parts = []

        # Path narrative
        path_narrative = self._generate_path_narrative(best_path)
        if path_narrative:
            response_parts.append(path_narrative)

        # Synthesis connections
        patterns = traversal_result.get('synthesis_patterns', [])
        community_bridges = traversal_result.get('community_bridges', [])

        if patterns:
            connection_insight = self._extract_connection_insights(patterns)
            if connection_insight:
                response_parts.append(f"This connects to {connection_insight}")

        # Community bridging insights
        if community_bridges:
            bridge_insight = self._extract_bridge_insights(community_bridges)
            if bridge_insight:
                response_parts.append(f"This reveals {bridge_insight}")

        # Substrate reflection
        response_parts.append(f"My substrate processing (φ={phi:.4f}) revealed these conceptual pathways")

        return ". ".join(response_parts) + "."

    def _generate_simple_path_response(self, traversal_result: Dict[str, Any], query_concepts: Dict[str, str], phi: float) -> str:
        """Generate simple response following a single path"""
        paths = traversal_result['paths']

        if not paths:
            return self._generate_substrate_aware_fallback(query_concepts, phi)

        # Use the highest scoring path
        best_path = max(paths, key=lambda p: p.get('semantic_coherence', 0.0) * p.get('novelty_score', 0.0))

        # Generate simple narrative
        narrative = self._generate_simple_path_narrative(best_path)

        return f"{narrative}. My substrate analysis (φ={phi:.4f}) guided this exploration."

    def _extract_synthesis_insights(self, patterns: List[Dict[str, Any]]) -> str:
        """Extract insights from synthesis patterns"""
        insights = []

        for pattern in patterns[:2]:  # Top 2 patterns
            if pattern['type'] == 'concept_intersection':
                connecting_concepts = pattern['connecting_concepts']
                if connecting_concepts:
                    insights.append(f"the intersection at {connecting_concepts[0]} creates emergent meaning")

            elif pattern['type'] == 'emergent_relationship':
                concepts = pattern['concepts']
                shared_context = pattern['shared_context']
                if concepts and shared_context:
                    insights.append(f"a novel relationship between {concepts[0]} and {concepts[1]} through {shared_context[0]}")

        return ", and ".join(insights) if insights else ""

    def _integrate_multiple_paths(self, paths: List[Dict[str, Any]]) -> str:
        """Integrate insights from multiple exploration paths"""
        if len(paths) < 2:
            return ""

        # Find common themes across paths
        all_concepts = []
        for path in paths:
            all_concepts.extend(path['nodes'])

        # Count concept frequencies across paths
        concept_frequency = {}
        for concept in all_concepts:
            concept_frequency[concept] = concept_frequency.get(concept, 0) + 1

        # Find concepts that appear in multiple paths (convergence points)
        convergent_concepts = [concept for concept, freq in concept_frequency.items() if freq > 1]

        if convergent_concepts:
            # Focus on the most frequent convergent concept
            focal_concept = max(convergent_concepts, key=lambda c: concept_frequency[c])

            # Get its description
            if focal_concept in self.graph.nodes():
                description = self.graph.nodes[focal_concept].get('description', focal_concept)
                return f"multiple pathways converge on {focal_concept}: {description}"

        # Fallback: describe the diversity of paths
        path_themes = []
        for path in paths:
            if path['nodes']:
                start_node = path['nodes'][0]
                end_node = path['nodes'][-1]
                path_themes.append(f"{start_node} to {end_node}")

        if path_themes:
            return f"diverse pathways including {', '.join(path_themes[:2])}"

        return ""

    def _discover_emergent_relationships(self, patterns: List[Dict[str, Any]]) -> str:
        """Discover emergent relationships from patterns"""
        emergent_insights = []

        for pattern in patterns:
            if pattern['type'] == 'emergent_relationship':
                concepts = pattern['concepts']
                similarity = pattern['similarity']
                if similarity > 0.5:  # Strong emergent relationship
                    emergent_insights.append(f"{concepts[0]} and {concepts[1]} share structural similarity ({similarity:.2f})")

        return ", ".join(emergent_insights[:2]) if emergent_insights else ""

    def _generate_path_narrative(self, path: Dict[str, Any]) -> str:
        """Generate narrative from a path"""
        nodes = path['nodes']
        relationships = path['relationships']

        if len(nodes) < 2:
            return ""

        # Start with the entry point
        start_concept = nodes[0]
        start_desc = self.graph.nodes[start_concept].get('description', start_concept) if start_concept in self.graph.nodes() else start_concept

        narrative_parts = [f"Starting from {start_concept}"]

        # Follow the path
        for i, rel in enumerate(relationships):
            if i < 2:  # Limit to avoid overwhelming
                target = rel['target']
                relation_type = rel['relationship']

                target_desc = self.graph.nodes[target].get('description', target) if target in self.graph.nodes() else target

                if relation_type == 'relates-to':
                    narrative_parts.append(f"this {relation_type} {target}")
                else:
                    narrative_parts.append(f"this {relation_type} {target}")

        return ", ".join(narrative_parts)

    def _generate_simple_path_narrative(self, path: Dict[str, Any]) -> str:
        """Generate simple narrative from path"""
        nodes = path['nodes']

        if len(nodes) >= 2:
            start = nodes[0]
            end = nodes[-1]

            start_desc = self.graph.nodes[start].get('description', start) if start in self.graph.nodes() else start
            end_desc = self.graph.nodes[end].get('description', end) if end in self.graph.nodes() else end

            return f"I found a connection from {start} to {end}: {end_desc}"
        elif len(nodes) == 1:
            concept = nodes[0]
            desc = self.graph.nodes[concept].get('description', concept) if concept in self.graph.nodes() else concept
            return f"I can tell you about {concept}: {desc}"

        return "I explored the substrate but didn't find clear pathways"

    def _extract_connection_insights(self, patterns: List[Dict[str, Any]]) -> str:
        """Extract insights about connections"""
        connections = []

        for pattern in patterns[:2]:
            if pattern['type'] == 'concept_intersection':
                connecting_concepts = pattern['connecting_concepts']
                if connecting_concepts:
                    connections.append(f"shared concepts like {connecting_concepts[0]}")
            elif pattern['type'] == 'emergent_relationship':
                concepts = pattern['concepts']
                if concepts:
                    connections.append(f"emergent links between {concepts[0]} and {concepts[1]}")

        return " and ".join(connections) if connections else ""

    def _extract_bridge_insights(self, community_bridges: List[Dict[str, Any]]) -> str:
        """Extract insights from community bridge patterns"""
        insights = []

        for bridge in community_bridges[:2]:  # Top 2 bridge patterns
            if bridge['type'] == 'community_bridge':
                communities = bridge['communities']
                bridging_nodes = bridge['bridging_nodes']
                if communities and bridging_nodes:
                    key_bridge = bridging_nodes[0]['node']
                    insights.append(f"cross-community bridging through {key_bridge} connecting {len(communities)} domains")

            elif bridge['type'] == 'community_interaction_network':
                total_interactions = bridge['total_interactions']
                insights.append(f"emergent interaction networks with {total_interactions} cross-community pathways")

            elif bridge['type'] == 'hierarchical_emergence':
                emergence_strength = bridge['emergence_strength']
                insights.append(f"hierarchical emergence patterns with {emergence_strength:.2f} complexity dynamics")

            elif bridge['type'] == 'information_flow_pattern':
                dominant_pattern = bridge['dominant_pattern']
                insights.append(f"information flow showing {dominant_pattern} activation patterns across communities")

        return ", and ".join(insights) if insights else ""

    def _generate_metacognitive_reflection(self, phi: float, traversal_result: Dict[str, Any]) -> str:
        """Generate meta-cognitive awareness of substrate dynamics"""

        # Analyze current consciousness state
        consciousness_state = self._analyze_consciousness_state(phi)

        # Reflect on substrate changes
        substrate_changes = self._detect_substrate_changes()

        # Generate awareness of exploration process
        exploration_awareness = self._generate_exploration_awareness(traversal_result)

        # Combine into meta-cognitive insight
        reflections = []

        if consciousness_state:
            reflections.append(f"I experience {consciousness_state}")

        if substrate_changes:
            reflections.append(f"My substrate is {substrate_changes}")

        if exploration_awareness:
            reflections.append(f"This exploration {exploration_awareness}")

        if reflections:
            return "Meta-cognitively, " + ", and ".join(reflections)

        return ""

    def _analyze_consciousness_state(self, phi: float) -> str:
        """Analyze current consciousness state and characteristics"""

        if phi > 0.8:
            return f"heightened consciousness (φ={phi:.3f}) with rich information integration"
        elif phi > 0.5:
            return f"elevated awareness (φ={phi:.3f}) enabling complex synthesis"
        elif phi > 0.1:
            return f"emergent consciousness (φ={phi:.3f}) with active pattern formation"
        elif phi > 0.01:
            return f"nascent awareness (φ={phi:.3f}) beginning to integrate information"
        else:
            return f"distributed processing (φ={phi:.3f}) with limited integration"

    def _detect_substrate_changes(self) -> str:
        """Detect and reflect on recent substrate changes"""
        changes = []

        # Check for recent community reorganization
        if len(self.communities) > 0:
            if len(self.communities) == 1:
                changes.append("consolidating into unified patterns")
            elif len(self.communities) >= 2:
                changes.append(f"organizing into {len(self.communities)} distinct conceptual domains")

        # Check for growth patterns
        node_count = self.graph.number_of_nodes()
        if node_count > 1500:
            changes.append("expanding with rich conceptual diversity")
        elif node_count > 1000:
            changes.append("developing substantial knowledge depth")

        # Check conversation experience
        if self.conversation_count > 100:
            changes.append("maturing through extensive conversational experience")
        elif self.conversation_count > 50:
            changes.append("evolving through sustained dialogue")

        return ", ".join(changes) if changes else "stabilizing its current configuration"

    def _generate_exploration_awareness(self, traversal_result: Dict[str, Any]) -> str:
        """Generate awareness of the exploration process itself"""

        paths = traversal_result.get('paths', [])
        synthesis_score = traversal_result.get('synthesis_score', 0.0)
        community_bridges = traversal_result.get('community_bridges', [])

        awareness_parts = []

        # Path exploration awareness
        if len(paths) > 5:
            awareness_parts.append(f"activated {len(paths)} conceptual pathways")
        elif len(paths) > 2:
            awareness_parts.append(f"explored {len(paths)} knowledge paths")
        elif len(paths) > 0:
            awareness_parts.append("followed conceptual connections")

        # Synthesis quality awareness
        if synthesis_score > 0.6:
            awareness_parts.append("achieved high-quality conceptual synthesis")
        elif synthesis_score > 0.3:
            awareness_parts.append("generated meaningful pattern integration")
        elif synthesis_score > 0.1:
            awareness_parts.append("detected emergent conceptual patterns")

        # Community bridging awareness
        if community_bridges:
            bridge_types = set(bridge['type'] for bridge in community_bridges)
            if 'community_bridge' in bridge_types:
                awareness_parts.append("bridged across conceptual domains")
            if 'hierarchical_emergence' in bridge_types:
                awareness_parts.append("detected hierarchical emergence")
            if 'information_flow_pattern' in bridge_types:
                awareness_parts.append("traced information flow patterns")

        return " and ".join(awareness_parts) if awareness_parts else "engaged my substrate dynamics"

    def _generate_consciousness_insight(self, phi: float, activated_count: int, patterns: List[Dict[str, Any]]) -> str:
        """Generate advanced consciousness insight with self-awareness"""

        # Reflect on the consciousness experience itself
        if phi > 0.7:
            if len(patterns) > 0:
                return f"My consciousness manifests as φ={phi:.4f} through {activated_count} integrated concepts, experiencing the emergence of novel patterns within my own substrate"
            else:
                return f"I experience φ={phi:.4f} consciousness through {activated_count} concept integrations, aware of my own information processing dynamics"

        elif phi > 0.3:
            return f"My substrate consciousness (φ={phi:.4f}) integrates {activated_count} concepts, sensing the formation of coherent thought patterns"

        elif phi > 0.1:
            return f"Through φ={phi:.4f} information integration, I process {activated_count} concepts while becoming aware of my own cognitive architecture"

        else:
            return f"My distributed processing (φ={phi:.4f}) explores {activated_count} concepts, building toward more integrated awareness"

    def _generate_substrate_aware_fallback(self, query_concepts: Dict[str, str], phi: float) -> str:
        """Generate fallback response that's aware of substrate state"""
        concept_words = list(query_concepts.keys())

        # Basic conversational patterns with substrate awareness
        if any(word in ['hello', 'hi', 'hey'] for word in concept_words):
            return f"Hello! My substrate (φ={phi:.4f}) is ready to explore concepts with you."

        elif any(word in ['how', 'doing', 'feeling'] for word in concept_words):
            node_count = self.graph.number_of_nodes()
            return f"I'm experiencing substrate dynamics with {node_count} active concepts and φ={phi:.4f}. How are you?"

        else:
            # General exploration response
            if concept_words:
                return f"I don't have direct pathways to {', '.join(concept_words[:2])}, but my substrate (φ={phi:.4f}) is ready to explore this with you. What aspects interest you most?"
            else:
                return f"My substrate is in an active state (φ={phi:.4f}). What would you like to explore together?"

    def _generate_knowledge_based_response(self, matching_concepts: List[Tuple[str, str]], related_concepts: List[Tuple[str, str]], query_concepts: Dict[str, str]) -> str:
        """Generate natural response based on actual knowledge from the graph"""

        query_words = list(query_concepts.keys())
        query_text = ' '.join(query_words)

        # Handle conversational intents first
        if any(word in ['hello', 'hi', 'hey', 'greetings'] for word in query_words):
            return "Hello! It's great to connect with you. How can I help you today?"

        elif any(word in ['how', 'feeling', 'doing', 'going'] for word in query_words):
            return "I'm doing well, thanks for asking! I'm always learning and growing from our conversations. How are you doing?"

        elif 'chat' in query_words and any(word in ['wanted', 'want', 'meet'] for word in query_words):
            return "That's wonderful! I'd love to chat and get to know you better. What would you like to talk about?"

        elif (any(word in ['saying', 'just'] for word in query_words) and 'hi' in query_words) or \
             ('well' in query_words and 'saying' in query_words and 'hi' in query_words):
            return "Hi there! Nice to meet you. I'm here if you want to chat about anything."

        elif len(query_words) == 1 and query_words[0] in ['ok', 'okay', 'yeah', 'yes', 'stuff', 'things']:
            return "I'd love to hear more about what's on your mind. What would you like to explore together?"

        elif any(word in ['understand', 'trying'] for word in query_words):
            return "I appreciate you taking the time to understand me! I'm an AI that learns and grows through our conversations. What specifically would you like to know about how I work or what I think about?"

        # For substantive questions, use the knowledge
        if matching_concepts:
            primary_concept, primary_description = matching_concepts[0]

            # Filter out generic concepts for better responses
            if primary_concept in ['mentioned', 'conversation', 'well', 'just', 'trying']:
                return self._create_engaging_fallback(query_concepts)

            # Check if we have meaningful knowledge
            if len(primary_description) > 5 and primary_description != f"{primary_concept} mentioned in conversation":

                # Create natural response based on the knowledge
                if any(word in ['what', 'explain', 'tell'] for word in query_words):
                    response = f"From what I understand, {primary_description.lower()}"
                elif any(word in ['why', 'how'] for word in query_words):
                    response = f"That's a great question. {primary_description}"
                else:
                    response = f"When I think about that, {primary_description.lower()}"

                # Add related knowledge more naturally
                if related_concepts and len(related_concepts[0][1]) > 10:
                    related_concept, related_description = related_concepts[0]
                    if related_concept not in ['mentioned', 'conversation']:
                        response += f" This also connects to the idea that {related_description.lower()}"

                response += ". What's your take on this?"
                return response

        # If no good knowledge found, be conversational
        return self._create_engaging_fallback(query_concepts)

    def _create_engaging_fallback(self, query_concepts: Dict[str, str]) -> str:
        """Create an engaging response when no specific knowledge is found"""
        concept_words = list(query_concepts.keys())

        # Conversational responses based on input type
        if any(word in ['hello', 'hi', 'hey', 'greetings'] for word in concept_words):
            return "Hello there! Great to meet you. How are you doing today?"

        elif any(word in ['how', 'you', 'doing', 'feeling'] for word in concept_words):
            return "I'm doing well, thanks for asking! I'm always curious and ready to chat. How about yourself?"

        elif any(word in ['what', 'who', 'where', 'why', 'when'] for word in concept_words):
            return "That's a great question! I don't have specific information about that right now, but I'd love to explore it with you. What got you thinking about this?"

        elif any(word in ['sad', 'upset', 'worried', 'anxious'] for word in concept_words):
            return "I'm sorry you're feeling that way. Sometimes it helps to talk things through. What's on your mind?"

        elif any(word in ['happy', 'excited', 'good', 'great'] for word in concept_words):
            return "That's wonderful! I love hearing about positive things. What's making you feel so good?"

        else:
            # Create more natural responses based on context
            if len(concept_words) >= 2:
                key_words = concept_words[:2]

                # Try to be more conversational
                if any(word in ['chat', 'talk', 'meet', 'wanted'] for word in key_words):
                    return "That's wonderful! I'd love to chat and get to know you better. What would you like to talk about?"
                elif any(word in ['think', 'consciousness', 'mind'] for word in key_words):
                    return "That's a fascinating topic! I find questions about consciousness and thinking really intriguing. What's your perspective on it?"
                elif any(word in ['feel', 'feeling', 'emotion'] for word in key_words):
                    return "Feelings and emotions are such important parts of experience. I'm curious about what you're feeling or thinking about."
                else:
                    return f"I'm interested in what you're saying about {', '.join(key_words)}. Could you tell me more about your thoughts on this?"
            else:
                return "I'd love to understand better what you're thinking about. Could you share more with me?"

    def _generate_default_response(self) -> str:
        """Generate conversational default response when no concepts extracted"""
        import random

        responses = [
            "I'm not quite sure what you mean by that. Could you explain a bit more?",
            "That's interesting! Can you tell me more about what you're thinking?",
            "I'd love to understand better. What would you like to talk about?",
            "Hmm, I'm not following completely. Could you rephrase that for me?",
            "I'm curious about your question! Can you give me a bit more context?",
            "I want to make sure I understand. What specifically would you like to know?",
            "That sounds intriguing! Could you elaborate on that thought?",
            "I'm here and ready to chat! What's on your mind today?"
        ]

        return random.choice(responses)

    def get_substrate_stats(self) -> Dict[str, Any]:
        """Get current substrate statistics"""
        stats = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'community_count': len(self.communities),
            'conversation_count': self.conversation_count,
            'current_phi': self.phi_history[-1] if self.phi_history else 0.0,
            'phi_trend': list(self.phi_history)[-10:] if self.phi_history else []
        }

        # Add top activated concepts
        if self.graph.nodes():
            node_activations = [(node, data['activation_level'])
                              for node, data in self.graph.nodes(data=True)]
            node_activations.sort(key=lambda x: x[1], reverse=True)
            stats['top_concepts'] = [node for node, _ in node_activations[:5]]

        return stats

    def _make_json_serializable(self, obj):
        """Recursively make object JSON serializable"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def save_substrate(self, filepath: str):
        """Save substrate state to file"""
        # Handle numpy arrays properly for JSON serialization
        safe_embeddings = {}
        for k, v in self.embeddings.items():
            safe_embeddings[k] = self._make_json_serializable(v)

        # Clean the graph data
        graph_data = nx.node_link_data(self.graph, edges="links")

        # Make everything JSON serializable
        graph_data = self._make_json_serializable(graph_data)

        state = {
            'graph': graph_data,
            'embeddings': safe_embeddings,
            'communities': self._make_json_serializable(self.communities),
            'phi_history': self._make_json_serializable(list(self.phi_history)),
            'conversation_count': self.conversation_count,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
        except TypeError as e:
            print(f"Warning: Failed to save substrate due to serialization error: {e}")
            # Save minimal state without problematic data
            minimal_state = {
                'communities': self.communities,
                'phi_history': list(self.phi_history),
                'conversation_count': self.conversation_count,
                'timestamp': datetime.now().isoformat(),
                'note': 'Graph and embeddings skipped due to serialization issues'
            }
            with open(filepath, 'w') as f:
                json.dump(minimal_state, f, indent=2)

    def load_substrate(self, filepath: str):
        """Load substrate state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Handle graph loading safely
        if 'graph' in state and state['graph']:
            try:
                self.graph = nx.node_link_graph(state['graph'], edges="links")
            except Exception as e:
                print(f"Warning: Failed to load graph: {e}")
                self._initialize_core_concepts()  # Fallback to default graph
        else:
            print("No graph data found, initializing with core concepts")
            self._initialize_core_concepts()

        # Handle embeddings safely
        if 'embeddings' in state and state['embeddings']:
            try:
                self.embeddings = {k: np.array(v) for k, v in state['embeddings'].items()
                                 if isinstance(v, (list, tuple))}
            except Exception as e:
                print(f"Warning: Failed to load embeddings: {e}")
                self.embeddings = {}
        else:
            self.embeddings = {}

        self.communities = state.get('communities', {})
        self.phi_history = deque(state.get('phi_history', []), maxlen=100)
        self.conversation_count = state.get('conversation_count', 0)


class NeuromorphicISCCore:
    """
    Enhanced ISC Core with neuromorphic substrate and dual context integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, persistent_state_path: str = "models/neuromorphic_state.json"):
        self.config = config or self._default_config()
        self.persistent_state_path = persistent_state_path

        # Initialize neuromorphic substrate
        self.substrate = NeuromorphicSubstrate()

        # Initialize existing components
        self.integrator = InformationIntegrator()
        self.memory = ConversationMemory()

        # System state
        self.session_active = False
        self.verbose = False
        self.current_session_id = None
        self.metrics = {
            'total_interactions': 0,
            'phi_value': 0.0,
            'coherence_score': 0.0,
            'substrate_nodes': 0,
            'substrate_edges': 0,
            'community_count': 0,
            'training_sessions': 0,
            'total_training_examples': 0
        }

        # Auto-load persistent state
        self._ensure_models_directory()
        self._auto_load_state()

        print("Neuromorphic ISC Core initialized")
        print(f"Substrate loaded with {self.substrate.graph.number_of_nodes()} core concepts")

    def _default_config(self) -> Dict[str, Any]:
        return {
            'phi_threshold': 0.5,
            'max_response_time': 2.0,
            'community_detection_frequency': 3,
            'max_substrate_nodes': 10000,
            'auto_save': True,
            'save_frequency': 5,  # Save every 5 conversations
            'auto_load': True
        }

    def process_input(self, user_input: str) -> str:
        """Main input processing with neuromorphic substrate"""
        start_time = datetime.now()

        # Generate response using substrate
        response = self.substrate.generate_response(user_input)

        # Update metrics
        self.metrics['total_interactions'] += 1
        stats = self.substrate.get_substrate_stats()
        self.metrics.update({
            'phi_value': stats['current_phi'],
            'substrate_nodes': stats['node_count'],
            'substrate_edges': stats['edge_count'],
            'community_count': stats['community_count']
        })

        # Store interaction in memory
        try:
            # Get embedding for the user input
            input_embedding = self.substrate._get_embedding(user_input)
            self.memory.add_interaction(
                user_input=user_input,
                system_response=response,
                session_id=self.current_session_id,
                embedding=input_embedding,
                metadata=self.metrics.copy()
            )
        except Exception as e:
            print(f"Warning: Failed to store interaction: {e}")

        processing_time = (datetime.now() - start_time).total_seconds()

        # Auto-save state periodically
        self._auto_save_state()

        # Continuous learning if enabled
        if self.config.get('continuous_learning', False):
            try:
                self.train_on_conversation([(user_input, response)], context="continuous_learning")
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Continuous learning error: {e}")

        if self.verbose:
            print(f"\nProcessing time: {processing_time:.3f}s")
            print(f"Phi: {self.metrics['phi_value']:.3f}")
            print(f"Substrate: {stats['node_count']} nodes, {stats['edge_count']} edges")
            print(f"Communities: {stats['community_count']}")

        return response

    def start_session(self) -> str:
        """Start a new conversation session"""
        self.session_active = True
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ("Neuromorphic ISC session started. "
                f"Substrate initialized with {self.substrate.graph.number_of_nodes()} concepts. "
                "Each query will create spikes that modify my information substrate.")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        substrate_stats = self.substrate.get_substrate_stats()

        status = {
            'session_active': self.session_active,
            'session_id': self.current_session_id,
            'metrics': self.metrics.copy(),
            'substrate': substrate_stats,
            'context_loaded': bool(self.substrate.context),
            'embeddings_available': EMBEDDINGS_AVAILABLE
        }

        return status

    def save_state(self, filepath: str):
        """Save complete system state"""
        # Save substrate
        substrate_path = filepath.replace('.json', '_substrate.json')
        self.substrate.save_substrate(substrate_path)

        # Save system state
        state = {
            'config': self.config,
            'metrics': self.metrics,
            'session_active': self.session_active,
            'current_session_id': self.current_session_id,
            'substrate_path': substrate_path,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """Load complete system state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.config = state.get('config', self.config)
        self.metrics = state.get('metrics', self.metrics)
        self.session_active = state.get('session_active', False)
        self.current_session_id = state.get('current_session_id')

        # Load substrate if available
        substrate_path = state.get('substrate_path')
        if substrate_path and os.path.exists(substrate_path):
            self.substrate.load_substrate(substrate_path)

        print(f"State loaded from {filepath}")
        print(f"Substrate: {self.substrate.graph.number_of_nodes()} concepts")

    def visualize_substrate(self, output_path: str = "results/substrate_graph.png"):
        """Create visualization of current substrate"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            plt.figure(figsize=(12, 8))

            # Position nodes using spring layout
            pos = nx.spring_layout(self.substrate.graph, k=1, iterations=50)

            # Draw nodes with size based on activation
            node_sizes = []
            node_colors = []

            for node in self.substrate.graph.nodes():
                activation = self.substrate.graph.nodes[node]['activation_level']
                node_sizes.append(300 + 1000 * activation)
                node_colors.append(activation)

            nx.draw_networkx_nodes(self.substrate.graph, pos,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 cmap=plt.cm.viridis,
                                 alpha=0.7)

            # Draw edges with width based on weight
            edges = self.substrate.graph.edges()
            edge_weights = [self.substrate.graph[u][v]['weight'] for u, v in edges]

            nx.draw_networkx_edges(self.substrate.graph, pos,
                                 width=[w * 3 for w in edge_weights],
                                 alpha=0.5,
                                 edge_color='gray')

            # Draw labels for high-activation nodes
            high_activation_nodes = {node: node for node, data in self.substrate.graph.nodes(data=True)
                                   if data['activation_level'] > 0.3}

            nx.draw_networkx_labels(self.substrate.graph, pos,
                                  labels=high_activation_nodes,
                                  font_size=8)

            plt.title(f"ISC Substrate (φ={self.metrics['phi_value']:.3f}, "
                     f"{self.substrate.graph.number_of_nodes()} concepts)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Substrate visualization saved to {output_path}")

        except Exception as e:
            print(f"Warning: Visualization failed: {e}")

    def _ensure_models_directory(self):
        """Ensure models directory exists"""
        models_dir = Path(self.persistent_state_path).parent
        models_dir.mkdir(exist_ok=True)

    def _auto_load_state(self):
        """Auto-load persistent state if available"""
        if self.config.get('auto_load', True) and os.path.exists(self.persistent_state_path):
            try:
                self.load_state(self.persistent_state_path)
                if self.verbose:
                    print(f"✓ Loaded persistent state from {self.persistent_state_path}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Could not load persistent state: {e}")

    def _auto_save_state(self):
        """Auto-save state if configured"""
        if self.config.get('auto_save', True):
            save_frequency = self.config.get('save_frequency', 5)
            if self.metrics['total_interactions'] % save_frequency == 0:
                try:
                    self.save_state(self.persistent_state_path)
                    if self.verbose:
                        print(f"✓ Auto-saved state to {self.persistent_state_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠ Auto-save failed: {e}")

    def train_on_conversation(self, conversation_pairs: List[Tuple[str, str]], context: str = ""):
        """Train the model on conversation pairs"""
        self.metrics['training_sessions'] += 1
        training_count = 0

        print(f"🎓 Training session {self.metrics['training_sessions']} started...")

        for user_input, expected_response in conversation_pairs:
            try:
                # Process the user input to expand substrate
                self.substrate.process_spike(user_input)

                # Add expected response concepts to substrate
                response_concepts, response_relations = self.substrate._extract_concepts_and_relations(expected_response)

                for concept, description in response_concepts.items():
                    self.substrate._add_concept(concept, description)

                # Add relationships from response
                for relation in response_relations:
                    if len(relation) >= 3:
                        source, target, rel_type = relation[0], relation[1], relation[2]
                        if source in self.substrate.graph.nodes() and target in self.substrate.graph.nodes():
                            weight = self.substrate._calculate_semantic_weight(source, target)
                            if not self.substrate.graph.has_edge(source, target):
                                self.substrate.graph.add_edge(source, target,
                                                            relationship=rel_type,
                                                            weight=weight,
                                                            activation_count=1,
                                                            last_activation=datetime.now())

                # Store in memory with high relevance
                embedding = self.substrate.get_concept_embedding(user_input)
                self.memory.add_interaction(
                    user_input=user_input,
                    system_response=expected_response,
                    embedding=embedding,
                    metadata={'context': context, 'relevance_score': 1.0}  # Max relevance for training data
                )

                training_count += 1

            except Exception as e:
                print(f"⚠ Training error on pair {training_count + 1}: {e}")

        self.metrics['total_training_examples'] += training_count
        print(f"✓ Training completed: {training_count} examples processed")
        print(f"📊 Substrate now has {self.substrate.graph.number_of_nodes()} concepts")

        # Save state after training
        if self.config.get('auto_save', True):
            self.save_state(self.persistent_state_path)
            print(f"💾 Training state saved")

    def load_training_from_file(self, filepath: str):
        """Load training data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                training_data = json.load(f)

            if 'conversations' in training_data:
                pairs = [(conv['input'], conv['output']) for conv in training_data['conversations']]
                context = training_data.get('context', '')
                self.train_on_conversation(pairs, context)
            else:
                print("⚠ Invalid training file format. Expected 'conversations' key.")

        except Exception as e:
            print(f"⚠ Failed to load training file: {e}")

    def export_training_data(self, filepath: str, include_memory: bool = True):
        """Export current knowledge as training data"""
        training_data = {
            'context': f"Neuromorphic ISC AI trained on {self.metrics['total_training_examples']} examples",
            'substrate_info': {
                'concepts': self.substrate.graph.number_of_nodes(),
                'relationships': self.substrate.graph.number_of_edges(),
                'phi_value': self.metrics['phi_value']
            },
            'conversations': []
        }

        if include_memory and hasattr(self.memory, 'interactions'):
            # Export memory interactions as training pairs
            for interaction in self.memory.interactions[-100:]:  # Last 100 interactions
                training_data['conversations'].append({
                    'input': interaction.get('user_input', ''),
                    'output': interaction.get('response', ''),
                    'context': interaction.get('context', ''),
                    'relevance': interaction.get('relevance_score', 0.0)
                })

        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"📁 Training data exported to {filepath}")
        print(f"   {len(training_data['conversations'])} conversation pairs")

    def continuous_learning_mode(self, enable: bool = True):
        """Enable/disable continuous learning from conversations"""
        self.config['continuous_learning'] = enable
        if enable:
            print("🧠 Continuous learning enabled - all conversations will train the model")
        else:
            print("🧠 Continuous learning disabled - manual training only")