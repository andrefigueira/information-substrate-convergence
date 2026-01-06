"""
Emergent Reasoning System

This module implements a genuinely novel approach to reasoning where:
1. Phi (integrated information) DRIVES reasoning, not just measures it
2. The substrate LEARNS new reasoning patterns from experience
3. Emergent capabilities can arise from the architecture
4. We can demonstrate that phi predicts reasoning accuracy

Key insight: Instead of hardcoding reasoning types, we let them emerge
from a self-organizing substrate that optimizes for integrated information.

Based on:
- Integrated Information Theory (Tononi et al.)
- Self-Organizing Maps (Kohonen)
- Neural Darwinism (Edelman)
- Predictive Processing (Friston)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random
import json
from datetime import datetime
import hashlib


@dataclass
class ReasoningTrace:
    """A trace of reasoning that can be learned from"""
    premise: str
    question: str
    answer: str
    success: bool
    confidence: float
    pathway: List[str]  # Which nodes were activated
    phi_contribution: float  # How much this trace contributed to phi
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_pattern(self) -> str:
        """Convert trace to a learnable pattern signature"""
        # Create a pattern from the pathway, not the specific content
        return "->".join(self.pathway)


@dataclass
class ReasoningNode:
    """A node in the reasoning substrate that can be activated"""
    node_id: str
    node_type: str  # 'input', 'hidden', 'output', 'emergent'
    activation: float = 0.0
    phi_contribution: float = 0.0
    connection_weights: Dict[str, float] = field(default_factory=dict)
    activation_history: List[float] = field(default_factory=list)
    learned_patterns: List[str] = field(default_factory=list)

    def update_phi_contribution(self, global_phi: float, local_activation: float):
        """Update this node's contribution to integrated information"""
        # Phi contribution = how much removing this node would reduce global phi
        # Approximated by activation * connectivity * uniqueness
        connectivity = len(self.connection_weights)
        uniqueness = 1.0 / (1.0 + np.std(self.activation_history) if self.activation_history else 1.0)
        self.phi_contribution = local_activation * connectivity * uniqueness * 0.1


class PhiDrivenSubstrate:
    """
    A reasoning substrate where phi actively drives computation.

    Key properties:
    - Nodes compete for activation based on phi contribution
    - High-phi pathways are strengthened (Hebbian + phi)
    - Low-phi pathways are pruned
    - New nodes can emerge from successful patterns
    """

    def __init__(self, initial_nodes: int = 50):
        self.nodes: Dict[str, ReasoningNode] = {}
        self.global_phi: float = 0.0
        self.phi_history: List[float] = []
        self.traces: List[ReasoningTrace] = []
        self.learned_rules: Dict[str, Dict[str, Any]] = {}
        self.emergent_patterns: List[Dict[str, Any]] = []
        self.generation: int = 0
        self.recent_pathway: List[str] = []  # Track recent active pathway

        # Initialize substrate with random connectivity
        self._initialize_substrate(initial_nodes)

    @property
    def edges(self) -> List[Tuple[str, str, float]]:
        """Return all edges as (source, target, weight) tuples"""
        edge_list = []
        for node_id, node in self.nodes.items():
            for source_id, weight in node.connection_weights.items():
                edge_list.append((source_id, node_id, weight))
        return edge_list

    def get_activation_pattern(self) -> np.ndarray:
        """Get current activation pattern as numpy array"""
        return np.array([n.activation for n in self.nodes.values()])

    def get_connectivity_matrix(self) -> np.ndarray:
        """Get connectivity as a flattened numpy array"""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        matrix = np.zeros((n, n))

        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        for target_id, node in self.nodes.items():
            target_idx = node_idx[target_id]
            for source_id, weight in node.connection_weights.items():
                if source_id in node_idx:
                    source_idx = node_idx[source_id]
                    matrix[source_idx, target_idx] = weight

        return matrix.flatten()

    def strengthen_recent_path(self, boost: float = 0.1):
        """Strengthen the most recently activated pathway"""
        if self.recent_pathway:
            self.strengthen_pathway(self.recent_pathway, success=True, magnitude=boost)

    def weaken_recent_path(self, penalty: float = 0.05):
        """Weaken the most recently activated pathway"""
        if self.recent_pathway:
            self.strengthen_pathway(self.recent_pathway, success=False, magnitude=penalty)

    def _initialize_substrate(self, n_nodes: int):
        """Create initial substrate with random but structured connectivity"""
        # Create input nodes (for premise/question encoding)
        for i in range(n_nodes // 5):
            node_id = f"input_{i}"
            self.nodes[node_id] = ReasoningNode(
                node_id=node_id,
                node_type='input',
                connection_weights={}
            )

        # Create hidden nodes (for reasoning)
        for i in range(n_nodes // 2):
            node_id = f"hidden_{i}"
            # Connect to random subset of input nodes
            connections = {}
            for inp_id in list(self.nodes.keys()):
                if random.random() < 0.3:  # 30% connectivity
                    connections[inp_id] = random.gauss(0.5, 0.2)

            self.nodes[node_id] = ReasoningNode(
                node_id=node_id,
                node_type='hidden',
                connection_weights=connections
            )

        # Create output nodes (for answer generation)
        hidden_ids = [n for n in self.nodes.keys() if n.startswith('hidden')]
        for i in range(n_nodes // 5):
            node_id = f"output_{i}"
            connections = {}
            for hid_id in hidden_ids:
                if random.random() < 0.4:
                    connections[hid_id] = random.gauss(0.5, 0.2)

            self.nodes[node_id] = ReasoningNode(
                node_id=node_id,
                node_type='output',
                connection_weights=connections
            )

        # Add some recurrent connections (self-reference)
        for node_id, node in self.nodes.items():
            if node.node_type == 'hidden' and random.random() < 0.1:
                # Connect to other hidden nodes
                other_hidden = [n for n in hidden_ids if n != node_id]
                if other_hidden:
                    target = random.choice(other_hidden)
                    node.connection_weights[target] = random.gauss(0.3, 0.1)

    def calculate_phi(self) -> float:
        """
        Calculate integrated information (phi) of the current substrate state.

        Phi = information generated by the whole above and beyond its parts.
        We approximate this using:
        - Mutual information between node clusters
        - Integration (how much the whole exceeds the sum of parts)
        """
        if len(self.nodes) < 2:
            return 0.0

        # Get activation pattern
        activations = np.array([n.activation for n in self.nodes.values()])

        if np.std(activations) < 0.01:
            return 0.0  # No differentiation = no phi

        # Calculate integration: correlation structure
        # High phi = nodes are correlated but not redundant

        # Split into partitions and compare information
        node_list = list(self.nodes.values())
        n = len(node_list)

        # Whole system entropy (approximated by activation variance)
        whole_entropy = np.var(activations) + 0.01

        # Partition system and calculate parts entropy
        mid = n // 2
        part1_activations = activations[:mid]
        part2_activations = activations[mid:]

        part1_entropy = np.var(part1_activations) + 0.01 if len(part1_activations) > 0 else 0.01
        part2_entropy = np.var(part2_activations) + 0.01 if len(part2_activations) > 0 else 0.01

        # Mutual information approximation
        # If parts are independent, their joint entropy = sum of individual
        # If integrated, joint entropy < sum (they share information)
        parts_sum = part1_entropy + part2_entropy

        # Phi = how much the whole exceeds the parts
        # Normalized to [0, 1]
        if parts_sum > 0:
            integration = 1.0 - (whole_entropy / parts_sum)
            phi = max(0.0, min(1.0, integration))
        else:
            phi = 0.0

        # Add contribution from connectivity structure
        avg_connectivity = np.mean([len(n.connection_weights) for n in self.nodes.values()])
        max_connectivity = len(self.nodes)
        connectivity_factor = avg_connectivity / max_connectivity if max_connectivity > 0 else 0

        # Final phi combines activation integration and structural integration
        phi = 0.7 * phi + 0.3 * connectivity_factor

        self.global_phi = phi
        self.phi_history.append(phi)

        # Update node phi contributions
        for node in self.nodes.values():
            node.update_phi_contribution(phi, node.activation)

        return phi

    def propagate_activation(self, input_pattern: Dict[str, float], steps: int = 5) -> Dict[str, float]:
        """
        Propagate activation through the substrate.

        Key innovation: Activation is MODULATED by phi contribution.
        High-phi nodes get boosted, low-phi nodes get suppressed.
        """
        # Set input activations
        for node_id, activation in input_pattern.items():
            if node_id in self.nodes:
                self.nodes[node_id].activation = activation

        # Propagate for several steps
        for step in range(steps):
            new_activations = {}

            for node_id, node in self.nodes.items():
                if node.node_type == 'input':
                    continue  # Input nodes don't update

                # Sum weighted inputs
                total_input = 0.0
                for source_id, weight in node.connection_weights.items():
                    if source_id in self.nodes:
                        source_activation = self.nodes[source_id].activation
                        # MODULATE by source's phi contribution
                        phi_boost = 1.0 + self.nodes[source_id].phi_contribution
                        total_input += source_activation * weight * phi_boost

                # Apply activation function (sigmoid)
                new_activation = 1.0 / (1.0 + np.exp(-total_input))

                # Phi-driven modulation: boost high-phi nodes
                phi_modulation = 1.0 + 0.5 * node.phi_contribution
                new_activation *= phi_modulation

                new_activations[node_id] = min(1.0, new_activation)

            # Update activations
            for node_id, activation in new_activations.items():
                self.nodes[node_id].activation = activation
                self.nodes[node_id].activation_history.append(activation)
                # Keep history bounded
                if len(self.nodes[node_id].activation_history) > 100:
                    self.nodes[node_id].activation_history.pop(0)

            # Recalculate phi after each step
            self.calculate_phi()

        # Track recent pathway (highly activated nodes)
        self.recent_pathway = [
            node_id for node_id, node in self.nodes.items()
            if node.activation > 0.5
        ]

        # Return output activations
        outputs = {
            node_id: node.activation
            for node_id, node in self.nodes.items()
            if node.node_type == 'output'
        }
        return outputs

    def strengthen_pathway(self, pathway: List[str], success: bool, magnitude: float = 0.1):
        """
        Strengthen or weaken a reasoning pathway based on success.

        This is Hebbian learning + phi:
        - Successful pathways get stronger
        - High-phi pathways get extra boost
        """
        modifier = magnitude if success else -magnitude * 0.5

        for i in range(len(pathway) - 1):
            source_id = pathway[i]
            target_id = pathway[i + 1]

            if source_id in self.nodes and target_id in self.nodes:
                target_node = self.nodes[target_id]

                if source_id in target_node.connection_weights:
                    # Phi-weighted learning: high-phi connections learn more
                    phi_boost = 1.0 + self.nodes[source_id].phi_contribution
                    delta = modifier * phi_boost

                    target_node.connection_weights[source_id] += delta
                    # Clamp weights
                    target_node.connection_weights[source_id] = max(
                        -1.0, min(1.0, target_node.connection_weights[source_id])
                    )

    def prune_low_phi_connections(self, threshold: float = 0.1):
        """Remove connections that don't contribute to phi"""
        for node in self.nodes.values():
            to_remove = []
            for source_id, weight in node.connection_weights.items():
                if source_id in self.nodes:
                    source_phi = self.nodes[source_id].phi_contribution
                    # Remove if low weight AND low phi contribution
                    if abs(weight) < threshold and source_phi < threshold:
                        to_remove.append(source_id)

            for source_id in to_remove:
                del node.connection_weights[source_id]

    def spawn_emergent_node(self, parent_nodes: List[str], pattern_signature: str):
        """
        Create a new node that captures a learned pattern.

        This is where emergence happens: successful patterns become
        first-class reasoning primitives.
        """
        node_id = f"emergent_{self.generation}_{len([n for n in self.nodes if n.startswith('emergent')])}"

        # New node connects to its parent nodes
        connections = {}
        for parent_id in parent_nodes:
            if parent_id in self.nodes:
                connections[parent_id] = 0.5  # Initial weight

        # Also connect to random hidden nodes for integration
        hidden_nodes = [n for n in self.nodes.keys() if n.startswith('hidden')]
        for hid in random.sample(hidden_nodes, min(3, len(hidden_nodes))):
            connections[hid] = random.gauss(0.3, 0.1)

        new_node = ReasoningNode(
            node_id=node_id,
            node_type='emergent',
            connection_weights=connections,
            learned_patterns=[pattern_signature]
        )

        self.nodes[node_id] = new_node

        # Connect some output nodes to this emergent node
        output_nodes = [n for n in self.nodes.keys() if n.startswith('output')]
        for out in random.sample(output_nodes, min(2, len(output_nodes))):
            self.nodes[out].connection_weights[node_id] = 0.4

        self.emergent_patterns.append({
            'node_id': node_id,
            'pattern': pattern_signature,
            'parent_nodes': parent_nodes,
            'generation': self.generation,
            'timestamp': datetime.now().isoformat()
        })

        return node_id


class EmergentReasoner:
    """
    A reasoning system that learns and evolves.

    Key innovations:
    1. Phi drives reasoning (not just measures it)
    2. Successful patterns become new reasoning primitives
    3. The system self-organizes toward high-phi configurations
    4. Emergent capabilities can arise
    """

    def __init__(self, substrate: Optional[PhiDrivenSubstrate] = None):
        self.substrate = substrate if substrate is not None else PhiDrivenSubstrate(initial_nodes=50)
        self.experience_buffer: List[ReasoningTrace] = []
        self.learned_mappings: Dict[str, str] = {}  # pattern -> answer
        self.accuracy_by_phi: List[Tuple[float, bool]] = []  # (phi, success) pairs
        self.generation = 0
        self.total_problems = 0
        self.correct_problems = 0

    def encode_input(self, premise: str, question: str) -> Dict[str, float]:
        """Encode premise and question as activation pattern"""
        # Simple bag-of-words encoding to input nodes
        text = f"{premise} {question}".lower()
        words = text.split()

        input_nodes = [n for n in self.substrate.nodes.keys() if n.startswith('input')]
        pattern = {}

        for i, node_id in enumerate(input_nodes):
            # Hash words to node indices
            activation = 0.0
            for word in words:
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                if word_hash % len(input_nodes) == i:
                    activation += 0.3
            pattern[node_id] = min(1.0, activation)

        return pattern

    def decode_output(self, output_activations: Dict[str, float], question: str) -> Tuple[str, float]:
        """Decode output activations to an answer"""
        if not output_activations:
            return "unknown", 0.3

        # Find most active output
        max_node = max(output_activations.items(), key=lambda x: x[1])
        max_activation = max_node[1]

        # Check learned mappings first
        pattern_key = self._get_activation_pattern_key(output_activations)
        if pattern_key in self.learned_mappings:
            return self.learned_mappings[pattern_key], max_activation

        # Determine answer type from question
        question_lower = question.lower()

        if any(q in question_lower for q in ['is it', 'are they', 'does', 'did', 'can', 'will']):
            # Yes/no question
            if max_activation > 0.6:
                return "yes", max_activation
            elif max_activation < 0.4:
                return "no", 1.0 - max_activation
            else:
                return "uncertain", 0.5

        # For other questions, return based on activation pattern
        return f"response_{max_node[0]}", max_activation

    def _get_activation_pattern_key(self, activations: Dict[str, float]) -> str:
        """Convert activation pattern to a hashable key"""
        # Discretize activations
        discrete = tuple(
            (k, round(v, 1))
            for k, v in sorted(activations.items())
        )
        return hashlib.md5(str(discrete).encode()).hexdigest()[:8]

    def reason(self, premise: str, question: str) -> Dict[str, Any]:
        """
        Perform reasoning using the phi-driven substrate.

        Returns answer and metadata about the reasoning process.
        """
        self.total_problems += 1

        # Encode input
        input_pattern = self.encode_input(premise, question)

        # Get active pathway (which nodes fire)
        initial_phi = self.substrate.calculate_phi()

        # Propagate through substrate
        output_activations = self.substrate.propagate_activation(input_pattern, steps=5)

        # Calculate final phi
        final_phi = self.substrate.calculate_phi()

        # Decode answer
        answer, confidence = self.decode_output(output_activations, question)

        # Get the pathway (highly activated nodes)
        pathway = [
            node_id for node_id, node in self.substrate.nodes.items()
            if node.activation > 0.5
        ]

        return {
            'answer': answer,
            'confidence': confidence,
            'initial_phi': initial_phi,
            'final_phi': final_phi,
            'phi_change': final_phi - initial_phi,
            'pathway': pathway,
            'pathway_length': len(pathway),
            'emergent_nodes_used': len([p for p in pathway if p.startswith('emergent')])
        }

    def learn_from_feedback(self, premise: str, question: str, correct_answer: str,
                           predicted_answer: str, was_correct: bool):
        """
        Learn from reasoning outcome.

        This is where the magic happens:
        1. Strengthen/weaken pathways based on success
        2. Extract patterns from successful reasoning
        3. Spawn emergent nodes for new patterns
        4. Track phi-accuracy correlation
        """
        # Get current phi
        current_phi = self.substrate.global_phi

        # Record phi-accuracy pair
        self.accuracy_by_phi.append((current_phi, was_correct))

        if was_correct:
            self.correct_problems += 1

        # Create trace
        pathway = [
            node_id for node_id, node in self.substrate.nodes.items()
            if node.activation > 0.5
        ]

        trace = ReasoningTrace(
            premise=premise,
            question=question,
            answer=correct_answer if was_correct else predicted_answer,
            success=was_correct,
            confidence=self.substrate.global_phi,
            pathway=pathway,
            phi_contribution=current_phi
        )

        self.experience_buffer.append(trace)

        # Strengthen or weaken the pathway
        self.substrate.strengthen_pathway(pathway, was_correct)

        # Learn mapping if successful
        if was_correct:
            input_pattern = self.encode_input(premise, question)
            output_activations = self.substrate.propagate_activation(input_pattern, steps=1)
            pattern_key = self._get_activation_pattern_key(output_activations)
            self.learned_mappings[pattern_key] = correct_answer

        # Check for emergent pattern opportunities
        if len(self.experience_buffer) >= 5:
            self._check_for_emergent_patterns()

        # Periodically prune low-phi connections
        if self.total_problems % 10 == 0:
            self.substrate.prune_low_phi_connections()
            self.generation += 1
            self.substrate.generation = self.generation

    def _check_for_emergent_patterns(self):
        """
        Look for recurring successful patterns that should become emergent nodes.

        This is how new reasoning capabilities emerge:
        - If the same pathway succeeds multiple times
        - Create a new node that captures that pathway
        """
        # Get recent successful traces
        recent_successes = [
            t for t in self.experience_buffer[-20:]
            if t.success
        ]

        if len(recent_successes) < 3:
            return

        # Find recurring pathway patterns
        pattern_counts = defaultdict(list)
        for trace in recent_successes:
            pattern = trace.to_pattern()
            pattern_counts[pattern].append(trace)

        # Spawn emergent nodes for patterns that appear 3+ times
        for pattern, traces in pattern_counts.items():
            if len(traces) >= 3:
                # Check if we already have this pattern
                existing = any(
                    pattern in node.learned_patterns
                    for node in self.substrate.nodes.values()
                    if node.node_type == 'emergent'
                )

                if not existing:
                    # Get parent nodes from the pathway
                    parent_nodes = traces[0].pathway[:5]  # First 5 nodes

                    new_node_id = self.substrate.spawn_emergent_node(
                        parent_nodes, pattern
                    )

                    print(f"[EMERGENT] New reasoning primitive: {new_node_id}")
                    print(f"  Pattern: {pattern[:50]}...")
                    print(f"  Based on {len(traces)} successful traces")

    def get_phi_accuracy_correlation(self) -> Dict[str, Any]:
        """
        Analyze whether phi predicts reasoning accuracy.

        This is the key experiment: does integrated information
        actually correlate with reasoning capability?
        """
        if len(self.accuracy_by_phi) < 10:
            return {'error': 'Not enough data', 'n_samples': len(self.accuracy_by_phi)}

        phis = np.array([p[0] for p in self.accuracy_by_phi])
        accuracies = np.array([1.0 if p[1] else 0.0 for p in self.accuracy_by_phi])

        # Correlation coefficient
        if np.std(phis) > 0 and np.std(accuracies) > 0:
            correlation = np.corrcoef(phis, accuracies)[0, 1]
        else:
            correlation = 0.0

        # Bin by phi and calculate accuracy per bin
        phi_bins = np.linspace(0, 1, 6)
        bin_accuracies = []
        for i in range(len(phi_bins) - 1):
            mask = (phis >= phi_bins[i]) & (phis < phi_bins[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(accuracies[mask])
                bin_accuracies.append({
                    'phi_range': f"{phi_bins[i]:.2f}-{phi_bins[i+1]:.2f}",
                    'accuracy': bin_acc,
                    'n_samples': int(np.sum(mask))
                })

        return {
            'correlation': correlation,
            'n_samples': len(self.accuracy_by_phi),
            'mean_phi': float(np.mean(phis)),
            'mean_accuracy': float(np.mean(accuracies)),
            'accuracy_by_phi_bin': bin_accuracies,
            'interpretation': self._interpret_correlation(correlation)
        }

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret the phi-accuracy correlation"""
        if corr > 0.5:
            return "STRONG POSITIVE: Higher phi strongly predicts better reasoning"
        elif corr > 0.2:
            return "MODERATE POSITIVE: Phi somewhat predicts reasoning accuracy"
        elif corr > -0.2:
            return "WEAK/NONE: Phi does not clearly predict reasoning accuracy"
        elif corr > -0.5:
            return "MODERATE NEGATIVE: Higher phi associated with worse reasoning (unexpected)"
        else:
            return "STRONG NEGATIVE: Higher phi strongly predicts worse reasoning (paradoxical)"

    def get_emergence_report(self) -> Dict[str, Any]:
        """Report on emergent capabilities"""
        emergent_nodes = [
            n for n in self.substrate.nodes.values()
            if n.node_type == 'emergent'
        ]

        return {
            'n_emergent_nodes': len(emergent_nodes),
            'emergent_patterns': self.substrate.emergent_patterns,
            'generation': self.generation,
            'total_nodes': len(self.substrate.nodes),
            'learned_mappings': len(self.learned_mappings),
            'experience_buffer_size': len(self.experience_buffer)
        }

    def get_substrate_stats(self) -> Dict[str, Any]:
        """Get statistics about the substrate"""
        nodes = list(self.substrate.nodes.values())

        return {
            'total_nodes': len(nodes),
            'node_types': {
                'input': len([n for n in nodes if n.node_type == 'input']),
                'hidden': len([n for n in nodes if n.node_type == 'hidden']),
                'output': len([n for n in nodes if n.node_type == 'output']),
                'emergent': len([n for n in nodes if n.node_type == 'emergent'])
            },
            'avg_connectivity': np.mean([len(n.connection_weights) for n in nodes]),
            'current_phi': self.substrate.global_phi,
            'phi_history_length': len(self.substrate.phi_history),
            'avg_phi': np.mean(self.substrate.phi_history) if self.substrate.phi_history else 0,
            'accuracy': self.correct_problems / self.total_problems if self.total_problems > 0 else 0,
            'total_problems': self.total_problems
        }


def run_emergence_experiment(n_iterations: int = 100) -> Dict[str, Any]:
    """
    Run an experiment to test if:
    1. Phi correlates with accuracy
    2. Emergent patterns form
    3. The system improves over time
    """
    reasoner = EmergentReasoner()

    # Test problems of varying types
    test_problems = [
        # Deductive
        ("If it rains, the ground gets wet. It is raining.", "Is the ground wet?", "yes"),
        ("All birds can fly. Tweety is a bird.", "Can Tweety fly?", "yes"),
        ("If A then B. Not B.", "Is A true?", "no"),

        # Simple yes/no
        ("The sky is blue.", "Is the sky blue?", "yes"),
        ("Water freezes at 0 degrees Celsius.", "Does water freeze at 0C?", "yes"),
        ("The sun rises in the east.", "Does the sun rise in the west?", "no"),

        # Causal
        ("Smoking causes cancer.", "Does smoking cause cancer?", "yes"),
        ("Ice cream sales correlate with drowning.", "Does ice cream cause drowning?", "no"),

        # Varied
        ("2 + 2 = 4", "Is 2 + 2 equal to 4?", "yes"),
        ("Paris is in France.", "Is Paris in France?", "yes"),
    ]

    results_over_time = []

    for iteration in range(n_iterations):
        # Pick a random problem
        premise, question, correct = random.choice(test_problems)

        # Reason
        result = reasoner.reason(premise, question)
        predicted = result['answer']

        # Check if correct (flexible matching)
        was_correct = (
            predicted.lower() == correct.lower() or
            correct.lower() in predicted.lower()
        )

        # Learn from feedback
        reasoner.learn_from_feedback(
            premise, question, correct, predicted, was_correct
        )

        # Record progress every 10 iterations
        if iteration % 10 == 0:
            stats = reasoner.get_substrate_stats()
            results_over_time.append({
                'iteration': iteration,
                'accuracy': stats['accuracy'],
                'phi': stats['current_phi'],
                'emergent_nodes': stats['node_types']['emergent'],
                'total_nodes': stats['total_nodes']
            })

    # Final analysis
    correlation = reasoner.get_phi_accuracy_correlation()
    emergence = reasoner.get_emergence_report()
    final_stats = reasoner.get_substrate_stats()

    return {
        'final_accuracy': final_stats['accuracy'],
        'final_phi': final_stats['current_phi'],
        'phi_accuracy_correlation': correlation,
        'emergence_report': emergence,
        'results_over_time': results_over_time,
        'substrate_stats': final_stats
    }


def run_phi_driven_experiment(n_iterations: int = 500) -> Dict[str, Any]:
    """
    Run a longer experiment that specifically tests whether:
    1. High-phi configurations reason better
    2. The system self-organizes toward high-phi states
    3. Learning actually improves over time

    This is the key experiment for validating the ISC thesis.
    """
    reasoner = EmergentReasoner()

    # More diverse test problems
    test_problems = [
        # Modus ponens
        ("If it rains, the ground gets wet. It is raining.", "Is the ground wet?", "yes"),
        ("If you study, you pass. You studied.", "Did you pass?", "yes"),
        ("If the alarm rings, wake up. The alarm is ringing.", "Should you wake up?", "yes"),

        # Modus tollens
        ("If it rains, the ground gets wet. The ground is dry.", "Is it raining?", "no"),
        ("If guilty, evidence exists. No evidence exists.", "Is the person guilty?", "no"),

        # Syllogisms
        ("All humans are mortal. Socrates is human.", "Is Socrates mortal?", "yes"),
        ("All dogs are mammals. Fido is a dog.", "Is Fido a mammal?", "yes"),
        ("No reptiles are mammals. Snakes are reptiles.", "Are snakes mammals?", "no"),

        # Causal
        ("Smoking causes cancer. John smokes.", "Is John at risk for cancer?", "yes"),
        ("Correlation exists between X and Y. No mechanism known.", "Does X cause Y?", "no"),

        # Negations
        ("The door is closed.", "Is the door open?", "no"),
        ("It is not raining.", "Is it raining?", "no"),
        ("The test passed.", "Did the test fail?", "no"),

        # Affirmations
        ("The sky is blue.", "Is the sky blue?", "yes"),
        ("Water is wet.", "Is water wet?", "yes"),
        ("Fire is hot.", "Is fire hot?", "yes"),
    ]

    # Track metrics over time
    accuracy_over_time = []
    phi_over_time = []
    emergence_over_time = []

    # Track per-problem-type accuracy
    problem_type_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

    print("Running Phi-Driven Emergence Experiment...")
    print("=" * 60)

    for iteration in range(n_iterations):
        premise, question, correct = random.choice(test_problems)

        # Determine problem type
        if 'if' in premise.lower() and 'not' not in premise.lower():
            if 'dry' in premise.lower() or 'no evidence' in premise.lower():
                problem_type = 'modus_tollens'
            else:
                problem_type = 'modus_ponens'
        elif 'all' in premise.lower() or 'no ' in premise.lower():
            problem_type = 'syllogism'
        elif 'cause' in premise.lower() or 'correlation' in premise.lower():
            problem_type = 'causal'
        elif 'not' in premise.lower() or 'closed' in premise.lower():
            problem_type = 'negation'
        else:
            problem_type = 'affirmation'

        # Reason
        result = reasoner.reason(premise, question)
        predicted = result['answer']

        was_correct = (
            predicted.lower() == correct.lower() or
            correct.lower() in predicted.lower()
        )

        # Track per-type accuracy
        problem_type_accuracy[problem_type]['total'] += 1
        if was_correct:
            problem_type_accuracy[problem_type]['correct'] += 1

        # Learn
        reasoner.learn_from_feedback(premise, question, correct, predicted, was_correct)

        # Record every 50 iterations
        if iteration % 50 == 0:
            stats = reasoner.get_substrate_stats()
            accuracy_over_time.append(stats['accuracy'])
            phi_over_time.append(stats['current_phi'])
            emergence_over_time.append(stats['node_types']['emergent'])

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Acc={stats['accuracy']*100:.1f}%, "
                      f"Phi={stats['current_phi']:.3f}, "
                      f"Emergent={stats['node_types']['emergent']}")

    # Final analysis
    correlation = reasoner.get_phi_accuracy_correlation()
    emergence = reasoner.get_emergence_report()
    final_stats = reasoner.get_substrate_stats()

    # Calculate improvement
    if len(accuracy_over_time) >= 2:
        early_accuracy = np.mean(accuracy_over_time[:3])
        late_accuracy = np.mean(accuracy_over_time[-3:])
        improvement = late_accuracy - early_accuracy
    else:
        improvement = 0

    # Calculate per-type results
    type_results = {}
    for ptype, data in problem_type_accuracy.items():
        if data['total'] > 0:
            type_results[ptype] = {
                'accuracy': data['correct'] / data['total'],
                'total': data['total']
            }

    return {
        'final_accuracy': final_stats['accuracy'],
        'final_phi': final_stats['current_phi'],
        'phi_accuracy_correlation': correlation,
        'emergence_report': emergence,
        'substrate_stats': final_stats,
        'improvement': improvement,
        'accuracy_over_time': accuracy_over_time,
        'phi_over_time': phi_over_time,
        'emergence_over_time': emergence_over_time,
        'accuracy_by_problem_type': type_results,
        'did_phi_drive_improvement': correlation['correlation'] > 0.15 and improvement > 0.05,
        'did_emergence_occur': emergence['n_emergent_nodes'] > 5,
        'did_self_organize': np.mean(phi_over_time[-3:]) > np.mean(phi_over_time[:3]) if len(phi_over_time) >= 6 else False
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ISC THESIS VALIDATION EXPERIMENT")
    print("Testing: Does integrated information drive reasoning capability?")
    print("=" * 70 + "\n")

    results = run_phi_driven_experiment(n_iterations=500)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n1. ACCURACY")
    print(f"   Final: {results['final_accuracy']*100:.1f}%")
    print(f"   Improvement over time: {results['improvement']*100:+.1f}%")

    print(f"\n2. PHI (Integrated Information)")
    print(f"   Final Phi: {results['final_phi']:.3f}")
    print(f"   Phi trend: {'INCREASING' if results['did_self_organize'] else 'STABLE/DECREASING'}")

    print(f"\n3. PHI-ACCURACY CORRELATION")
    corr = results['phi_accuracy_correlation']
    print(f"   Correlation: {corr['correlation']:.3f}")
    print(f"   Interpretation: {corr['interpretation']}")

    print(f"\n4. EMERGENCE")
    print(f"   Emergent nodes created: {results['emergence_report']['n_emergent_nodes']}")
    print(f"   Learned mappings: {results['emergence_report']['learned_mappings']}")
    print(f"   Emergence occurred: {'YES' if results['did_emergence_occur'] else 'NO'}")

    print(f"\n5. ACCURACY BY PROBLEM TYPE")
    for ptype, data in sorted(results['accuracy_by_problem_type'].items()):
        print(f"   {ptype}: {data['accuracy']*100:.0f}% (n={data['total']})")

    print("\n" + "=" * 70)
    print("ISC THESIS VALIDATION")
    print("=" * 70)

    validations = []
    if results['did_phi_drive_improvement']:
        validations.append("+ Phi correlates with reasoning accuracy")
    else:
        validations.append("- Phi does NOT clearly correlate with accuracy")

    if results['did_emergence_occur']:
        validations.append("+ Emergent reasoning primitives formed")
    else:
        validations.append("- No significant emergence observed")

    if results['did_self_organize']:
        validations.append("+ System self-organized toward higher phi")
    else:
        validations.append("- System did NOT self-organize toward higher phi")

    if results['improvement'] > 0.05:
        validations.append("+ System improved through learning")
    else:
        validations.append("- System did NOT improve significantly")

    for v in validations:
        print(f"   {v}")

    thesis_support = sum(1 for v in validations if v.startswith('+'))
    print(f"\n   THESIS SUPPORT: {thesis_support}/4 criteria met")

    if thesis_support >= 3:
        print("   CONCLUSION: Strong support for ISC thesis")
    elif thesis_support >= 2:
        print("   CONCLUSION: Partial support for ISC thesis")
    else:
        print("   CONCLUSION: Weak/no support for ISC thesis")
