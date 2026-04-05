"""
Novel Reasoning Generation through Cognitive Model Evolution

This module evolves novel reasoning strategies by:
1. Treating cognitive patterns as "genomes"
2. Using evolutionary algorithms to combine and mutate patterns
3. Evaluating fitness based on problem-solving effectiveness
4. Discovering reasoning approaches no single human has developed

This is inspired by the cellular automata evolution in the ISC project
but applied to cognitive reasoning patterns.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import random
import hashlib
from datetime import datetime
import json
from pathlib import Path

from .cognitive_architecture import CognitiveProfile, ReasoningPattern, CognitiveComposer


@dataclass
class ReasoningGenome:
    """
    A "genome" representing a reasoning strategy.

    Encodes:
    - Cognitive style weights
    - Concept association patterns
    - Reasoning sequence templates
    - Domain transfer rules
    """
    genome_id: str
    generation: int = 0

    # Cognitive style genes (0-1)
    analytical_gene: float = 0.5
    intuitive_gene: float = 0.5
    systematic_gene: float = 0.5
    creative_gene: float = 0.5
    detail_gene: float = 0.5
    abstraction_gene: float = 0.5

    # Reasoning sequence genes
    reasoning_steps: List[str] = field(default_factory=lambda: ['observe', 'connect', 'conclude'])

    # Domain transfer genes (which domains to pull from)
    domain_weights: Dict[str, float] = field(default_factory=dict)

    # Fitness metrics
    fitness_score: float = 0.0
    evaluation_count: int = 0

    # Lineage
    parent_ids: List[str] = field(default_factory=list)

    def mutate(self, mutation_rate: float = 0.1) -> 'ReasoningGenome':
        """Create a mutated copy of this genome"""
        mutated = ReasoningGenome(
            genome_id=hashlib.md5(
                f"{self.genome_id}_{random.random()}".encode()
            ).hexdigest()[:10],
            generation=self.generation + 1,
            parent_ids=[self.genome_id]
        )

        # Mutate cognitive style genes
        mutated.analytical_gene = self._mutate_gene(self.analytical_gene, mutation_rate)
        mutated.intuitive_gene = self._mutate_gene(self.intuitive_gene, mutation_rate)
        mutated.systematic_gene = self._mutate_gene(self.systematic_gene, mutation_rate)
        mutated.creative_gene = self._mutate_gene(self.creative_gene, mutation_rate)
        mutated.detail_gene = self._mutate_gene(self.detail_gene, mutation_rate)
        mutated.abstraction_gene = self._mutate_gene(self.abstraction_gene, mutation_rate)

        # Mutate reasoning steps
        mutated.reasoning_steps = self.reasoning_steps.copy()
        if random.random() < mutation_rate:
            possible_steps = [
                'observe', 'analyze', 'connect', 'abstract', 'synthesize',
                'question', 'hypothesize', 'test', 'conclude', 'generalize',
                'specialize', 'analogize', 'decompose', 'integrate'
            ]
            if random.random() < 0.5 and len(mutated.reasoning_steps) > 2:
                # Remove a step
                mutated.reasoning_steps.pop(random.randint(0, len(mutated.reasoning_steps) - 1))
            else:
                # Add or replace a step
                new_step = random.choice(possible_steps)
                if len(mutated.reasoning_steps) < 7:
                    mutated.reasoning_steps.insert(
                        random.randint(0, len(mutated.reasoning_steps)),
                        new_step
                    )

        # Mutate domain weights
        mutated.domain_weights = self.domain_weights.copy()
        if random.random() < mutation_rate:
            domains = ['science', 'technology', 'philosophy', 'psychology', 'art', 'social', 'practical']
            domain = random.choice(domains)
            mutated.domain_weights[domain] = random.random()

        return mutated

    def _mutate_gene(self, value: float, mutation_rate: float) -> float:
        """Mutate a single gene value"""
        if random.random() < mutation_rate:
            # Add Gaussian noise
            value += random.gauss(0, 0.2)
            value = max(0.0, min(1.0, value))
        return value

    def crossover(self, other: 'ReasoningGenome') -> 'ReasoningGenome':
        """Create offspring by crossing over with another genome"""
        child = ReasoningGenome(
            genome_id=hashlib.md5(
                f"{self.genome_id}_{other.genome_id}_{random.random()}".encode()
            ).hexdigest()[:10],
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.genome_id, other.genome_id]
        )

        # Crossover cognitive genes (random selection)
        child.analytical_gene = random.choice([self.analytical_gene, other.analytical_gene])
        child.intuitive_gene = random.choice([self.intuitive_gene, other.intuitive_gene])
        child.systematic_gene = random.choice([self.systematic_gene, other.systematic_gene])
        child.creative_gene = random.choice([self.creative_gene, other.creative_gene])
        child.detail_gene = random.choice([self.detail_gene, other.detail_gene])
        child.abstraction_gene = random.choice([self.abstraction_gene, other.abstraction_gene])

        # Crossover reasoning steps (interleave)
        child.reasoning_steps = []
        max_len = max(len(self.reasoning_steps), len(other.reasoning_steps))
        for i in range(max_len):
            if i < len(self.reasoning_steps) and random.random() < 0.5:
                child.reasoning_steps.append(self.reasoning_steps[i])
            if i < len(other.reasoning_steps) and random.random() < 0.5:
                child.reasoning_steps.append(other.reasoning_steps[i])

        if not child.reasoning_steps:
            child.reasoning_steps = ['observe', 'conclude']

        # Crossover domain weights
        all_domains = set(self.domain_weights.keys()).union(other.domain_weights.keys())
        for domain in all_domains:
            w1 = self.domain_weights.get(domain, 0.5)
            w2 = other.domain_weights.get(domain, 0.5)
            child.domain_weights[domain] = random.choice([w1, w2])

        return child

    def to_cognitive_profile(self) -> CognitiveProfile:
        """Convert genome to a cognitive profile for reasoning"""
        profile = CognitiveProfile(profile_id=f"evolved_{self.genome_id}")
        profile.analytical_tendency = self.analytical_gene
        profile.intuitive_tendency = self.intuitive_gene
        profile.systematic_tendency = self.systematic_gene
        profile.creative_tendency = self.creative_gene
        profile.detail_orientation = self.detail_gene
        profile.big_picture_orientation = self.abstraction_gene

        # Convert reasoning steps to a pattern
        pattern = ReasoningPattern(
            pattern_id=f"evolved_{self.genome_id}",
            pattern_type="evolved",
            trigger_concepts=set(),
            typical_transitions=[(self.reasoning_steps[i], self.reasoning_steps[i+1])
                                for i in range(len(self.reasoning_steps) - 1)],
            confidence=self.fitness_score,
            frequency=self.evaluation_count
        )
        profile.reasoning_patterns[pattern.pattern_id] = pattern

        # Set domain preferences
        profile.preferred_domains = self.domain_weights.copy()

        return profile

    def to_dict(self) -> Dict:
        return {
            'genome_id': self.genome_id,
            'generation': self.generation,
            'analytical_gene': self.analytical_gene,
            'intuitive_gene': self.intuitive_gene,
            'systematic_gene': self.systematic_gene,
            'creative_gene': self.creative_gene,
            'detail_gene': self.detail_gene,
            'abstraction_gene': self.abstraction_gene,
            'reasoning_steps': self.reasoning_steps,
            'domain_weights': self.domain_weights,
            'fitness_score': self.fitness_score,
            'evaluation_count': self.evaluation_count,
            'parent_ids': self.parent_ids
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReasoningGenome':
        genome = cls(
            genome_id=data['genome_id'],
            generation=data.get('generation', 0)
        )
        genome.analytical_gene = data.get('analytical_gene', 0.5)
        genome.intuitive_gene = data.get('intuitive_gene', 0.5)
        genome.systematic_gene = data.get('systematic_gene', 0.5)
        genome.creative_gene = data.get('creative_gene', 0.5)
        genome.detail_gene = data.get('detail_gene', 0.5)
        genome.abstraction_gene = data.get('abstraction_gene', 0.5)
        genome.reasoning_steps = data.get('reasoning_steps', ['observe', 'connect', 'conclude'])
        genome.domain_weights = data.get('domain_weights', {})
        genome.fitness_score = data.get('fitness_score', 0.0)
        genome.evaluation_count = data.get('evaluation_count', 0)
        genome.parent_ids = data.get('parent_ids', [])
        return genome

    @classmethod
    def from_cognitive_profile(cls, profile: CognitiveProfile) -> 'ReasoningGenome':
        """Create a genome from an existing cognitive profile"""
        genome = cls(
            genome_id=f"from_{profile.profile_id[:8]}",
            generation=0
        )
        genome.analytical_gene = profile.analytical_tendency
        genome.intuitive_gene = profile.intuitive_tendency
        genome.systematic_gene = profile.systematic_tendency
        genome.creative_gene = profile.creative_tendency
        genome.detail_gene = profile.detail_orientation
        genome.abstraction_gene = profile.big_picture_orientation
        genome.domain_weights = profile.preferred_domains.copy()

        # Extract reasoning steps from patterns
        if profile.reasoning_patterns:
            pattern = list(profile.reasoning_patterns.values())[0]
            if pattern.typical_transitions:
                steps = [pattern.typical_transitions[0][0]]
                for _, step in pattern.typical_transitions:
                    steps.append(step)
                genome.reasoning_steps = steps[:7]

        return genome


class ReasoningEvolutionEngine:
    """
    Evolves novel reasoning strategies through genetic algorithms.

    This is the core of the "novel reasoning generation" feature.
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.1,
        save_path: str = "models/reasoning_evolution"
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.population: List[ReasoningGenome] = []
        self.generation = 0
        self.best_genome: Optional[ReasoningGenome] = None
        self.fitness_history: List[float] = []

        # Load or initialize population
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing population or create new one"""
        state_path = self.save_path / "evolution_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.population = [
                    ReasoningGenome.from_dict(g)
                    for g in state.get('population', [])
                ]
                self.generation = state.get('generation', 0)
                self.fitness_history = state.get('fitness_history', [])
                if state.get('best_genome'):
                    self.best_genome = ReasoningGenome.from_dict(state['best_genome'])
            except Exception as e:
                print(f"Warning: Could not load evolution state: {e}")
                self._initialize_population()
        else:
            self._initialize_population()

    def _initialize_population(self):
        """Create initial random population"""
        self.population = []

        # Create diverse initial population
        reasoning_archetypes = [
            {'analytical': 0.9, 'systematic': 0.8, 'detail': 0.8},
            {'intuitive': 0.9, 'creative': 0.8, 'abstraction': 0.8},
            {'systematic': 0.9, 'detail': 0.9, 'analytical': 0.6},
            {'creative': 0.9, 'abstraction': 0.9, 'intuitive': 0.7},
            {'analytical': 0.7, 'creative': 0.7, 'abstraction': 0.7},  # Balanced
        ]

        # Create some from archetypes
        for archetype in reasoning_archetypes:
            genome = ReasoningGenome(
                genome_id=hashlib.md5(str(archetype).encode()).hexdigest()[:10]
            )
            genome.analytical_gene = archetype.get('analytical', 0.5)
            genome.intuitive_gene = archetype.get('intuitive', 0.5)
            genome.systematic_gene = archetype.get('systematic', 0.5)
            genome.creative_gene = archetype.get('creative', 0.5)
            genome.detail_gene = archetype.get('detail', 0.5)
            genome.abstraction_gene = archetype.get('abstraction', 0.5)
            self.population.append(genome)

        # Fill rest with random
        while len(self.population) < self.population_size:
            genome = ReasoningGenome(
                genome_id=hashlib.md5(str(random.random()).encode()).hexdigest()[:10]
            )
            genome.analytical_gene = random.random()
            genome.intuitive_gene = random.random()
            genome.systematic_gene = random.random()
            genome.creative_gene = random.random()
            genome.detail_gene = random.random()
            genome.abstraction_gene = random.random()
            genome.reasoning_steps = random.sample(
                ['observe', 'analyze', 'connect', 'abstract', 'synthesize',
                 'question', 'hypothesize', 'conclude'],
                k=random.randint(3, 6)
            )
            self.population.append(genome)

    def seed_from_profiles(self, profiles: List[CognitiveProfile]):
        """Seed the population with genomes derived from real cognitive profiles"""
        for profile in profiles[:self.population_size // 2]:
            genome = ReasoningGenome.from_cognitive_profile(profile)
            # Replace a random low-fitness genome
            if self.population:
                worst_idx = min(
                    range(len(self.population)),
                    key=lambda i: self.population[i].fitness_score
                )
                self.population[worst_idx] = genome

    def evaluate_genome(
        self,
        genome: ReasoningGenome,
        problem_context: str,
        evaluator: Optional[Callable[[ReasoningGenome, str], float]] = None
    ) -> float:
        """
        Evaluate a genome's fitness for a given problem.

        If no evaluator is provided, uses heuristic fitness.
        """
        if evaluator:
            fitness = evaluator(genome, problem_context)
        else:
            fitness = self._heuristic_fitness(genome, problem_context)

        # Update genome's fitness (moving average)
        genome.evaluation_count += 1
        genome.fitness_score = (
            (genome.fitness_score * (genome.evaluation_count - 1) + fitness)
            / genome.evaluation_count
        )

        return fitness

    def _heuristic_fitness(self, genome: ReasoningGenome, problem_context: str) -> float:
        """
        Heuristic fitness function based on genome coherence and diversity.

        A good reasoning strategy should:
        1. Have coherent (not contradictory) cognitive styles
        2. Have diverse reasoning steps
        3. Match problem domain requirements
        """
        fitness = 0.5  # Base fitness

        # Coherence: analytical+systematic or intuitive+creative tend to work together
        analytical_cluster = (genome.analytical_gene + genome.systematic_gene + genome.detail_gene) / 3
        creative_cluster = (genome.intuitive_gene + genome.creative_gene + genome.abstraction_gene) / 3

        # Reward specialization or balanced approach
        specialization = abs(analytical_cluster - creative_cluster)
        fitness += 0.2 * (specialization if specialization > 0.3 else 1 - specialization)

        # Diversity in reasoning steps
        unique_steps = len(set(genome.reasoning_steps))
        total_steps = len(genome.reasoning_steps)
        step_diversity = unique_steps / max(total_steps, 1)
        fitness += 0.2 * step_diversity

        # Reasonable step count (not too few, not too many)
        step_count_score = 1.0 - abs(len(genome.reasoning_steps) - 4) / 4
        fitness += 0.1 * max(0, step_count_score)

        return min(1.0, max(0.0, fitness))

    def evolve_generation(
        self,
        problem_contexts: List[str],
        evaluator: Optional[Callable[[ReasoningGenome, str], float]] = None
    ) -> Dict[str, Any]:
        """
        Evolve one generation of the population.

        Returns statistics about the evolution.
        """
        # Evaluate all genomes
        for genome in self.population:
            for context in problem_contexts:
                self.evaluate_genome(genome, context, evaluator)

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)

        # Track best
        if self.best_genome is None or self.population[0].fitness_score > self.best_genome.fitness_score:
            self.best_genome = self.population[0]

        # Record fitness
        avg_fitness = np.mean([g.fitness_score for g in self.population])
        self.fitness_history.append(avg_fitness)

        # Create next generation
        new_population = []

        # Keep elite
        elite_count = int(self.population_size * self.elite_ratio)
        new_population.extend(self.population[:elite_count])

        # Create offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = parent1.crossover(parent2)
            else:
                # Clone and mutate
                parent = self._tournament_select()
                child = parent.mutate(self.mutation_rate)

            # Always apply some mutation
            child = child.mutate(self.mutation_rate * 0.5)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Save state
        self._save_state()

        return {
            'generation': self.generation,
            'best_fitness': self.best_genome.fitness_score if self.best_genome else 0,
            'avg_fitness': avg_fitness,
            'best_genome_id': self.best_genome.genome_id if self.best_genome else None
        }

    def _tournament_select(self, tournament_size: int = 3) -> ReasoningGenome:
        """Select a genome using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness_score)

    def get_best_strategy_for_problem(
        self,
        problem_context: str,
        top_k: int = 3
    ) -> List[ReasoningGenome]:
        """
        Get the best evolved reasoning strategies for a specific problem.
        """
        # Evaluate all genomes for this specific problem
        scored = []
        for genome in self.population:
            fitness = self._heuristic_fitness(genome, problem_context)
            scored.append((genome, fitness))

        scored.sort(key=lambda x: -x[1])
        return [g for g, _ in scored[:top_k]]

    def generate_novel_strategy(self) -> ReasoningGenome:
        """
        Generate a novel reasoning strategy by combining best traits
        from multiple successful genomes.
        """
        if len(self.population) < 3:
            return self.population[0] if self.population else ReasoningGenome(genome_id="default")

        # Get top performers
        top_genomes = sorted(self.population, key=lambda g: -g.fitness_score)[:5]

        # Create a super-genome combining best traits
        novel = ReasoningGenome(
            genome_id=f"novel_{self.generation}_{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}"
        )

        # Take best gene from each trait
        novel.analytical_gene = max(g.analytical_gene for g in top_genomes)
        novel.systematic_gene = max(g.systematic_gene for g in top_genomes)

        # Average for balance traits
        novel.intuitive_gene = np.mean([g.intuitive_gene for g in top_genomes])
        novel.creative_gene = np.mean([g.creative_gene for g in top_genomes])
        novel.detail_gene = np.mean([g.detail_gene for g in top_genomes])
        novel.abstraction_gene = np.mean([g.abstraction_gene for g in top_genomes])

        # Combine reasoning steps from best genomes
        all_steps = []
        for g in top_genomes:
            all_steps.extend(g.reasoning_steps)

        # Keep most common steps
        step_counts = defaultdict(int)
        for step in all_steps:
            step_counts[step] += 1

        novel.reasoning_steps = [
            step for step, _ in sorted(step_counts.items(), key=lambda x: -x[1])[:5]
        ]

        # Combine domain weights
        for g in top_genomes:
            for domain, weight in g.domain_weights.items():
                novel.domain_weights[domain] = max(
                    novel.domain_weights.get(domain, 0),
                    weight
                )

        novel.parent_ids = [g.genome_id for g in top_genomes]

        return novel

    def describe_strategy(self, genome: ReasoningGenome) -> str:
        """Generate a human-readable description of a reasoning strategy"""
        parts = []

        # Describe cognitive style
        style_parts = []
        if genome.analytical_gene > 0.7:
            style_parts.append("highly analytical")
        if genome.intuitive_gene > 0.7:
            style_parts.append("intuitive")
        if genome.systematic_gene > 0.7:
            style_parts.append("systematic")
        if genome.creative_gene > 0.7:
            style_parts.append("creative")

        if style_parts:
            parts.append(f"This strategy is {', '.join(style_parts)}.")

        # Describe reasoning process
        if genome.reasoning_steps:
            steps_str = " -> ".join(genome.reasoning_steps)
            parts.append(f"It follows this reasoning flow: {steps_str}.")

        # Describe domain focus
        if genome.domain_weights:
            top_domains = sorted(genome.domain_weights.items(), key=lambda x: -x[1])[:2]
            domains_str = " and ".join([d for d, _ in top_domains])
            parts.append(f"It draws particularly from {domains_str} knowledge.")

        # Describe fitness
        parts.append(f"Fitness score: {genome.fitness_score:.2f} (evaluated {genome.evaluation_count} times).")

        return " ".join(parts)

    def _save_state(self):
        """Save evolution state to disk"""
        state = {
            'population': [g.to_dict() for g in self.population],
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None
        }

        state_path = self.save_path / "evolution_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
