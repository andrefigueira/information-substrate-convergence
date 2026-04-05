"""
ISC Integration Benchmark Suite

Fair, reproducible comparison of reasoning capabilities.

Comparison approaches:
1. ISC vs Random Baseline (lower bound)
2. ISC vs Pattern Matching (simple heuristic)
3. ISC vs Published LLM Results (where available)

All tests use:
- Same problems for all systems
- Objective scoring (exact match or semantic equivalence)
- Multiple runs for statistical significance
- Bootstrap confidence intervals
"""

import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats
from collections import defaultdict


@dataclass
class BenchmarkProblem:
    """A single benchmark problem"""
    id: str
    category: str  # deductive, inductive, causal, etc.
    premise: str
    question: str
    correct_answer: str
    options: List[str]  # Multiple choice options
    difficulty: float  # 0-1
    source: str  # Where this problem came from


@dataclass
class SystemResult:
    """Result from a system on a single problem"""
    problem_id: str
    predicted: str
    correct: bool
    confidence: float
    latency_ms: float
    reasoning_trace: List[str]


@dataclass
class BenchmarkResult:
    """Full benchmark results for a system"""
    system_name: str
    total_problems: int
    correct: int
    accuracy: float
    accuracy_ci: Tuple[float, float]  # 95% CI
    mean_latency_ms: float
    mean_confidence: float
    accuracy_by_category: Dict[str, float]
    per_problem_results: List[SystemResult]
    timestamp: str


class ReasoningBenchmarks:
    """Standard reasoning benchmark problems"""

    def __init__(self):
        self.problems: List[BenchmarkProblem] = []
        self._load_problems()

    def _load_problems(self):
        """Load benchmark problems"""
        # Deductive reasoning (syllogisms, modus ponens)
        self.problems.extend([
            BenchmarkProblem(
                id="ded_001", category="deductive",
                premise="All mammals are warm-blooded. All dogs are mammals.",
                question="Are dogs warm-blooded?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "sometimes"],
                difficulty=0.2, source="syllogism"
            ),
            BenchmarkProblem(
                id="ded_002", category="deductive",
                premise="If it rains, the ground gets wet. It is raining.",
                question="Is the ground wet?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "maybe"],
                difficulty=0.2, source="modus_ponens"
            ),
            BenchmarkProblem(
                id="ded_003", category="deductive",
                premise="All birds have feathers. Penguins are birds.",
                question="Do penguins have feathers?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "some do"],
                difficulty=0.2, source="syllogism"
            ),
            BenchmarkProblem(
                id="ded_004", category="deductive",
                premise="No reptiles are mammals. All snakes are reptiles.",
                question="Are snakes mammals?",
                correct_answer="no",
                options=["yes", "no", "unknown", "some are"],
                difficulty=0.3, source="syllogism"
            ),
            BenchmarkProblem(
                id="ded_005", category="deductive",
                premise="If A implies B, and B implies C, then A implies C. A implies B. B implies C.",
                question="Does A imply C?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "sometimes"],
                difficulty=0.4, source="transitivity"
            ),
            BenchmarkProblem(
                id="ded_006", category="deductive",
                premise="All squares are rectangles. All rectangles have four sides.",
                question="Do squares have four sides?",
                correct_answer="yes",
                options=["yes", "no", "depends", "unknown"],
                difficulty=0.3, source="syllogism"
            ),
            BenchmarkProblem(
                id="ded_007", category="deductive",
                premise="If someone is a doctor, they went to medical school. Jane is a doctor.",
                question="Did Jane go to medical school?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "probably"],
                difficulty=0.3, source="modus_ponens"
            ),
            BenchmarkProblem(
                id="ded_008", category="deductive",
                premise="All prime numbers greater than 2 are odd. 7 is a prime number greater than 2.",
                question="Is 7 odd?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "depends"],
                difficulty=0.3, source="syllogism"
            ),
            BenchmarkProblem(
                id="ded_009", category="deductive",
                premise="If the light is red, cars must stop. The light is red.",
                question="Must cars stop?",
                correct_answer="yes",
                options=["yes", "no", "sometimes", "unknown"],
                difficulty=0.2, source="modus_ponens"
            ),
            BenchmarkProblem(
                id="ded_010", category="deductive",
                premise="No fish can breathe air directly. Salmon are fish.",
                question="Can salmon breathe air directly?",
                correct_answer="no",
                options=["yes", "no", "sometimes", "unknown"],
                difficulty=0.3, source="syllogism"
            ),
        ])

        # Causal reasoning
        self.problems.extend([
            BenchmarkProblem(
                id="cau_001", category="causal",
                premise="Smoking causes lung cancer. John smokes.",
                question="Is John at increased risk for lung cancer?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "unrelated"],
                difficulty=0.3, source="causal_chain"
            ),
            BenchmarkProblem(
                id="cau_002", category="causal",
                premise="Deforestation leads to soil erosion. Soil erosion causes flooding.",
                question="Does deforestation contribute to flooding?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "unrelated"],
                difficulty=0.4, source="causal_chain"
            ),
            BenchmarkProblem(
                id="cau_003", category="causal",
                premise="Exercise increases metabolism. Increased metabolism burns more calories.",
                question="Does exercise burn calories?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "sometimes"],
                difficulty=0.3, source="causal_chain"
            ),
            BenchmarkProblem(
                id="cau_004", category="causal",
                premise="Lack of sleep impairs concentration. Impaired concentration reduces work performance.",
                question="Does lack of sleep affect work performance?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "unrelated"],
                difficulty=0.4, source="causal_chain"
            ),
            BenchmarkProblem(
                id="cau_005", category="causal",
                premise="Rising CO2 causes global warming. Global warming melts ice caps.",
                question="Does rising CO2 contribute to melting ice caps?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "unrelated"],
                difficulty=0.4, source="causal_chain"
            ),
        ])

        # Analogical reasoning
        self.problems.extend([
            BenchmarkProblem(
                id="ana_001", category="analogical",
                premise="Puppy is to dog as kitten is to ___.",
                question="What completes the analogy?",
                correct_answer="cat",
                options=["cat", "mouse", "animal", "pet"],
                difficulty=0.2, source="word_analogy"
            ),
            BenchmarkProblem(
                id="ana_002", category="analogical",
                premise="Hot is to cold as up is to ___.",
                question="What completes the analogy?",
                correct_answer="down",
                options=["down", "top", "sky", "above"],
                difficulty=0.2, source="word_analogy"
            ),
            BenchmarkProblem(
                id="ana_003", category="analogical",
                premise="Author is to book as composer is to ___.",
                question="What completes the analogy?",
                correct_answer="music",
                options=["music", "piano", "singer", "concert"],
                difficulty=0.3, source="word_analogy"
            ),
            BenchmarkProblem(
                id="ana_004", category="analogical",
                premise="Bird is to nest as human is to ___.",
                question="What completes the analogy?",
                correct_answer="house",
                options=["house", "city", "family", "earth"],
                difficulty=0.3, source="word_analogy"
            ),
            BenchmarkProblem(
                id="ana_005", category="analogical",
                premise="Eye is to see as ear is to ___.",
                question="What completes the analogy?",
                correct_answer="hear",
                options=["hear", "sound", "listen", "noise"],
                difficulty=0.2, source="word_analogy"
            ),
        ])

        # Inductive reasoning
        self.problems.extend([
            BenchmarkProblem(
                id="ind_001", category="inductive",
                premise="Swan 1 is white. Swan 2 is white. Swan 3 is white. Swan 4 is white.",
                question="What color is swan 5 likely to be?",
                correct_answer="white",
                options=["white", "black", "gray", "unknown"],
                difficulty=0.2, source="pattern"
            ),
            BenchmarkProblem(
                id="ind_002", category="inductive",
                premise="2, 4, 6, 8, ___",
                question="What number comes next?",
                correct_answer="10",
                options=["10", "9", "12", "16"],
                difficulty=0.2, source="sequence"
            ),
            BenchmarkProblem(
                id="ind_003", category="inductive",
                premise="Monday, Tuesday, Wednesday, Thursday, ___",
                question="What day comes next?",
                correct_answer="Friday",
                options=["Friday", "Saturday", "Sunday", "Monday"],
                difficulty=0.1, source="sequence"
            ),
            BenchmarkProblem(
                id="ind_004", category="inductive",
                premise="1, 1, 2, 3, 5, 8, ___",
                question="What number comes next?",
                correct_answer="13",
                options=["13", "11", "10", "16"],
                difficulty=0.4, source="fibonacci"
            ),
            BenchmarkProblem(
                id="ind_005", category="inductive",
                premise="Apple, Banana, Cherry, ___",
                question="Following alphabetical fruit pattern, what comes next?",
                correct_answer="date",
                options=["date", "elderberry", "fig", "grape"],
                difficulty=0.4, source="pattern"
            ),
        ])

        # Abductive reasoning
        self.problems.extend([
            BenchmarkProblem(
                id="abd_001", category="abductive",
                premise="The grass is wet. It is morning.",
                question="What is the most likely explanation?",
                correct_answer="dew",
                options=["dew", "flood", "sprinkler", "ocean"],
                difficulty=0.3, source="explanation"
            ),
            BenchmarkProblem(
                id="abd_002", category="abductive",
                premise="The patient has fever, cough, and body aches.",
                question="What is the most likely cause?",
                correct_answer="flu",
                options=["flu", "broken bone", "allergy", "hunger"],
                difficulty=0.3, source="diagnosis"
            ),
            BenchmarkProblem(
                id="abd_003", category="abductive",
                premise="The car won't start. The lights don't turn on.",
                question="What is the most likely problem?",
                correct_answer="dead battery",
                options=["dead battery", "flat tire", "empty tank", "broken window"],
                difficulty=0.3, source="diagnosis"
            ),
            BenchmarkProblem(
                id="abd_004", category="abductive",
                premise="The cookies are gone. There are crumbs on the child's face.",
                question="What most likely happened?",
                correct_answer="child ate cookies",
                options=["child ate cookies", "dog ate them", "they evaporated", "stolen"],
                difficulty=0.2, source="explanation"
            ),
            BenchmarkProblem(
                id="abd_005", category="abductive",
                premise="The plant is wilting. The soil is dry.",
                question="What is the most likely cause?",
                correct_answer="lack of water",
                options=["lack of water", "too much sun", "disease", "frost"],
                difficulty=0.2, source="diagnosis"
            ),
        ])

        # HARD: Multi-hop reasoning (keyword matching will fail)
        self.problems.extend([
            BenchmarkProblem(
                id="multi_001", category="multi_hop",
                premise="Alice is taller than Bob. Bob is taller than Carol. Carol is taller than David.",
                question="Is Alice taller than David?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "equal"],
                difficulty=0.6, source="transitive"
            ),
            BenchmarkProblem(
                id="multi_002", category="multi_hop",
                premise="X causes Y. Y causes Z. Z causes W.",
                question="Does X contribute to W?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "maybe"],
                difficulty=0.6, source="causal_chain"
            ),
            BenchmarkProblem(
                id="multi_003", category="multi_hop",
                premise="All A are B. All B are C. All C are D. Item X is an A.",
                question="Is X a D?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "sometimes"],
                difficulty=0.7, source="multi_syllogism"
            ),
            BenchmarkProblem(
                id="multi_004", category="multi_hop",
                premise="Company profit depends on sales. Sales depend on marketing. Marketing depends on budget. Budget was cut.",
                question="What happens to company profit?",
                correct_answer="decreases",
                options=["increases", "decreases", "unchanged", "unknown"],
                difficulty=0.7, source="causal_chain"
            ),
            BenchmarkProblem(
                id="multi_005", category="multi_hop",
                premise="Room A connects to Room B. Room B connects to Room C. Room C connects to Room D. You are in Room A.",
                question="Can you reach Room D?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "maybe"],
                difficulty=0.5, source="graph_traversal"
            ),
            BenchmarkProblem(
                id="multi_006", category="multi_hop",
                premise="Tom is Mary's brother. Mary is John's mother. John is Lisa's father.",
                question="What is Tom's relation to Lisa?",
                correct_answer="great-uncle",
                options=["great-uncle", "grandfather", "father", "cousin"],
                difficulty=0.8, source="family_relations"
            ),
            BenchmarkProblem(
                id="multi_007", category="multi_hop",
                premise="Process A produces chemical X. Chemical X is required for Process B. Process B produces chemical Y. Chemical Y is toxic to fish.",
                question="Does Process A indirectly affect fish?",
                correct_answer="yes",
                options=["yes", "no", "unknown", "unrelated"],
                difficulty=0.7, source="causal_chain"
            ),
            BenchmarkProblem(
                id="multi_008", category="multi_hop",
                premise="If pressure increases, temperature increases. If temperature increases, volume expands. If volume expands, container stress increases.",
                question="If pressure increases, what happens to container stress?",
                correct_answer="increases",
                options=["increases", "decreases", "unchanged", "unknown"],
                difficulty=0.7, source="causal_chain"
            ),
            BenchmarkProblem(
                id="multi_009", category="multi_hop",
                premise="Server A sends data to Server B. Server B processes and sends to Server C. Server C stores in Database D. Server B is down.",
                question="Can data from Server A reach Database D?",
                correct_answer="no",
                options=["yes", "no", "sometimes", "unknown"],
                difficulty=0.6, source="graph_analysis"
            ),
            BenchmarkProblem(
                id="multi_010", category="multi_hop",
                premise="Species A eats Species B. Species B eats Species C. Species C eats Plants. Plants need sunlight.",
                question="If sunlight decreases, what happens to Species A?",
                correct_answer="decreases",
                options=["increases", "decreases", "unchanged", "unknown"],
                difficulty=0.8, source="ecological_chain"
            ),
        ])

    def get_all(self) -> List[BenchmarkProblem]:
        return self.problems

    def get_by_category(self, category: str) -> List[BenchmarkProblem]:
        return [p for p in self.problems if p.category == category]

    def get_sample(self, n: int, seed: int = None) -> List[BenchmarkProblem]:
        if seed:
            random.seed(seed)
        return random.sample(self.problems, min(n, len(self.problems)))


class BaselineRandom:
    """Random baseline - picks random answer"""

    def __init__(self):
        self.name = "Random Baseline"

    def solve(self, problem: BenchmarkProblem) -> Tuple[str, float, List[str]]:
        answer = random.choice(problem.options)
        confidence = 1.0 / len(problem.options)
        trace = ["Random selection"]
        return answer, confidence, trace


class BaselineKeyword:
    """Keyword matching baseline"""

    def __init__(self):
        self.name = "Keyword Baseline"

    def solve(self, problem: BenchmarkProblem) -> Tuple[str, float, List[str]]:
        premise_lower = problem.premise.lower()
        question_lower = problem.question.lower()

        # Simple heuristics
        if 'all' in premise_lower and 'are' in premise_lower:
            if 'yes' in problem.options:
                return 'yes', 0.6, ["Detected universal statement, likely yes"]

        if 'no ' in premise_lower or 'not ' in premise_lower:
            if 'no' in problem.options:
                return 'no', 0.6, ["Detected negation"]

        # Check for keyword overlap
        best_option = None
        best_score = 0
        for opt in problem.options:
            if opt.lower() in premise_lower:
                score = premise_lower.count(opt.lower())
                if score > best_score:
                    best_score = score
                    best_option = opt

        if best_option:
            return best_option, 0.5, [f"Keyword match: {best_option}"]

        # Default to first option
        return problem.options[0], 0.25, ["Default selection"]


class ISCSystem:
    """ISC reasoning system"""

    def __init__(self):
        self.name = "ISC Reasoner"
        self.reasoner = None
        self._init_reasoner()

    def _init_reasoner(self):
        try:
            from .reasoning_api import ISCReasoner, ReasoningType
            self.reasoner = ISCReasoner()
            self.ReasoningType = ReasoningType
        except ImportError:
            print("Warning: Could not import ISCReasoner")

    def solve(self, problem: BenchmarkProblem) -> Tuple[str, float, List[str]]:
        if not self.reasoner:
            return problem.options[0], 0.0, ["Reasoner not available"]

        # Map category to reasoning type
        type_map = {
            'deductive': self.ReasoningType.DEDUCTIVE,
            'inductive': self.ReasoningType.INDUCTIVE,
            'abductive': self.ReasoningType.ABDUCTIVE,
            'causal': self.ReasoningType.CAUSAL,
            'analogical': self.ReasoningType.ANALOGICAL
        }

        reasoning_type = type_map.get(problem.category, self.ReasoningType.AUTO)

        result = self.reasoner.reason(
            query=problem.question,
            context=[problem.premise],
            reasoning_type=reasoning_type
        )

        # Match answer to options
        answer_lower = result.answer.lower()
        best_match = problem.options[0]
        best_score = 0

        for opt in problem.options:
            if opt.lower() in answer_lower or answer_lower in opt.lower():
                score = len(opt)
                if score > best_score:
                    best_score = score
                    best_match = opt

        # Special case for yes/no
        if 'yes' in answer_lower and 'yes' in problem.options:
            best_match = 'yes'
        elif 'no' in answer_lower and 'no' in problem.options:
            best_match = 'no'

        return best_match, result.confidence, result.reasoning_chain


class BenchmarkRunner:
    """Run benchmarks and collect results"""

    def __init__(self):
        self.benchmarks = ReasoningBenchmarks()

    def run_system(
        self,
        system: Any,
        problems: List[BenchmarkProblem] = None,
        seed: int = 42
    ) -> BenchmarkResult:
        """Run a system on benchmark problems"""

        if problems is None:
            problems = self.benchmarks.get_all()

        results = []
        correct_count = 0

        for problem in problems:
            start = time.time()
            predicted, confidence, trace = system.solve(problem)
            latency = (time.time() - start) * 1000

            is_correct = predicted.lower() == problem.correct_answer.lower()
            if is_correct:
                correct_count += 1

            results.append(SystemResult(
                problem_id=problem.id,
                predicted=predicted,
                correct=is_correct,
                confidence=confidence,
                latency_ms=latency,
                reasoning_trace=trace
            ))

        accuracy = correct_count / len(problems) if problems else 0

        # Bootstrap CI
        correct_array = np.array([r.correct for r in results])
        ci = self._bootstrap_ci(correct_array)

        # Accuracy by category
        by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r, p in zip(results, problems):
            by_category[p.category]['total'] += 1
            if r.correct:
                by_category[p.category]['correct'] += 1

        accuracy_by_cat = {
            cat: d['correct'] / d['total'] if d['total'] > 0 else 0
            for cat, d in by_category.items()
        }

        return BenchmarkResult(
            system_name=system.name,
            total_problems=len(problems),
            correct=correct_count,
            accuracy=accuracy,
            accuracy_ci=ci,
            mean_latency_ms=np.mean([r.latency_ms for r in results]),
            mean_confidence=np.mean([r.confidence for r in results]),
            accuracy_by_category=accuracy_by_cat,
            per_problem_results=results,
            timestamp=datetime.now().isoformat()
        )

    def _bootstrap_ci(self, correct_array: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate 95% CI via bootstrap"""
        bootstrap_means = []
        n = len(correct_array)

        for _ in range(n_bootstrap):
            sample = np.random.choice(correct_array, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        return (np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5))

    def compare_systems(
        self,
        systems: List[Any],
        n_runs: int = 5,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Compare multiple systems with statistical testing"""

        all_results = {s.name: [] for s in systems}

        for run in range(n_runs):
            random.seed(seed + run)
            np.random.seed(seed + run)

            problems = self.benchmarks.get_all()

            for system in systems:
                result = self.run_system(system, problems)
                all_results[system.name].append(result)

        # Statistical comparison
        comparison = {
            'systems': {},
            'pairwise_tests': [],
            'summary': {}
        }

        for name, results in all_results.items():
            accuracies = [r.accuracy for r in results]
            comparison['systems'][name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'accuracies': accuracies,
                'mean_latency': np.mean([r.mean_latency_ms for r in results]),
                'mean_confidence': np.mean([r.mean_confidence for r in results])
            }

        # Pairwise t-tests
        system_names = list(all_results.keys())
        for i in range(len(system_names)):
            for j in range(i + 1, len(system_names)):
                name1, name2 = system_names[i], system_names[j]
                acc1 = comparison['systems'][name1]['accuracies']
                acc2 = comparison['systems'][name2]['accuracies']

                t_stat, p_value = stats.ttest_ind(acc1, acc2)
                effect_size = (np.mean(acc1) - np.mean(acc2)) / np.sqrt((np.var(acc1) + np.var(acc2)) / 2)

                comparison['pairwise_tests'].append({
                    'system1': name1,
                    'system2': name2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05,
                    'better': name1 if np.mean(acc1) > np.mean(acc2) else name2
                })

        # Summary
        best_system = max(comparison['systems'].items(), key=lambda x: x[1]['mean_accuracy'])
        comparison['summary'] = {
            'best_system': best_system[0],
            'best_accuracy': best_system[1]['mean_accuracy'],
            'n_runs': n_runs,
            'n_problems': len(self.benchmarks.get_all()),
            'timestamp': datetime.now().isoformat()
        }

        return comparison


def run_benchmark():
    """Run full benchmark comparison"""
    print("=" * 60)
    print("ISC INTEGRATION BENCHMARK")
    print("=" * 60)
    print()

    runner = BenchmarkRunner()

    # Initialize systems
    systems = [
        BaselineRandom(),
        BaselineKeyword(),
        ISCSystem()
    ]

    print(f"Systems: {[s.name for s in systems]}")
    print(f"Problems: {len(runner.benchmarks.get_all())}")
    print()

    # Run comparison
    print("Running benchmark (5 runs for statistical significance)...")
    comparison = runner.compare_systems(systems, n_runs=5)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print(f"{'System':<20} | {'Accuracy':<12} | {'Std':<8} | {'Latency':<10} | {'Confidence':<10}")
    print("-" * 70)

    for name, data in comparison['systems'].items():
        print(f"{name:<20} | {data['mean_accuracy']*100:>10.1f}% | {data['std_accuracy']*100:>6.1f}% | {data['mean_latency']:>8.1f}ms | {data['mean_confidence']:>8.2f}")

    print()
    print("STATISTICAL TESTS:")
    for test in comparison['pairwise_tests']:
        sig = "*" if test['significant'] else ""
        print(f"  {test['system1']} vs {test['system2']}: p={test['p_value']:.4f}{sig}, d={test['effect_size']:.2f}, better={test['better']}")

    print()
    print(f"BEST SYSTEM: {comparison['summary']['best_system']} ({comparison['summary']['best_accuracy']*100:.1f}%)")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "benchmark_comparison.json"
    output_path.parent.mkdir(exist_ok=True)

    # Create serializable version
    serializable = {
        'systems': {},
        'pairwise_tests': comparison['pairwise_tests'],
        'summary': comparison['summary']
    }
    for name, data in comparison['systems'].items():
        serializable['systems'][name] = {
            'mean_accuracy': float(data['mean_accuracy']),
            'std_accuracy': float(data['std_accuracy']),
            'mean_latency': float(data['mean_latency']),
            'mean_confidence': float(data['mean_confidence'])
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return comparison


if __name__ == "__main__":
    run_benchmark()
