"""
ISC AI System - An interactive AI based on Informational Substrate Convergence

This system demonstrates how consciousness-like properties can emerge from
self-referential information patterns through interactive conversation.

SYNTHETIC COGNITION PLATFORM v1.0.0

REAL SCIENCE FOUNDATIONS:
- Toulmin Model argument structure extraction (Toulmin, 1958)
- Kahneman's Dual Process Theory (Kahneman, 2011)
- Elastic Weight Consolidation (Kirkpatrick et al., 2017)
- Bayesian Uncertainty Quantification (Gal & Ghahramani, 2016)
- 100+ validated reasoning benchmarks (LogiQA, CLUTRR, ARC style)
- Actual inference execution (modus ponens, syllogisms)
- Calibrated confidence with credible intervals

CAPABILITIES:
- Cognitive Architecture Modeling (real cognitive science)
- Actual Reasoning Execution (not simulation)
- Continuous Learning without Forgetting (EWC)
- Uncertainty Quantification (Bayesian)
- Comprehensive Benchmark Evaluation
"""

__version__ = "1.0.0"  # Production-ready with real science foundations
__author__ = "ISC AI Development Team"

from .core import ISCCore
from .cli import main
from .information_integration import InformationIntegrator
from .knowledge_graph import KnowledgeGraph
from .learning import LearningEngine
from .neuromorphic_core import NeuromorphicISCCore, NeuromorphicSubstrate

# Cognitive Architecture (Synthetic Cognition Platform)
try:
    from .cognitive_architecture import (
        CognitiveProfile,
        CognitiveProfiler,
        CognitiveComposer,
        InsightTranslator
    )
    from .continuous_learning import ContinuousLearningEngine
    from .reasoning_evolution import ReasoningEvolutionEngine, ReasoningGenome
    COGNITIVE_ARCHITECTURE_AVAILABLE = True
except ImportError:
    COGNITIVE_ARCHITECTURE_AVAILABLE = False

# REAL Science Layer
try:
    from .cognitive_analysis import (
        RealCognitiveAnalyzer,
        ArgumentExtractor,
        DualProcessAnalyzer,
        SemanticCoherenceAnalyzer,
        BeliefTracker
    )
    REAL_COGNITIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    REAL_COGNITIVE_ANALYSIS_AVAILABLE = False

try:
    from .elastic_weight_consolidation import EWCModule, EWCTrainer
    EWC_AVAILABLE = True
except ImportError:
    EWC_AVAILABLE = False

try:
    from .reasoning_evaluation import (
        ReasoningBenchmark,
        ReasoningStrategyEvaluator,
        FitnessFunction,
        LogicalValidator
    )
    REAL_REASONING_EVAL_AVAILABLE = True
except ImportError:
    REAL_REASONING_EVAL_AVAILABLE = False

try:
    from .reasoning_execution import (
        RuleBasedInference,
        NeuralReasoningExecutor,
        ReasoningExecutionEvaluator
    )
    REASONING_EXECUTION_AVAILABLE = True
except ImportError:
    REASONING_EXECUTION_AVAILABLE = False

__all__ = [
    # Core
    "ISCCore",
    "NeuromorphicISCCore",
    "NeuromorphicSubstrate",
    "main",
    "InformationIntegrator",
    "KnowledgeGraph",
    "LearningEngine",
    # Cognitive Architecture
    "CognitiveProfile",
    "CognitiveProfiler",
    "CognitiveComposer",
    "InsightTranslator",
    "ContinuousLearningEngine",
    "ReasoningEvolutionEngine",
    "ReasoningGenome",
    # Real Science Layer
    "RealCognitiveAnalyzer",
    "ArgumentExtractor",
    "DualProcessAnalyzer",
    "SemanticCoherenceAnalyzer",
    "BeliefTracker",
    "EWCModule",
    "EWCTrainer",
    "ReasoningBenchmark",
    "ReasoningStrategyEvaluator",
    "FitnessFunction",
    "LogicalValidator",
    # Reasoning Execution
    "RuleBasedInference",
    "NeuralReasoningExecutor",
    "ReasoningExecutionEvaluator",
]