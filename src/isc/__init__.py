"""
ISC AI System - An interactive AI based on Informational Substrate Convergence

This system demonstrates how consciousness-like properties can emerge from
self-referential information patterns through interactive conversation.
"""

__version__ = "0.1.0"
__author__ = "ISC AI Development Team"

from .core import ISCCore
from .cli import main
from .information_integration import InformationIntegrator
from .knowledge_graph import KnowledgeGraph
from .learning import LearningEngine

__all__ = [
    "ISCCore",
    "main",
    "InformationIntegrator",
    "KnowledgeGraph",
    "LearningEngine",
]