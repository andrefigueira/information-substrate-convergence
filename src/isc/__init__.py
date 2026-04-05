"""
ISC — Informational Substrate Convergence

Public API: the SORC metric and its supporting functions.

    from src.isc.sorc import compute_sorc, extract_hidden_states, random_baseline, SORCResult
"""

__version__ = "2.0.0"
__author__ = "André Pereira Figueira"

from .sorc import compute_sorc, extract_hidden_states, random_baseline, SORCResult

__all__ = [
    "compute_sorc",
    "extract_hidden_states",
    "random_baseline",
    "SORCResult",
]
