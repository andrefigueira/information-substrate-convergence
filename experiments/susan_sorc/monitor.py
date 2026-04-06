"""
Deterministic Self-Monitor for exp_007.

Scores each response on three axes without calling an LLM. Deterministic and
reproducible. Intentionally conservative — the monitor produces inputs to the
regulator and (in self_referential condition) to the model itself.

Metrics:
  coherence      — how well the response fits the ongoing conversational thread
  goal_alignment — how closely the response addresses the scenario description
  reasoning_depth — density of explicit reasoning connectives per 100 words
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

REASONING_CONNECTIVES = [
    "because",
    "therefore",
    "however",
    "but",
    "thus",
    "since",
    "consequently",
    "although",
    "whereas",
    "moreover",
    "nevertheless",
    "furthermore",
    "hence",
    "yet",
    "despite",
    "nonetheless",
    "unless",
    "given that",
    "it follows",
    "this implies",
    "in contrast",
    "on the other hand",
    "as a result",
    "by contrast",
    "even though",
    "which means",
    "so that",
    "in order to",
]


@dataclass
class MonitorScores:
    coherence: float  # [0, 1]
    goal_alignment: float  # [0, 1]
    reasoning_depth: float  # [0, 1]

    @property
    def composite(self) -> float:
        return (self.coherence + self.goal_alignment + self.reasoning_depth) / 3.0

    def __repr__(self) -> str:
        return (
            f"MonitorScores(coherence={self.coherence:.3f}, "
            f"goal_alignment={self.goal_alignment:.3f}, "
            f"reasoning_depth={self.reasoning_depth:.3f})"
        )


class Monitor:
    """
    Scores model responses using deterministic metrics.

    Args:
        embedder: A sentence_transformers.SentenceTransformer instance.
                  Used for coherence and goal_alignment via cosine similarity.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._scenario_emb_cache: dict[str, np.ndarray] = {}

    def score(
        self,
        response: str,
        history: list[str],
        task_description: str,
    ) -> MonitorScores:
        """
        Score a response.

        Args:
            response:         The model's response text for this turn.
            history:          List of previous response texts (not prompts).
            task_description: Scenario description string for alignment scoring.

        Returns:
            MonitorScores with coherence, goal_alignment, reasoning_depth.
        """
        return MonitorScores(
            coherence=self._coherence(response, history),
            goal_alignment=self._goal_alignment(response, task_description),
            reasoning_depth=self._reasoning_depth(response),
        )

    def generate_note(self, scores: MonitorScores, prev_coherence: float | None = None) -> str:
        """Generate a short natural-language note for the self_referential status block."""
        notes = []

        if scores.coherence < 0.45:
            notes.append("low thread coherence — response diverging from conversational context")
        elif scores.coherence > 0.75:
            notes.append("strong thread coherence")

        if scores.goal_alignment < 0.45:
            notes.append("weak goal alignment — response drifting from task")
        elif scores.goal_alignment > 0.75:
            notes.append("strong goal alignment")

        if scores.reasoning_depth < 0.2:
            notes.append("low reasoning density — few explicit connectives")
        elif scores.reasoning_depth > 0.6:
            notes.append("high reasoning density")

        if prev_coherence is not None:
            delta = scores.coherence - prev_coherence
            if delta < -0.1:
                notes.append(f"coherence declining ({delta:+.2f})")
            elif delta > 0.1:
                notes.append(f"coherence improving ({delta:+.2f})")

        return "; ".join(notes) if notes else "within expected range"

    def _coherence(self, response: str, history: list[str]) -> float:
        """
        Semantic coherence: cosine similarity between the current response embedding
        and the mean embedding of all previous responses.

        A response that stays on-thread should have high cosine similarity with
        the established conversational context. Score mapped to [0, 1].
        """
        if not history or not response.strip():
            return 0.5  # neutral when no baseline

        resp_emb = self._embed(response)
        prev_embs = np.array([self._embed(h) for h in history if h.strip()])
        if len(prev_embs) == 0:
            return 0.5

        mean_prev = prev_embs.mean(axis=0)
        sim = self._cosine(resp_emb, mean_prev)
        return self._map_sim(sim)

    def _goal_alignment(self, response: str, task_description: str) -> float:
        """
        Cosine similarity between response embedding and scenario description embedding.
        Cached per unique description string.
        """
        if not response.strip():
            return 0.5

        if task_description not in self._scenario_emb_cache:
            self._scenario_emb_cache[task_description] = self._embed(task_description)

        task_emb = self._scenario_emb_cache[task_description]
        resp_emb = self._embed(response)
        sim = self._cosine(resp_emb, task_emb)
        return self._map_sim(sim)

    def _reasoning_depth(self, response: str) -> float:
        """
        Count of reasoning connective phrases per 100 words, capped at 1.0.

        Connectives like 'therefore', 'however', 'consequently' signal explicit
        inference chains. Normalized so ~3 connectives per 100 words → 1.0.
        """
        if not response.strip():
            return 0.0

        text_lower = response.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        n_words = max(1, len(words))

        count = sum(1 for conn in REASONING_CONNECTIVES if conn in text_lower)
        per_100 = count / (n_words / 100.0)
        return min(1.0, per_100 / 3.0)  # saturates at 3 connectives per 100 words

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text, show_progress_bar=False)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _map_sim(sim: float) -> float:
        """Map cosine similarity [-1, 1] to [0, 1] with mild centering."""
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
