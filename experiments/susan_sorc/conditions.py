"""
Context builders for the four experimental conditions in exp_007.

Each condition builds a different context string from the same raw history.
SORC is measured on the full context string before generation.

Conditions:
  control          — current turn only, no history, no status
  history_only     — full history + current turn, no status
  feedback_blind   — history (possibly truncated by regulator) + current turn, no status
  self_referential — same as feedback_blind but with a status block prepended

The status block in self_referential is the key ISC manipulation: it gives the model
an explicit self-model (its own quality metrics and trajectory). ISC predicts this
will increase SORC by introducing second-order relational structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.susan_sorc.monitor import MonitorScores

SYSTEM_PROMPT = (
    "You are a thoughtful expert assistant engaged in a sustained analytical conversation. "
    "Build on previous turns, reason carefully, and give substantive responses."
)

CONDITIONS = ["control", "history_only", "feedback_blind", "self_referential"]


@dataclass
class ContextResult:
    text: str  # full context string fed to model (and measured by SORC)
    n_history_turns: int  # how many prior turns are included


def build_context(
    condition: str,
    turn_prompts: list[str],
    turn_responses: list[str],
    current_prompt: str,
    scores: MonitorScores | None,
    monitor_note: str,
    coherence_trend: list[float],
    regulator_status: str,
    context_retention: float,
) -> ContextResult:
    """
    Build the full context string for the given condition.

    Args:
        condition:         One of CONDITIONS.
        turn_prompts:      Previous turn prompts (same length as turn_responses).
        turn_responses:    Previous turn responses.
        current_prompt:    The prompt for the current turn.
        scores:            Monitor scores from the previous turn (None on turn 0).
        monitor_note:      Natural-language note from the monitor.
        coherence_trend:   List of recent coherence scores (last 3).
        regulator_status:  Regulator status line string.
        context_retention: Fraction of history to retain (for feedback_blind and self_referential).

    Returns:
        ContextResult with the full context string and count of included history turns.
    """
    assert condition in CONDITIONS, f"Unknown condition: {condition!r}"
    assert len(turn_prompts) == len(turn_responses)

    if condition == "control":
        return _build_control(current_prompt)

    history = list(zip(turn_prompts, turn_responses))

    if condition == "history_only":
        return _build_with_history(history, current_prompt, retention=1.0)

    if condition == "feedback_blind":
        return _build_with_history(history, current_prompt, retention=context_retention)

    # self_referential
    return _build_self_referential(
        history=history,
        current_prompt=current_prompt,
        retention=context_retention,
        scores=scores,
        monitor_note=monitor_note,
        coherence_trend=coherence_trend,
        regulator_status=regulator_status,
    )


def _build_control(current_prompt: str) -> ContextResult:
    text = f"{SYSTEM_PROMPT}\n\nUser: {current_prompt}\nAssistant:"
    return ContextResult(text=text, n_history_turns=0)


def _build_with_history(
    history: list[tuple[str, str]],
    current_prompt: str,
    retention: float,
) -> ContextResult:
    included = _retained_history(history, retention)
    lines = [SYSTEM_PROMPT, ""]
    for prompt, response in included:
        lines.append(f"User: {prompt}")
        lines.append(f"Assistant: {response}")
        lines.append("")
    lines.append(f"User: {current_prompt}")
    lines.append("Assistant:")
    return ContextResult(text="\n".join(lines), n_history_turns=len(included))


def _build_self_referential(
    history: list[tuple[str, str]],
    current_prompt: str,
    retention: float,
    scores: MonitorScores | None,
    monitor_note: str,
    coherence_trend: list[float],
    regulator_status: str,
) -> ContextResult:
    included = _retained_history(history, retention)
    turn_n = len(history) + 1

    status_block = _format_status_block(
        turn_n=turn_n,
        scores=scores,
        monitor_note=monitor_note,
        coherence_trend=coherence_trend,
        regulator_status=regulator_status,
    )

    lines = [SYSTEM_PROMPT, "", status_block, ""]
    for prompt, response in included:
        lines.append(f"User: {prompt}")
        lines.append(f"Assistant: {response}")
        lines.append("")
    lines.append(f"User: {current_prompt}")
    lines.append("Assistant:")
    return ContextResult(text="\n".join(lines), n_history_turns=len(included))


def _format_status_block(
    turn_n: int,
    scores: MonitorScores | None,
    monitor_note: str,
    coherence_trend: list[float],
    regulator_status: str,
) -> str:
    if scores is None:
        return (
            f"[System Status — Turn {turn_n}]\n"
            "No prior turn data.\n"
            f"Regulator: {regulator_status}\n"
            "[End Status]"
        )

    trend_str = (
        "[" + ", ".join(f"{v:.2f}" for v in coherence_trend) + "]" if coherence_trend else "[]"
    )

    return (
        f"[System Status — Turn {turn_n}]\n"
        f"Coherence: {scores.coherence:.2f} | "
        f"Goal Alignment: {scores.goal_alignment:.2f} | "
        f"Reasoning Depth: {scores.reasoning_depth:.2f}\n"
        f"Monitor note: {monitor_note}\n"
        f"Regulator: {regulator_status}\n"
        f"Coherence trend (last 3): {trend_str}\n"
        "[End Status]"
    )


def _retained_history(
    history: list[tuple[str, str]],
    retention: float,
) -> list[tuple[str, str]]:
    """Return the most recent fraction of history turns."""
    if not history:
        return []
    n_keep = max(1, round(len(history) * retention))
    return history[-n_keep:]
