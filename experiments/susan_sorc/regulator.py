"""
Homeostatic Regulator for exp_007.

PID controller that adjusts temperature and context_retention based on the
monitor's coherence score. Replicates SUSAN's regulator design in Python.

Behaviour:
  - Coherence below target → reduce disruption (lower temperature, more context)
  - Coherence above target → increase disruption (higher temperature, less context)
  - Homeostatic, not reward-based — maintains a quality band, not maximises

The regulator runs in feedback_blind and self_referential conditions.
In self_referential, the model sees the regulator's current state in the status block.
In feedback_blind, the regulator acts silently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegulatorState:
    temperature: float = 0.7
    context_retention: float = 1.0
    integral: float = 0.0
    prev_error: float = 0.0
    n_updates: int = 0


class Regulator:
    """
    PID controller over temperature and context_retention.

    Args:
        target:   Target coherence score (0-1). Default 0.65.
        kp:       Proportional gain.
        ki:       Integral gain (prevents steady-state drift).
        kd:       Derivative gain (damps oscillation).
        temp_range:      (min, max) for temperature output.
        retention_range: (min, max) for context_retention output.
    """

    TEMP_MIN = 0.5
    TEMP_MAX = 1.0
    RETENTION_MIN = 0.3
    RETENTION_MAX = 1.0
    INTEGRAL_CLAMP = 2.0

    def __init__(
        self,
        target: float = 0.65,
        kp: float = 0.3,
        ki: float = 0.05,
        kd: float = 0.1,
    ) -> None:
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._state = RegulatorState()

    @property
    def temperature(self) -> float:
        return self._state.temperature

    @property
    def context_retention(self) -> float:
        return self._state.context_retention

    def update(self, coherence: float) -> None:
        """
        Update regulator state given current coherence reading.

        Positive error (coherence below target) → positive output → reduce disruption:
          lower temperature, increase context retention.
        Negative error (coherence above target) → negative output → increase disruption:
          raise temperature, reduce context retention.
        """
        error = self.target - coherence
        self._state.integral = float(
            max(
                -self.INTEGRAL_CLAMP,
                min(self.INTEGRAL_CLAMP, self._state.integral + error),
            )
        )
        derivative = error - self._state.prev_error
        self._state.prev_error = error
        self._state.n_updates += 1

        output = self.kp * error + self.ki * self._state.integral + self.kd * derivative

        # Positive output → reduce disruption
        self._state.temperature = float(
            max(self.TEMP_MIN, min(self.TEMP_MAX, self._state.temperature - output * 0.1))
        )
        self._state.context_retention = float(
            max(
                self.RETENTION_MIN,
                min(self.RETENTION_MAX, self._state.context_retention + output * 0.05),
            )
        )

    def reset(self) -> None:
        """Reset to initial state for a new scenario/condition run."""
        self._state = RegulatorState()

    def status_line(self) -> str:
        """One-line status for self_referential status block."""
        return (
            f"temperature={self._state.temperature:.2f}, "
            f"context_retention={self._state.context_retention:.2f}"
        )
