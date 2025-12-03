"""
Shared reliability / failure modeling for components.

Provides:
- FailureMode: HEALTHY, MINOR (derate), MAJOR (outage)
- ReliabilityParams: hazard rates, MTTRs, derate factors, exogenous drivers
- ReliabilityState: current mode and remaining outage hours
- ReliabilityMixin: applies hazard sampling, derate, cost, and downtime flag
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Sequence


class FailureMode(Enum):
    HEALTHY = "HEALTHY"
    MINOR = "MINOR"   # power derate
    MAJOR = "MAJOR"   # complete outage


@dataclass
class ReliabilityParams:
    base_fail_rate_per_hour: float
    minor_prob_fraction: float
    minor_derate: float
    mttr_minor_hours: float
    mttr_major_hours: float
    maint_cost_per_hour: float
    exo_keys: Sequence[str] = ()
    exo_weight: float = 0.0
    exo_shift: float = 0.0


@dataclass
class ReliabilityState:
    mode: FailureMode = FailureMode.HEALTHY
    remaining_hours: float = 0.0


class ReliabilityMixin:
    """Adds stochastic reliability with exogenous-driven hazard."""

    def __init__(self, reliability_params: ReliabilityParams | None = None,
                 reliability_rng: random.Random | None = None):
        self.reliability_params = reliability_params
        self.reliability_state = ReliabilityState()
        self._reliability_rng = reliability_rng or random.Random()

    def _reset_reliability(self):
        self.reliability_state = ReliabilityState()

    def _update_reliability(self, exogenous: dict, dt_hours: float):
        rp = self.reliability_params
        if rp is None:
            return

        st = self.reliability_state

        # Decrement ongoing outage
        if st.mode in (FailureMode.MINOR, FailureMode.MAJOR):
            st.remaining_hours = max(0.0, st.remaining_hours - dt_hours)
            if st.remaining_hours == 0.0:
                st.mode = FailureMode.HEALTHY
            return

        # Compute hazard with exogenous influence
        exo_term = 0.0
        for key in rp.exo_keys:
            if key in exogenous:
                try:
                    val = float(exogenous.get(key, 0.0))
                except Exception:
                    val = 0.0
                exo_term = max(exo_term, val - rp.exo_shift)

        hazard = rp.base_fail_rate_per_hour * (1.0 + rp.exo_weight * max(0.0, exo_term))
        hazard = max(hazard, 0.0)
        p_fail = 1.0 - math.exp(-hazard * dt_hours)

        if self._reliability_rng.random() < p_fail:
            is_minor = self._reliability_rng.random() < rp.minor_prob_fraction
            st.mode = FailureMode.MINOR if is_minor else FailureMode.MAJOR
            mttr = rp.mttr_minor_hours if is_minor else rp.mttr_major_hours
            st.remaining_hours = self._reliability_rng.expovariate(1.0 / mttr) if mttr > 0 else 0.0

    def _reliability_derate(self, power: float) -> float:
        rp = self.reliability_params
        st = self.reliability_state
        if rp is None:
            return power
        if st.mode is FailureMode.MAJOR:
            return 0.0
        if st.mode is FailureMode.MINOR:
            return power * rp.minor_derate
        return power

    def _reliability_cost(self, dt_hours: float, apply_cost: bool = True) -> float:
        rp = self.reliability_params
        st = self.reliability_state
        if not apply_cost:
            return 0.0
        if rp and st.mode in (FailureMode.MINOR, FailureMode.MAJOR):
            return - rp.maint_cost_per_hour * dt_hours
        return 0.0

    def _reliability_downtime_flag(self) -> float:
        st = self.reliability_state
        if st.mode is FailureMode.MAJOR:
            return 1.0
        return 0.0
