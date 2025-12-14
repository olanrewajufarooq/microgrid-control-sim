"""
microgrid_sim/components/base.py

Abstract base classes (interfaces) for microgrid components.

Sign conventions:
  - Power (kW): generation > 0, consumption < 0.
  - Cash flow (currency per step): NEGATIVE = expense you pay, POSITIVE = revenue you receive.

References
----------
Bordons, C.; García-Torres, F.; Ridao, M. (2020). Model Predictive Control of Microgrids. Springer.
Concrete classes will cite specific chapters/equations where used.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """
    Abstract base class for any component in the microgrid.

    Methods
    -------
    step(t, **kwargs)
        Advance the component by one time step. Control inputs (actions)
        and exogenous signals are passed via kwargs.
    get_power() -> float
        Power contribution for the last step (kW). Generation > 0; consumption < 0.
    get_cost() -> float
        Cash flow for the last step:
        NEGATIVE for payment (expense), POSITIVE for revenue (income).
    reset()
        Reset to initial state.
    """

    def __init__(self, name: str):
        self.name = name
        self._cost: float = 0.0  # NEG=expense, POS=revenue
        self._downtime: float = 0.0  # 1.0 = down/offline, 0.0 = available

    @abstractmethod
    def step(self, t: int, **kwargs):
        """Advance by one discrete step."""
        ...

    @abstractmethod
    def get_power(self) -> float:
        """Return last-step power (kW): generation > 0; consumption < 0."""
        ...

    def get_cost(self) -> float:
        """Return last-step cash flow (NEG=expense, POS=revenue)."""
        return self._cost

    def get_downtime(self) -> float:
        """Return downtime flag for the last step (1.0=down/unavailable, 0.0=available)."""
        return self._downtime

    def set_seed(self, seed: int):
        """
        Optional hook for deterministic behavior.
        Reseeds any internal random generators if present.
        """
        try:
            if hasattr(self, "_rng") and hasattr(self._rng, "seed"):
                self._rng.seed(seed)
        except Exception:
            pass
        try:
            if hasattr(self, "_reliability_rng") and hasattr(self._reliability_rng, "seed"):
                self._reliability_rng.seed(seed)
        except Exception:
            pass

    @abstractmethod
    def reset(self):
        """Reset internal state and cost trackers."""
        self._cost = 0.0
        self._downtime = 0.0


class BaseGenerator(BaseComponent):
    """Abstract base class for generators (including a Grid/slack component)."""

    def __init__(self, name: str):
        super().__init__(name)
        self._power_output: float = 0.0

    def get_power(self) -> float:
        return self._power_output

    def reset(self):
        super().reset()
        self._power_output = 0.0


class BaseStorage(BaseComponent):
    """
    Abstract base class for storage units.
    Adds State of Charge (SoC) access for controllers and logging.
    """

    def __init__(self, name: str, capacity_kwh: float, initial_soc: float = 0.5):
        super().__init__(name)
        self.capacity_kwh = float(capacity_kwh)
        self.initial_soc = float(initial_soc)
        self._soc: float = float(initial_soc)
        self._power_flow: float = 0.0  # discharge > 0, charge < 0

    def get_power(self) -> float:
        """Storage power flow (kW): discharging > 0; charging < 0."""
        return self._power_flow

    @abstractmethod
    def get_soc(self) -> float:
        """Return current SoC in [0, 1]."""
        return self._soc

    def reset(self):
        super().reset()
        self._soc = self.initial_soc
        self._power_flow = 0.0


class BaseLoad(BaseComponent):
    """Abstract base class for loads (consumption < 0)."""

    def __init__(self, name: str):
        super().__init__(name)
        self._power_demand: float = 0.0  # positive internal demand (returned as negative power)

    def get_power(self) -> float:
        """Return last-step demand as a negative number (kW)."""
        return -self._power_demand

    def reset(self):
        super().reset()
        self._power_demand = 0.0
