"""
microgrid_sim/components/storage.py

Battery storage implementation with:
- ON/OFF state
- Charge/discharge limits (kW)
- SOC bounds [min_soc, max_soc]
- Charge/discharge efficiencies
- Step-level cash flow for degradation cost (negative when you pay)

Sign conventions:
- Power: discharging > 0 (generation), charging < 0 (consumption).
- Cash flow: NEGATIVE = expense, POSITIVE = revenue.

References
----------
Bordons, C.; Garcia-Torres, F.; Ridao, M. (2020). Model Predictive Control of Microgrids. Springer.
We follow a control-oriented SoC update commonly used in EMS/MPC:
   E_{k+1} = E_k + eta_c * P_ch * Deltat  - (1/eta_d) * P_dis * Deltat,
with P_ch >= 0, P_dis >= 0.
Implemented under the sign convention P_batt > 0 (discharge), P_batt < 0 (charge).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random
from typing import Literal

from microgrid_sim.types import BatteryAction
from .base import BaseStorage
from .reliability import ReliabilityMixin, ReliabilityParams, FailureMode


class PowerState(Enum):
    OFF = "OFF"
    ON = "ON"


@dataclass
class BatteryParams:
    capacity_kwh: float
    time_step_minutes: float
    initial_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.9
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    degradation_cost_per_kwh: float = 0.0  # expense per kWh throughput (NEG cash flow)


class BatteryStorage(BaseStorage, ReliabilityMixin):
    """
    Concrete battery with ON/OFF, power limits, SOC bounds, and efficiency.

    Actions (dict)
    --------------
    - set_state: "ON" | "OFF"  (optional)
    - power_setpoint: float (kW). Positive -> discharge; Negative -> charge.
      The realized power is clamped by limits and SOC feasibility.

    Cash-flow model
    ---------------
    Only degradation is accounted here:
        step_throughput_kWh = |P_actual| * Deltat_hours
        cash_flow = - degradation_cost_per_kwh * step_throughput_kWh
    NEGATIVE means you paid for degradation (expense).
    """

    def __init__(
        self,
        name: str,
        capacity_kwh: float,
        time_step_minutes: float,
        initial_soc: float = 0.5,
        min_soc: float = 0.1,
        max_soc: float = 0.9,
        max_charge_kw: float = 50.0,
        max_discharge_kw: float = 50.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        degradation_cost_per_kwh: float = 0.0,
        reliability: ReliabilityParams | None = None,
        reliability_rng: random.Random | None = None,
    ):
        super().__init__(name, capacity_kwh=capacity_kwh, initial_soc=initial_soc)
        ReliabilityMixin.__init__(self, reliability, reliability_rng)
        self.params = BatteryParams(
            capacity_kwh=float(capacity_kwh),
            time_step_minutes=float(time_step_minutes),
            initial_soc=float(initial_soc),
            min_soc=float(min_soc),
            max_soc=float(max_soc),
            max_charge_kw=float(max_charge_kw),
            max_discharge_kw=float(max_discharge_kw),
            charge_efficiency=float(charge_efficiency),
            discharge_efficiency=float(discharge_efficiency),
            degradation_cost_per_kwh=float(degradation_cost_per_kwh),
        )
        self.state: PowerState = PowerState.OFF

    # ----- helpers

    @property
    def dt_hours(self) -> float:
        return self.params.time_step_minutes / 60.0

    def get_soc(self) -> float:
        return self._soc

    # ----- API

    def reset(self):
        super().reset()
        self._soc = self.params.initial_soc
        self._power_flow = 0.0
        self.state = PowerState.OFF
        self._cost = 0.0
        self._reset_reliability()

    def step(self, t: int, **kwargs):
        """
        Parameters
        ----------
        t : int
            current step index (not used in dynamics, available for logging)
        kwargs : dict
            action dict with optional keys:
                - set_state: "ON" | "OFF"
                - power_setpoint: float (kW), +discharge / -charge
        """
        # Parse action
        action: BatteryAction | float | None = kwargs.get("action", None)
        set_state: Literal["ON", "OFF"] | None = None
        power_setpoint: float = 0.0
        exo = kwargs.get("exogenous", {}) or {}
        self._update_reliability(exo, self.dt_hours)

        if isinstance(action, dict):
            set_state = action.get("set_state")
            power_setpoint = float(action.get("power_setpoint", 0.0))
        elif action is not None:
            # Allow direct numeric as shorthand for setpoint
            try:
                power_setpoint = float(action)
            except Exception:
                power_setpoint = 0.0

        if set_state is not None:
            self.state = PowerState(set_state)

        # If OFF, no power and no degradation cost
        if self.state is PowerState.OFF:
            self._power_flow = 0.0
            self._cost = 0.0
            return

        # Reliability major outage -> force offline
        if self.reliability_state.mode is FailureMode.MAJOR:
            self._power_flow = 0.0
            self._cost = self._reliability_cost(self.dt_hours)
            return

        # Clamp power by hardware limits
        p_cmd = float(power_setpoint)
        p_cmd = max(-self.params.max_charge_kw, min(self.params.max_discharge_kw, p_cmd))

        # Compute feasible power wrt SOC and efficiencies
        soc = self._soc
        cap = self.params.capacity_kwh
        dt = self.dt_hours
        eta_c = self.params.charge_efficiency
        eta_d = self.params.discharge_efficiency

        p_actual = p_cmd

        if p_cmd >= 0.0:
            # Discharge: SoC decreases by (p / eta_d) * dt / cap
            max_allowable_discharge_kwh = (soc - self.params.min_soc) * cap  # energy we can take out
            max_discharge_kw_soc = (max_allowable_discharge_kwh * eta_d) / dt if dt > 0 else 0.0
            p_actual = min(p_cmd, max(0.0, max_discharge_kw_soc))
            delta_soc = -(p_actual / eta_d) * dt / cap
            soc = max(self.params.min_soc, soc + delta_soc)
        else:
            # Charge: SoC increases by (|p| * eta_c) * dt / cap
            p_ch = -p_cmd  # positive charge magnitude
            max_allowable_charge_kwh = (self.params.max_soc - soc) * cap
            max_charge_kw_soc = (max_allowable_charge_kwh / eta_c) / dt if dt > 0 else 0.0
            p_actual_mag = min(p_ch, max(0.0, max_charge_kw_soc))
            p_actual = -p_actual_mag
            delta_soc = (p_actual_mag * eta_c) * dt / cap
            soc = min(self.params.max_soc, soc + delta_soc)

        p_actual = self._reliability_derate(p_actual)

        self._power_flow = p_actual
        self._soc = soc

        # Degradation cash flow (NEG=expense). Throughput in kWh this step:
        throughput_kwh = abs(self._power_flow) * dt
        self._cost = - self.params.degradation_cost_per_kwh * throughput_kwh
        self._cost += self._reliability_cost(self.dt_hours)
        self._downtime = self._reliability_downtime_flag()
