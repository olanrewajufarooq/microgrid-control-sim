"""
microgrid_sim/control/ems_rule.py

Rule-Based Baseline Energy Management System (Topic 10 of 20).
No forecasting, no beliefs — purely heuristic, transparent rules.

This controller reads the current system state (SoC and hour-of-day)
and sets actions for:
  - BatteryStorage: charge (negative kW) or discharge (positive kW)
  - FossilGenerator: ON/OFF with power setpoint

Grid exchange is handled by the GridIntertie (slack) inside the environment.

Sign conventions
---------------
- Generation > 0 kW
- Consumption < 0 kW
- Battery: discharging (+kW), charging (-kW)

Costs
-----
Cashflow is computed by components (diesel fuel, battery degradation, grid tariffs).
The EMS does not directly compute cost — it decides actions which affect costs.

References
----------
1. Bordons, C., García-Torres, F., & Ridao (2020). Model Predictive Control of Microgrids. Springer.
2. Lasseter, R. H. (2011). Smart Distribution: Coupled Microgrids. Proceedings of the IEEE.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EMSRuleParams:
    """
    Tunable parameters for the Rule-Based EMS.
    """
    # Battery policy
    charge_kw: float = -6.0            # negative = charging
    discharge_kw: float = 6.0          # positive = discharging
    soc_charge_band: float = 0.90      # charge if SOC below this during charge window
    soc_discharge_band: float = 0.20   # discharge if SOC above this during discharge window

    # Time-of-day windows (inclusive hour indices in [0..23])
    charge_hours_start: int = 11
    charge_hours_end: int   = 15
    discharge_hours_start: int = 18
    discharge_hours_end: int   = 21

    # Diesel policy (night support)
    diesel_night_on: bool = True
    diesel_night_start: int = 20
    diesel_night_end: int   = 6        # cyclic range: [20..23] U [0..6)
    diesel_night_kw: float  = 4.0


class RuleBasedEMS:
    """
    Simple, fully deterministic EMS for baseline benchmarking.

    Usage
    -----
    ems = RuleBasedEMS()
    actions = ems.decide(hour=14, soc=0.42, exogenous={"house": {"load_kw": 1.2}})
    """

    def __init__(self, params: EMSRuleParams | None = None,
                 battery_name: str = "bat",
                 diesel_name: str = "diesel"):
        self.p = params or EMSRuleParams()
        self.battery_name = str(battery_name)
        self.diesel_name = str(diesel_name)

    @staticmethod
    def _in_range_cyclic(h: int, start: int, end: int) -> bool:
        """Check if hour h ∈ [start..end] with wrap-around (e.g., 20..6)."""
        if start <= end:
            return start <= h <= end
        return h >= start or h <= end

    def decide(self, hour: int, soc: float, exogenous: Dict[str, Dict[str, float]] | None = None) -> Dict[str, Any]:
        """
        Decide component actions for this step.

        Parameters
        ----------
        hour : int
            Hour-of-day [0..23].
        soc : float
            Battery state-of-charge [0..1].
        exogenous : dict, optional
            Current exogenous signals (not used in this simple EMS,
            kept for future extensions and API symmetry).

        Returns
        -------
        actions : dict
            Mapping suitable for MicrogridEnv.step(actions=...)
        """
        p = self.p
        actions: Dict[str, Any] = {}

        # Diesel heuristic (night support window)
        diesel_on = p.diesel_night_on and self._in_range_cyclic(hour, p.diesel_night_start, p.diesel_night_end)
        actions[self.diesel_name] = {
            "on": bool(diesel_on),
            "power_setpoint": float(p.diesel_night_kw if diesel_on else 0.0),
        }

        # Battery heuristic: charge window vs discharge window with SOC bands
        bat_sp = 0.0
        if p.charge_hours_start <= hour <= p.charge_hours_end and soc < p.soc_charge_band:
            bat_sp = p.charge_kw  # charge (negative)
        elif p.discharge_hours_start <= hour <= p.discharge_hours_end and soc > p.soc_discharge_band:
            bat_sp = p.discharge_kw  # discharge (positive)
        else:
            bat_sp = 0.0

        actions[self.battery_name] = {"set_state": "ON", "power_setpoint": float(bat_sp)}
        return actions
