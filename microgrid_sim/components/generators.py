"""
microgrid_sim/components/generators.py

Validated deterministic generator models with cost and control features.

Each class now includes:
- Literature-based steady-state power equations.
- Optional start/shutdown or connect/disconnect control.
- Cost helper methods (fuel, maintenance, startup/shutdown).
- Consistent sign conventions (generation > 0, consumption < 0).

References
----------
1. Bordons, C., García-Torres, F., & Ridao, M. (2020).
   *Model Predictive Control of Microgrids.* Springer.
2. NREL PVWatts® Model Documentation, 2014.
3. IEC 61400-12-1: Wind Turbine Power Performance Testing.
4. Hansen, A. D. (2008). *Wind Turbine Power Curves: An Analysis.*
5. Penche, C. (1998). *Layman's Handbook on How to Develop a Small Hydro Site.*
6. IEA (2022). *Emission Factors for Electricity and Heat.*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import BaseGenerator

# ============================================================
# --- Shared Mixin for Cost Calculations ---------------------
# ============================================================

class GeneratorCostMixin:
    """Reusable economic-cost helper methods (NEG = expense)."""

    def fuel_cost(self, energy_kwh: float, rate_per_kwh: float) -> float:
        """Variable fuel expense ($)."""
        return -abs(energy_kwh * rate_per_kwh)

    def maintenance_cost(self, hours: float, rate_per_hour: float) -> float:
        """Fixed maintenance cost ($) for run-hours."""
        return -abs(hours * rate_per_hour)

    def startup_cost(self, cost: float) -> float:
        """One-time startup expense ($)."""
        return -abs(cost)

    def shutdown_cost(self, cost: float) -> float:
        """One-time shutdown expense ($)."""
        return -abs(cost)


# ============================================================
# --- PV Generator (PVWatts-style) ----------------------------
# ============================================================

@dataclass
class PVParams:
    capacity_kw: float
    time_step_minutes: float
    derate: float = 0.9
    temp_coeff_per_degC: float = -0.0045  # per °C
    cell_temp_C: float = 25.0
    p_min_kw: float = 0.0
    p_max_kw: Optional[float] = None


class PVGenerator(BaseGenerator):
    """
    PV array model using PVWatts-style scaling with optional temperature factor.

    Equation
    --------
    P_PV = P_rated * derate * (G_t / 1000) * (1 + γ (T_cell − 25))
    where:
        G_t [W/m²] = irradiance,
        γ = temp_coeff_per_degC,
        P_rated [kW] = nameplate capacity.

    When disconnected → P = 0.
    """

    def __init__(self, name: str, capacity_kw: float, time_step_minutes: float,
                 derate: float = 0.9, p_min_kw: float = 0.0,
                 p_max_kw: Optional[float] = None,
                 temp_coeff_per_degC: float = -0.0045,
                 cell_temp_C: float = 25.0):
        super().__init__(name)
        self.params = PVParams(
            capacity_kw=float(capacity_kw),
            time_step_minutes=float(time_step_minutes),
            derate=float(derate),
            p_min_kw=float(p_min_kw),
            p_max_kw=float(p_max_kw) if p_max_kw is not None else float(capacity_kw),
            temp_coeff_per_degC=float(temp_coeff_per_degC),
            cell_temp_C=float(cell_temp_C),
        )
        self._connected = True

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def reset(self):
        super().reset()
        self._connected = True

    def step(self, t: int, **kwargs):
        if not self._connected:
            self._power_output = 0.0
            self._cost = 0.0
            return

        exo = kwargs.get("exogenous", {}) or {}
        g = float(exo.get("irradiance_Wm2", 0.0))
        p_stc = self.params.capacity_kw * self.params.derate * (g / 1000.0)
        temp_factor = 1.0 + self.params.temp_coeff_per_degC * (self.params.cell_temp_C - 25.0)
        p = p_stc * temp_factor
        p = max(self.params.p_min_kw, min(self.params.p_max_kw, p))
        self._power_output = p
        self._cost = 0.0


# ============================================================
# --- Wind Turbine (IEC cubic) -------------------------------
# ============================================================

@dataclass
class WindParams:
    rated_kw: float
    time_step_minutes: float
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0


class WindTurbine(BaseGenerator):
    """
    IEC-style cubic wind turbine model with connect/disconnect.

    Equation
    --------
    P(v) =
        0,                                   v < v_ci or v > v_co
        P_r * ((v − v_ci)/(v_r − v_ci))³,    v_ci ≤ v < v_r
        P_r,                                 v_r ≤ v ≤ v_co
    """

    def __init__(self, name: str, rated_kw: float, time_step_minutes: float,
                 cut_in_ms: float = 3.0, rated_ms: float = 12.0, cut_out_ms: float = 25.0):
        super().__init__(name)
        self.params = WindParams(float(rated_kw), float(time_step_minutes),
                                 float(cut_in_ms), float(rated_ms), float(cut_out_ms))
        self._connected = True

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def reset(self):
        super().reset()
        self._connected = True

    def step(self, t: int, **kwargs):
        if not self._connected:
            self._power_output = 0.0
            self._cost = 0.0
            return

        v = float((kwargs.get("exogenous", {}) or {}).get("wind_speed_ms", 0.0))
        if v < self.params.cut_in_ms or v >= self.params.cut_out_ms:
            p = 0.0
        elif v < self.params.rated_ms:
            x = (v - self.params.cut_in_ms) / (self.params.rated_ms - self.params.cut_in_ms)
            p = self.params.rated_kw * (x ** 3)
        else:
            p = self.params.rated_kw
        self._power_output = max(0.0, p)
        self._cost = 0.0


# ============================================================
# --- Hydro Generator (linearized) ---------------------------
# ============================================================

@dataclass
class HydroParams:
    k_w_per_cms: float
    time_step_minutes: float
    p_max_kw: Optional[float] = None
    maintenance_cost_per_hour: float = 0.0
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0


class HydroGenerator(BaseGenerator, GeneratorCostMixin):
    """
    Linearized hydro model: P = k * Q.

    Origin
    ------
    Derived from P = ρ g η Q H,
    where ρ=1000 kg/m³, g=9.81 m/s², η~0.8, H=effective head.
    At EMS level we use constant k [kW per m³/s].

    Start/shutdown & maintenance costs are included.
    """

    def __init__(self, name: str, k_w_per_cms: float, time_step_minutes: float,
                 p_max_kw: Optional[float] = None,
                 maintenance_cost_per_hour: float = 0.0,
                 startup_cost: float = 0.0,
                 shutdown_cost: float = 0.0):
        super().__init__(name)
        self.params = HydroParams(float(k_w_per_cms), float(time_step_minutes),
                                  float(p_max_kw) if p_max_kw is not None else None,
                                  float(maintenance_cost_per_hour),
                                  float(startup_cost),
                                  float(shutdown_cost))
        self._on = True

    @property
    def dt_hours(self): return self.params.time_step_minutes / 60.0

    def start(self):
        if not self._on:
            self._on = True
            self._cost += self.startup_cost(self.params.startup_cost)

    def shutdown(self):
        if self._on:
            self._on = False
            self._cost += self.shutdown_cost(self.params.shutdown_cost)

    def reset(self):
        super().reset(); self._on = True

    def step(self, t: int, **kwargs):
        if not self._on:
            self._power_output = 0; self._cost = 0; return
        flow = float((kwargs.get("exogenous", {}) or {}).get("flow_cms", 0.0))
        p = self.params.k_w_per_cms * flow
        if self.params.p_max_kw: p = min(self.params.p_max_kw, p)
        p = max(0.0, p)
        self._power_output = p
        self._cost = self.maintenance_cost(self.dt_hours, self.params.maintenance_cost_per_hour)


# ============================================================
# --- Fossil Generator (dispatchable) ------------------------
# ============================================================

@dataclass
class FossilParams:
    p_min_kw: float
    p_max_kw: float
    time_step_minutes: float
    fuel_cost_per_kwh: float
    maintenance_cost_per_hour: float = 0.0
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0
    efficiency: float = 0.35  # electrical efficiency (optional)


class FossilGenerator(BaseGenerator, GeneratorCostMixin):
    """
    Dispatchable fossil generator.

    Steady-state model
    ------------------
    E = P Δt,
    C_fuel = p_fuel * E,
    η_gen = P_elec / P_fuel_in.

    Optional emission factor: 0.7 kgCO₂/kWh typical diesel.

    Costs
    -----
    fuel_cost_per_kwh, maintenance_cost_per_hour, startup/shutdown_cost.

    Control
    -------
    - Float action: power setpoint (kW) → >0 ON, ≤0 OFF.
    - Dict  action: {"on": bool, "power_setpoint": float}.
    """

    def __init__(self, name: str,
                 p_min_kw: float, p_max_kw: float, time_step_minutes: float,
                 fuel_cost_per_kwh: float,
                 maintenance_cost_per_hour: float = 0.0,
                 startup_cost: float = 0.0,
                 shutdown_cost: float = 0.0,
                 efficiency: float = 0.35):
        super().__init__(name)
        self.params = FossilParams(float(p_min_kw), float(p_max_kw),
                                   float(time_step_minutes),
                                   float(fuel_cost_per_kwh),
                                   float(maintenance_cost_per_hour),
                                   float(startup_cost),
                                   float(shutdown_cost),
                                   float(efficiency))
        self._on = False

    @property
    def dt_hours(self): return self.params.time_step_minutes / 60.0

    def start(self):
        if not self._on:
            self._on = True
            self._cost += self.startup_cost(self.params.startup_cost)

    def shutdown(self):
        if self._on:
            self._on = False
            self._cost += self.shutdown_cost(self.params.shutdown_cost)

    def emissions(self, energy_kwh: float, kg_per_kwh: float = 0.7) -> float:
        """Approximate CO₂ emissions (kg)."""
        return energy_kwh * kg_per_kwh

    def reset(self):
        super().reset(); self._on = False

    def step(self, t: int, **kwargs):
        a = kwargs.get("action", 0.0)
        if isinstance(a, dict):
            on_cmd = bool(a.get("on", self._on))
            sp = float(a.get("power_setpoint", 0.0))
        else:
            sp = float(a); on_cmd = sp > 0
        prev_on = self._on; self._on = on_cmd

        if not self._on:
            if prev_on: self._cost += self.shutdown_cost(self.params.shutdown_cost)
            self._power_output = 0; return
        if self._on and not prev_on:
            self._cost += self.startup_cost(self.params.startup_cost)

        p = max(self.params.p_min_kw, min(self.params.p_max_kw, sp))
        self._power_output = p
        energy = p * self.dt_hours
        self._cost += self.fuel_cost(energy, self.params.fuel_cost_per_kwh)
        self._cost += self.maintenance_cost(self.dt_hours, self.params.maintenance_cost_per_hour)


# ============================================================
# --- Grid Intertie (connect/disconnect) ---------------------
# ============================================================

@dataclass
class GridParams:
    time_step_minutes: float
    price_import_per_kwh: float
    price_export_per_kwh: float
    import_limit_kw: Optional[float] = None
    export_limit_kw: Optional[float] = None


class GridIntertie(BaseGenerator):
    """
    Grid interface with connect/disconnect capability.

    - connect(): rejoin main grid.
    - disconnect(): island the microgrid.
    - When disconnected: P=0, cost=0.

    Limits: import_limit_kw, export_limit_kw.
    """

    def __init__(self, name: str, time_step_minutes: float,
                 price_import_per_kwh: float, price_export_per_kwh: float,
                 import_limit_kw: Optional[float] = None,
                 export_limit_kw: Optional[float] = None):
        super().__init__(name)
        self.params = GridParams(float(time_step_minutes),
                                 float(price_import_per_kwh),
                                 float(price_export_per_kwh),
                                 None if import_limit_kw is None else float(import_limit_kw),
                                 None if export_limit_kw is None else float(export_limit_kw))
        self._connected = True

    def connect(self): self._connected = True
    def disconnect(self): self._connected = False
    def is_connected(self) -> bool: return bool(self._connected)

    @property
    def dt_hours(self): return self.params.time_step_minutes / 60.0

    def _clamp_by_limits(self, desired_p: float) -> float:
        if desired_p >= 0:
            if self.params.import_limit_kw is None: return desired_p
            return min(desired_p, self.params.import_limit_kw)
        else:
            if self.params.export_limit_kw is None: return desired_p
            return max(desired_p, -self.params.export_limit_kw)

    def step(self, t: int, **kwargs):
        if not self._connected:
            self._power_output = 0; self._cost = 0; return
        a = kwargs.get("action", 0.0)
        try: desired_p = float(a)
        except Exception: desired_p = 0.0
        p = self._clamp_by_limits(desired_p)
        self._power_output = p
        e = abs(p) * self.dt_hours
        if p >= 0: self._cost = - self.params.price_import_per_kwh * e
        else:      self._cost = + self.params.price_export_per_kwh * e
