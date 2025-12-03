"""
microgrid_sim/components/generators.py

Validated stochastic generator models with cost and control features.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .base import BaseGenerator
from .reliability import FailureMode, ReliabilityMixin, ReliabilityParams

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
    temp_coeff_per_degC: float = -0.0045  # per degC
    cell_temp_C: float = 25.0
    p_min_kw: float = 0.0
    p_max_kw: Optional[float] = None
    operational_cost_per_kwh: float = 0.0


class PVGenerator(BaseGenerator, ReliabilityMixin):
    """
    PV array model using PVWatts-style scaling with optional temperature factor.

    Equation
    --------
    P_PV = P_rated * derate * (G_t / 1000) * (1 + gamma (T_cell - 25))
    where:
        G_t [W/m^2] = irradiance,
        gamma = temp_coeff_per_degC,
        P_rated [kW] = nameplate capacity.

    When disconnected -> P = 0.
    """

    def __init__(
        self, name: str, capacity_kw: float, time_step_minutes: float,
        derate: float = 0.9, p_min_kw: float = 0.0,
        p_max_kw: Optional[float] = None,
        temp_coeff_per_degC: float = -0.0045,
        cell_temp_C: float = 25.0,
        operational_cost_per_kwh: float = 0.0,
        reliability: ReliabilityParams | None = None,
        reliability_rng: random.Random | None = None,
        ):
        super().__init__(name)
        ReliabilityMixin.__init__(self, reliability, reliability_rng)
        self.params = PVParams(
            capacity_kw=float(capacity_kw),
            time_step_minutes=float(time_step_minutes),
            derate=float(derate),
            p_min_kw=float(p_min_kw),
            p_max_kw=float(p_max_kw) if p_max_kw is not None else float(capacity_kw),
            temp_coeff_per_degC=float(temp_coeff_per_degC),
            cell_temp_C=float(cell_temp_C),
            operational_cost_per_kwh=float(operational_cost_per_kwh),
        )
        self._connected = True

    @property
    def dt_hours(self) -> float:
        return self.params.time_step_minutes / 60.0

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def reset(self):
        super().reset()
        self._connected = True
        self._reset_reliability()

    def step(self, t: int, **kwargs):
        exo = kwargs.get("exogenous", {}) or {}
        self._update_reliability(exo, self.dt_hours)

        if not self._connected:
            self._power_output = 0.0
            self._cost = 0.0
            return

        g = float(exo.get("irradiance_Wm2", 0.0))
        p_stc = self.params.capacity_kw * self.params.derate * (g / 1000.0)
        temp_factor = 1.0 + self.params.temp_coeff_per_degC * (self.params.cell_temp_C - 25.0)
        p = p_stc * temp_factor
        p = max(self.params.p_min_kw, min(self.params.p_max_kw, p))
        p = self._reliability_derate(p)
        self._power_output = p

        # O&M cost per generated kWh (NEGATIVE = expense)
        e_kwh = max(0.0, p) * self.dt_hours
        self._cost = -self.params.operational_cost_per_kwh * e_kwh
        self._cost += self._reliability_cost(self.dt_hours)
        self._downtime = self._reliability_downtime_flag()


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
    operational_cost_per_kwh: float = 0.0

class WindTurbine(BaseGenerator, ReliabilityMixin):
    """
    IEC-style cubic wind turbine model with connect/disconnect.

    Equation
    --------
    P(v) =
        0,                                   v < v_ci or v > v_co
        P_r * ((v - v_ci)/(v_r - v_ci))^3,    v_ci <= v < v_r
        P_r,                                 v_r <= v <= v_co
    """

    def __init__(
        self, name: str,
        rated_kw: float, time_step_minutes: float,
        cut_in_ms: float = 3.0,
        rated_ms: float = 12.0,
        cut_out_ms: float = 25.0,
        operational_cost_per_kwh: float = 0.0,
        reliability: ReliabilityParams | None = None,
        reliability_rng: random.Random | None = None,
        ):
        super().__init__(name)
        ReliabilityMixin.__init__(self, reliability, reliability_rng)
        self.params = WindParams(
            float(rated_kw), float(time_step_minutes),
            float(cut_in_ms), float(rated_ms), float(cut_out_ms),
            float(operational_cost_per_kwh),
            )
        self._connected = True

    @property
    def dt_hours(self) -> float:
        return self.params.time_step_minutes / 60.0

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def reset(self):
        super().reset()
        self._connected = True
        self._reset_reliability()

    def step(self, t: int, **kwargs):
        exo = kwargs.get("exogenous", {}) or {}
        self._update_reliability(exo, self.dt_hours)

        if not self._connected:
            self._power_output = 0.0
            self._cost = 0.0
            return

        v = float(exo.get("wind_speed_ms", 0.0))
        if v < self.params.cut_in_ms or v >= self.params.cut_out_ms:
            p = 0.0
        elif v < self.params.rated_ms:
            x = (v - self.params.cut_in_ms) / (self.params.rated_ms - self.params.cut_in_ms)
            p = self.params.rated_kw * (x ** 3)
        else:
            p = self.params.rated_kw
        p = self._reliability_derate(p)
        self._power_output = max(0.0, p)

        # O&M cost per generated kWh (NEGATIVE = expense)
        e_kwh = self._power_output * self.dt_hours
        self._cost = -self.params.operational_cost_per_kwh * e_kwh
        self._cost += self._reliability_cost(self.dt_hours)
        self._downtime = self._reliability_downtime_flag()


# ============================================================
# --- Hydro Generator (linearized) ---------------------------
# ============================================================

@dataclass
class HydroParams:
    k_w_per_cms: float
    time_step_minutes: float
    p_max_kw: Optional[float] = None
    operational_cost_per_hour: float = 0.0
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0


class HydroGenerator(BaseGenerator, GeneratorCostMixin, ReliabilityMixin):
    """
    Linearized hydro model: P = k * Q.

    Origin
    ------
    Derived from P = rho g eta Q H,
    where rho=1000 kg/m^3, g=9.81 m/s^2, eta~0.8, H=effective head.
    At EMS level we use constant k [kW per m^3/s].

    Start/shutdown & maintenance costs are included.
    """

    def __init__(
        self, name: str, k_w_per_cms: float, time_step_minutes: float,
        p_max_kw: Optional[float] = None,
        operational_cost_per_hour: float = 0.0,
        startup_cost: float = 0.0,
        shutdown_cost: float = 0.0,
        reliability: ReliabilityParams | None = None,
        reliability_rng: random.Random | None = None,
        ):
        super().__init__(name)
        ReliabilityMixin.__init__(self, reliability, reliability_rng)
        self.params = HydroParams(
            float(k_w_per_cms), float(time_step_minutes),
            float(p_max_kw) if p_max_kw is not None else None,
            float(operational_cost_per_hour),
            float(startup_cost),
            float(shutdown_cost)
            )
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
        self._reset_reliability()

    def step(self, t: int, **kwargs):
        exo = kwargs.get("exogenous", {}) or {}
        self._update_reliability(exo, self.dt_hours)

        if self.reliability_state.mode is FailureMode.MAJOR:
            self._power_output = 0.0
            self._cost = self._reliability_cost(self.dt_hours)
            self._downtime = 1.0
            return

        if not self._on:
            self._power_output = 0; self._cost = 0; return
        flow = float(exo.get("flow_cms", 0.0))
        p = self.params.k_w_per_cms * flow
        if self.params.p_max_kw: p = min(self.params.p_max_kw, p)
        p = max(0.0, p)
        p = self._reliability_derate(p)
        self._power_output = p
        self._cost = self.maintenance_cost(self.dt_hours, self.params.operational_cost_per_hour)
        self._cost += self._reliability_cost(self.dt_hours)
        self._downtime = self._reliability_downtime_flag()


# ============================================================
# --- Fossil Generator (dispatchable) ------------------------
# ============================================================

@dataclass
class FossilParams:
    p_min_kw: float
    p_max_kw: float
    time_step_minutes: float
    fuel_cost_per_kwh: float
    operational_cost_per_hour: float = 0.0
    startup_cost: float = 0.0
    shutdown_cost: float = 0.0
    efficiency: float = 0.35  # electrical efficiency (optional)

class FossilGenerator(BaseGenerator, GeneratorCostMixin, ReliabilityMixin):
    """
    Dispatchable fossil generator.

    Steady-state model
    ------------------
    E = P Deltat,
    C_fuel = p_fuel * E,
    eta_gen = P_elec / P_fuel_in.

    Optional emission factor: 0.7 kgCO_2/kWh typical diesel.

    Costs
    -----
    fuel_cost_per_kwh, maintenance_cost_per_hour, startup/shutdown_cost.

    Control
    -------
    - Float action: power setpoint (kW) -> >0 ON, <=0 OFF.
    - Dict  action: {"on": bool, "power_setpoint": float}.
    """

    def __init__(
        self, name: str,
        p_min_kw: float, p_max_kw: float, time_step_minutes: float,
        fuel_cost_per_kwh: float,
        operational_cost_per_hour: float = 0.0,
        startup_cost: float = 0.0,
        shutdown_cost: float = 0.0,
        efficiency: float = 0.35,
        reliability: ReliabilityParams | None = None,
        reliability_rng: random.Random | None = None,
        ):
        super().__init__(name)
        ReliabilityMixin.__init__(self, reliability, reliability_rng)
        self.params = FossilParams(
            float(p_min_kw), float(p_max_kw),
            float(time_step_minutes),
            float(fuel_cost_per_kwh),
            float(operational_cost_per_hour),
            float(startup_cost),
            float(shutdown_cost),
            float(efficiency),
            )
        self._on = False
        self._cost = 0.0

    @property
    def dt_hours(self): return self.params.time_step_minutes / 60.0

    def start(self):
        if not self._on:
            self._on = True
            # self._cost += self.startup_cost(self.params.startup_cost)

    def shutdown(self):
        if self._on:
            self._on = False
            # self._cost += self.shutdown_cost(self.params.shutdown_cost)

    def emissions(self, energy_kwh: float, kg_per_kwh: float = 0.7) -> float:
        """Approximate CO_2 emissions (kg)."""
        return energy_kwh * kg_per_kwh

    def reset(self):
        super().reset();
        self._on = False
        self._cost = 0.0
        self._reset_reliability()

    def step(self, t: int, **kwargs):
        """
        Expected action patterns:
        - float: power_setpoint (kW) -> >0 turns ON, <=0 turns OFF
        - dict : {"on": bool, "power_setpoint": float}
        Cashflow convention: NEGATIVE = expense, POSITIVE = revenue.
        """
        a = kwargs.get("action", 0.0)
        exo = kwargs.get("exogenous", {}) or {}
        dt_h = self.dt_hours

        # Update reliability first; a failure may force outage
        self._update_reliability(exo, dt_h)
        if self.reliability_state.mode is FailureMode.MAJOR:
            self._power_output = 0.0
            self._cost = self._reliability_cost(dt_h)
            self._on = False
            return

        # Parse action
        on_cmd = self._on
        sp = None  # None means "no setpoint provided"
        if isinstance(a, dict):
            on_cmd = bool(a.get("on", self._on))
            sp_raw = a.get("power_setpoint", None)
            if sp_raw is not None:
                try:
                    sp = float(sp_raw)
                except (TypeError, ValueError):
                    sp = None  # treat invalid as not provided
        else:
            # numeric or string -> try cast
            try:
                sp = float(a)
                on_cmd = sp > 0.0
            except (TypeError, ValueError):
                sp = None
                on_cmd = self._on

        prev_on = self._on
        self._on = on_cmd

        # --- Dispatch power ---
        if not self._on:
            dispatched_kw = 0.0
        else:
            if sp is None:
                # ON but no setpoint -> default to p_min
                dispatched_kw = max(self.params.p_min_kw, 0.0)
            else:
                dispatched_kw = max(self.params.p_min_kw, min(self.params.p_max_kw, sp))

        self._power_output = max(0.0, dispatched_kw)
        if self.reliability_state.mode is FailureMode.MINOR:
            self._power_output = self._reliability_derate(self._power_output)

        # --- Per-step costs (NEGATIVE = expense) ---
        energy_kwh = self._power_output * dt_h

        fuel_cost     = self.fuel_cost(energy_kwh, self.params.fuel_cost_per_kwh) if self._on else 0.0
        maint_cost    = self.maintenance_cost(dt_h, self.params.operational_cost_per_hour) if self._on else 0.0
        start_cost    = self.startup_cost(self.params.startup_cost)   if (self._on and not prev_on) else 0.0
        shutdown_cost = self.shutdown_cost(self.params.shutdown_cost) if (prev_on and not self._on) else 0.0

        self._cost = fuel_cost + maint_cost + start_cost + shutdown_cost
        self._cost += self._reliability_cost(dt_h)
        self._downtime = self._reliability_downtime_flag()


# ============================================================
# --- Grid Intertie (connect/disconnect) ---------------------
# ============================================================

@dataclass
class GridParams:
    time_step_minutes: float
    price_import_per_kwh: float
    price_export_per_kwh: float
    import_limit_kw: Optional[float] = None # Default: None = infinity
    export_limit_kw: Optional[float] = None # Default: None = infinity


class GridIntertie(BaseGenerator, ReliabilityMixin):
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
        export_limit_kw: Optional[float] = None,
        reliability: ReliabilityParams | None = None,
        reliability_rng: random.Random | None = None,
        ):

        super().__init__(name)
        ReliabilityMixin.__init__(self, reliability, reliability_rng)

        self.params = GridParams(
            float(time_step_minutes),
            float(price_import_per_kwh),
            float(price_export_per_kwh),
            import_limit_kw,
            export_limit_kw,
            )
        self._connected = True

    def connect(self):
        self._connected = True
    def disconnect(self):
        self._connected = False
    def is_connected(self) -> bool:
        return bool(self._connected)
    def reset(self):
        super().reset()
        self._connected = True
        self._reset_reliability()

    @property
    def dt_hours(self):
        return self.params.time_step_minutes / 60.0

    def _clamp_by_limits(
        self,
        desired_p: float,
        import_limit: Optional[float],
        export_limit: Optional[float]
        ) -> float:
        """
        Clamps the desired power based on the DYNAMIC limits.
        'None' is treated as infinity.
        """
        if desired_p >= 0:  # Importing
            if import_limit is None:
                return desired_p  # No limit
            return min(desired_p, import_limit)
        else:  # Exporting
            if export_limit is None:
                return desired_p  # No limit
            return max(desired_p, -export_limit)

    def step(self, t: int, **kwargs):
        """
        Steps the grid, using dynamic prices and limits from exogenous data.
        """
        exo = kwargs.get("exogenous", {}) or {}
        self._update_reliability(exo, self.dt_hours)

        if self.reliability_state.mode is FailureMode.MAJOR:
            self._connected = False
        if not self._connected:
            self._power_output = 0
            self._cost = 0
            return

        # Get DYNAMIC parameters from exogenous data

        # Get dynamic prices, falling back to init params if not provided
        price_import = exo.get("price_import_per_kwh", self.params.price_import_per_kwh)
        price_export = exo.get("price_export_per_kwh", self.params.price_export_per_kwh)

        # Get dynamic limits, falling back to init params (which default to None)
        import_limit = exo.get("import_limit_kw", self.params.import_limit_kw)
        export_limit = exo.get("export_limit_kw", self.params.export_limit_kw)
        if self.reliability_state.mode is FailureMode.MINOR and self.reliability_params:
            scale = self.reliability_params.minor_derate
            import_limit = scale * import_limit if import_limit is not None else None
            export_limit = scale * export_limit if export_limit is not None else None

        # Get action (the power requested by the system)
        a = kwargs.get("action", 0.0)
        try:
            desired_p = float(a)
        except Exception:
            desired_p = 0.0

        # Apply limits and calculate cost
        p = self._clamp_by_limits(desired_p, import_limit, export_limit)

        self._power_output = p
        e = abs(p) * self.dt_hours

        if p >= 0: # Importing
            self._cost =- price_import * e
        else: # Exporting
            self._cost =+ price_export * e

        self._downtime = self._reliability_downtime_flag()
        # No maintenance cost applied to grid
# ============================================================
# --- Replay Generator (Data-Driven) -------------------------
# ============================================================

class ReplayGenerator(BaseGenerator):
    """
    A simple, data-driven generator that "replays" a power
    time series from the exogenous data feed.

    It looks for an exogenous key "power_kw" and outputs
    that value directly.

    This component does support connect/disconnect for
    curtailment rules.
    """
    def __init__(self, name: str, time_step_minutes: float):
        super().__init__(name)
        self.params = {"time_step_minutes": time_step_minutes}
        self._connected = True

    def connect(self):
        """Connects the component to the grid."""
        self._connected = True

    def disconnect(self):
        """Disconnects the component, forcing power to 0."""
        self._connected = False

    def reset(self):
        """Resets the component to its initial state."""
        super().reset()
        self._connected = True

    def step(self, t: int, **kwargs):
        """
        Steps the component by reading the 'power_kw' from
        the exogenous data and setting it as the output.
        """

        action = kwargs.get("action")
        if action == "disconnect":
            self.disconnect()
        elif action == "connect":
            self.connect()

        if not self._connected:
            self._power_output = 0.0
            self._cost = 0.0
            return

        exo = kwargs.get("exogenous", {}) or {}
        p = float(exo.get("power_kw", 0.0))

        # We assume the data is already cleaned (non-negative)
        self._power_output = max(0.0, p)
        self._cost = 0.0 # Replay generators have no marginal cost
