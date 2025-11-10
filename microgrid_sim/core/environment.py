"""
microgrid_sim/core/environment.py

MicrogridEnv v2 — timeline & logging orchestrator that delegates
electrical bus balance to microgrid_sim.system.MicrogridSystem.

Responsibilities
----------------
- Hold component registries (generators, storage, loads, grid).
- Advance time steps, feeding actions + exogenous inputs to components.
- Use MicrogridSystem to compute bus balance and KPIs per step.
- Log per-component power/cost/SOC + aggregate KPIs into a history table.

Sign Conventions
----------------
- Power (kW): generation > 0; consumption / charging < 0.
- Cash flow per step: NEGATIVE = expense (you paid), POSITIVE = revenue.

Outputs (history keys)
----------------------
Per-component:
    "<name>_power", "<name>_cost", and "<name>_soc" (if storage)
Aggregates:
    "gen_total_kw", "load_total_kw", "storage_total_kw",
    "grid_slack_kw", "net_power_unbalanced",
    "unmet_load_kw", "curtailed_gen_kw", "downtime",
    "total_cashflow", "t"

Usage
-----
env = MicrogridEnv(simulation_steps=24)
env.add_component(pv); env.add_component(wind)
env.add_component(house); env.add_component(factory)
env.add_component(battery)
env.add_component(grid, is_grid=True)
env.reset()
env.run(actions_list, exogenous_list)
df = env.get_results()

References
----------
Bordons, C.; García-Torres, F.; Ridao, M. (2020).
Model Predictive Control of Microgrids. Springer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

try:
    import pandas as pd
except Exception:
    pd = None

from microgrid_sim.components.base import (BaseComponent, BaseGenerator,
                                           BaseLoad, BaseStorage)
from microgrid_sim.components.generators import GridIntertie
from microgrid_sim.system import MicrogridSystem

class EmsController(Protocol):
    """
    Defines the interface for any controller we want to pass to the environment.
    The notebook's RuleBasedEMS already matches this.
    """
    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]) -> Dict[str, Any]:
        ...

class MicrogridEnv:
    """
    Time-based orchestrator for microgrid simulation.

    Parameters
    ----------
    simulation_steps : int
        Number of time steps per run (e.g., 24 for a day if hourly).
    """

    def __init__(self,
                 simulation_hours: int,
                 control_interval_minutes: int,
                 sim_dt_minutes: int):

        self.simulation_hours = int(simulation_hours)
        self.control_dt = int(control_interval_minutes)
        self.sim_dt = int(sim_dt_minutes)

        if self.control_dt % self.sim_dt != 0:
            raise ValueError("Control interval must be a multiple of simulation timestep.")

        self.steps_per_control_interval = self.control_dt // self.sim_dt
        self.total_simulation_steps = self.simulation_hours * self.steps_per_control_interval

        self.current_step = 0

        # Component registries
        self.generators: List[BaseGenerator] = []
        self.storage: List[BaseStorage] = []
        self.loads: List[BaseLoad] = []
        self.grid_component: Optional[GridIntertie] = None
        self._system: Optional[MicrogridSystem] = None
        self.history: Dict[str, List[float]] = {}

    # ---------------------------------------------------------------------
    # Component management
    # ---------------------------------------------------------------------
    def add_component(self, component: BaseComponent, is_grid: bool = False):
        """
        Register a component with the environment.

        Args
        ----
        component : BaseComponent
        is_grid   : bool
            Mark True if this is the GridIntertie (slack). Only one allowed.
        """
        if is_grid:
            if not isinstance(component, GridIntertie):
                raise TypeError("Grid component must be a GridIntertie.")
            if self.grid_component is not None:
                raise ValueError("Grid component already added.")
            self.grid_component = component

        if isinstance(component, BaseGenerator):
            self.generators.append(component)
        elif isinstance(component, BaseStorage):
            self.storage.append(component)
        elif isinstance(component, BaseLoad):
            self.loads.append(component)

        self._wire_system()

    def _wire_system(self):
        """(Re)build the MicrogridSystem from current registries."""
        # Ensure the grid is not duplicated in generators list for the system's own accounting
        gens = [g for g in self.generators if not isinstance(g, GridIntertie)]
        self._system = MicrogridSystem(
            generators=gens,
            storage=list(self.storage),
            loads=list(self.loads),
            grid=self.grid_component,
        )

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------
    def _initialize_history(self):
        """Set up historical logs for every component and aggregate KPI fields."""
        self.history = {}

        # Use an ordered listing for deterministic column order
        comp_order: List[BaseComponent] = []
        comp_order: List[BaseComponent] = [*self.generators, *self.storage, *self.loads]
        if self.grid_component and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        # Per-component slots
        for comp in comp_order:
            self.history[f"{comp.name}_power"] = []
            self.history[f"{comp.name}_cost"] = []
            if isinstance(comp, BaseStorage):
                self.history[f"{comp.name}_soc"] = []

        # Aggregate slots
        for key in ["gen_total_kw", "load_total_kw", "storage_total_kw", "grid_slack_kw",
                    "net_power_unbalanced", "unmet_load_kw", "curtailed_gen_kw",
                    "downtime", "total_cashflow", "t"]:
            self.history[key] = []
        self.current_step = 0

    def _log_step(self, summary: Dict[str, float]):
        comp_order: List[BaseComponent] = [*self.generators, *self.storage, *self.loads]
        if self.grid_component and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        for comp in comp_order:
            self.history[f"{comp.name}_power"].append(comp.get_power())
            self.history[f"{comp.name}_cost"].append(comp.get_cost())
            if isinstance(comp, BaseStorage):
                self.history[f"{comp.name}_soc"].append(comp.get_soc())

        self.history["gen_total_kw"].append(summary["gen_kw"])
        self.history["load_total_kw"].append(summary["load_kw"])
        self.history["storage_total_kw"].append(summary["storage_kw"])
        self.history["grid_slack_kw"].append(summary["grid_kw"]) # This is grid_kw
        self.history["net_power_unbalanced"].append(
            summary["gen_kw"] + summary["storage_kw"] + summary["load_kw"]
        )
        self.history["unmet_load_kw"].append(summary["unmet_kw"])
        self.history["curtailed_gen_kw"].append(summary["curtailed_kw"])
        self.history["downtime"].append(1.0 if summary["unmet_kw"] > 0.0 else 0.0)
        self.history["total_cashflow"].append(summary["total_cost"])
        self.history["t"].append(self.current_step)

    # ---------------------------------------------------------------------
    # Simulation loop
    # ---------------------------------------------------------------------
    def reset(self):
        """Reset all components and (re)initialize logs."""
        self.current_step = 0
        for comp in self.generators + self.storage + self.loads:
            comp.reset()
        if self.grid_component is not None:
            self.grid_component.reset()

        # Ensure system is wired (in case add_component occurred before reset)
        self._wire_system()
        self._initialize_history()

    def step(self, actions: Optional[Dict[str, object]] = None,
             exogenous: Optional[Dict[str, object]] = None):
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        actions   : dict[name -> action], optional
            Controller actions for this step.
        exogenous : dict[name -> payload], optional
            External data (e.g., irradiance, wind, loads) for this step.
        """
        if self._system is None:
            raise RuntimeError("System not wired. Add components before stepping.")

        # Delegate physical balance to MicrogridSystem
        summary = self._system.step(actions=actions, exogenous=exogenous)

        # Log and advance time
        self._log_step(summary)
        self.current_step += 1

    def run(self, controller: EmsController, exogenous_list: List[Dict[str, Any]]):
        """
        Runs the full simulation.

        Parameters:
        -----------
        controller : EmsController
            An object with a .decide(hour, soc, exog) method.
        exogenous_list : list
            The *high-fidelity, per-simulation-step* list of exogenous data.
            (e.g., length 1440 for a 24-hour, 1-min-step sim).
        """
        if len(exogenous_list) != self.total_simulation_steps:
            raise ValueError(f"Exogenous list length ({len(exogenous_list)}) does not "
                             f"match total simulation steps ({self.total_simulation_steps}).")

        self.reset()

        current_ems_action = {}

        for k in range(self.total_simulation_steps):

            # Check if we are at the start of a new control interval
            if k % self.steps_per_control_interval == 0:
                hour = k // self.steps_per_control_interval

                # Get battery SOC (assumes one battery named "bat")
                soc = 0.5
                for s in self.storage:
                    if getattr(s, "name", "") == "bat":
                        soc = float(s.get_soc())

                # Controller makes a decision
                current_ems_action = controller.decide(
                    hour=hour,
                    soc=soc,
                    exogenous=exogenous_list[k] # Give it the data for this instant
                )

            # --- DYNAMIC GRID CONTROL (example for islanding) ---
            # This logic is now handled in the notebook's loop
            # But we must pass any grid actions from the controller
            if self.grid_component and "grid" in current_ems_action:
                 grid_ctrl = current_ems_action.get("grid")
                 if grid_ctrl == "disconnect":
                     self.grid_component.disconnect()
                 elif grid_ctrl == "connect":
                     self.grid_component.connect()

            # Run one simulation step using the held action
            self.step(
                actions=current_ems_action,
                exogenous=exogenous_list[k]
            )

    # ---------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------
    def get_results(self, as_dataframe: bool = True):
        """
        Return the simulation history.
        If some component histories are shorter/longer, we pad with NaN.
        This preserves all available data and lets pandas handle alignment.

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, return a pandas DataFrame; otherwise, return the raw dict.

        Returns
        -------
        pandas.DataFrame | dict
        """
        if not self.history:
            return pd.DataFrame() if (as_dataframe and pd is not None) else {}

        L = int(self.current_step)

        if as_dataframe and pd is not None:
            aligned = {}
            for k, v in self.history.items():
                s = pd.Series(v, dtype=float)
                s = s.iloc[:L]            # clip anything longer than executed steps
                s = s.reindex(range(L))   # pad gaps with NaN
                aligned[k] = s
            df = pd.DataFrame(aligned)
            df.index.name = "step (sim_dt)"
            return df

        # fallback: just return dict clipped to L (no NaN padding for plain lists)
        return {k: list(v[:L]) for k, v in self.history.items()}
