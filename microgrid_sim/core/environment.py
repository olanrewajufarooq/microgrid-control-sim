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

from typing import Dict, List, Optional, Any

try:
    import pandas as pd
except Exception:
    pd = None

from microgrid_sim.components.base import (BaseComponent, BaseGenerator,
                                           BaseLoad, BaseStorage)
from microgrid_sim.components.generators import GridIntertie
from microgrid_sim.system import MicrogridSystem


class MicrogridEnv:
    """
    Time-based orchestrator for microgrid simulation.

    Parameters
    ----------
    simulation_steps : int
        Number of time steps per run (e.g., 24 for a day if hourly).
    """

    def __init__(self, simulation_steps: int):
        self.simulation_steps = int(simulation_steps)
        self.current_step = 0

        # Component registries
        self.generators: List[BaseGenerator] = []
        self.storage: List[BaseStorage] = []
        self.loads: List[BaseLoad] = []
        self.grid_component: Optional[GridIntertie] = None

        # System model (wired after components are added)
        self._system: Optional[MicrogridSystem] = None

        # Logged data
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
                raise TypeError("Grid component must be an instance of GridIntertie.")
            if self.grid_component is not None:
                raise ValueError("A grid component has already been added.")
            self.grid_component = component

        if isinstance(component, BaseGenerator):
            # If it's the grid, it will also pass this isinstance; safe to include.
            self.generators.append(component)
        elif isinstance(component, BaseStorage):
            self.storage.append(component)
        elif isinstance(component, BaseLoad):
            self.loads.append(component)
        else:
            raise TypeError(f"Unsupported component type for: {getattr(component, 'name', component)}")

        # Re-wire the system whenever components change
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
        comp_order.extend([g for g in self.generators])
        comp_order.extend(self.storage)
        comp_order.extend(self.loads)
        if self.grid_component is not None and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        # Per-component slots
        for comp in comp_order:
            self.history[f"{comp.name}_power"] = []
            self.history[f"{comp.name}_cost"] = []
            if isinstance(comp, BaseStorage):
                self.history[f"{comp.name}_soc"] = []

        # Aggregate slots
        self.history["gen_total_kw"] = []
        self.history["load_total_kw"] = []
        self.history["storage_total_kw"] = []
        self.history["grid_slack_kw"] = []
        self.history["net_power_unbalanced"] = []  # sum of non-grid powers
        self.history["unmet_load_kw"] = []
        self.history["curtailed_gen_kw"] = []
        self.history["downtime"] = []              # 1 if unmet > 0 else 0
        self.history["total_cashflow"] = []

        # Time index
        self.history["t"] = []

        self.current_step = 0

    def _log_step(self, summary: Dict[str, float]):
        """Append the current step’s readings to history."""
        # Per-component logs
        comp_order: List[BaseComponent] = []
        comp_order.extend([g for g in self.generators])
        comp_order.extend(self.storage)
        comp_order.extend(self.loads)
        if self.grid_component is not None and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        for comp in comp_order:
            self.history[f"{comp.name}_power"].append(comp.get_power())
            self.history[f"{comp.name}_cost"].append(comp.get_cost())
            if isinstance(comp, BaseStorage):
                # type: ignore[attr-defined]
                self.history[f"{comp.name}_soc"].append(comp.get_soc())

        # Aggregate logs
        self.history["gen_total_kw"].append(summary["gen_kw"])
        self.history["load_total_kw"].append(summary["load_kw"])
        self.history["storage_total_kw"].append(summary["storage_kw"])
        self.history["grid_slack_kw"].append(summary["grid_kw"])
        self.history["net_power_unbalanced"].append(
            summary["gen_kw"] + summary["storage_kw"] + summary["load_kw"]
        )
        self.history["unmet_load_kw"].append(summary["unmet_kw"])
        self.history["curtailed_gen_kw"].append(summary["curtailed_kw"])
        self.history["downtime"].append(1.0 if summary["unmet_kw"] > 0.0 else 0.0)
        self.history["total_cashflow"].append(summary["total_cost"])

        # Time index
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
        if self.current_step >= self.simulation_steps:
            return  # finished

        # Delegate physical balance to MicrogridSystem
        summary = self._system.step(actions=actions, exogenous=exogenous)

        # Log and advance time
        self._log_step(summary)
        self.current_step += 1

    def run(self,
            actions_list: Optional[List[Dict[str, object]]] = None,
            exogenous_list: Optional[List[Dict[str, object]]] = None):
        """
        Run the full simulation horizon.

        Parameters
        ----------
        actions_list   : list of dict, length == simulation_steps (optional)
        exogenous_list : list of dict, length == simulation_steps (optional)
        """
        self.reset()

        # Fallback to empty dicts if lists not provided
        if actions_list is None:
            actions_list = [{} for _ in range(self.simulation_steps)]
        if exogenous_list is None:
            exogenous_list = [{} for _ in range(self.simulation_steps)]

        if len(actions_list) != self.simulation_steps:
            raise ValueError("actions_list length must match simulation_steps.")
        if len(exogenous_list) != self.simulation_steps:
            raise ValueError("exogenous_list length must match simulation_steps.")

        for t in range(self.simulation_steps):
            self.step(actions=actions_list[t], exogenous=exogenous_list[t])

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
            df.index.name = "step"
            return df

        # fallback: just return dict clipped to L (no NaN padding for plain lists)
        return {k: list(v[:L]) for k, v in self.history.items()}
