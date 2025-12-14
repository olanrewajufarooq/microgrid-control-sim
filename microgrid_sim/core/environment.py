"""
microgrid_sim/core/environment.py

MicrogridEnv v2 - timeline & logging orchestrator that delegates
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
Bordons, C.; Garcia-Torres, F.; Ridao, M. (2020).
Model Predictive Control of Microgrids. Springer.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Protocol
import random

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from microgrid_sim.components.base import (BaseComponent, BaseGenerator,
                                           BaseLoad, BaseStorage)
from microgrid_sim.components.generators import GridIntertie
from microgrid_sim.system import MicrogridSystem

# Define the controller protocol (interface)
class EmsController(Protocol):
    """
    Interface (Protocol) for any controller that can be passed to the Environment.

    This ensures that any "plug-and-play" controller, whether it's
    rule-based or an RL agent, has the required .decide() method.
    """
    def decide(self,
               hour: int,
               soc: float,
               exogenous: Dict[str, Any]
               ) -> Dict[str, Any]:
        ...

class MicrogridEnv:
    """
    Time-based orchestrator for microgrid simulation.

    This environment runs on two timescales:
    1.  Simulation Timestep (e.g., 1 minute): The physics step,
        defined by `sim_dt_minutes`.
    2.  Control Interval (e.g., 60 minutes): The decision-making step,
        defined by `control_interval_minutes`.

    The `run()` method automatically handles this two-timescale loop.
    The `step()` method provides a low-level, single-step API.
    """

    def __init__(self,
                 simulation_hours: int,
                 control_interval_minutes: int,
                 sim_dt_minutes: int):
        """
        Args:
            simulation_hours (int): The total number of hours to simulate (e.g., 24, 72).
            control_interval_minutes (int): How often the controller
                makes a decision (e.g., 60).
            sim_dt_minutes (int): The resolution of the physics
                simulation (e.g., 1).
        """
        self.simulation_hours = int(simulation_hours)
        self.control_dt = int(control_interval_minutes)
        self.sim_dt = int(sim_dt_minutes)

        if self.control_dt % self.sim_dt != 0:
            raise ValueError("Control interval must be a multiple of simulation timestep.")

        self.steps_per_control_interval = self.control_dt // self.sim_dt

        # Total simulation steps = (total hours * 60 min/hr) / (sim_dt_min/step)
        self.total_simulation_steps = (self.simulation_hours * 60) // self.sim_dt

        self.current_step = 0

        # Component registries
        self.generators: List[BaseGenerator] = []
        self.storage: List[BaseStorage] = []
        self.loads: List[BaseLoad] = []
        self.grid_component: Optional[GridIntertie] = None
        self._system: Optional[MicrogridSystem] = None
        self.history: Dict[str, List[float]] = {}

    # --- Component management ---

    def add_component(self, component: BaseComponent, is_grid: bool = False):
        """
        Registers a new component (like a PV array, Battery, or Load)
        with the environment.

        Args:
            component (BaseComponent): The component instance to add.
            is_grid (bool): True if this component is the main GridIntertie
                (slack bus). Only one is allowed.
        """
        if is_grid:
            if not isinstance(component, GridIntertie):
                raise TypeError("Grid component must be a GridIntertie.")
            if self.grid_component is not None:
                raise ValueError("A grid component has already been added.")
            self.grid_component = component

        if isinstance(component, BaseGenerator):
            self.generators.append(component)
        elif isinstance(component, BaseStorage):
            self.storage.append(component)
        elif isinstance(component, BaseLoad):
            self.loads.append(component)

        self._wire_system()

    def _wire_system(self):
        """
        (Re)builds the internal MicrogridSystem from the current
        component registries. This is called automatically by add_component.
        """
        # Ensure the grid is not duplicated in the generators list
        gens = [g for g in self.generators if not isinstance(g, GridIntertie)]
        self._system = MicrogridSystem(
            generators=gens,
            storage=list(self.storage),
            loads=list(self.loads),
            grid=self.grid_component,
        )

    # --- Logging helpers ---

    def _initialize_history(self):
        """Sets up the history dictionary to log data for every step."""
        self.history = {}
        comp_order: List[BaseComponent] = [*self.generators, *self.storage, *self.loads]
        if self.grid_component and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        for comp in comp_order:
            self.history[f"{comp.name}_power"] = []
            self.history[f"{comp.name}_cost"] = []
            if isinstance(comp, BaseStorage):
                self.history[f"{comp.name}_soc"] = []
            # log downtime if supported
            try:
                comp.get_downtime()
                self.history[f"{comp.name}_downtime"] = []
            except Exception:
                pass

        # Aggregates
        for key in ["gen_total_kw", "load_total_kw", "storage_total_kw", "grid_slack_kw",
                    "net_power_unbalanced", "unmet_load_kw", "curtailed_gen_kw",
                    "downtime", "total_cashflow", "t"]:
            self.history[key] = []
        self.current_step = 0

    def _log_step(self, summary: Dict[str, float]):
        """Appends the results of a single step to the history."""
        comp_order: List[BaseComponent] = [*self.generators, *self.storage, *self.loads]
        if self.grid_component and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        for comp in comp_order:
            self.history[f"{comp.name}_power"].append(comp.get_power())
            self.history[f"{comp.name}_cost"].append(comp.get_cost())
            if isinstance(comp, BaseStorage):
                self.history[f"{comp.name}_soc"].append(comp.get_soc())
            # downtime if available
            try:
                self.history[f"{comp.name}_downtime"].append(comp.get_downtime())
            except Exception:
                pass

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
        self.history["t"].append(self.current_step)

    # --- Simulation Loop ---

    def _apply_seed(self, seed: int):
        """Seed python/numpy and per-component RNGs for reproducibility."""
        random.seed(seed)
        try:
            np.random.seed(seed)
        except Exception:
            pass

        comp_order: List[BaseComponent] = [*self.generators, *self.storage, *self.loads]
        if self.grid_component and self.grid_component not in comp_order:
            comp_order.append(self.grid_component)

        for idx, comp in enumerate(comp_order):
            try:
                if hasattr(comp, "set_seed"):
                    comp.set_seed(seed + idx)
            except Exception:
                continue

    def reset(self, seed: Optional[int] = None):
        """Resets all components and clears the simulation history."""
        if seed is not None:
            self._apply_seed(seed)

        self.current_step = 0
        for comp in self.generators + self.storage + self.loads:
            comp.reset()
        if self.grid_component is not None:
            self.grid_component.reset()
        self._wire_system()
        self._initialize_history()

    def step(self, actions: Dict[str, object], exogenous: Dict[str, object]):
        """
        Runs one single simulation step (e.g., 1 minute).
        This is the low-level, Gym-style API.

        Args:
            actions (dict): The action dictionary from the controller.
            exogenous (dict): The external data for this step.
        """
        if self._system is None:
            raise RuntimeError("System not wired. Call .add_component() first.")

        # Handle special string-based actions like "disconnect"
        if self.grid_component:
            grid_action = actions.get(self.grid_component.name)
            if grid_action == "disconnect":
                self.grid_component.disconnect()
            elif grid_action == "connect":
                self.grid_component.connect()

        # Step the physical system
        summary = self._system.step(t=self.current_step, actions=actions, exogenous=exogenous)

        # Log results
        self._log_step(summary)
        self.current_step += 1

    def run(self, controller: EmsController, exogenous_list: List[Dict[str, Any]], seed: Optional[int] = None):
        """
        Runs the full simulation using a high-level controller.

        This method automatically manages the two-timescale loop, calling
        the controller's `.decide()` method once per control interval
        and the environment's `.step()` method once per simulation step.

        Args:
            controller (EmsController): An object with a
                `.decide(hour, soc, exog)` method.
            exogenous_list (list): The *high-fidelity, per-simulation-step* list of exogenous data (e.g., length 1440).
        Args:
            controller (EmsController): An object with a
                `.decide(hour, soc, exog)` method.
            exogenous_list (list): The *high-fidelity, per-simulation-step* list of exogenous data (e.g., length 1440).
            seed (int, optional): If provided, seeds all RNGs before running for reproducibility.
        """
        required = self.total_simulation_steps

        if len(exogenous_list) < required:
            raise ValueError(
                f"Exogenous list too short ({len(exogenous_list)}) for "
                f"{required} simulation steps at sim_dt={self.sim_dt} min."
            )

        # If it's longer, slice instead of crashing (warn once)
        if len(exogenous_list) > required:
            if getattr(self, "_warned_exog_slice", False) is False:
                print(
                    f"[MicrogridEnv] Warning: exogenous_list longer than required "
                    f"({len(exogenous_list)} > {required}); slicing to {required}."
                )
                self._warned_exog_slice = True
            exogenous_list = exogenous_list[:required]

        self.reset(seed=seed)

        current_ems_action = {}

        for k in range(self.total_simulation_steps):

            # Check if we are at the start of a new control interval
            if k % self.steps_per_control_interval == 0:

                # Get total hours *of the simulation*
                total_hour = k // self.steps_per_control_interval
                # Get the hour *of the day* (0-23)
                hour_of_day = total_hour % 24

                # Get battery SOC (assumes one battery named "bat")
                soc = 0.5 # default
                for s in self.storage:
                    if getattr(s, "name", "") == "bat":
                        soc = float(s.get_soc())

                # Controller makes a decision
                current_ems_action = controller.decide(
                    hour=hour_of_day, # <-- Pass the correct hour (0-23)
                    soc=soc,
                    exogenous=exogenous_list[k]
                )

            # Run one simulation step (1 minute)
            self.step(
                actions=current_ems_action,
                exogenous=exogenous_list[k]
            )

    # --- Results ---

    def get_results(self, as_dataframe: bool = True):
        """
        Returns the simulation history.

        Args:
            as_dataframe (bool): If True, returns a pandas DataFrame.
                If False, returns the raw dictionary of lists.

        Returns:
            pd.DataFrame | dict: The simulation log.
        """
        if not self.history:
            return pd.DataFrame() if (as_dataframe and pd is not None) else {}

        L = int(self.current_step)

        if as_dataframe and pd is not None:
            # Align all history lists to the same length (L)
            aligned = {}
            for k, v in self.history.items():
                s = pd.Series(v, dtype=float)
                s = s.iloc[:L]
                s = s.reindex(range(L))
                aligned[k] = s
            df = pd.DataFrame(aligned)
            df.index.name = "step (sim_dt)" # Index is now per-minute
            return df

        # Fallback: return dict clipped to L
        return {k: list(v[:L]) for k, v in self.history.items()}
