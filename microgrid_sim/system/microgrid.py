"""
microgrid_sim/system/microgrid.py

Composite Microgrid System model.

Aggregates all components, enforces instantaneous bus balance,
and computes overall KPIs (power, energy, cost, unmet load, curtailment).

This sits one level below MicrogridEnv:
- MicrogridSystem handles electrical interactions & metrics.
- MicrogridEnv handles simulation timeline, control, and data logging.

References
----------
Bordons, C.; García-Torres, F.; Ridao, M. (2020).
*Model Predictive Control of Microgrids*, Springer – Ch. 1-3 (hierarchical architecture).
"""

from __future__ import annotations
from typing import Dict, List
from ..components.base import BaseComponent, BaseGenerator, BaseStorage, BaseLoad
from ..components.generators import GridIntertie

class MicrogridSystem:
    """
    Electrical microgrid bus model (steady-state per time-step).

    Attributes
    ----------
    generators : list[BaseGenerator]
    storage    : list[BaseStorage]
    loads      : list[BaseLoad]
    grid       : GridIntertie | None

    Methods
    -------
    step(actions, exogenous)
        Steps each component one tick and enforces P_bus ≈ 0
    power_summary()
        Returns dict with totals and imbalance.
    """

    def __init__(self,
                 generators: List[BaseGenerator],
                 storage: List[BaseStorage],
                 loads: List[BaseLoad],
                 grid: GridIntertie | None = None):
        self.generators = generators
        self.storage = storage
        self.loads = loads
        self.grid = grid

    # ---------------------------------------------------------
    def step(self,
             actions: Dict[str, Dict] | None = None,
             exogenous: Dict[str, Dict] | None = None
             ) -> Dict[str, float]:
        """
        Advance all components one step and compute power balance.

        Parameters
        ----------
        actions : dict[name -> action]     control inputs
        exogenous : dict[name -> exogenous] external data

        Returns
        -------
        summary : dict with bus power terms:
            gen_kw, load_kw, storage_kw, grid_kw,
            imbalance_kw, unmet_kw, curtailed_kw, total_cost.
        """
        actions = actions or {}
        exogenous = exogenous or {}

        # step each component
        totals = dict(gen=0.0, load=0.0, stor=0.0, grid=0.0, cost=0.0)
        all_comps: List[BaseComponent] = [*self.generators, *self.storage, *self.loads]
        if self.grid:
            all_comps.append(self.grid)

        for comp in all_comps:
            act = actions.get(comp.name)
            exo = exogenous.get(comp.name)
            comp.step(0, action=act, exogenous=exo)
            p = comp.get_power()
            c = comp.get_cost()

            if isinstance(comp, BaseGenerator) and not isinstance(comp, GridIntertie):
                totals["gen"] += p
            elif isinstance(comp, BaseStorage):
                totals["stor"] += p
            elif isinstance(comp, BaseLoad):
                totals["load"] += p
            elif isinstance(comp, GridIntertie):
                totals["grid"] += p
            totals["cost"] += c

        # bus balance
        imbalance = totals["gen"] + totals["stor"] + totals["grid"] + totals["load"]

        # classify residuals
        unmet = curtailed = 0.0
        if imbalance < 0:
            unmet = -imbalance
        elif imbalance > 0:
            curtailed = imbalance

        return dict(
            gen_kw=totals["gen"],
            load_kw=totals["load"],
            storage_kw=totals["stor"],
            grid_kw=totals["grid"],
            imbalance_kw=imbalance,
            unmet_kw=unmet,
            curtailed_kw=curtailed,
            total_cost=totals["cost"],
        )
    
    def power_summary(self) -> Dict[str, float]:
        """Return zeroed summary (for logging headers)."""
        return dict(
            gen_kw=0.0, load_kw=0.0, storage_kw=0.0,
            grid_kw=0.0, imbalance_kw=0.0,
            unmet_kw=0.0, curtailed_kw=0.0, total_cost=0.0
        )
