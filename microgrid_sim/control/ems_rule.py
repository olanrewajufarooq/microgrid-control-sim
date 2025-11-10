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
from typing import Dict, Any, List, Tuple
from .rules import BaseRule


class RuleBasedEMS:
    """
    A "master" EMS controller that is composed of a list of
    individual "Rule" strategies.

    This class's `decide` method implements the conflict resolution
    logic to merge all rule decisions into a final action dictionary.
    """
    def __init__(self, rules: List[BaseRule]):
        """
        Args:
            rules (List[BaseRule]): An array of "Rule" objects that will
                be executed to build the final action.
        """
        self.rules = rules

    def decide(self,
               hour: int,
               soc: float,
               exogenous: Dict[str, Dict[str, float]] | None = None
               ) -> Dict[str, Any]:
        """
        Generates a final action dictionary by merging all individual rules.

        This method correctly resolves conflicts, especially for the grid:
        Priority: "disconnect" > float (setpoint) > "connect"

        For all other components, the last rule in the list wins.
        """
        final_actions: Dict[str, Any] = {}

        # Helper to store the winning grid action
        # We use a tuple: (priority, value)
        # 3 = disconnect, 2 = setpoint, 1 = connect, 0 = None
        grid_action: Tuple[int, Any] = (0, None)

        for rule in self.rules:
            # Get the decision from the individual rule
            action_piece = rule.decide(hour, soc, exogenous)

            for key, value in action_piece.items():
                if key == "grid":
                    # Priority 1 (High): "disconnect"
                    if value == "disconnect":
                        if grid_action[0] < 3:
                            grid_action = (3, "disconnect")
                    # Priority 2: float (Setpoint)
                    elif isinstance(value, float):
                        if grid_action[0] < 2:
                            grid_action = (2, value)
                    # Priority 3 (Low): "connect"
                    elif value == "connect":
                        if grid_action[0] < 1:
                            grid_action = (1, "connect")
                else:
                    # For all other components, the last rule wins
                    final_actions[key] = value
        # After the loop, add the winning grid action if one was set
        if grid_action[1] is not None:
            final_actions["grid"] = grid_action[1]

        return final_actions
