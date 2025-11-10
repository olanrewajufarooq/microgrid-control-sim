"""
Shared types and interfaces for the microgrid simulator.
"""
from typing import Protocol, Dict, Any, TypedDict, Literal

# --- 1. Controller Interface ---
# This formalizes your "plug-and-play" idea.
# Any controller must have a 'decide' method with this signature.
class EmsController(Protocol):
    """
    Interface for any controller that can be passed to the Environment.
    """
    def decide(self,
               hour: int,
               soc: float,
               exogenous: Dict[str, Any]
               ) -> Dict[str, Any]:
        ...

# --- 2. Standard Action Payloads (like TypeScript interfaces) ---
# This makes the 'actions' dictionary explicit.

class BatteryAction(TypedDict, total=False):
    """Action for a BatteryStorage component."""
    set_state: Literal["ON", "OFF"]
    power_setpoint: float

class FossilAction(TypedDict, total=False):
    """Action for a FossilGenerator component."""
    on: bool
    power_setpoint: float

# You can also define the grid action
GridAction = float or Literal["connect", "disconnect"]
