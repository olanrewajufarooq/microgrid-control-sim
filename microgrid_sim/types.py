"""
Shared types and interfaces for the microgrid simulator.
"""
from typing import List, Protocol, Dict, Any, TypedDict, Literal

# Controller Interface
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

# Standard Action Payloads

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

class DataBuilder(Protocol):
    """
    Interface (Protocol) for any data source.

    This is the "wrapper" that allows different data sources
    (synthetic, Liège, etc.) to be used by the simulation.
    """
    def build_list(self) -> List[Dict[str, Dict[str, float]]]:
        """
        Builds and returns the complete, high-fidelity (per-minute)
        exogenous data list for the entire simulation period.
        """
        ...
