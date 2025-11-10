"""
microgrid_sim.control

Control algorithms for the microgrid.
This package defines the "brains" of the EMS.

- RuleBasedEMS: A "master" controller that composes a list of "Rules".
- rules (module): A collection of plug-and-play, single-purpose
  "Rule" objects (e.g., BatteryRule, GridRule) that define
  a specific control strategy.
- (Later) MDP/RL agents, etc.
"""

from .ems_rule import RuleBasedEMS
from .rules import (
    BaseRule,
    TimeSchedule,
    SetpointSchedule,
    BatteryRule,
    DieselRule,
    GridRule,
    RenewableDisconnectRule
)

__all__ = [
    "RuleBasedEMS",
    "BaseRule",
    "TimeSchedule",
    "SetpointSchedule",
    "BatteryRule",
    "DieselRule",
    "GridRule",
    "RenewableDisconnectRule",
]
