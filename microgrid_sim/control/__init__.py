"""
microgrid_sim.control

Control algorithms for the microgrid.
This package defines the "brains" of the EMS.

- RuleBasedEMS: A "master" controller that composes a list of "Rules".
- rules (module): A collection of plug-and-play, single-purpose
  "Rule" objects (e.g., BatteryRule, GridRule) that define
  a specific control strategy.
- (Later) MDP/RL agents, etc.
- RLController: Reinforcement Learning controller using PPO
"""

from .ems_rule import RuleBasedEMS
from .rl_controller import RLController
from .rules import (BaseRule, BatteryRule, DieselRule, GridRule,
                    RenewableDisconnectRule, SetpointSchedule, TimeSchedule)

__all__ = [
    "RuleBasedEMS",
    "BaseRule",
    "TimeSchedule",
    "SetpointSchedule",
    "BatteryRule",
    "DieselRule",
    "GridRule",
    "RenewableDisconnectRule",
    "RLController",
]
