from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

# --- Re-usable Dataclasses for Schedules ---

@dataclass
class TimeSchedule:
    """A simple 24-hour cyclic time window.

    Attributes:
        start_hour (int): The hour the schedule begins (inclusive, 0-23).
        end_hour (int): The hour the schedule ends (inclusive, 0-23).
    """
    start_hour: int
    end_hour: int

@dataclass
class SetpointSchedule:
    """A time window with a specific power value.

    Attributes:
        start_hour (int): The hour the schedule begins (inclusive, 0-23).
        end_hour (int): The hour the schedule ends (inclusive, 0-23).
        setpoint_kw (float): The power (kW) to set.
    """
    start_hour: int
    end_hour: int
    setpoint_kw: float

# --- Base Class for all Rules ---

class BaseRule(ABC):
    """
    The abstract base class (interface) for a single, pluggable control rule.

    Each 'Rule' is a "strategy" that defines a piece of the total
    control logic for the microgrid.
    """
    @abstractmethod
    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]
               ) -> Dict[str, Any]:
        """
        Make a control decision for this rule.

        Args:
            hour (int): The current hour of the day (0-23).
            soc (float): The current state of charge (0-1) of the main battery.
            exogenous (dict): A dictionary of external data (e.g., PV, wind).

        Returns:
            Dict[str, Any]: A piece of the action dictionary, e.g., {"bat": ...}
        """
        pass

    @staticmethod
    def _in_range_cyclic(h: int, start: int, end: int) -> bool:
        """Helper to check if hour h is in [start...end] cyclically."""
        if start <= end:
            return start <= h <= end
        # Handles wrap-around (e.g., 20:00 to 06:00)
        return h >= start or h <= end

# --- Concrete "Plug-and-Play" Rule Classes ---

class BatteryRule(BaseRule):
    """
    Controls a battery based on a time-of-day charging schedule
    and the battery's current state of charge.
    """
    def __init__(self,
                 name: str = "bat",
                 charge_schedule: TimeSchedule = TimeSchedule(10, 18),
                 discharge_schedule: TimeSchedule = TimeSchedule(20, 8),
                 charge_kw: float = -6.0,
                 discharge_kw: float = 6.0,
                 soc_min_trigger: float = 0.2,
                 soc_max_trigger: float = 0.9):
        """
        Args:
            name (str): The name of the battery component to control.
            charge_schedule (TimeSchedule): Window to allow charging.
            discharge_schedule (TimeSchedule): Window to allow discharging.
            charge_kw (float): Power (kW) to use when charging.
            discharge_kw (float): Power (kW) to use when discharging.
            soc_min_trigger (float): SoC must be *above* this to discharge.
            soc_max_trigger (float): SoC must be *below* this to charge.
        """
        self.name = name
        self.charge_sch = charge_schedule
        self.discharge_sch = discharge_schedule
        self.charge_kw = charge_kw
        self.discharge_kw = discharge_kw
        self.soc_min = soc_min_trigger
        self.soc_max = soc_max_trigger

    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]) -> Dict[str, Any]:
        """Decides the battery's power setpoint based on time and SoC."""
        bat_sp = 0.0
        # Check for charging
        if (self._in_range_cyclic(hour, self.charge_sch.start_hour, self.charge_sch.end_hour)
            and soc < self.soc_max):
            bat_sp = self.charge_kw
        # Check for discharging
        elif (self._in_range_cyclic(hour, self.discharge_sch.start_hour, self.discharge_sch.end_hour)
              and soc > self.soc_min):
            bat_sp = self.discharge_kw

        return {self.name: {"set_state": "ON", "power_setpoint": bat_sp}}

class DieselRule(BaseRule):
    """Controls a diesel generator based on a time-of-day schedule."""
    def __init__(self,
                 name: str = "diesel",
                 run_schedule: TimeSchedule = TimeSchedule(20, 6),
                 setpoint_kw: float = 4.0):
        """
        Args:
            name (str): The name of the diesel component to control.
            run_schedule (TimeSchedule): Window when the diesel should be ON.
            setpoint_kw (float): Power (kW) to output when ON.
        """
        self.name = name
        self.schedule = run_schedule
        self.kw = setpoint_kw

    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]) -> Dict[str, Any]:
        """Decides the diesel's ON/OFF state and setpoint based on time."""
        # This rule ignores soc and exogenous, but must accept them
        is_on = self._in_range_cyclic(hour, self.schedule.start_hour, self.schedule.end_hour)
        return {self.name: {"on": is_on, "power_setpoint": self.kw if is_on else 0.0}}

class GridRule(BaseRule):
    """
    Manages all grid interactions, including islanding and
    scheduled setpoints (e.g., forced buy/sell).
    """
    def __init__(self,
                 name: str = "grid",
                 island_schedule: Optional[TimeSchedule] = None,
                 setpoint_schedule: Optional[SetpointSchedule] = None):
        """
        Args:
            name (str): The name of the grid component to control.
            island_schedule (TimeSchedule, optional): Window to DISCONNECT.
            setpoint_schedule (SetpointSchedule, optional): Window to force a
                specific buy (positive kW) or sell (negative kW) setpoint.
        """
        self.name = name
        self.island_sch = island_schedule
        self.setpoint_sch = setpoint_schedule

        # --- Internal Validator ---
        if self.island_sch and self.setpoint_sch:
            # A simple check for obvious schedule overlap
            istart, iend = self.island_sch.start_hour, self.island_sch.end_hour
            sstart, send = self.setpoint_sch.start_hour, self.setpoint_sch.end_hour
            if max(istart, sstart) <= min(iend, send):
                print(f"Warning: GridRule for '{name}' has overlapping island "
                      f"[{istart}-{iend}] and setpoint [{sstart}-{send}] schedules. "
                      "Islanding will take priority.")

    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]) -> Dict[str, Any]:
        """Decides the grid's state (disconnect, setpoint, or connect)"""
        # This rule ignores soc and exogenous, but must accept them

        # Priority 1: Islanding
        if (self.island_sch and
            self._in_range_cyclic(hour, self.island_sch.start_hour, self.island_sch.end_hour)):
            return {self.name: "disconnect"}

        # Priority 2: Scheduled Setpoint
        if (self.setpoint_sch and
            self._in_range_cyclic(hour, self.setpoint_sch.start_hour, self.setpoint_sch.end_hour)):
            return {self.name: self.setpoint_sch.setpoint_kw}

        # Priority 3: Default (Connected, Slack)
        return {self.name: "connect"}

class RenewableDisconnectRule(BaseRule):
    """
    A generic rule to disconnect (curtail) a component, like PV or Wind,
    based on a time-of-day schedule.
    """
    def __init__(self,
                 name: str,
                 disconnect_schedule: Optional[TimeSchedule] = None):
        """
        Args:
            name (str): The name of the component to control (e.g., "pv").
            disconnect_schedule (TimeSchedule, optional): Window to DISCONNECT.
        """
        self.name = name
        self.schedule = disconnect_schedule

    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]) -> Dict[str, Any]:
        """Decides the component's connect/disconnect state based on time."""
        # This rule ignores soc and exogenous, but must accept them

        if (self.schedule and
            self._in_range_cyclic(hour, self.schedule.start_hour, self.schedule.end_hour)):
            return {self.name: "disconnect"}

        return {self.name: "connect"}
