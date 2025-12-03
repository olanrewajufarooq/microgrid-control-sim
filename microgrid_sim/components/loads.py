"""
microgrid_sim/components/loads.py

Realistic load models for residential and factory demand with calendar-aware
profiles and optional event/season multipliers.

Features
--------
- Deterministic daily baseline profile (hour-of-day shape).
- Optional random noise for stochastic variation.
- Elastic / inelastic split (for DR or MPC flexibility).
- Data-driven override via exogenous["load_kw"].
- Calendar-aware profiles: per-weekday and per-month shapes.
- Seasonal and special-day multipliers (e.g., Ramadan/Eid).
- Peak hour multipliers and day-of-month multipliers.
- Sign convention: consumption < 0 kW.
"""

from __future__ import annotations

import datetime
import random
from typing import Optional, Sequence, Dict

import numpy as np

from .base import BaseLoad


class _BaseLoadProfile(BaseLoad):
    """
    Shared base class for profile-based or data-driven loads.

    If `data_driven=True`, it will ignore its internal profile
    and only use `load_kw` from the exogenous data feed.

    Synthetic mode supports calendar-aware behavior:
    - per-weekday and per-month shapes
    - month multipliers, day-of-month multipliers
    - special-day multipliers (e.g., holidays) via "MM-DD" keys
    - peak hour multipliers
    """

    def __init__(
        self,
        name: str,
        base_kw: float,
        profile_shape: Optional[Sequence[float]] = None,
        noise_std: float = 0.0,
        elastic_fraction: float = 0.0,
        data_driven: bool = False,
        calendar_weekday_shapes: Optional[Dict[int, Sequence[float]]] = None,
        calendar_month_shapes: Optional[Dict[int, Sequence[float]]] = None,
        day_of_month_multipliers: Optional[Dict[int, float]] = None,
        peak_hour_multipliers: Optional[Dict[int, float]] = None,
                 month_multipliers: Optional[Dict[int, float]] = None,
                 special_day_multipliers: Optional[Dict[str, float]] = None,
                 start_date: Optional[datetime.date] = None):
        """
        Args:
            name (str): Name of the component.
            base_kw (float): Base power (kW) for the synthetic profile.
            profile_shape (list): 24-hour shape for the synthetic profile.
            noise_std (float): Std. dev. of noise for synthetic profile.
            elastic_fraction (float): Fraction of load considered 'elastic'.
            data_driven (bool): If True, disables synthetic profile and
                relies *only* on exogenous `load_kw` data.
            calendar_weekday_shapes (dict[int -> 24-shape]): Optional override per weekday (0=Mon...6=Sun).
            calendar_month_shapes (dict[int -> 24-shape]): Optional override per month (1-12).
            day_of_month_multipliers (dict[int -> float]): Optional scaler for specific days (e.g., holidays).
            peak_hour_multipliers (dict[int -> float]): Optional scaler for specific hours (0-23) to model peaks.
            month_multipliers (dict[int -> float]): Optional scaler per month (1-12) for seasonal effects.
            special_day_multipliers (dict["MM-DD" -> float]): Optional scaler for specific calendar dates (e.g., holidays).
            start_date (datetime.date): Optional simulation start date; if None, randomly picked in the year.
        """
        super().__init__(name)
        self.data_driven = data_driven
        self._rng = random.Random()
        chosen_date = start_date or (datetime.date(2024, 1, 1) + datetime.timedelta(days=self._rng.randint(0, 364)))
        self._start_datetime = datetime.datetime.combine(chosen_date, datetime.time())

        # These are used for the synthetic profile
        self.base_kw = float(base_kw)
        self.noise_std = float(noise_std)
        self.elastic_fraction = float(elastic_fraction)
        self._profile = np.array(profile_shape if profile_shape is not None else [1.0] * 24, dtype=float)
        self._calendar_weekday_shapes = {
            k: np.array(v, dtype=float) for k, v in (calendar_weekday_shapes or {}).items()
        }
        self._calendar_month_shapes = {
            k: np.array(v, dtype=float) for k, v in (calendar_month_shapes or {}).items()
        }
        self._day_of_month_multipliers = {int(k): float(v) for k, v in (day_of_month_multipliers or {}).items()}
        self._peak_hour_multipliers = {int(k): float(v) for k, v in (peak_hour_multipliers or {}).items()}
        self._month_multipliers = {int(k): float(v) for k, v in (month_multipliers or {}).items()}
        # Special dates as "MM-DD" strings (zero-padded)
        self._special_day_multipliers = {}
        for k, v in (special_day_multipliers or {}).items():
            try:
                key = str(k)
                if "-" in key:
                    mm, dd = key.split("-", 1)
                    key = f"{int(mm):02d}-{int(dd):02d}"
                self._special_day_multipliers[key] = float(v)
            except Exception:
                continue

        self._t = 0 # Internal step counter (for synthetic profile)

    def _hour_index(self, t: int) -> int:
        """Gets the 0-23 hour index for the synthetic profile."""
        # Assumes t is the step index for the *control interval*
        return t % len(self._profile)

    def _sample_demand(self, t: int) -> float:
        """Generates a single demand value from the internal synthetic profile."""
        idx = self._hour_index(t)
        p = self.base_kw * self._profile[idx]
        if self.noise_std > 0:
            p += np.random.normal(0, self.noise_std * self.base_kw)
        return max(p, 0.0)

    def _calendar_shape_value(self, hour: int, dow: Optional[int], month: Optional[int]) -> float:
        """Pick a calendar-adjusted shape value for the given hour."""
        shape = self._profile
        if month is not None and month in self._calendar_month_shapes:
            shape = self._calendar_month_shapes[month]
        if dow is not None and dow in self._calendar_weekday_shapes:
            shape = self._calendar_weekday_shapes[dow]
        if shape.size == 0:
            return 1.0
        return float(shape[hour % len(shape)])

    def step(self, t: int, **kwargs):
        """
        Advances the load by one step.

        Priority:
        1. Uses `exogenous["load_kw"]` if provided.
        2. If `data_driven=True` and no data is provided, demand is 0.
        3. If `data_driven=False`, falls back to internal synthetic profile.
        """
        exo = kwargs.get("exogenous", {}) or {}
        load_kw = exo.get("load_kw")
        peak_multiplier_exo = exo.get("peak_multiplier")

        current_dt = self._start_datetime + datetime.timedelta(minutes=int(t))
        day_of_week = current_dt.weekday()  # 0=Mon
        month = current_dt.month
        day_of_month = current_dt.day

        if load_kw is not None:
            # Priority 1: Use exogenous data if provided
            demand = float(load_kw)
        elif self.data_driven:
            # Priority 2: Data-driven, but no data provided. Demand is 0.
            demand = 0.0
            # You could add a warning here, but it might be noisy:
            # print(f"Warning: Load '{self.name}' is data-driven but received no 'load_kw' at step {t}.")
        else:
            # Priority 3: Fallback to internal synthetic profile
            # We use 't' which, from the environment, is the per-minute step.
            # We need the *hour* for the profile.
            # Assuming 1-min sim_dt and 60-min control_dt
            hour_of_day = (t // 60) % 24
            shape_val = self._calendar_shape_value(hour_of_day, day_of_week, month)
            demand = self.base_kw * shape_val
            if self.noise_std > 0:
                demand += np.random.normal(0, self.noise_std * self.base_kw)
            if self._peak_hour_multipliers:
                demand *= self._peak_hour_multipliers.get(hour_of_day, 1.0)
            if peak_multiplier_exo is not None:
                try:
                    demand *= float(peak_multiplier_exo)
                except Exception:
                    pass
            if month is not None and month in self._month_multipliers:
                demand *= self._month_multipliers[month]
            if day_of_month is not None and day_of_month in self._day_of_month_multipliers:
                demand *= self._day_of_month_multipliers[day_of_month]
            if month is not None and day_of_month is not None:
                key = f"{int(month):02d}-{int(day_of_month):02d}"
                if key in self._special_day_multipliers:
                    demand *= self._special_day_multipliers[key]
            demand = max(demand, 0.0)

        self._power_demand = demand
        self._cost = 0.0  # loads have no internal cost (tariff via grid)

# --------------------------------------------------------------------
# Residential Load
# --------------------------------------------------------------------

class ResidentialLoad(_BaseLoadProfile):
    """
    Typical residential demand (24-h normalized shape).

    Defaults
    --------
    - Weekday and weekend hourly shapes (built in).
    - Month multipliers with summer/winter bumps.
    - Special-day multipliers for common holidays (approx Ramadan/Eid dates).
    """

    def __init__(self,
                 name: str,
                 base_kw: float = 1.0,
                 profile_shape: Optional[Sequence[float]] = None,
                 noise_std: float = 0.05,
                 elastic_fraction: float = 0.1,
                 data_driven: bool = False,
                 **kwargs):
        # Default profile (weekday) + weekend override
        weekday_shape = [
            0.45, 0.40, 0.38, 0.38, 0.42, 0.60, 0.85, 0.90, 0.80,
            0.70, 0.65, 0.60, 0.65, 0.80, 1.00, 1.25, 1.50, 1.30,
            1.05, 0.90, 0.80, 0.70, 0.60, 0.50
        ]
        weekend_shape = [
            0.55, 0.50, 0.48, 0.48, 0.55, 0.80, 1.05, 1.10, 1.00,
            0.90, 0.85, 0.80, 0.85, 1.00, 1.20, 1.45, 1.60, 1.40,
            1.10, 0.95, 0.85, 0.75, 0.65, 0.55
        ]
        shape = profile_shape or weekday_shape
        # Inject default calendar shapes if user did not provide
        if "calendar_weekday_shapes" not in kwargs:
            kwargs["calendar_weekday_shapes"] = {
                0: weekday_shape, 1: weekday_shape, 2: weekday_shape,
                3: weekday_shape, 4: weekday_shape,
                5: weekend_shape, 6: weekend_shape
            }
        # Default month multipliers (seasonal) and special days (approximate for fixed calendar)
        if "month_multipliers" not in kwargs:
            kwargs["month_multipliers"] = {
                1: 1.05, 2: 1.00, 3: 1.02, 4: 1.05, 5: 1.08, 6: 1.10,
                7: 1.15, 8: 1.12, 9: 1.05, 10: 1.03, 11: 1.04, 12: 1.10
            }
        if "special_day_multipliers" not in kwargs:
            # Example fixed-date multipliers (user can override with real lunar dates)
            kwargs["special_day_multipliers"] = {
                "04-10": 1.20,  # Eid al-Fitr (approx, configurable)
                "06-18": 1.15,  # Eid al-Adha (approx)
                "03-23": 1.10,  # Ramadan start (approx)
                "04-15": 1.10,  # Ramadan mid (approx)
            }
        super().__init__(
            name, base_kw, shape, noise_std, elastic_fraction,
            data_driven=data_driven,
            **kwargs
        )


# --------------------------------------------------------------------
# Factory Load
# --------------------------------------------------------------------

class FactoryLoad(_BaseLoadProfile):
    """
    Factory / industrial load (weekday-like shape).

    Defaults
    --------
    - Weekday production shape and reduced weekend shape.
    - Month multipliers with lighter summer loads.
    - Special-day multipliers for downtime around common holidays.
    """

    def __init__(self,
        name: str,
        base_kw: float = 5.0,
        profile_shape: Optional[Sequence[float]] = None,
        noise_std: float = 0.02,
        elastic_fraction: float = 0.05,
        data_driven: bool = False,
        **kwargs):
        # Default weekday (production) and weekend (reduced staffing)
        weekday_shape = [
            0.15, 0.15, 0.15, 0.25, 0.45, 0.75, 0.95, 1.05, 1.05,
            1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.85,
            0.65, 0.55, 0.45, 0.35, 0.25, 0.20
        ]
        weekend_shape = [
            0.10, 0.10, 0.10, 0.12, 0.15, 0.25, 0.35, 0.40, 0.45,
            0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.40, 0.35,
            0.30, 0.25, 0.20, 0.15, 0.12, 0.10
        ]
        shape = profile_shape or weekday_shape
        if "calendar_weekday_shapes" not in kwargs:
            kwargs["calendar_weekday_shapes"] = {
                0: weekday_shape, 1: weekday_shape, 2: weekday_shape,
                3: weekday_shape, 4: weekday_shape,
                5: weekend_shape, 6: weekend_shape
            }
        if "month_multipliers" not in kwargs:
            kwargs["month_multipliers"] = {
                1: 0.95, 2: 0.95, 3: 0.98, 4: 1.00, 5: 1.02, 6: 0.98,
                7: 0.90, 8: 0.92, 9: 0.96, 10: 0.98, 11: 0.97, 12: 0.94
            }
        if "special_day_multipliers" not in kwargs:
            kwargs["special_day_multipliers"] = {
                "04-10": 0.70,  # Eid al-Fitr downtime
                "06-18": 0.75,  # Eid al-Adha downtime
                "03-23": 0.85,  # Ramadan start
                "04-15": 0.85,  # Ramadan mid
            }
        super().__init__(
            name, base_kw, shape, noise_std, elastic_fraction,
            data_driven=data_driven,
            **kwargs
        )
