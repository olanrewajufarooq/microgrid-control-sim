"""
microgrid_sim/components/loads.py

Realistic load models for residential and factory demand.

Features
--------
- Deterministic daily baseline profile (hour-of-day shape).
- Optional random noise for stochastic variation.
- Elastic / inelastic split (for DR or MPC flexibility).
- Data-driven override via exogenous["load_kw"].
- New `data_driven` flag to disable synthetic profiles.
- Sign convention: consumption < 0 kW.

References
----------
1. Bordons et al. (2020) *Model Predictive Control of Microgrids* – Ch. 2 (load modeling).
2. NREL End-Use Load Profiles (EULP) 2021 – baseline shapes.
3. DOE (2013) *Commercial and Residential Building Energy Consumption Survey*.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import BaseLoad


class _BaseLoadProfile(BaseLoad):
    """
    Shared base class for profile-based or data-driven loads.

    If `data_driven=True`, it will ignore its internal profile
    and only use `load_kw` from the exogenous data feed.
    """

    def __init__(self,
                 name: str,
                 base_kw: float,
                 profile_shape: Optional[Sequence[float]] = None,
                 noise_std: float = 0.0,
                 elastic_fraction: float = 0.0,
                 data_driven: bool = False):
        """
        Args:
            name (str): Name of the component.
            base_kw (float): Base power (kW) for the synthetic profile.
            profile_shape (list): 24-hour shape for the synthetic profile.
            noise_std (float): Std. dev. of noise for synthetic profile.
            elastic_fraction (float): Fraction of load considered 'elastic'.
            data_driven (bool): If True, disables synthetic profile and
                relies *only* on exogenous `load_kw` data.
        """
        super().__init__(name)
        self.data_driven = data_driven

        # These are used for the synthetic profile
        self.base_kw = float(base_kw)
        self.noise_std = float(noise_std)
        self.elastic_fraction = float(elastic_fraction)
        self._profile = np.array(profile_shape if profile_shape is not None else [1.0] * 24, dtype=float)

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
            demand = self._sample_demand(hour_of_day)

        self._power_demand = demand
        self._cost = 0.0  # loads have no internal cost (tariff via grid)


# --------------------------------------------------------------------
# Residential Load
# --------------------------------------------------------------------

class ResidentialLoad(_BaseLoadProfile):
    """
    Typical residential demand (24-h normalized shape).

    Default normalized shape (hour of day):
        [0.6, 0.55, 0.5, 0.5, 0.55, 0.8, 1.0, 0.9, 0.8,
         0.7, 0.65, 0.6, 0.65, 0.8, 1.0, 1.2, 1.4, 1.2,
         1.0, 0.9, 0.8, 0.7, 0.65, 0.6]

    References: NREL EULP, typical household diurnal pattern.
    """

    def __init__(self,
                 name: str,
                 base_kw: float = 1.0,
                 profile_shape: Optional[Sequence[float]] = None,
                 noise_std: float = 0.05,
                 elastic_fraction: float = 0.1,
                 data_driven: bool = False): # <-- NEW
        shape = profile_shape or [
            0.6, 0.55, 0.5, 0.5, 0.55, 0.8, 1.0, 0.9, 0.8,
            0.7, 0.65, 0.6, 0.65, 0.8, 1.0, 1.2, 1.4, 1.2,
            1.0, 0.9, 0.8, 0.7, 0.65, 0.6
        ]
        super().__init__(
            name, base_kw, shape, noise_std, elastic_fraction,
            data_driven=data_driven # <-- PASS FLAG
        )


# --------------------------------------------------------------------
# Factory Load
# --------------------------------------------------------------------

class FactoryLoad(_BaseLoadProfile):
    """
    Factory / industrial load (weekday-like shape).

    Default normalized shape (hour of day):
        [0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.1,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8,
         0.6, 0.5, 0.4, 0.3, 0.2, 0.2]

    Typical load factor ≈ 0.8, little evening spike.

    References: DOE CBECS (2013), industrial baseline.
    """

    def __init__(self,
                 name: str,
                 base_kw: float = 5.0,
                 profile_shape: Optional[Sequence[float]] = None,
                 noise_std: float = 0.02,
                 elastic_fraction: float = 0.05,
                 data_driven: bool = False): # <-- NEW
        shape = profile_shape or [
            0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.1,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8,
            0.6, 0.5, 0.4, 0.3, 0.2, 0.2
        ]
        super().__init__(
            name, base_kw, shape, noise_std, elastic_fraction,
            data_driven=data_driven # <-- PASS FLAG
        )
