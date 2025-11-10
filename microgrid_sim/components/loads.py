"""
microgrid_sim/components/loads.py

Realistic load models for residential and factory demand.

Features
--------
- Deterministic daily baseline profile (hour-of-day shape).
- Optional random noise for stochastic variation.
- Elastic / inelastic split (for DR or MPC flexibility).
- Data-driven override via exogenous["load_kw"].
- Sign convention: consumption < 0 kW.

References
----------
1. Bordons et al. (2020) *Model Predictive Control of Microgrids* – Ch. 2 (load modeling).
2. NREL End-Use Load Profiles (EULP) 2021 – baseline shapes.
3. DOE (2013) *Commercial and Residential Building Energy Consumption Survey*.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence
from .base import BaseLoad


class _BaseLoadProfile(BaseLoad):
    """Shared deterministic + stochastic load behavior."""

    def __init__(self,
                 name: str,
                 base_kw: float,
                 profile_shape: Optional[Sequence[float]] = None,
                 noise_std: float = 0.0,
                 elastic_fraction: float = 0.0):
        super().__init__(name)
        self.base_kw = float(base_kw)
        self.noise_std = float(noise_std)
        self.elastic_fraction = float(elastic_fraction)
        self._profile = np.array(profile_shape if profile_shape is not None else [1.0] * 24, dtype=float)
        self._t = 0

    def _hour_index(self, t: int) -> int:
        return t % len(self._profile)

    def _sample_demand(self, t: int) -> float:
        idx = self._hour_index(t)
        p = self.base_kw * self._profile[idx]
        if self.noise_std > 0:
            p += np.random.normal(0, self.noise_std * self.base_kw)
        return max(p, 0.0)

    def step(self, t: int, **kwargs):
        exo = kwargs.get("exogenous", {}) or {}
        load_kw = exo.get("load_kw")
        if load_kw is not None:
            demand = float(load_kw)
        else:
            demand = self._sample_demand(t)
        # split elastic vs inelastic parts for controllers (not used yet)
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
                 elastic_fraction: float = 0.1):
        shape = profile_shape or [
            0.6, 0.55, 0.5, 0.5, 0.55, 0.8, 1.0, 0.9, 0.8,
            0.7, 0.65, 0.6, 0.65, 0.8, 1.0, 1.2, 1.4, 1.2,
            1.0, 0.9, 0.8, 0.7, 0.65, 0.6
        ]
        super().__init__(name, base_kw, shape, noise_std, elastic_fraction)


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
                 elastic_fraction: float = 0.05):
        shape = profile_shape or [
            0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.1,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8,
            0.6, 0.5, 0.4, 0.3, 0.2, 0.2
        ]
        super().__init__(name, base_kw, shape, noise_std, elastic_fraction)
