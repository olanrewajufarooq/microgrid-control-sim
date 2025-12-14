"""
Contains the BaseDataBuilder class, which provides shared logic
for all data builders (synthetic, real-world, etc.).
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from microgrid_sim.types import DataBuilder
from microgrid_sim.utils.data_loader import DataSeries, ExogenousSpec, ExogenousBuilder

class BaseDataBuilder(DataBuilder, ABC):
    """
    Abstract base class for all data builders.

    It handles the final `build_list` step and provides helpers
    for interpolation and synthetic price generation.
    """
    def __init__(self, total_hours: int, sim_dt_minutes: int, seed: int = 1):
        self.total_hours = total_hours
        self.num_days = int(np.ceil(total_hours / 24.0))
        self.sim_dt = sim_dt_minutes
        self.resample_str = f"{sim_dt_minutes}T"
        self.steps_per_hour = 60 // sim_dt_minutes
        self.total_steps = total_hours * self.steps_per_hour
        # Use Generator API for reproducible randomness
        self.rng = np.random.default_rng(seed)

        self._spec: Dict[str, Dict[str, DataSeries]] = {}

        # Create time axes for interpolation
        self.hourly_x = np.arange(self.total_hours + 1)
        self.minute_x = np.linspace(0, self.total_hours, self.total_steps, endpoint=False)

    def set_seed(self, seed: int):
        """Reset the builder RNG for reproducible data generation."""
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def load_data(self) -> None:
        """
        The main method for child classes. This is responsible for
        loading/generating data and populating the `self._spec` dictionary.
        """
        pass

    def _generate_smooth_series(self, hourly_points: np.ndarray, noise_std_dev: float) -> np.ndarray:
        # (This is the same as the previous version)
        if len(hourly_points) != (self.total_hours + 1):
            raise ValueError(f"Hourly points array has length {len(hourly_points)}, expected {self.total_hours + 1}")
        interpolated = np.interp(self.minute_x, self.hourly_x, hourly_points)
        if noise_std_dev > 0:
            noise = self.rng.normal(0, noise_std_dev, self.total_steps) * np.mean(interpolated[interpolated > 0])
            series = interpolated + noise
        else:
            series = interpolated
        return np.maximum(0, series)

    def add_synthetic_grid_prices(self, name: str, **kwargs):
        # (This is the same as the previous version)
        # ... (generates synthetic prices) ...
        all_import_prices, all_export_prices = [], []

        peak_import_price=kwargs.get("peak_import_price", 0.30)
        offpeak_import_price=kwargs.get("offpeak_import_price", 0.10)
        peak_export_price=kwargs.get("peak_export_price", 0.08)
        offpeak_export_price=kwargs.get("offpeak_export_price", 0.05)
        peak_start_hour=kwargs.get("peak_start_hour", 16)
        peak_end_hour=kwargs.get("peak_end_hour", 20)

        for _ in range(self.num_days):
            peak_imp = peak_import_price * self.rng.uniform(0.95, 1.05)
            offpeak_imp = offpeak_import_price * self.rng.uniform(0.95, 1.05)
            peak_exp = peak_export_price * self.rng.uniform(0.95, 1.05)
            offpeak_exp = offpeak_export_price * self.rng.uniform(0.95, 1.05)

            day_import = np.full(24, offpeak_imp)
            day_export = np.full(24, offpeak_exp)

            day_import[peak_start_hour : peak_end_hour + 1] = peak_imp
            day_export[peak_start_hour : peak_end_hour + 1] = peak_exp

            all_import_prices.append(day_import)
            all_export_prices.append(day_export)

        hourly_import = np.concatenate(all_import_prices)[:self.total_hours]
        hourly_export = np.concatenate(all_export_prices)[:self.total_hours]

        hourly_import = np.append(hourly_import, hourly_import[0])
        hourly_export = np.append(hourly_export, hourly_export[0])

        import_series = np.interp(self.minute_x, self.hourly_x, hourly_import)
        export_series = np.interp(self.minute_x, self.hourly_x, hourly_export)

        self._spec[name] = {
            "price_import_per_kwh": DataSeries.from_array(import_series),
            "price_export_per_kwh": DataSeries.from_array(export_series),
        }

    def build_list(self) -> List[Dict[str, Dict[str, float]]]:
        """Builds and returns the final exogenous data list."""
        self.load_data()
        spec = ExogenousSpec(self._spec)
        builder = ExogenousBuilder(spec)
        return builder.build(steps=self.total_steps, start_index=0)
