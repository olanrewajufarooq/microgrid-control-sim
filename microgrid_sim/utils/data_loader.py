"""
Data-driven exogenous input pipeline for the microgrid environment.

- DataSeries: Wraps a 1D numeric series.
- ExogenousSpec: Maps component names to their DataSeries.
- ExogenousBuilder: Builds the per-step list for the environment.
- SyntheticDataBuilder: Generates realistic, interpolated, multi-day
  synthetic data for testing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Mapping, Any

import math
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class DataSeries:
    """A simple 1D numeric container with optional metadata."""
    values: List[float]
    name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_array(arr: Sequence[float], name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> "DataSeries":
        vals = [float(x) for x in arr]
        return DataSeries(values=vals, name=name, meta=meta)

    @staticmethod
    def from_csv(path: str, time_col: Optional[str], value_col: str,
                 start: Optional[int] = None, stop: Optional[int] = None,
                 name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> "DataSeries":
        if pd is None:
            raise ImportError("pandas is required to load CSV files.")
        df = pd.read_csv(path)
        if value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not found in {path}. Found: {list(df.columns)}")
        series = df[value_col].astype(float).tolist()
        if start is not None or stop is not None:
            series = series[slice(start, stop)]
        nm = name if name is not None else value_col
        meta2 = dict(meta or {})
        if time_col is not None and time_col in df.columns:
            meta2["time_col"] = time_col
        meta2["source_csv"] = path
        return DataSeries(series, name=nm, meta=meta2)

    def window(self, start_index: int, steps: int) -> List[float]:
        """Return a contiguous sub-list of length `steps`."""
        end = start_index + steps
        if end > len(self.values):
            raise ValueError(f"Requested window [{start_index}:{end}) exceeds DataSeries length {len(self.values)}")
        return self.values[start_index:end]


class ExogenousSpec:
    """A mapping from component name -> { exogenous_key: DataSeries }"""
    def __init__(self, mapping: Mapping[str, Mapping[str, DataSeries]]):
        self._map: Dict[str, Dict[str, DataSeries]] = {}
        for comp, inner in mapping.items():
            self._map[str(comp)] = {}
            for key, ds in inner.items():
                if not isinstance(ds, DataSeries):
                    raise TypeError(f"ExogenousSpec expects DataSeries for '{comp}:{key}', got {type(ds)}")
                self._map[str(comp)][str(key)] = ds

    def components(self) -> List[str]:
        return list(self._map.keys())

    def keys_for(self, component: str) -> List[str]:
        return list(self._map[component].keys())

    def series_for(self, component: str, key: str) -> DataSeries:
        return self._map[component][key]


class ExogenousBuilder:
    """
    Builds the per-step `exogenous_list` for MicrogridEnv.run(...).
    """
    def __init__(self, spec: ExogenousSpec):
        self.spec = spec

    def build(self, steps: int, start_index: int = 0) -> List[Dict[str, Dict[str, float]]]:
        comp_names = self.spec.components()
        # Pre-slice everything to fail fast on length
        windows: Dict[str, Dict[str, List[float]]] = {}
        for comp in comp_names:
            windows[comp] = {}
            for key in self.spec.keys_for(comp):
                ds = self.spec.series_for(comp, key)
                windows[comp][key] = ds.window(start_index, steps)

        # Stitch per-step dicts
        exog_list: List[Dict[str, Dict[str, float]]] = []
        for i in range(steps):
            step_payload: Dict[str, Dict[str, float]] = {}
            for comp in comp_names:
                inner: Dict[str, float] = {}
                for key in self.spec.keys_for(comp):
                    inner[key] = float(windows[comp][key][i])
                step_payload[comp] = inner
            exog_list.append(step_payload)
        return exog_list


class SyntheticDataBuilder:
    """
    Generates realistic, interpolated synthetic data for the simulation.

    This replaces the messy data generation block in the notebook and
    solves the "stair-case" data problem by using linear interpolation.

    **This version correctly tiles data for multi-day simulations.**
    """
    def __init__(self, total_hours: int, sim_dt_minutes: int, seed: int = 1):
        self.total_hours = total_hours
        self.sim_dt = sim_dt_minutes
        self.steps_per_hour = 60 // sim_dt_minutes
        self.total_steps = total_hours * self.steps_per_hour
        self.rng = np.random.RandomState(seed)
        self._spec: Dict[str, Dict[str, DataSeries]] = {}

        # Create the x-axis for interpolation
        # hourly_x goes from 0 to total_hours (e.g., 0...72 for 3 days)
        self.hourly_x = np.arange(self.total_hours + 1)
        # minute_x is the high-res axis
        self.minute_x = np.linspace(0, self.total_hours, self.total_steps, endpoint=False)

    def _generate_smooth_series(self, hourly_points: np.ndarray, noise_std_dev: float) -> np.ndarray:
        """Interpolates hourly points to minutes and adds per-minute noise."""
        if len(hourly_points) != (self.total_hours + 1):
            raise ValueError(f"Hourly points array has length {len(hourly_points)}, expected {self.total_hours + 1}")

        # Interpolate hourly data to per-minute
        interpolated = np.interp(self.minute_x, self.hourly_x, hourly_points)
        # Add per-minute noise
        if noise_std_dev > 0:
            noise = self.rng.normal(0, noise_std_dev, self.total_steps) * np.mean(interpolated)
            series = interpolated + noise
        else:
            series = interpolated
        return np.maximum(0, series) # Ensure no negative values

    def add_grid_prices(self,
                        name: str,
                        peak_import_price: float = 0.30,
                        offpeak_import_price: float = 0.10,
                        peak_export_price: float = 0.08,
                        offpeak_export_price: float = 0.05,
                        peak_start_hour: int = 16,
                        peak_end_hour: int = 20):
        """
        Generates dynamic, interpolated Time-of-Use (TOU) prices for the grid.
        """
        # --- Create a 24-hour base profile ---
        base_import_prices = np.full(24, offpeak_import_price)
        base_export_prices = np.full(24, offpeak_export_price)

        # Apply peak prices to the 24-hour base
        for h in range(peak_start_hour, peak_end_hour + 1):
             base_import_prices[h] = peak_import_price
             base_export_prices[h] = peak_export_price

        # --- Tile the base profile to match total_hours ---
        hourly_import = np.tile(base_import_prices, self.total_hours // 24 + 2)[:self.total_hours]
        hourly_export = np.tile(base_export_prices, self.total_hours // 24 + 2)[:self.total_hours]

        # Append the first point to the end to make a (total_hours + 1) array
        hourly_import_prices = np.append(hourly_import, hourly_import[0])
        hourly_export_prices = np.append(hourly_export, hourly_export[0])

        # Interpolate (no noise for prices)
        import_price_series = np.interp(self.minute_x, self.hourly_x, hourly_import_prices)
        export_price_series = np.interp(self.minute_x, self.hourly_x, hourly_export_prices)

        # Add to spec
        self._spec[name] = {
            "price_import_per_kwh": DataSeries.from_array(import_price_series),
            "price_export_per_kwh": DataSeries.from_array(export_price_series),
        }

    def add_pv(self, name: str, peak_irr: float = 900.0, noise_std_dev: float = 0.05):
        """Generates a smooth PV irradiance profile (W/m^2)."""
        # --- Create a 24-hour base profile ---
        t_base = np.arange(24)
        base_pv = np.sin((t_base - 6) * np.pi / 12.0)
        base_pv[t_base < 6] = 0
        base_pv[t_base > 18] = 0
        base_pv = base_pv * peak_irr

        # --- Tile the base profile to match total_hours ---
        hourly_pv = np.tile(base_pv, self.total_hours // 24 + 2)[:self.total_hours]
        # Append the first point (which is 0) to the end
        hourly_pv = np.append(hourly_pv, hourly_pv[0])

        pv_series = self._generate_smooth_series(hourly_pv, noise_std_dev)
        self._spec[name] = {"irradiance_Wm2": DataSeries.from_array(pv_series)}

    def add_wind(self, name: str, mean_speed: float = 6.0, noise_std_dev: float = 0.1):
        """Generates a smooth wind speed profile (m/s)."""
        # --- Create a 24-hour base profile ---
        t_base = np.arange(24)
        base_wind = mean_speed + 2.0 * np.sin(t_base * 2 * np.pi / 24 + 0.5)

        # --- Tile the base profile to match total_hours ---
        hourly_wind = np.tile(base_wind, self.total_hours // 24 + 2)[:self.total_hours]
        # Append the first point to the end
        hourly_wind = np.append(hourly_wind, hourly_wind[0])

        wind_series = self._generate_smooth_series(hourly_wind, noise_std_dev)
        self._spec[name] = {"wind_speed_ms": DataSeries.from_array(wind_series)}

    def add_load(self, name: str, base_kw: float, profile: str = "residential", noise_std_dev: float = 0.1):
        """Generates a smooth load profile (kW)."""
        # --- Create a 24-hour base profile ---
        if profile == "residential":
            shape = [0.6, 0.55, 0.5, 0.5, 0.55, 0.8, 1.0, 0.9, 0.8, 0.7,
                     0.65, 0.6, 0.65, 0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.9,
                     0.8, 0.7, 0.65, 0.6]
        else: # factory
            shape = [0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.1, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.5,
                     0.4, 0.3, 0.2, 0.2]

        base_load = np.array(shape) * base_kw

        # --- Tile the base profile to match total_hours ---
        hourly_load = np.tile(base_load, self.total_hours // 24 + 2)[:self.total_hours]
        # Append the first point to the end
        hourly_load = np.append(hourly_load, hourly_load[0])

        load_series = self._generate_smooth_series(hourly_load, noise_std_dev)
        self._spec[name] = {"load_kw": DataSeries.from_array(load_series)}

    def build_list(self) -> List[Dict[str, Dict[str, float]]]:
        """Builds the final per-minute exogenous list."""
        spec = ExogenousSpec(self._spec)
        builder = ExogenousBuilder(spec)
        return builder.build(steps=self.total_steps, start_index=0)
