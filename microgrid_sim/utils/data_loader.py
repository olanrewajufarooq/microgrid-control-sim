"""
microgrid_sim/utils/data_loader.py

Data-driven exogenous input pipeline for the microgrid environment.

What this gives you
-------------------
1) DataSeries: wrap a 1D numeric series (from CSV or array)
2) ExogenousSpec: map component -> { exogenous_key: DataSeries }
3) ExogenousBuilder: produce the per-step `exogenous_list` the env expects

Supported exogenous keys by our components
------------------------------------------
- PVGenerator        : "irradiance_Wm2"   (float)
- WindTurbine        : "wind_speed_ms"    (float)
- HydroGenerator     : "flow_cms"         (float)
- Residential/Factory: "load_kw"          (float)

Typical usage
-------------
from microgrid_sim.utils.data_loader import DataSeries, ExogenousSpec, ExogenousBuilder

ghi = DataSeries.from_csv("data/solar_irradiance.csv", time_col="timestamp", value_col="ghi_Wm2")
wind = DataSeries.from_csv("data/wind_speed.csv", time_col="timestamp", value_col="wind_ms")
house= DataSeries.from_csv("data/residential_load.csv", time_col="timestamp", value_col="kW")
factory=DataSeries.from_csv("data/factory_load.csv", time_col="timestamp", value_col="kW")

spec = ExogenousSpec({
    "pv"     : {"irradiance_Wm2": ghi},
    "wind"   : {"wind_speed_ms" : wind},
    "house"  : {"load_kw"       : house},
    "factory": {"load_kw"       : factory},
})

builder = ExogenousBuilder(spec)
exog_list = builder.build(steps=24, start_index=0)   # returns list[dict] length==steps

Citations / True online data
----------------------------
- Solar irradiance: NREL NSRDB API (GHI/DNI/DHI). See docs and API.
  Add your API key & pull CSV/JSON as needed.
- Wind speeds/weather: ERA5 / C3S (global hourly).
- Open PV/Wind synthetic time series: Renewables.ninja.
- Hydrology/flow: USGS Water Data APIs (US only).
- Building loads: NREL End-Use Load Profiles (US), or other open load sets.

(Links in the docstrings below.)

Notes
-----
- This loader is intentionally **opinionated** but minimal. It does not resample/time-align
  by timestamps automatically (everyone’s calendars differ). Instead, you preload your CSV
  into equal-length arrays for the horizon you plan to simulate, or slice by indices.
- If you want full timestamp alignment/resampling, we can add a `TimeIndexer` utility in the next topic.

Author: You
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Mapping, Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass
class DataSeries:
    """
    A simple 1D numeric container with optional metadata.

    Create from CSV:
        ds = DataSeries.from_csv("file.csv", time_col="timestamp", value_col="ghi_Wm2")

    Or from array/list:
        ds = DataSeries.from_array([..], name="ghi_Wm2")

    Length checks:
        - You choose how many steps to consume downstream via ExogenousBuilder.build(...).
        - If the requested window overruns the data length, we raise ValueError (to fail fast).

    References (data sources)
    -------------------------
    Solar irradiance:
      - NREL NSRDB: satellite-derived GHI/DNI/DHI and met vars. API & docs.
        https://developer.nrel.gov/docs/solar/nsrdb/  (API)  ;  https://nrel.github.io/nsrdb/ (code)
    Wind/weather:
      - ERA5 (C3S/ECMWF): hourly reanalysis since 1940.
        https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels  ; https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
      - Renewables.ninja: synthesized PV/Wind time series, research-backed.  https://www.renewables.ninja/
    Hydrology:
      - USGS Water Data APIs (streamflow, etc., US).  https://api.waterdata.usgs.gov/
    Building loads:
      - NREL End-Use Load Profiles for U.S. buildings.  https://www.nrel.gov/buildings/end-use-load-profiles
    """
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
        """
        Minimal CSV loader: reads a single value column. `time_col` is optional and not used here
        beyond documentation; alignment is left to you.

        Args:
            path: CSV path
            time_col: name of timestamp column (optional; informational)
            value_col: the numeric column to extract
            start, stop: optional slice indices to subset the series
            name: override series name
            meta: optional metadata dict
        """
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
        """Return a contiguous sub-list of length `steps` starting at `start_index`."""
        end = start_index + steps
        if end > len(self.values):
            raise ValueError(f"Requested window [{start_index}:{end}) exceeds DataSeries length {len(self.values)}")
        return self.values[start_index:end]


class ExogenousSpec:
    """
    A mapping from component name -> { exogenous_key: DataSeries }

    Example:
        spec = ExogenousSpec({
            "pv": {"irradiance_Wm2": ghi},
            "wind": {"wind_speed_ms": wind},
            "house": {"load_kw": house_load},
        })
    """
    def __init__(self, mapping: Mapping[str, Mapping[str, DataSeries]]):
        # deep-copy-ish but keep references
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

    build(steps, start_index) returns:
        [
          {
            "pv": {"irradiance_Wm2":  ..., },
            "wind": {"wind_speed_ms": ..., },
            "house": {"load_kw": ...},
          },
          ...
        ]  # length == steps

    If any component/key is missing data for the requested window, we raise ValueError.
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
    """
    def __init__(self, total_hours: int, sim_dt_minutes: int, seed: int = 1):
        self.total_hours = total_hours
        self.sim_dt = sim_dt_minutes
        self.steps_per_hour = 60 // sim_dt_minutes
        self.total_steps = total_hours * self.steps_per_hour
        self.rng = np.random.RandomState(seed)
        self._spec = {} # component_name -> {key: DataSeries}

        # Create the x-axis for interpolation
        self.hourly_x = np.arange(total_hours + 1) # 0..24
        self.minute_x = np.linspace(0, total_hours, self.total_steps, endpoint=False)

    def _generate_smooth_series(self, hourly_points: np.ndarray, noise_std_dev: float) -> np.ndarray:
        """Interpolates hourly points to minutes and adds per-minute noise."""
        # Interpolate hourly data to per-minute
        interpolated = np.interp(self.minute_x, self.hourly_x, hourly_points)
        # Add per-minute noise
        noise = self.rng.normal(0, noise_std_dev, self.total_steps) * interpolated
        series = interpolated + noise
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

        Note: This does NOT generate limits. The grid component will
        use its initialized defaults (None = infinity) for limits.
        """

        # Create hourly profiles (total_hours + 1 points for interpolation)
        hourly_import_prices = np.full(self.total_hours + 1, offpeak_import_price)
        hourly_export_prices = np.full(self.total_hours + 1, offpeak_export_price)

        # Apply peak prices.
        # We tile the peak hours for multi-day simulations.
        for day in range(self.total_hours // 24 + 1):
            start = day * 24 + peak_start_hour
            end = day * 24 + peak_end_hour + 1 # +1 for slicing
            if start > len(hourly_import_prices):
                break
            hourly_import_prices[start:end] = peak_import_price
            hourly_export_prices[start:end] = peak_export_price

        # Ensure wrap-around for smooth interpolation
        hourly_import_prices[0] = hourly_import_prices[-1]
        hourly_export_prices[0] = hourly_export_prices[-1]

        # Interpolate (no noise for prices, they are fixed schedules)
        import_price_series = np.interp(self.minute_x, self.hourly_x, hourly_import_prices)
        export_price_series = np.interp(self.minute_x, self.hourly_x, hourly_export_prices)

        # Add to spec
        self._spec[name] = {
            "price_import_per_kwh": DataSeries.from_array(import_price_series),
            "price_export_per_kwh": DataSeries.from_array(export_price_series),
        }

    def add_pv(self, name: str, peak_irr: float = 900.0, noise_std_dev: float = 0.05):
        """Generates a smooth PV irradiance profile (W/m^2)."""
        t = self.hourly_x
        # Clear-sky bell curve (peaking at noon)
        hourly_pv = np.sin((t - 6) * np.pi / 12.0)
        hourly_pv[t < 6] = 0
        hourly_pv[t > 18] = 0
        hourly_pv = hourly_pv * peak_irr
        # Create a 24-hour cycle and repeat, ensuring 0 at hour 0 and 24
        hourly_pv[0] = 0
        hourly_pv[-1] = 0 # This ensures smooth interpolation

        # Cycle for multiple days if needed
        hourly_pv = np.tile(hourly_pv[:-1], self.total_hours // 24 + 1)[:self.total_hours + 1]

        pv_series = self._generate_smooth_series(hourly_pv, noise_std_dev)
        self._spec[name] = {"irradiance_Wm2": DataSeries.from_array(pv_series)}

    def add_wind(self, name: str, mean_speed: float = 6.0, noise_std_dev: float = 0.1):
        """Generates a smooth wind speed profile (m/s)."""
        t = self.hourly_x
        hourly_wind = mean_speed + 2.0 * np.sin(t * 2 * np.pi / 24 + 0.5)

        # Repeat for multiple days
        hourly_wind = np.tile(hourly_wind[:-1], self.total_hours // 24 + 1)[:self.total_hours + 1]

        wind_series = self._generate_smooth_series(hourly_wind, noise_std_dev)
        self._spec[name] = {"wind_speed_ms": DataSeries.from_array(wind_series)}

    def add_load(self, name: str, base_kw: float, profile: str = "residential", noise_std_dev: float = 0.1):
        """Generates a smooth load profile (kW)."""
        if profile == "residential":
            shape = [0.6, 0.55, 0.5, 0.5, 0.55, 0.8, 1.0, 0.9, 0.8, 0.7,
                     0.65, 0.6, 0.65, 0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.9,
                     0.8, 0.7, 0.65, 0.6]
        else: # factory
            shape = [0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.1, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.5,
                     0.4, 0.3, 0.2, 0.2]

        hourly_load = np.array(shape) * base_kw

        # Repeat for multiple days and add the 25th hour (end of day)
        hourly_load = np.tile(hourly_load, self.total_hours // 24 + 1)[:self.total_hours + 1]
        hourly_load[-1] = hourly_load[0] # Wrap around

        load_series = self._generate_smooth_series(hourly_load, noise_std_dev)
        self._spec[name] = {"load_kw": DataSeries.from_array(load_series)}

    def build_list(self) -> List[Dict[str, Dict[str, float]]]:
        """Builds the final per-minute exogenous list."""
        spec = ExogenousSpec(self._spec)
        builder = ExogenousBuilder(spec)
        return builder.build(steps=self.total_steps, start_index=0)
