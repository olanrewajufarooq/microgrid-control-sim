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

import math

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
