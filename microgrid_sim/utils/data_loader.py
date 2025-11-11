"""
Data-driven exogenous input pipeline for the microgrid environment.

- DataSeries: Wraps a 1D numeric series.
- ExogenousSpec: Maps component names to their DataSeries.
- ExogenousBuilder: Builds the per-step list for the environment.
- SyntheticDataBuilder: Generates realistic, interpolated, multi-day
  synthetic data with day-to-day variability.
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
