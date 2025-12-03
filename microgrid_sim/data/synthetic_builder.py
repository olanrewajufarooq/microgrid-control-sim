from typing import List, Tuple
import numpy as np

from microgrid_sim.utils.data_loader import DataSeries
from .base_builder import BaseDataBuilder

class SyntheticDataBuilder(BaseDataBuilder):
    """
    Generates fully synthetic, interpolated data with day-to-day
    variability for all components.

    Notes:
    - Wind series now include occasional gust events.
    - PV series now include cloud dips to mimic fast irradiance drops.
    - Load synthetic generation is skipped; use component-side profiles instead.
    """

    def __init__(self, total_hours: int, sim_dt_minutes: int, seed: int = 1):
        """Initializes the synthetic builder."""
        super().__init__(total_hours, sim_dt_minutes, seed)
        # These will store the component names passed by the user
        self._pv_names: List[Tuple[str, float]] = []
        self._wind_names: List[Tuple[str, float]] = []
        self._load_profiles: List[Tuple[str, float, str]] = []  # retained for backward compatibility (ignored)
        self._grid_names: List[str] = []

    def add_pv(self, name: str, peak_irr: float = 900.0, noise_std_dev: float = 0.05):
        """Registers a PV component to be generated."""
        self._pv_names.append((name, peak_irr))

    def add_wind(self, name: str, mean_speed: float = 6.0, noise_std_dev: float = 0.1):
        """Registers a Wind component to be generated."""
        self._wind_names.append((name, mean_speed))

    def add_load(self, name: str, base_kw: float, profile: str = "residential", noise_std_dev: float = 0.1):
        """
        Registers a Load component to be generated (deprecated).

        Synthetic load generation is skipped because ResidentialLoad/FactoryLoad
        already implement profiles. This call is kept for API compatibility.
        """
        # No-op on purpose
        return

    def add_grid_prices(self, name: str, **kwargs):
        """Registers a Grid component to be generated."""
        # We'll call the base class method in load_data
        self._grid_names.append((name, kwargs))

    def load_data(self) -> None:
        """
        Generates all registered synthetic data and populates self._spec.
        """
        # --- Generate PV ---
        for name, peak_irr in self._pv_names:
            all_hourly_pv = []
            t_base = np.arange(24)
            for _ in range(self.num_days):
                day_peak_irr = peak_irr * self.rng.uniform(0.6, 1.0)
                peak_shift = self.rng.normal(0.0, 0.25)
                base_pv = np.sin((t_base - 6 - peak_shift) * np.pi / 12.0)
                base_pv[t_base < 6] = 0
                base_pv[t_base > 18] = 0
                base_pv = np.maximum(0, base_pv * day_peak_irr)
                all_hourly_pv.append(base_pv)
            hourly_pv = np.concatenate(all_hourly_pv)[:self.total_hours]
            hourly_pv = np.append(hourly_pv, hourly_pv[0])
            pv_series = self._generate_smooth_series(hourly_pv, 0.05)

            # Apply cloud cover dips (stochastic irradiance reductions)
            cloud_series = pv_series.copy()
            n_cloud_events = max(1, int(0.1 * self.num_days))
            for _ in range(n_cloud_events):
                # pick a random start minute and duration (15-90 minutes)
                start_idx = int(self.rng.integers(0, len(cloud_series)))
                duration = int(self.rng.integers(int(15 * 60 / self.sim_dt), int(90 * 60 / self.sim_dt) + 1))
                end_idx = min(len(cloud_series), start_idx + duration)
                dip_factor = self.rng.uniform(0.3, 0.7)
                cloud_series[start_idx:end_idx] *= dip_factor

            pv_series = cloud_series
            self._spec[name] = {"irradiance_Wm2": DataSeries.from_array(pv_series)}

        # --- Generate Wind ---
        for name, mean_speed in self._wind_names:
            all_hourly_wind = []
            t_base = np.arange(24)
            for _ in range(self.num_days):
                day_mean = mean_speed * self.rng.uniform(0.7, 1.3)
                day_phase = self.rng.uniform(0, 2 * np.pi)
                base_wind = day_mean + 2.0 * np.sin(t_base * 2 * np.pi / 24 + day_phase)
                all_hourly_wind.append(np.maximum(0, base_wind))
            hourly_wind = np.concatenate(all_hourly_wind)[:self.total_hours]
            hourly_wind = np.append(hourly_wind, hourly_wind[0])
            wind_series = self._generate_smooth_series(hourly_wind, 0.1)

            # Inject gust events (short-duration spikes)
            n_gusts = max(1, int(0.2 * self.num_days))
            gust_series = wind_series.copy()
            for _ in range(n_gusts):
                start_idx = int(self.rng.integers(0, len(gust_series)))
                duration = int(self.rng.integers(int(5 * 60 / self.sim_dt), int(30 * 60 / self.sim_dt) + 1))
                end_idx = min(len(gust_series), start_idx + duration)
                gust_factor = self.rng.uniform(1.3, 2.0)
                gust_series[start_idx:end_idx] *= gust_factor

            wind_series = gust_series
            self._spec[name] = {"wind_speed_ms": DataSeries.from_array(wind_series)}

        # --- Generate Grid Prices ---
        for name, kwargs in self._grid_names:
            self.add_synthetic_grid_prices(name, **kwargs)
