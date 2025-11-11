import pandas as pd
from .base_builder import BaseDataBuilder
from microgrid_sim.utils.data_loader import DataSeries
import os

class LiegeDataBuilder(BaseDataBuilder):
    """
    Data builder for the "Liège Microgrid Open Data" dataset.
    This builder:
    1. Loads `miris_load.csv` and `miris_pv.csv`.
    2. Cleans the data using user-defined min/max clamps.
    3. Resamples 5-sec data to appropriate intervals.
    4. Aligns and merges all data.
    5. Randomly selects a valid 12:00 AM start date.

    Provides (as Replay Data):
    - 'pv': PV power (kW)
    - 'site_load': The total site consumption (kW)
    - 'grid': Synthesized grid prices.

    **Note: This builder does NOT provide wind data.**
    """

    def __init__(self,
                 folder_path: str,
                 total_hours: int,
                 sim_dt_minutes: int,
                 seed: int = 1,
                 min_pv_power: float = 0.0,
                 max_pv_power: float = 0.8,  # Default from EDA
                 min_load_power: float = 0.0,
                 max_load_power: float = 0.3  # Default from EDA
                 ):
        """
        Initializes the Liège data builder.

        Args:
            folder_path (str): Path to the 'liege' folder.
            total_hours (int): The total number of hours to simulate.
            sim_dt_minutes (int): The simulation's time step (e.g., 1).
            seed (int): Random seed for selecting the start date.
            min_pv_power (float): Min value to clamp PV power.
            max_pv_power (float): Max value to clamp PV power.
            min_load_power (float): Min value to clamp Load power.
            max_load_power (float): Max value to clamp Load power.
        """
        super().__init__(total_hours, sim_dt_minutes, seed)
        self.load_path = os.path.join(folder_path, "miris_load.csv")
        self.pv_path = os.path.join(folder_path, "miris_pv.csv")
        self.min_pv = min_pv_power
        self.max_pv = max_pv_power
        self.min_load = min_load_power
        self.max_load = max_load_power

    def load_data(self) -> None:
        """
        Loads, cleans, resamples, and randomly slices the Liège dataset.
        This method is called by the parent `build_list()`.
        """
        print("Loading and resampling Liège dataset...")

        # Load & Clean Consumption (5-sec)
        df_load = pd.read_csv(self.load_path, usecols=['DateTime', 'Conso'])
        df_load['DateTime'] = pd.to_datetime(df_load['DateTime'])
        df_load = df_load.set_index('DateTime')

        print(f"Cleaning raw data. Clamping PV to [{self.min_pv}, {self.max_pv}] kW.")
        print(f"Clamping Load to [{self.min_load}, {self.max_load}] kW.")
        df_load['Conso'] = df_load['Conso'].clip(lower=self.min_load, upper=self.max_load)

        df_load_resampled = df_load.resample(self.resample_str).mean()
        df_load_resampled.rename(columns={'Conso': 'site_load'}, inplace=True)

        # Load & Clean PV
        df_pv = pd.read_csv(self.pv_path, usecols=['DateTime', 'PV'])
        df_pv['DateTime'] = pd.to_datetime(df_pv['DateTime'])
        df_pv = df_pv.set_index('DateTime')

        df_pv['PV'] = df_pv['PV'].clip(lower=self.min_pv, upper=self.max_pv)

        df_pv_resampled = df_pv.resample(self.resample_str).mean()
        df_pv_resampled.rename(columns={'PV': 'pv_power'}, inplace=True)

        # Align and Find Random Start
        # Convert all to UTC to be safe, then remove timezone info
        df_load_resampled.index = df_load_resampled.index.tz_convert('UTC').tz_localize(None)
        df_pv_resampled.index = df_pv_resampled.index.tz_convert('UTC').tz_localize(None)

        df_all = pd.concat([df_load_resampled, df_pv_resampled], axis=1)
        df_all = df_all.dropna() # Find the intersection of valid data

        possible_start_times = df_all.at_time('00:00').index
        duration_needed = pd.Timedelta(hours=self.total_hours)
        last_valid_start = df_all.index.max() - duration_needed
        valid_start_dates = possible_start_times[possible_start_times <= last_valid_start]

        if valid_start_dates.empty:
            raise ValueError(f"No valid 12:00 AM start date found with {self.total_hours} "
                             f"hours of data. Max available: {len(df_all) * self.sim_dt / 60} hours.")

        start_date = self.rng.choice(valid_start_dates)
        start_index = df_all.index.get_loc(start_date)
        df_final = df_all.iloc[start_index : start_index + self.total_steps]
        print(f"Randomly selected start date: {start_date}")

        # Populate self._spec

        # Use 'power_kw' key for ReplayGenerator
        pv_power = df_final['pv_power'].values
        self._spec['pv'] = {"power_kw": DataSeries.from_array(pv_power)}

        load_series = df_final['site_load'].values
        self._spec['site_load'] = {"load_kw": DataSeries.from_array(load_series)}

        print("Note: Grid price data not included. Using synthetic prices.")
        self.add_synthetic_grid_prices('grid')
