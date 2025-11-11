"""
Data builder for the Mesa Del Sol Microgrid dataset.
Loads, cleans, and resamples real-world data.
"""
import pandas as pd
from .base_builder import BaseDataBuilder
from microgrid_sim.utils.data_loader import DataSeries
import os

class MesaDataBuilder(BaseDataBuilder):
    """
    Data builder for the "Mesa Del Sol Microgrid" dataset.
    https://www.kaggle.com/datasets/yekenot/power-data-from-mesa-del-sol-microgrid

    This builder:
    1. Loads all .csv files from the specified folder.
    2. Concatenates and sorts them into one continuous timeline.
    3. Cleans the data using user-defined min/max clamps.
    4. Resamples 10-second data to sim_dt averages.
    5. Randomly selects a valid 12:00 AM start date.

    Provides (as Replay Data):
    - 'pv': PVPCS_Active_Power (kW)
    - 'site_load': GE_Active_Power (kW)
    - 'grid': Synthesized grid prices
    """

    def __init__(self,
                 csv_folder_path: str,
                 total_hours: int,
                 sim_dt_minutes: int,
                 seed: int = 1,
                 min_pv_power: float = 0.0,
                 max_pv_power: float = 50.0,  # Default from EDA
                 min_load_power: float = 0.0,
                 max_load_power: float = 250.0 # Default from EDA
                 ):
        """
        Initializes the Mesa data builder.

        Args:
            csv_folder_path (str): Path to the folder containing Mesa CSVs.
            total_hours (int): The total number of hours to simulate.
            sim_dt_minutes (int): The simulation's time step (e.g., 1).
            seed (int): Random seed for selecting the start date.
            min_pv_power (float): Min value to clamp PV power.
            max_pv_power (float): Max value to clamp PV power (removes outliers).
            min_load_power (float): Min value to clamp Load power (removes negatives).
            max_load_power (float): Max value to clamp Load power (removes outliers).
        """
        super().__init__(total_hours, sim_dt_minutes, seed)
        self.csv_folder = csv_folder_path
        self.min_pv = min_pv_power
        self.max_pv = max_pv_power
        self.min_load = min_load_power
        self.max_load = max_load_power

    def load_data(self) -> None:
        """
        Loads, cleans, concatenates, and randomly slices the Mesa dataset.
        This method is called by the parent `build_list()`.
        """
        print("Loading and concatenating all Mesa Del Sol CSVs...")

        all_files = [os.path.join(self.csv_folder, f) for f in os.listdir(self.csv_folder) if f.endswith('.csv')]
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_folder}")

        df_list = []
        for f in all_files:
            df = pd.read_csv(f, usecols=['Timestamp', 'PVPCS_Active_Power', 'GE_Active_Power'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M:%S')
            df_list.append(df)

        df_raw = pd.concat(df_list).sort_values(by='Timestamp').set_index('Timestamp')

        # Data Cleaning Step
        print(f"Cleaning raw data. Clamping PV to [{self.min_pv}, {self.max_pv}] kW.")
        print(f"Clamping Load to [{self.min_load}, {self.max_load}] kW.")

        df_raw['PVPCS_Active_Power'] = df_raw['PVPCS_Active_Power'].clip(
            lower=self.min_pv, upper=self.max_pv
        )
        df_raw['GE_Active_Power'] = df_raw['GE_Active_Power'].clip(
            lower=self.min_load, upper=self.max_load
        )

        #  Resample
        print("Resampling data...")
        df_resampled = df_raw.resample(self.resample_str).mean().ffill().bfill()

        # Find a Random but Valid Start Date
        possible_start_times = df_resampled.at_time('00:00').index
        duration_needed = pd.Timedelta(hours=self.total_hours)
        last_valid_start = df_resampled.index.max() - duration_needed
        valid_start_dates = possible_start_times[possible_start_times <= last_valid_start]

        if valid_start_dates.empty:
            raise ValueError(f"No valid 12:00 AM start date found with {self.total_hours} "
                             f"hours of data. Max available: {len(df_resampled) * self.sim_dt / 60} hours.")

        start_date = self.rng.choice(valid_start_dates)
        start_index = df_resampled.index.get_loc(start_date)
        df_final = df_resampled.iloc[start_index : start_index + self.total_steps]
        print(f"Randomly selected start date: {start_date}")

        # Populate self._spec

        # Use 'power_kw' key for ReplayGenerator
        pv_power = df_final['PVPCS_Active_Power'].values
        self._spec['pv'] = {"power_kw": DataSeries.from_array(pv_power)}

        load_series = df_final['GE_Active_Power'].values
        self._spec['site_load'] = {"load_kw": DataSeries.from_array(load_series)}

        print("Note: Grid price data not included. Using synthetic prices.")
        self.add_synthetic_grid_prices('grid')
