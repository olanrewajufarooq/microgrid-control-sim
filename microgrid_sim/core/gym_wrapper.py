"""
microgrid_sim/core/gym_wrapper.py

Gymnasium-compatible wrapper for MicrogridEnv to enable RL training.

This wrapper converts the MicrogridEnv into a standard Gym environment with:
- Discrete/Continuous action spaces for battery, diesel, grid, renewables
- Observation space with SOC, generation, loads, and grid prices
- Reward function based on operational costs and reliability
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .environment import MicrogridEnv
# from ..components.base import BaseGenerator, BaseStorage, BaseLoad
# from ..components.generators import GridIntertie
from ..types import DataBuilder

from ..components.generators import PVGenerator, WindTurbine, FossilGenerator, GridIntertie
from ..components.storage import BatteryStorage
from ..components.loads import ResidentialLoad, FactoryLoad

class MicrogridGymEnv(gym.Env):
    """
    Gymnasium wrapper for MicrogridEnv.

    Action Space:
    -------------
    Box(8,) - Continuous actions:
        [0] Battery action: -1 (max charge) to +1 (max discharge), 0 (off)
        [1] Battery magnitude: 0 to 1 (fraction of max charge/discharge)
        [2] Diesel on/off: 0 (off) or 1 (on)
        [3] Grid mode: -1 (island), 0 (slack/connect), +1 (scheduled trade)
        [4] Grid trade amount: -1 (max buy) to +1 (max sell)
        [5] PV connect: 0 (disconnect) or 1 (connect)
        [6] Wind connect: 0 (disconnect) or 1 (connect)
        [7] Diesel setpoint: 0 to 1 (fraction of max power)

    Observation Space:
    ------------------
    Box(8,) - Continuous observations:
        [0] Battery SOC (0-1)
        [1] Diesel power output (normalized 0-1)
        [2] Grid import price (normalized)
        [3] Grid export price (normalized)
        [4] PV power (normalized 0-1)
        [5] Wind power (normalized 0-1)
        [6] Factory load (normalized 0-1)
        [7] Residential load (normalized 0-1)
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        data_builder: DataBuilder,
        simulation_hours: int = 24,
        control_interval_minutes: int = 60,
        sim_dt_minutes: int = 1,
        battery_capacity_kwh: float = 50.0,
        battery_max_charge_kw: float = 8.0,
        battery_max_discharge_kw: float = 8.0,
        diesel_max_kw: float = 15.0,
        pv_capacity_kw: float = 5.0,
        wind_capacity_kw: float = 8.5,
        max_factory_load_kw: float = 15.0,
        max_residential_load_kw: float = 5.0,
        grid_max_import_kw: float = 50.0,
        grid_max_export_kw: float = 50.0,
        max_grid_price: float = 0.50,
        reward_weights: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Gym environment.

        Args:
            data_builder: DataBuilder instance to generate exogenous data
            simulation_hours: Total hours to simulate
            control_interval_minutes: Control decision interval
            sim_dt_minutes: Physics simulation timestep
            battery_capacity_kwh: Battery capacity
            battery_max_charge_kw: Max battery charging rate
            battery_max_discharge_kw: Max battery discharging rate
            diesel_max_kw: Max diesel generator output
            pv_capacity_kw: PV array capacity
            wind_capacity_kw: Wind turbine capacity
            max_factory_load_kw: Max factory load (for normalization)
            max_residential_load_kw: Max residential load (for normalization)
            grid_max_import_kw: Max grid import
            grid_max_export_kw: Max grid export
            max_grid_price: Max grid price (for normalization)
            reward_weights: Custom reward function weights
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        self.render_mode = render_mode

        # Store normalization constants
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_charge_kw = battery_max_charge_kw
        self.battery_max_discharge_kw = battery_max_discharge_kw
        self.diesel_max_kw = diesel_max_kw
        self.pv_capacity_kw = pv_capacity_kw
        self.wind_capacity_kw = wind_capacity_kw
        self.max_factory_load_kw = max_factory_load_kw
        self.max_residential_load_kw = max_residential_load_kw
        self.grid_max_import_kw = grid_max_import_kw
        self.grid_max_export_kw = grid_max_export_kw
        self.max_grid_price = max_grid_price

        # Simulation parameters
        self.simulation_hours = simulation_hours
        self.control_dt = control_interval_minutes
        self.sim_dt = sim_dt_minutes
        self.steps_per_control = control_interval_minutes // sim_dt_minutes
        self.max_steps = (simulation_hours * 60) // control_interval_minutes

        # Data builder
        self.data_builder = data_builder
        self.exog_list: List[Dict[str, Dict[str, float]]] = []

        # Environment (will be built in reset)
        self.env: Optional[MicrogridEnv] = None
        self.current_step = 0

        # Reward weights (cost, unmet, curtailment, soc_penalty)
        self.reward_weights = reward_weights or {
            "cost": -1.0,
            "unmet": -10.0,
            "curtailment": -0.1,
            "soc_deviation": -0.5
        }

        # Define action space: 8 continuous actions
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0, -1, -1, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Define observation space: 8 continuous observations
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def _build_microgrid(self) -> MicrogridEnv:
        """Builds the microgrid environment with all components."""
        env = MicrogridEnv(
            simulation_hours=self.simulation_hours,
            control_interval_minutes=self.control_dt,
            sim_dt_minutes=self.sim_dt
        )

        # Add components
        pv = PVGenerator("pv", capacity_kw=self.pv_capacity_kw,
                        time_step_minutes=self.sim_dt)
        wind = WindTurbine("wind", rated_kw=self.wind_capacity_kw,
                          time_step_minutes=self.sim_dt)
        diesel = FossilGenerator("diesel", p_min_kw=0.0, p_max_kw=self.diesel_max_kw,
                                time_step_minutes=self.sim_dt, fuel_cost_per_kwh=0.25)

        battery = BatteryStorage(
            "bat", capacity_kwh=self.battery_capacity_kwh,
            time_step_minutes=self.sim_dt, initial_soc=0.5,
            max_charge_kw=self.battery_max_charge_kw,
            max_discharge_kw=self.battery_max_discharge_kw
        )

        factory = FactoryLoad("factory", base_kw=self.max_factory_load_kw * 0.7)
        house = ResidentialLoad("house", base_kw=self.max_residential_load_kw * 0.6)

        grid = GridIntertie("grid", time_step_minutes=self.sim_dt,
                           price_import_per_kwh=0.20, price_export_per_kwh=0.05,
                           import_limit_kw=self.grid_max_import_kw,
                           export_limit_kw=self.grid_max_export_kw)

        env.add_component(pv)
        env.add_component(wind)
        env.add_component(diesel)
        env.add_component(battery)
        env.add_component(factory)
        env.add_component(house)
        env.add_component(grid, is_grid=True)

        return env

    def _get_observation(self, exog_step: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Extracts observation from current environment state and exogenous data.

        Returns:
            np.ndarray: Normalized observation vector [8,]
        """
        obs = np.zeros(8, dtype=np.float32)

        # [0] Battery SOC
        for storage in self.env.storage:
            if storage.name == "bat":
                obs[0] = float(storage.get_soc())
                break

        # [1] Diesel power (normalized)
        for gen in self.env.generators:
            if gen.name == "diesel":
                obs[1] = float(gen.get_power() / self.diesel_max_kw)
                break

        # [2-3] Grid prices (normalized)
        grid_exog = exog_step.get("grid", {})
        obs[2] = grid_exog.get("price_import_per_kwh", 0.20) / self.max_grid_price
        obs[3] = grid_exog.get("price_export_per_kwh", 0.05) / self.max_grid_price

        # [4] PV power (normalized)
        pv_exog = exog_step.get("pv", {})
        pv_irr = pv_exog.get("irradiance_Wm2", 0.0)
        obs[4] = min(1.0, (self.pv_capacity_kw * 0.9 * pv_irr / 1000.0) / self.pv_capacity_kw)

        # [5] Wind power (normalized)
        wind_exog = exog_step.get("wind", {})
        wind_speed = wind_exog.get("wind_speed_ms", 0.0)
        # Approximate wind power curve
        if wind_speed < 3 or wind_speed > 25:
            wind_p = 0
        elif wind_speed < 12:
            wind_p = ((wind_speed - 3) / (12 - 3)) ** 3
        else:
            wind_p = 1.0
        obs[5] = wind_p

        # [6] Factory load (normalized)
        factory_exog = exog_step.get("factory", {})
        obs[6] = factory_exog.get("load_kw", 0.0) / self.max_factory_load_kw

        # [7] Residential load (normalized)
        house_exog = exog_step.get("house", {})
        obs[7] = house_exog.get("load_kw", 0.0) / self.max_residential_load_kw

        # Clip to [0, 1]
        obs = np.clip(obs, 0.0, 1.0)

        return obs

    def _parse_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Converts normalized action vector to environment action dictionary.

        Args:
            action: np.ndarray [8,] with values in defined ranges

        Returns:
            dict: Action dictionary for MicrogridEnv
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        actions = {}

        # [0-1] Battery: direction (-1 to 1) and magnitude (0 to 1)
        bat_dir = float(action[0])  # -1 = charge, +1 = discharge
        bat_mag = float(action[1])  # 0 to 1

        if abs(bat_dir) < 0.1:  # Dead zone for "off"
            bat_setpoint = 0.0
        elif bat_dir > 0:  # Discharge
            bat_setpoint = bat_mag * self.battery_max_discharge_kw
        else:  # Charge
            bat_setpoint = -bat_mag * self.battery_max_charge_kw

        actions["bat"] = {
            "set_state": "ON" if abs(bat_setpoint) > 0.01 else "OFF",
            "power_setpoint": bat_setpoint
        }

        # [2, 7] Diesel: on/off and setpoint
        diesel_on = float(action[2]) > 0.5
        diesel_sp = float(action[7]) * self.diesel_max_kw if diesel_on else 0.0
        actions["diesel"] = {
            "on": diesel_on,
            "power_setpoint": diesel_sp
        }

        # [3-4] Grid: mode and trade amount
        grid_mode = float(action[3])
        grid_trade = float(action[4])

        if grid_mode < -0.5:  # Island mode
            actions["grid"] = "disconnect"
        elif grid_mode > 0.5:  # Scheduled trade
            if grid_trade > 0:  # Sell
                actions["grid"] = -grid_trade * self.grid_max_export_kw
            else:  # Buy
                actions["grid"] = -grid_trade * self.grid_max_import_kw
        else:  # Slack/connect mode
            actions["grid"] = "connect"

        # [5] PV connect/disconnect
        actions["pv"] = "connect" if float(action[5]) > 0.5 else "disconnect"

        # [6] Wind connect/disconnect
        actions["wind"] = "connect" if float(action[6]) > 0.5 else "disconnect"

        return actions

    def _compute_reward(self, prev_step: int) -> float:
        """
        Computes reward based on operational costs and reliability.

        Reward = w_cost * (-cost) + w_unmet * (-unmet) + w_curt * (-curtailment) + w_soc * (-soc_penalty)
        """
        df = self.env.get_results(as_dataframe=True)

        if len(df) == 0 or prev_step >= len(df):
            return 0.0

        # Get metrics for the last control interval
        start_idx = max(0, prev_step)
        end_idx = min(len(df), self.env.current_step)

        if start_idx >= end_idx:
            return 0.0

        window = df.iloc[start_idx:end_idx]

        # Cost (negative is good, we want to minimize expenses)
        step_cost = window["total_cashflow"].sum()

        # Unmet energy (bad)
        step_unmet = window["unmet_load_kw"].sum() * (self.sim_dt / 60.0)

        # Curtailed energy (slightly bad, wasted renewable)
        step_curtailed = window["curtailed_gen_kw"].sum() * (self.sim_dt / 60.0)

        # SOC deviation from 0.5 (penalize extreme SOC)
        soc_vals = []
        for storage in self.env.storage:
            if storage.name == "bat":
                soc_vals = [storage.get_soc()]
        soc_penalty = np.mean([(s - 0.5)**2 for s in soc_vals]) if soc_vals else 0.0

        # Compute weighted reward
        reward = (
            self.reward_weights["cost"] * step_cost +
            self.reward_weights["unmet"] * step_unmet +
            self.reward_weights["curtailment"] * step_curtailed +
            self.reward_weights["soc_deviation"] * soc_penalty
        )

        return float(reward)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Build fresh environment
        self.env = self._build_microgrid()
        self.env.reset()

        # Generate new exogenous data
        self.exog_list = self.data_builder.build_list()

        self.current_step = 0

        # Get initial observation
        obs = self._get_observation(self.exog_list[0])

        info = {"step": 0}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action: Action vector [8,]

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional info
        """
        # Parse action
        env_action = self._parse_action(action)

        # Store current step for reward calculation
        prev_step = self.env.current_step

        # Execute action for all sub-steps in this control interval
        for _ in range(self.steps_per_control):
            if self.env.current_step >= len(self.exog_list):
                break
            exog = self.exog_list[self.env.current_step]
            self.env.step(actions=env_action, exogenous=exog)

        self.current_step += 1

        # Compute reward
        reward = self._compute_reward(prev_step)

        # Get next observation
        next_exog_idx = min(self.env.current_step, len(self.exog_list) - 1)
        obs = self._get_observation(self.exog_list[next_exog_idx])

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Info
        info = {
            "step": self.current_step,
            "total_cost": self.env.get_results(as_dataframe=True)["total_cashflow"].sum() if self.env.history else 0.0
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")

    def close(self):
        """Clean up resources."""
