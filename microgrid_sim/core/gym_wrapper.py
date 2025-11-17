"""
microgrid_sim/core/gym_wrapper.py

Gymnasium-compatible wrapper supporting multiple instances of components.
FIXED: Corrects observation space, action parsing, and adapter integration.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, unflatten, flatten_space

from ..types import DataBuilder, EmsController
from ..components.generators import PVGenerator, WindTurbine, FossilGenerator, GridIntertie
from ..components.storage import BatteryStorage
from ..components.loads import ResidentialLoad, FactoryLoad, BaseLoad
from ..core.environment import MicrogridEnv


# --- Internal Adapter Class for MicrogridEnv.run() ---
class RLAdapter(EmsController):
    """
    Acts as an EmsController for MicrogridEnv.run().
    Translates the RL Controller's output (Dict or Box) into the
    MicrogridEnv action dictionary.
    """
    def __init__(self, rl_controller: Any, gym_env: 'MicrogridGymEnv', is_flattened: bool):
        self.rl_controller = rl_controller
        self.original_env = gym_env
        self.is_flattened = is_flattened

    def decide(self, hour: int, soc: float, exogenous: Dict[str, Any]) -> Dict[str, Any]:
        """Implements the EmsController protocol by translating the agent's action."""

        # 1. Construct the observation dict
        observation = self.original_env._get_observation(exog_step=exogenous)

        # 2. Get the action from the RL Controller
        action_output = self.rl_controller.decide(observation, deterministic=True)

        # 3. Unflatten if necessary
        if self.is_flattened:
            action_dict = unflatten(self.original_env.action_space, action_output)
        else:
            action_dict = action_output

        # 4. Parse the action dict into the MicrogridEnv action dictionary
        microgrid_action_dict = self.original_env._parse_action(action_dict)

        return microgrid_action_dict


# --- Internal Specialized Wrapper for SB3 PPO ---
class FlattenedMicrogridGymEnv(gym.Wrapper):
    """
    Presents a flat Box to PPO. Internally:
      - Discrete(k) keys consume k slots (logits) -> argmax -> integer in the original start..start+k-1
      - Box(1,) keys consume 1 slot in [-1, 1] and are linearly mapped if needed downstream
    """
    def __init__(self, env: 'MicrogridGymEnv'):
        super().__init__(env)
        self._schema = []  # list of (key, kind, size, start)
        total = 0
        for k, sp in env.action_space.spaces.items():
            if isinstance(sp, spaces.Discrete):
                self._schema.append((k, "disc", sp.n, getattr(sp, "start", 0)))
                total += sp.n
            elif isinstance(sp, spaces.Box):
                assert sp.shape == (1,), f"Only scalar Box actions supported for key {k}"
                self._schema.append((k, "box", 1, None))
                total += 1
            else:
                raise TypeError(f"Unsupported subspace for {k}: {type(sp)}")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(total,), dtype=np.float32)
        # observation_space stays the same as env (Dict)

    def _unflatten_action(self, flat: np.ndarray):
        flat = np.asarray(flat, dtype=np.float32).ravel()
        out = {}
        i = 0
        for k, kind, size, start in self._schema:
            if kind == "disc":
                logits = flat[i:i+size]; i += size
                cls = int(np.argmax(logits))
                out[k] = (start or 0) + cls
            else:
                val = float(flat[i]); i += 1
                out[k] = np.array([val], dtype=np.float32)
        return out

    def step(self, action: np.ndarray):
        dict_action = self._unflatten_action(action)
        return self.env.step(dict_action)


class MicrogridGymEnv(gym.Env):
    """
    Gymnasium wrapper dynamically sized based on the components in the provided MicrogridEnv.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        microgrid_env: MicrogridEnv,
        data_builder: DataBuilder,
        max_grid_price: float = 0.50,
        reward_weights: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None
    ):
        """Initializes the Gym environment by linking to an existing MicrogridEnv."""
        super().__init__()

        # --- 1. Environment and Parameters Discovery ---
        self.env = microgrid_env
        self.data_builder = data_builder
        self.render_mode = render_mode
        self.max_grid_price = max_grid_price

        # Collect ALL components by type
        self.batteries: List[BatteryStorage] = [c for c in self.env.storage if isinstance(c, BatteryStorage)]
        self.diesels: List[FossilGenerator] = [c for c in self.env.generators if isinstance(c, FossilGenerator)]
        self.pv_gens: List[PVGenerator] = [c for c in self.env.generators if isinstance(c, PVGenerator)]
        self.wind_gens: List[WindTurbine] = [c for c in self.env.generators if isinstance(c, WindTurbine)]
        self.all_loads: List[BaseLoad] = self.env.loads
        self.grid: Optional[GridIntertie] = self.env.grid_component

        # --- Mandatory Component Guardrails ---
        if not self.batteries or not self.diesels or not self.grid:
            raise ValueError("MicrogridGymEnv requires at least one BatteryStorage, one FossilGenerator, and the GridIntertie.")

        self.grid_name = self.grid.name

        # --- Dynamic Scaling Factors Dictionary ---
        self.MAX_CAPACITIES: Dict[str, float] = {}

        for bat in self.batteries:
            self.MAX_CAPACITIES[f"bat_chg_{bat.name}"] = bat.params.max_charge_kw
            self.MAX_CAPACITIES[f"bat_dis_{bat.name}"] = bat.params.max_discharge_kw
        for d in self.diesels:
            self.MAX_CAPACITIES[f"diesel_pmax_{d.name}"] = d.params.p_max_kw
        for pv in self.pv_gens:
            self.MAX_CAPACITIES[f"pv_cap_{pv.name}"] = pv.params.capacity_kw
        for wind in self.wind_gens:
            self.MAX_CAPACITIES[f"wind_rated_{wind.name}"] = wind.params.rated_kw
        for load in self.all_loads:
            self.MAX_CAPACITIES[f"load_max_{load.name}"] = getattr(load, 'base_kw', 1.0) * 1.5

        # Simulation parameters
        self.control_dt = self.env.control_dt
        self.sim_dt = self.env.sim_dt
        self.steps_per_control = self.control_dt // self.sim_dt
        self.max_steps = 0

        # Reward weights
        self.reward_weights = reward_weights or {
            "cost": -1.0, "unmet": -10.0, "curtailment": -0.1, "soc_deviation": -0.5
        }

        # --- 2. Dynamic Action and Observation Space Definition ---
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # Store exogenous list
        self.exog_list: List[Dict[str, Dict[str, float]]] = []
        self.current_step = 0

    # --- Factory Methods for RL Integration ---
    def create_flattened_env(self) -> FlattenedMicrogridGymEnv:
        """Creates and returns a wrapper of this environment with a single Box action space."""
        return FlattenedMicrogridGymEnv(self)

    def get_rl_adapter(self, rl_controller: Any, is_flattened: bool = False) -> RLAdapter:
        """Returns an instance of the RLAdapter, ready to be passed to MicrogridEnv.run()."""
        return RLAdapter(rl_controller, self, is_flattened)

    # --- Action and Observation Builder Methods ---
    def _build_action_space(self) -> spaces.Dict:
        """Dynamically constructs the action space (Dict of Discrete/Box)."""
        act_spaces = {}

        # 1. Grid
        act_spaces[f"{self.grid.name}_mode"] = spaces.Discrete(3, start=-1)
        act_spaces[f"{self.grid.name}_trade"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2. Batteries
        for bat in self.batteries:
            act_spaces[f"{bat.name}_mode"] = spaces.Discrete(3, start=-1)  # -1=charge, 0=off, 1=discharge
            act_spaces[f"{bat.name}_mag"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 3. Diesels
        for d in self.diesels:
            act_spaces[f"{d.name}_on"] = spaces.Discrete(2)
            act_spaces[f"{d.name}_sp"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 4. Renewables
        for pv in self.pv_gens:
            act_spaces[f"{pv.name}_con"] = spaces.Discrete(2)
        for wind in self.wind_gens:
            act_spaces[f"{wind.name}_con"] = spaces.Discrete(2)

        return spaces.Dict(act_spaces)

    def _build_observation_space(self) -> spaces.Dict:
        """Dynamically constructs the observation space."""
        obs_spaces = {}

        # 1. Grid Prices
        obs_spaces[f"{self.grid.name}_p_imp"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_spaces[f"{self.grid.name}_p_exp"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 2. Batteries (SOC)
        for bat in self.batteries:
            obs_spaces[f"{bat.name}_soc"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 3. Diesels (Power)
        for d in self.diesels:
            obs_spaces[f"{d.name}_p"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 4. Renewables (Power from exogenous)
        for pv in self.pv_gens:
            obs_spaces[f"{pv.name}_p"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        for wind in self.wind_gens:
            obs_spaces[f"{wind.name}_p"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 5. Loads
        for load in self.all_loads:
            obs_spaces[f"{load.name}_l"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(obs_spaces)

    def _get_observation(self, exog_step: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Extracts and normalizes observation from component states and exogenous data."""
        obs = {}

        # 1. Grid Prices
        grid_exog = exog_step.get(self.grid.name, {})
        obs[f"{self.grid.name}_p_imp"] = np.array([
            grid_exog.get("price_import_per_kwh", 0.20) / self.max_grid_price
        ], dtype=np.float32)
        obs[f"{self.grid.name}_p_exp"] = np.array([
            grid_exog.get("price_export_per_kwh", 0.05) / self.max_grid_price
        ], dtype=np.float32)

        # 2. Batteries (SOC)
        for bat in self.batteries:
            obs[f"{bat.name}_soc"] = np.array([bat.get_soc()], dtype=np.float32)

        # 3. Diesels (Power)
        for d in self.diesels:
            diesel_norm = d.get_power() / self.MAX_CAPACITIES[f"diesel_pmax_{d.name}"]
            obs[f"{d.name}_p"] = np.array([np.clip(diesel_norm, 0.0, 1.0)], dtype=np.float32)

        # 4. PV (Approximate Power from Exogenous)
        for pv in self.pv_gens:
            pv_exog = exog_step.get(pv.name, {})
            pv_irr = pv_exog.get("irradiance_Wm2", 0.0)

            if pv_irr > 0.0 and pv.params.capacity_kw > 0:
                pv_approx_power = (pv.params.capacity_kw * pv.params.derate * pv_irr / 1000.0)
            else:
                pv_approx_power = pv_exog.get("power_kw", 0.0)

            pv_norm = pv_approx_power / self.MAX_CAPACITIES[f"pv_cap_{pv.name}"]
            obs[f"{pv.name}_p"] = np.array([np.clip(pv_norm, 0.0, 1.0)], dtype=np.float32)

        # 5. Wind (Approximate Power from Exogenous)
        for wind in self.wind_gens:
            wind_exog = exog_step.get(wind.name, {})
            wind_speed = wind_exog.get("wind_speed_ms", 0.0)

            if wind_speed > 0.0 and wind.params.rated_kw > 0:
                v_ci = wind.params.cut_in_ms
                v_r = wind.params.rated_ms
                v_co = wind.params.cut_out_ms

                if wind_speed < v_ci or wind_speed > v_co:
                    wind_approx_power = 0.0
                elif wind_speed < v_r:
                    wind_p_norm = ((wind_speed - v_ci) / (v_r - v_ci)) ** 3
                    wind_approx_power = wind_p_norm * wind.params.rated_kw
                else:
                    wind_approx_power = wind.params.rated_kw
            else:
                wind_approx_power = wind_exog.get("power_kw", 0.0)

            wind_norm = wind_approx_power / self.MAX_CAPACITIES[f"wind_rated_{wind.name}"]
            obs[f"{wind.name}_p"] = np.array([np.clip(wind_norm, 0.0, 1.0)], dtype=np.float32)

        # 6. Loads
        for load in self.all_loads:
            load_exog = exog_step.get(load.name, {})
            load_kw = load_exog.get("load_kw", 0.0)
            load_norm = load_kw / self.MAX_CAPACITIES[f"load_max_{load.name}"]
            obs[f"{load.name}_l"] = np.array([np.clip(load_norm, 0.0, 1.0)], dtype=np.float32)

        return obs

    def _parse_action(self, action: Dict[str, Union[np.ndarray, int]]) -> Dict[str, Any]:
        """Converts the Dict action space into the MicrogridEnv action dictionary."""
        actions = {}
        grid_name = self.grid.name

        # 1. Grid
        grid_mode = int(action[f"{grid_name}_mode"])
        grid_trade_val = action[f"{grid_name}_trade"]
        grid_trade_mag = float(grid_trade_val[0]) if isinstance(grid_trade_val, np.ndarray) else float(grid_trade_val)

        if grid_mode == -1:
            actions[grid_name] = "disconnect"
        elif grid_mode == 1:
            if grid_trade_mag > 0:
                # Positive = sell/export
                actions[grid_name] = -grid_trade_mag * self.grid.params.export_limit_kw
            else:
                # Negative = buy/import
                actions[grid_name] = abs(grid_trade_mag) * self.grid.params.import_limit_kw
        else:
            actions[grid_name] = "connect"

        # 2. Batteries
        for bat in self.batteries:
            bat_mode = int(action[f"{bat.name}_mode"])
            bat_mag_val = action[f"{bat.name}_mag"]
            bat_mag = float(bat_mag_val[0]) if isinstance(bat_mag_val, np.ndarray) else float(bat_mag_val)

            if bat_mode == 0:
                bat_setpoint = 0.0
            elif bat_mode > 0:  # Discharge
                bat_setpoint = bat_mag * self.MAX_CAPACITIES[f"bat_dis_{bat.name}"]
            else:  # Charge
                bat_setpoint = -bat_mag * self.MAX_CAPACITIES[f"bat_chg_{bat.name}"]

            actions[bat.name] = {
                "set_state": "ON" if abs(bat_setpoint) > 0.01 else "OFF",
                "power_setpoint": bat_setpoint
            }

        # 3. Diesels
        for d in self.diesels:
            diesel_on = int(action[f"{d.name}_on"]) == 1
            diesel_sp_val = action[f"{d.name}_sp"]
            diesel_sp_mag = float(diesel_sp_val[0]) if isinstance(diesel_sp_val, np.ndarray) else float(diesel_sp_val)
            diesel_sp = diesel_sp_mag * self.MAX_CAPACITIES[f"diesel_pmax_{d.name}"] if diesel_on else 0.0

            actions[d.name] = {
                "on": diesel_on,
                "power_setpoint": diesel_sp
            }

        # 4. Renewables
        for pv in self.pv_gens:
            pv_connect = int(action[f"{pv.name}_con"]) == 1
            actions[pv.name] = "connect" if pv_connect else "disconnect"

        for wind in self.wind_gens:
            wind_connect = int(action[f"{wind.name}_con"]) == 1
            actions[wind.name] = "connect" if wind_connect else "disconnect"

        return actions

    # --- Gymnasium API Methods ---
    def _compute_reward(self, prev_step: int) -> float:
        """Computes reward based on operational costs, reliability, and SOC balancing."""
        if not self.env.history:
            return 0.0

        df = self.env.get_results(as_dataframe=True)
        start_idx = prev_step * self.steps_per_control
        end_idx = self.env.current_step

        if start_idx >= len(df) or start_idx >= end_idx:
            return 0.0

        window = df.iloc[start_idx:end_idx]

        step_cost = window["total_cashflow"].sum()
        dt_hours = self.sim_dt / 60.0
        step_unmet = window["unmet_load_kw"].sum() * dt_hours
        step_curtailed = window["curtailed_gen_kw"].sum() * dt_hours

        soc_vals = [bat.get_soc() for bat in self.batteries]
        soc_penalty = np.mean([(s - 0.5)**2 for s in soc_vals]) if soc_vals else 0.0

        reward = (
            self.reward_weights["cost"] * step_cost +
            -self.reward_weights["unmet"] * step_unmet +
            -self.reward_weights["curtailment"] * step_curtailed +
            -self.reward_weights["soc_deviation"] * soc_penalty
        )

        return float(reward)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        """Resets the environment and generates a new episode's worth of exogenous data."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.env.reset()
        self.exog_list = self.data_builder.build_list()
        self.env.total_simulation_steps = len(self.exog_list)
        self.max_steps = self.env.total_simulation_steps // self.steps_per_control
        self.current_step = 0

        obs = self._get_observation(self.exog_list[0])

        return obs, {"step": 0, "total_cost": 0.0}

    def step(self, action: Dict[str, Union[np.ndarray, int]]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """Executes one control interval step."""
        env_action = self._parse_action(action)
        prev_control_step = self.current_step

        # Execute action for all sub-steps in this control interval
        for _ in range(self.steps_per_control):
            sim_step = self.env.current_step
            if sim_step >= len(self.exog_list):
                break
            exog = self.exog_list[sim_step]
            self.env.step(actions=env_action, exogenous=exog)

        self.current_step += 1

        reward = self._compute_reward(prev_control_step)

        next_exog_idx = min(self.env.current_step, len(self.exog_list) - 1)
        obs = self._get_observation(self.exog_list[next_exog_idx])

        terminated = self.current_step >= self.max_steps
        truncated = False

        df = self.env.get_results(as_dataframe=True)
        total_cost = df["total_cashflow"].sum() if not df.empty else 0.0

        return obs, reward, terminated, truncated, {"step": self.current_step, "total_cost": total_cost}

    def render(self):
        if self.render_mode == "human":
            print(f"Control Step: {self.current_step}/{self.max_steps}")

    def close(self):
        pass
