"""
microgrid_sim/control/rl_controller.py

Reinforcement Learning controller using Stable-Baselines3 (PPO).

The PPO model automatically handles the Dict observation spaces
provided by MicrogridGymEnv.
"""

from __future__ import annotations
import os
from typing import Any, Optional, Tuple, Dict, Union
import numpy as np
from stable_baselines3 import A2C, DQN, SAC, TD3

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.logger import configure
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. RL features disabled.")

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DQN": DQN,
}

class RLController:
    """
    Reinforcement Learning controller for microgrid energy management using PPO.
    """

    def __init__(
        self,
        env: Optional[Any] = None,
        algo: str = "PPO",
        policy: str = "MultiInputPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 0,
        device: str = "auto"
    ):
        """Initialize the RL controller."""
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

        self.env = env
        self.algo_name = algo.upper()
        if self.algo_name not in ALGO_MAP:
            raise ValueError(f"Unsupported algo '{algo}'. Use one of {list(ALGO_MAP.keys())}.")
        self.model = None
        self.verbose = verbose

        self.config = {
            "policy": policy, "learning_rate": learning_rate, "n_steps": n_steps,
            "batch_size": batch_size, "n_epochs": n_epochs, "gamma": gamma,
            "gae_lambda": gae_lambda, "clip_range": clip_range, "ent_coef": ent_coef,
            "vf_coef": vf_coef, "max_grad_norm": max_grad_norm, "device": device,
            "verbose": verbose,
        }

        if env is not None:
            self._build_model()

    def _build_model(self):
        """Builds the PPO model with configured hyperparameters."""
        if self.env is None:
            raise ValueError("Environment must be provided to build model.")

        # MultiInputPolicy is mandatory for Dict observation spaces
        if self.config['policy'] == 'MlpPolicy':
            self.config['policy'] = 'MultiInputPolicy'

        algo_class = ALGO_MAP[self.algo_name]

        if self.algo_name in ["SAC", "TD3", "DQN"]:
            self.model = algo_class(
                policy=self.config["policy"], env=self.env, learning_rate=self.config["learning_rate"],
                batch_size=self.config["batch_size"], gamma=self.config["gamma"],
                verbose=self.verbose, device=self.config["device"]
            )
        elif self.algo_name == "A2C":
            self.model = algo_class(
                policy=self.config["policy"], env=self.env, learning_rate=self.config["learning_rate"],
                n_steps=self.config["n_steps"], gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"], ent_coef=self.config["ent_coef"],
                vf_coef=self.config["vf_coef"], max_grad_norm=self.config["max_grad_norm"],
                verbose=self.verbose, device=self.config["device"]
            )
        elif self.algo_name == "PPO":
            self.model = algo_class(
                policy=self.config["policy"], env=self.env, learning_rate=self.config["learning_rate"],
                n_steps=self.config["n_steps"], batch_size=self.config["batch_size"], n_epochs=self.config["n_epochs"],
                gamma=self.config["gamma"], gae_lambda=self.config["gae_lambda"], clip_range=self.config["clip_range"],
                ent_coef=self.config["ent_coef"], vf_coef=self.config["vf_coef"], max_grad_norm=self.config["max_grad_norm"],
                verbose=self.verbose, device=self.config["device"]
            )
        else:
            raise ValueError(f"Algorithm '{self.algo_name}' is not supported.")

        if self.verbose >= 1:
            print(f"{self.algo_name} model initialized with hyperparameters:")
            for k, v in self.config.items():
                print(f"  {k}: {v}")

    def train(
        self,
        total_timesteps: int = 50000,
        log_dir: Optional[str] = None,
        eval_env: Optional[Any] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
    ) -> "RLController":
        """Train the RL agent."""
        if self.model is None:
            self._build_model()

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            logger = configure(log_dir, ["csv", "tensorboard"])
            self.model.set_logger(logger)

        callbacks = []
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_dir if log_dir else "./logs/", log_path=log_dir if log_dir else "./logs/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True, render=False,
                verbose=self.verbose,
            )
            callbacks.append(eval_callback)

        callback_list = callbacks if callbacks else None

        if self.verbose >= 1:
            print(f"\nStarting training for {total_timesteps} timesteps...")

        self.model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)

        if self.verbose >= 1:
            print("Training complete!")

        return self

    def decide(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> Union[Dict[str, np.ndarray | int], np.ndarray]:
        """
        Get action from the trained model.

        Returns: Dict if the environment is Dict-spaced, or np.ndarray if Box-spaced.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)

        if self.verbose >= 1:
            print(f"Model saved to {path}.zip")

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[Any] = None,
        device: str = "auto",
        verbose: int = 1
    ) -> "RLController":
        """Load a trained model from disk."""
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required.")

        controller = cls(env=env, verbose=verbose, device=device, policy='MultiInputPolicy')
        controller.model = PPO.load(path, env=env, device=device)

        if verbose >= 1:
            print(f"Model loaded from {path}.zip")

        return controller

    def evaluate(
        self,
        env: Any,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Tuple[float, float]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        episode_rewards = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.decide(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if render: env.render()

            episode_rewards.append(episode_reward)

            if self.verbose >= 1:
                print(f"Episode {ep+1}/{n_episodes}: Reward = {episode_reward:.2f}")

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        if self.verbose >= 1:
            print(f"\nEvaluation Results ({n_episodes} episodes): Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward
