"""
microgrid_sim/control/rl_controller.py

Reinforcement Learning controller using Stable-Baselines3.

Features:
- PPO (Proximal Policy Optimization) agent
- Training and evaluation modes
- Model saving/loading
- Integration with MicrogridGymEnv
"""

from __future__ import annotations
import os
from typing import Any, Optional, Tuple
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    # from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. RL features disabled.")


class RLController:
    """
    Reinforcement Learning controller for microgrid energy management.

    This controller uses PPO (Proximal Policy Optimization) to learn
    optimal control policies for battery, diesel, and grid operations.

    Usage:
    ------
    Training:
        controller = RLController(env)
        controller.train(total_timesteps=50000)
        controller.save("models/ppo_microgrid")

    Inference:
        controller = RLController.load("models/ppo_microgrid", env)
        action = controller.decide(obs)
    """

    def __init__(
        self,
        env: Optional[Any] = None,
        policy: str = "MlpPolicy",
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
        verbose: int = 1,
        device: str = "auto"
    ):
        """
        Initialize the RL controller.

        Args:
            env: Gym environment (MicrogridGymEnv)
            policy: Policy network type ('MlpPolicy' recommended)
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each env per update
            batch_size: Minibatch size
            n_epochs: Number of epochs for policy update
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

        self.env = env
        self.model: Optional[PPO] = None
        self.verbose = verbose

        # Store hyperparameters for later use
        self.config = {
            "policy": policy,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "device": device
        }

        if env is not None:
            self._build_model()

    def _build_model(self):
        """Builds the PPO model with configured hyperparameters."""
        if self.env is None:
            raise ValueError("Environment must be provided to build model.")

        self.model = PPO(
            policy=self.config["policy"],
            env=self.env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            verbose=self.verbose,
            device=self.config["device"]
        )

        if self.verbose >= 1:
            print("PPO model initialized with hyperparameters:")
            for k, v in self.config.items():
                print(f"  {k}: {v}")

    def train(
        self,
        total_timesteps: int = 50000,
        log_dir: Optional[str] = None,
        eval_env: Optional[Any] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        callback: Optional[BaseCallback] = None
    ) -> "RLController":
        """
        Train the RL agent.

        Args:
            total_timesteps: Total number of timesteps to train
            log_dir: Directory for tensorboard logs
            eval_env: Separate environment for evaluation
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes for evaluation
            callback: Custom callback for training

        Returns:
            self: Returns self for method chaining
        """
        if self.model is None:
            self._build_model()

        # Configure logger
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
            self.model.set_logger(logger)

        # Setup evaluation callback
        callbacks = []
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_dir if log_dir else "./logs/",
                log_path=log_dir if log_dir else "./logs/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        if callback is not None:
            callbacks.append(callback)

        callback_list = callbacks if callbacks else None

        # Train
        if self.verbose >= 1:
            print(f"\nStarting training for {total_timesteps} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=self.verbose >= 1
        )

        if self.verbose >= 1:
            print("Training complete!")

        return self

    def decide(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get action from the trained model.

        Args:
            observation: Current observation from environment
            deterministic: If True, use deterministic policy (no exploration)

        Returns:
            action: Action vector to execute
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Create directory if needed
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
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model (without extension)
            env: Environment to use with loaded model
            device: Device to load model on
            verbose: Verbosity level

        Returns:
            RLController: Loaded controller instance
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required.")

        controller = cls(env=env, verbose=verbose, device=device)
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
        """
        Evaluate the trained model.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy
            render: Render environment during evaluation

        Returns:
            mean_reward: Mean episode reward
            std_reward: Standard deviation of rewards
        """
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

                if render:
                    env.render()

            episode_rewards.append(episode_reward)

            if self.verbose >= 1:
                print(f"Episode {ep+1}/{n_episodes}: Reward = {episode_reward:.2f}")

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        if self.verbose >= 1:
            print(f"\nEvaluation Results ({n_episodes} episodes):")
            print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called after every step."""
        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout."""
        if self.verbose >= 1:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"Rollout complete. Mean reward (last 100 ep): {mean_reward:.2f}")
