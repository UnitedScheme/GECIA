"""
Offline SAC Training with Ray RLlib
===================================

import os
import pandas as pd
import numpy as np
import gymnasium as gym
from ray.rllib.algorithms.sac import SACConfig
import torch
import warnings
from ray.rllib.policy.sample_batch import SampleBatch

warnings.filterwarnings("ignore")

# =========================
# 1. Load Offline Dataset
# =========================
print("▌Loading data...")
data = pd.read_csv("model-base.csv")
states = data.iloc[:, 0:34].values.astype(np.float32)
actions = data.iloc[:, 34:35].values.astype(np.float32)
rewards = data.iloc[:, 35:36].values.astype(np.float32)
terminals = data.iloc[:, 36:37].values.astype(np.float32)

# Set terminal flags for last 1000 samples if no terminals exist
if terminals.sum() == 0:
    terminals[-1000:, 0] = 1  # Default terminal state handling

print(f"✔ Data loaded | Samples: {len(states)} | Average reward: {rewards.mean():.3f}")

# =========================
# 2. Offline Environment
# =========================
class OfflineEnv(gym.Env):
    """
    Offline reinforcement learning environment for medical treatment data.
    
    This environment uses pre-collected medical treatment data instead of 
    interacting with a real environment. Suitable for offline RL algorithms.
    """
    
    def __init__(self, config=None):
        # Observation space: 34-dimensional medical features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        
        # Action space: 1-dimensional treatment dosage (normalized to [-1, 1])
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

        # Load offline dataset
        self._dataset = {
            'obs': states,
            'actions': actions,
            'rewards': rewards,
            'terminateds': terminals
        }
        self._index = 0
        self._current_episode_reward = 0.0
        self._current_episode_length = 0

    def reset(self, *, seed=None, options=None):
        """Reset environment to a random starting point in the dataset."""
        self._index = np.random.randint(0, len(self._dataset['obs']))
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        return self._dataset['obs'][self._index], {}

    def step(self, action):
        """
        Execute one step in the environment using pre-collected data.
        
        Args:
            action: Proposed action (treatment dosage)
            
        Returns:
            obs: Next observation
            reward: Immediate reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional info including episode reward
        """
        obs = self._dataset['obs'][self._index]
        reward = float(self._dataset['rewards'][self._index][0])
        terminated = bool(self._dataset['terminateds'][self._index][0])
        truncated = False

        # Track episode statistics
        self._current_episode_reward += reward
        self._current_episode_length += 1

        # Move to next data point (cyclic)
        self._index = (self._index + 1) % len(self._dataset['obs'])
        return obs, reward, terminated, truncated, {
            'episode_reward': self._current_episode_reward
        }

# =========================
# 3. SAC Algorithm Configuration
# =========================
config = (
    SACConfig()
    .environment(env=OfflineEnv, env_config={})
    .training(
        # Policy network architecture
        policy_model_config={
            "fcnet_hiddens": [256, 256], 
            "fcnet_activation": "tanh"
        },
        # Q-network architecture
        q_model_config={
            "fcnet_hiddens": [256, 256, 256], 
            "fcnet_activation": "relu"
        },
        # Learning rates
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # Soft update parameter
        tau=0.01,
        # Discount factor
        gamma=0.99,
        # Training batch size
        train_batch_size=1024,
        # Warm-up steps before learning starts
        num_steps_sampled_before_learning_starts=5000,
        # Experience replay buffer
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 100000,
        },
    )
    .framework("torch")
    .resources(num_gpus=1)
    .offline_data(
        input_="sampler",
        actions_in_input_normalized=True,
    )
    # Enable new API stack for better performance
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True
    )
)

# =========================
# 4. Training Execution
# =========================
if __name__ == "__main__":
    print("▌Initializing training...")
    algo = config.build()

    # Create checkpoint directory
    checkpoint_dir = "/root/ray/checkpoints_final"
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        best_reward = -float('inf')
        
        # Training loop
        for i in range(100):
            result = algo.train()
            avg_reward = result.get('episode_reward_mean', 0)
            avg_episode_length = result.get('episode_len_mean', 0)
            total_timesteps = result.get('num_env_steps_sampled_lifetime', 0)

            print(
                f"Iteration {i} | Avg reward: {avg_reward:.2f} | "
                f"Avg length: {avg_episode_length:.2f} | Steps: {total_timesteps}"
            )

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                checkpoint_path = algo.save(checkpoint_dir)
                print(f"↯ Saved best checkpoint (reward={avg_reward:.2f})")

            # Early stopping if no improvement
            if i > 20 and avg_reward <= 0:
                print("No reward improvement, early stopping")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final checkpoint
        checkpoint_path = algo.save(checkpoint_dir)
        print(f"✔ Final checkpoint saved to: {checkpoint_path}")

        # Export PyTorch model for deployment
        try:
            from ray.rllib.algorithms.algorithm import Algorithm

            if isinstance(algo, Algorithm):
                policy = algo.get_policy()
                if policy is not None:
                    torch.save(
                        policy.model.state_dict(),
                        os.path.join(checkpoint_dir, "torch_model.pt")
                    )
                    print(f"✔ PyTorch model saved to: {checkpoint_dir}/torch_model.pt")
        except Exception as e:
            print(f"Failed to save PyTorch model: {str(e)}")

        print("Process completed")
