"""
Project 2: Dynamic Pricing using Reinforcement Learning

Member 2: Custom Gym Environment

Implements the PricingEnv using Gymnasium.
Defines the environment structure, action space,
observation space, and reset functionality for
the Reinforcement Learning agent.

Author     : Shilpa S Nair
Week       : 1
File       : pricing_env.py
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PricingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.max_inventory = 100
        self.max_days = 30

        self.inventory = self.max_inventory
        self.days_left = self.max_days

        self.price_levels = [4000, 4500, 5000, 5500, 6000]

        self.action_space = spaces.Discrete(len(self.price_levels))

        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([100, 30]),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days

        state = np.array([self.inventory, self.days_left], dtype=np.int32)

        info = {}

        return state, info


# -----------------------------
# Testing the environment
# -----------------------------
if __name__ == "__main__":

    env = PricingEnv()

    state, info = env.reset()

    print("===== Pricing Environment =====")
    print(f"Inventory : {env.inventory}")
    print(f"Days Left : {env.days_left}")
    print(f"Price Levels : {env.price_levels}")
    print(f"Action Space : {env.action_space}")
    print(f"Observation Space : {env.observation_space}")
    print(f"Initial State : {state}")
    print(f"Info : {info}")