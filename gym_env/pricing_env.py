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




    # --------------------
    # Reset
    # --------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.inventory = self.max_inventory
        self.days_left = self.max_days

        state = np.array(
            [self.inventory, self.days_left],
            dtype=np.int32
        )

        info = {}

        return state, info



    # --------------------
    # Step
    # --------------------
    def step(self, action):

        print("\n===== STEP FUNCTION CALLED =====")

        selected_price = self.price_levels[action]
        print("Selected Action :", action)
        print("Selected Price  :", selected_price)

        bookings = 5
        print("Bookings :", bookings)

        reward = bookings * selected_price
        print("Reward :", reward)

        self.inventory = max(0, self.inventory - bookings)
        print("Remaining Inventory :", self.inventory)

        self.days_left -= 1
        print("Days Left :", self.days_left)

        terminated = (
            self.inventory == 0 or
            self.days_left == 0
        )

        truncated = False

        next_state = np.array(
            [self.inventory, self.days_left],
            dtype=np.int32
        )

        print("Next State :", next_state)

        info = {
            "Selected Price": selected_price,
            "Bookings": bookings
        }

        return next_state, reward, terminated, truncated, info


    # --------------------
    # Render
    # --------------------
    def render(self):

        print("\n===== Current Environment =====")
        print(f"Remaining Inventory : {self.inventory}")
        print(f"Days Left           : {self.days_left}")

# -----------------------------------
# Testing the Pricing Environment
# -----------------------------------

if __name__ == "__main__":

    # Create Environment
    env = PricingEnv()

    print("===================================")
    print("   PRICING ENVIRONMENT TEST")
    print("===================================\n")

    # Test Reset
    state, info = env.reset()

    print("1. Reset Function")
    print("-----------------")
    print("Initial State :", state)
    print("Info :", info)
    print()

    # Test Action Space
    print("2. Action Space")
    print("-----------------")
    print("Available Price Levels :", env.price_levels)
    print("Number of Actions :", env.action_space.n)
    print()

    # Test Observation Space
    print("3. Observation Space")
    print("-----------------")
    print(env.observation_space)
    print()

    # Test Step Function
    print("4. Step Function")
    print("-----------------")

    action = 2   # Price = 5000

    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nReturned Values")
    print("Next State :", next_state)
    print("Reward :", reward)
    print("Terminated :", terminated)
    print("Truncated :", truncated)
    print("Info :", info)
    print()

    # Test Render
    print("5. Render Function")
    print("-----------------")
    env.render()

    print("\n===================================")
    print("Environment Test Completed Successfully")
    print("===================================")