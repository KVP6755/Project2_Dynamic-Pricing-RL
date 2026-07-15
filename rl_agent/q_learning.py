"""
Project 2: Travel & Hospitality - Dynamic Pricing using Reinforcement Learning (RL)

Week 2: Q-Learning Agent

Member 2: Q-Learning Implementation

Implements the Q-Learning agent responsible for learning the optimal
pricing strategy by interacting with the PricingEnv environment.

Author  : Shilpa S Nair
Week    : 2
File    : q_learning.py
"""


# Import necessary libraries

import sys
import os
import numpy as np
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from gym_env.pricing_env import PricingEnv


class QLearningAgent:

    def __init__(self):

        # Create Pricing Environment
        self.env = PricingEnv()

        # Learning Parameters
        self.learning_rate = 0.1    #Controls how quickly the agent updates its knowledge.
        self.discount_factor = 0.95  #Determines how much future rewards matter.
        self.epsilon = 1.0          #Exploration rate for epsilon-greedy strategy.Controls exploration.


        # State and Action Sizes
        self.state_size = (               #Inventory can be from 0 to 100 → 101 possible values.
            self.env.max_inventory + 1,   #Days left can be from 0 to 30 → 31 possible values. 
            self.env.max_days + 1
        )

        self.action_size = len(self.env.price_levels)


        # Initialize Q-Table
        self.q_table = np.zeros(
            (
                self.state_size[0],
                self.state_size[1],
                self.action_size
            )
        )


if __name__ == "__main__":

    agent = QLearningAgent()

    print("=" * 40)
    print("Q-Learning Agent Initialized")
    print("=" * 40)

    print(f"State Size  : {agent.state_size}")
    print(f"Action Size : {agent.action_size}")
    print(f"Learning Rate : {agent.learning_rate}")
    print(f"Discount Factor : {agent.discount_factor}")
    print(f"Epsilon : {agent.epsilon}")

    print("\nQ-Table Shape :", agent.q_table.shape)



    #git add .
#git commit -m "Initialize Q-Learning agent with environment and Q-table"
#git push origin Shilpa