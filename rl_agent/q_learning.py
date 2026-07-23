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

        # Store rewards of each episode
        self.reward_history = []


# --------------------------------------------------------
# Choose an action using the Epsilon-Greedy Policy
# --------------------------------------------------------
    def choose_action(self, state):

        # Get inventory and days left from the current state
        inventory, days_left = state

        # Generate a random number between 0 and 1
        random_number = random.uniform(0, 1)

        # Exploration: choose a random action
        if random_number < self.epsilon:

            action = self.env.action_space.sample()

            print("\n===== Exploration =====")
            print("Random Number :", random_number)
            print("Random Action Selected :", action)
            print("Selected Price :", self.env.price_levels[action])

        # Exploitation: choose the best action from the Q-table
        else:

            action = np.argmax(
                self.q_table[inventory, days_left]
            )

            print("\n===== Exploitation =====")
            print("Random Number :", random_number)
            print("Best Action Selected :", action)
            print("Selected Price :", self.env.price_levels[action])

        return action

# --------------------------------------------------------
# Update Q-Table using the Q-Learning Equation
# --------------------------------------------------------
    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):

        # Get current state
        inventory, days_left = state

        # Get next state
        next_inventory, next_days_left = next_state

        # Current Q-Value
        current_q = self.q_table[
            inventory,
            days_left,
            action
        ]

        # Maximum Future Q-Value
        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(
                self.q_table[
                    next_inventory,
                    next_days_left
                ]
            )

        # Q-Learning Equation
        new_q = current_q + self.learning_rate * (
            reward +
            self.discount_factor * max_future_q -
            current_q
        )

        # Update the Q-Table
        self.q_table[
            inventory,
            days_left,
            action
        ] = new_q

        return new_q


    # --------------------------------------------------------
    # Day 4: Training Loop
    #
    # The agent interacts with the environment for multiple
    # episodes. In each episode it:
    # 1. Resets the environment
    # 2. Chooses an action
    # 3. Takes a step
    # 4. Receives a reward
    # 5. Updates the Q-table
    # 6. Continues until the episode ends
    # --------------------------------------------------------
# --------------------------------------------------------
# Train the Q-Learning Agent
# --------------------------------------------------------
    def train(
        self,
        episodes=3
    ):

        print("\n========================================")
        print("Training Started")
        print("========================================")

        # Clear previous rewards
        self.reward_history.clear()

        # Loop through each episode
        for episode in range(episodes):

            # Reset the environment
            state, info = self.env.reset()

            total_reward = 0
            steps = 0

            terminated = False
            truncated = False

            # Continue until the episode ends
            while not (terminated or truncated):

                # Choose an action
                action = self.choose_action(state)

                # Take one step in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Update the Q-Table
                self.update_q_table(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated
                )

                # Move to the next state
                state = next_state

                # Add reward
                total_reward += reward

                # Count steps
                steps += 1

            # Store episode reward
            self.reward_history.append(total_reward)

            print("\nEpisode :", episode + 1)
            print("Steps :", steps)
            print("Total Reward :", total_reward)

        print("\n========================================")
        print("Training Completed")
        print("========================================")

if __name__ == "__main__":

    # Create Q-Learning Agent
    agent = QLearningAgent()

    print("=" * 45)
    print("      Q-LEARNING AGENT TEST")
    print("=" * 45)

    # Agent Information
    print("\n1. Agent Initialization")
    print("-" * 30)
    print(f"State Size        : {agent.state_size}")
    print(f"Action Size       : {agent.action_size}")
    print(f"Learning Rate     : {agent.learning_rate}")
    print(f"Discount Factor   : {agent.discount_factor}")
    print(f"Epsilon           : {agent.epsilon}")
    print(f"Q-Table Shape     : {agent.q_table.shape}")

    # Reset Environment
    print("\n2. Environment Reset")
    print("-" * 30)

    state, info = agent.env.reset()

    print("Current State :", state)
    print("Inventory     :", state[0])
    print("Days Left     :", state[1])

    # Test Epsilon-Greedy Policy
    print("\n3. Epsilon-Greedy Action Selection")
    print("-" * 30)

    action = agent.choose_action(state)

    print("\nReturned Action :", action)
    print("Selected Price  :", agent.env.price_levels[action])

    # ----------------------------------------
    # Test Q-Table Update
    # ----------------------------------------

    print("\n4. Q-Table Update")
    print("-" * 30)

    next_state, reward, terminated, truncated, info = agent.env.step(action)

    updated_q = agent.update_q_table(
        state,
        action,
        reward,
        next_state,
        terminated
    )

    print("\nCurrent State :", state)
    print("Selected Action :", action)
    print("Reward :", reward)
    print("Next State :", next_state)

    print("\nUpdated Q-Value :", updated_q)

    inventory, days_left = state

    print("Stored Q-Value :",
          agent.q_table[
              inventory,
              days_left,
              action
          ])

    
 # ----------------------------------------
# Training Loop
# ----------------------------------------

# ----------------------------------------
# Training Loop
# ----------------------------------------

    print("\n5. Training Loop")
    print("-" * 30)

    agent.train(episodes=3)

    print("\n" + "=" * 45)
    print("Q-Learning Agent Tested Successfully")
    print("=" * 45)