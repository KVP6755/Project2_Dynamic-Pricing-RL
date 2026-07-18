"""
Project 2 Week 2: Environment Integration & Training
=====================================================
Member 3: Integrates Q-Learning agent with PricingEnv,
runs training for multiple episodes, and stores rewards
and training history.

Author  : Varnika Valliammai V
File    : training_pipeline.py
"""

import numpy as np
import pandas as pd

# Import MDP constants from Member 1 (Week 1)
from mdp_design import (
    PRICE_LEVELS,
    TOTAL_INVENTORY,
    TOTAL_DAYS,
    simulate_one_step
)

# ─────────────────────────────────────────
# 1. State Encoding
# ─────────────────────────────────────────
def encode_state(inventory, days):
    """
    Converts [inventory, days] into a single integer index
    for Q-table lookup.

    Q-table needs one row per state — we flatten the 2D
    state [inventory, days] into one unique integer.

    Parameters:
        inventory : rooms remaining (0-100)
        days      : days left (0-30)

    Returns:
        state_index : unique integer (0 to 3130)
    """
    return int(inventory * (TOTAL_DAYS + 1) + days)


# ─────────────────────────────────────────
# 2. Training Configuration
# ─────────────────────────────────────────
TOTAL_EPISODES = 500
RANDOM_STATE   = 42


# ─────────────────────────────────────────
# 3. Single Episode Runner
# ─────────────────────────────────────────
def run_episode(agent, episode_num, training=True):
    """
    Runs one complete 30-day booking episode.

    Parameters:
        agent       : QLearningAgent from Member 2
        episode_num : current episode number
        training    : if True agent learns, if False agent exploits

    Returns:
        episode_reward  : total revenue this episode
        episode_history : list of daily step logs
    """
    # Start state: full inventory, full days
    inventory = TOTAL_INVENTORY
    days      = TOTAL_DAYS
    state     = encode_state(inventory, days)

    episode_reward  = 0
    episode_history = []
    done            = False
    day             = 0

    while not done:

        # Agent picks action (price index 0-4)
        if training:
            action = agent.choose_action(state)
        else:
            action = agent.best_action(state)


        next_state_raw, reward, done, info = simulate_one_step(
            [inventory, days], action
        )

        next_inventory = next_state_raw[0]
        next_days      = next_state_raw[1]
        next_state     = encode_state(next_inventory, next_days)

        # Agent updates Q-table (training only)
        if training:
            agent.update(
                state      = state,
                action     = action,
                reward     = reward,
                next_state = next_state,
                done       = done
            )

        # Log this day
        episode_history.append({
            'episode'   : episode_num,
            'day'       : day + 1,
            'inventory' : inventory,
            'days_left' : days,
            'action'    : action,
            'price'     : PRICE_LEVELS[action],
            'bookings'  : info['bookings_made'],
            'reward'    : reward
        })

        episode_reward += reward
        inventory       = next_inventory
        days            = next_days
        state           = next_state
        day            += 1

    return episode_reward, episode_history

