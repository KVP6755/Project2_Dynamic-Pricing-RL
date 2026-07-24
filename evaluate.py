"""
================================================================================
evaluate.py
================================================================================
Project     : Travel & Hospitality — Dynamic Pricing using RL
              Infotact Technical Internship — Project 2
Author      : Subhashree Behera
Week        : Week 3 — Deep Reinforcement Learning (DQN/PPO)
Role        : Environment Integration & Performance Evaluation

--------------------------------------------------------------------------------
Responsibility:
    Integrate the trained DQN agent with the custom PricingEnv Gym
    environment, run evaluation episodes, collect performance metrics,
    and compare DQN performance against the Q-Learning baseline from
    Week 2.

    This file is the final judge — it answers:
    "Did the DQN agent actually learn a better pricing policy than
    the naive Q-Learning baseline?"

--------------------------------------------------------------------------------
What This File Does:
    1. Loads the trained DQN model (from Member 1: dqn_model.py)
    2. Loads the PricingEnv Gym environment (Member 2: gym environment)
    3. Runs N evaluation episodes with the DQN agent
    4. Runs N episodes with the Q-Learning baseline for comparison
    5. Collects metrics: total revenue, rooms sold, rooms unsold
    6. Plots comparison graphs (reward curves, bar charts)
    7. Generates a written performance summary report

--------------------------------------------------------------------------------
Metrics Tracked:
    - Episode total revenue (primary KPI)
    - Rooms sold per episode
    - Rooms unsold per episode (spoilage)
    - Mean, std, min, max across N episodes

--------------------------------------------------------------------------------
Comparison:
    DQN Agent vs Q-Learning Baseline vs Random Policy
    Winner = highest mean episode revenue across 100 episodes

--------------------------------------------------------------------------------
Dependencies:
    - dqn_model.py    (Member 1) — DQNAgent class
    - train_dqn.py    (Member 2) — trained model weights
    - mdp_design.py   (Week 1)   — environment constants
    - state_action_space.py (Week 2) — encode_state, PRICE_LEVELS

--------------------------------------------------------------------------------
Output Files:
    plots/reward_comparison.png      — DQN vs baseline episode rewards
    plots/revenue_bar_chart.png      — mean revenue comparison bar chart
    plots/learning_curve.png         — DQN reward improvement over episodes
    plots/rooms_sold_comparison.png  — rooms sold vs unsold comparison
================================================================================
"""


# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CONSTANTS
# ============================================================
# Must match mdp_design.py and state_action_space.py exactly

PRICE_LEVELS    = [50, 100, 150, 200, 250]
N_ACTIONS       = len(PRICE_LEVELS)
TOTAL_INVENTORY = 100
TOTAL_DAYS      = 30
RANDOM_STATE    = 42

# Evaluation settings
N_EVAL_EPISODES = 100    # episodes to run per policy
PLOTS_DIR       = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'dqn'      : '#2ecc71',   # green
    'qlearning': '#3498db',   # blue
    'random'   : '#e74c3c'    # red
}

print("Evaluate.py constants loaded:")
print(f"  Eval episodes : {N_EVAL_EPISODES}")
print(f"  Price levels  : {PRICE_LEVELS}")
print(f"  Plots output  : {PLOTS_DIR}/")

# ============================================================
# MOCK DQN AGENT (placeholder until Member 1 shares dqn_model.py)
# ============================================================
# This stub mimics the interface of Member 1's DQNAgent class.
# When dqn_model.py is ready, replace this block with:
#     from dqn_model import DQNAgent

class MockDQNAgent:
    """
    Placeholder DQN agent — mimics DQNAgent interface from Member 1.
    Replace with: from dqn_model import DQNAgent
    when Member 1's file is ready.

    Simulates a slightly-better-than-random pricing policy to
    allow full evaluation pipeline testing before DQN is trained.
    """
    def __init__(self):
        np.random.seed(RANDOM_STATE)
        self.name = "DQN Agent (Mock)"

    def select_action(self, state):
        """
        Select price action given current state.
        Mock: prefers mid-range prices (indices 1-3)
        Real DQN: argmax(Q-values) from neural network
        """
        inventory, days = state
        # Mock smart behavior: high price with lots of days/inventory
        # low price when running out of time
        if days > 20:
            return np.random.choice([2, 3, 4])  # higher prices
        elif days > 10:
            return np.random.choice([1, 2, 3])  # mid prices
        else:
            return np.random.choice([0, 1, 2])  # lower prices to clear


# ============================================================
# DEMAND FUNCTION (from mdp_design.py Week 1)
# ============================================================

def demand_function(price, days_remaining):
    """
    Stochastic customer demand — same formula as Week 1 mdp_design.py.
    Kept here so evaluate.py is self-contained for testing.
    """
    BASE_DEMAND_RATE  = 0.8
    PRICE_SENSITIVITY = 0.002
    URGENCY_FACTOR    = 0.3

    urgency  = URGENCY_FACTOR * (1 / max(days_remaining, 1))
    prob     = BASE_DEMAND_RATE - PRICE_SENSITIVITY * price + urgency
    prob     = max(0.0, min(1.0, prob))
    bookings = np.random.binomial(5, prob)
    return int(bookings)