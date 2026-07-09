"""
================================================================================
mdp_design.py
================================================================================
Project     : Travel & Hospitality — Dynamic Pricing using RL
              Infotact Technical Internship — Project 2
Author      : Subhashree Behera
Week - 1    : MDP Design & Documentation

--------------------------------------------------------------------------------
What is an MDP?
    A Markov Decision Process (MDP) is the mathematical framework used to
    formulate reinforcement learning problems. It defines:

    - WHO is making decisions     → the Agent (pricing algorithm)
    - WHAT the agent observes     → the State
    - WHAT the agent can do       → Actions (price levels)
    - WHAT the agent gets back    → Reward (revenue)
    - HOW the world changes       → Transition (customer demand)

--------------------------------------------------------------------------------
The Business Problem:
    An airline/hotel has 100 seats/rooms to sell over 30 days.
    If priced too high → customers don't book → unsold inventory (spoilage)
    If priced too low → sells out early → lost revenue opportunity

    Goal: learn an optimal pricing POLICY that maximizes total revenue
    across the full 30-day selling season.

--------------------------------------------------------------------------------
MDP Formulation:

    STATE  = [remaining_inventory, days_until_departure]
             remaining_inventory : integer, 0 to 100
             days_until_departure: integer, 0 to 30
             Total state space   : 101 × 31 = 3,131 possible states

    ACTION = price_level chosen by agent
             Discrete options: [50, 100, 150, 200, 250]
             Total action space : 5 actions

    REWARD = revenue earned at each step
             reward = price × bookings_made
             bookings_made depends on stochastic demand function

    TRANSITION:
             next_state = [remaining_inventory - bookings_made,
                           days_until_departure - 1]

    EPISODE:
             Starts at state [100, 30] (full inventory, 30 days)
             Ends when days_until_departure = 0 OR inventory = 0
             One episode = one complete booking season

--------------------------------------------------------------------------------
Demand Function (Stochastic):
    Customer purchase probability DECREASES as price increases.
    Customer purchase probability ALSO changes based on days left.
    This models real urgency — last-minute travelers accept higher prices.

    P(purchase) = max(0, base_rate - price_sensitivity × price_level
                         + urgency_factor × (1 / days_left))

--------------------------------------------------------------------------------
Why Markov Property Holds:
    The next state depends ONLY on the current state and action —
    not on the full history of past prices or bookings.
    [remaining_inventory=50, days=10] gives the same transition
    probabilities regardless of HOW we arrived at that state.

--------------------------------------------------------------------------------
Functions:
    define_state_space()     → documents and validates state space
    define_action_space()    → documents price level options
    define_reward_function() → implements reward calculation
    define_demand_function() → implements stochastic booking probability
    simulate_one_step()      → simulates one MDP transition
    simulate_episode()       → runs one complete 30-day episode
    summarize_mdp()          → prints full MDP specification report

--------------------------------------------------------------------------------
Output:
    This file serves as the single source of truth for the MDP design.
    Member 2 (Custom Gym Environment) will implement this specification
    inside PricingEnv using Gymnasium.
================================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd

# ============================================================
# MDP CONSTANTS
# ============================================================
# These are the fixed parameters of our pricing problem.
# Member 2 (Gym Environment) must use these exact same values
# to ensure consistency across the full RL pipeline.

# Inventory
TOTAL_INVENTORY = 100     # total rooms/seats per season
MAX_INVENTORY   = 100     # upper bound of state space

# Time
TOTAL_DAYS      = 30      # days in one booking season
MAX_DAYS        = 30      # upper bound of state space

# Price levels (Action space)
# 5 discrete price options the agent can choose from
PRICE_LEVELS = [50, 100, 150, 200, 250]
N_ACTIONS    = len(PRICE_LEVELS)

# Demand parameters
BASE_DEMAND_RATE    = 0.8   # base probability any customer books
PRICE_SENSITIVITY   = 0.002 # how much price reduces booking prob
URGENCY_FACTOR      = 0.3   # how much last-minute urgency increases prob
MAX_BOOKINGS_PER_STEP = 5   # max customers who arrive per time step

# Episode
RANDOM_STATE = 42  # reproducibility seed

print("MDP Constants loaded:")
print(f"  State space : {(MAX_INVENTORY+1) * (MAX_DAYS+1)} states")
print(f"  Action space: {N_ACTIONS} price levels → {PRICE_LEVELS}")
print(f"  Episode     : {TOTAL_DAYS} days, {TOTAL_INVENTORY} rooms")