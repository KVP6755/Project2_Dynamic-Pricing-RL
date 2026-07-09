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

# ============================================================
# FUNCTION 1: define_state_space
# ============================================================

def define_state_space():
    """
    Documents and validates the MDP state space.

    State = [remaining_inventory, days_until_departure]

    Why these two variables?
    - remaining_inventory tells the agent HOW DESPERATE it is to sell
      (0 rooms left = episode over, 100 rooms left = plenty of time)
    - days_until_departure tells the agent HOW MUCH TIME it has
      (0 days left = must sell now at any price or lose inventory)

    Together they capture the full context needed to make a
    pricing decision — this satisfies the Markov Property.

    Returns:
        dict: state space specification
    """
    state_spec = {
        'variables'   : ['remaining_inventory', 'days_until_departure'],
        'inventory'   : {'min': 0, 'max': MAX_INVENTORY, 'type': 'int'},
        'days'        : {'min': 0, 'max': MAX_DAYS,      'type': 'int'},
        'total_states': (MAX_INVENTORY + 1) * (MAX_DAYS + 1),
        'start_state' : [TOTAL_INVENTORY, TOTAL_DAYS],
        'terminal_conditions': [
            'days_until_departure == 0',
            'remaining_inventory == 0'
        ]
    }

    print("=" * 50)
    print("STATE SPACE")
    print("=" * 50)
    print(f"Variables      : {state_spec['variables']}")
    print(f"Inventory range: 0 to {MAX_INVENTORY}")
    print(f"Days range     : 0 to {MAX_DAYS}")
    print(f"Total states   : {state_spec['total_states']}")
    print(f"Start state    : {state_spec['start_state']}")
    print(f"Terminal when  : {state_spec['terminal_conditions']}")
    print("=" * 50)

    return state_spec


# ============================================================
# FUNCTION 2: define_action_space
# ============================================================

def define_action_space():
    """
    Documents the MDP action space.

    Action = price_level chosen by the agent at each time step.

    The agent picks ONE price per day from 5 discrete options.
    This is a simplified version of real airline/hotel pricing
    (which can have hundreds of fare classes) but captures the
    core trade-off: high price = more revenue per booking but
    fewer customers; low price = more bookings but less margin.

    Returns:
        dict: action space specification
    """
    action_spec = {
        'type'        : 'Discrete',
        'n_actions'   : N_ACTIONS,
        'price_levels': PRICE_LEVELS,
        'min_price'   : min(PRICE_LEVELS),
        'max_price'   : max(PRICE_LEVELS),
        'index_map'   : {i: p for i, p in enumerate(PRICE_LEVELS)}
    }

    print("=" * 50)
    print("ACTION SPACE")
    print("=" * 50)
    print(f"Type           : {action_spec['type']}")
    print(f"Number of actions: {action_spec['n_actions']}")
    print(f"Price options  : {action_spec['price_levels']}")
    print(f"Index mapping  : {action_spec['index_map']}")
    print("=" * 50)

    return action_spec


if __name__ == "__main__":
    define_state_space()
    define_action_space()

# ============================================================
# FUNCTION 3: define_reward_function
# ============================================================

def define_reward_function(price, bookings_made):
    """
    Calculate the immediate reward for one time step.

    Reward = price × bookings_made

    This is the REVENUE generated in one day at a given price.
    The agent's goal is to maximize CUMULATIVE reward across
    the full 30-day episode — not just one day's revenue.

    This creates the core tension:
    - High price → high reward per booking, but fewer bookings
    - Low price  → more bookings, but lower reward per booking
    - Near deadline → must drop price to clear remaining inventory

    Args:
        price         (int): price chosen by agent (from PRICE_LEVELS)
        bookings_made (int): number of customers who booked at that price

    Returns:
        float: revenue generated this step

    Examples:
        define_reward_function(200, 3) → 600  (3 bookings at ₹200)
        define_reward_function(50,  8) → 400  (8 bookings at ₹50)
        define_reward_function(250, 0) → 0    (nobody booked at ₹250)
    """
    reward = price * bookings_made

    return float(reward)


# Quick test
if __name__ == "__main__":
    print("=" * 50)
    print("REWARD FUNCTION TEST")
    print("=" * 50)
    test_cases = [(200, 3), (50, 8), (250, 0), (150, 5)]
    for price, bookings in test_cases:
        r = define_reward_function(price, bookings)
        print(f"Price={price}, Bookings={bookings} → Reward={r}")
    print("=" * 50)