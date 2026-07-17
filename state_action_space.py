"""
================================================================================
state_action_space.py
================================================================================
Project     : Travel & Hospitality — Dynamic Pricing using RL
              Infotact Technical Internship — Project 2
Author      : Subhashree Behera
Week        : Week 2 — Baseline Strategies & Q-Learning Setup
Role        : State & Action Space Design

--------------------------------------------------------------------------------
What This File Does:
    This file is the bridge between the MDP design (Week 1) and the
    Q-Learning agent (Member 2). It translates the human-readable MDP
    specification into the exact mathematical structures Q-Learning needs.

    Specifically, Q-Learning requires:
    1. A way to convert state [inventory, days] → single integer index
       (because a Q-table is a 2D array, not a dict of lists)
    2. A validated action space with clear index → price mapping
    3. An initialized Q-table of the correct shape
    4. Helper functions for Member 2 to use cleanly

--------------------------------------------------------------------------------
Why State Encoding Matters:
    Q-Learning stores Q-values in a table:
        Q[state_index][action_index] = expected_cumulative_reward

    Our state is 2D: [remaining_inventory (0-100), days_left (0-30)]
    We need a single integer to index the Q-table row.

    Encoding formula:
        state_index = inventory × (MAX_DAYS + 1) + days_left

    This maps every unique (inventory, days) pair to a unique integer:
        [100, 30] → 3130  (start state)
        [0, 0]   → 0     (terminal state)
        [50, 15] → 1565  (middle of episode)

    Total unique states: 101 × 31 = 3,131
    Total Q-table size : 3,131 states × 5 actions = 15,655 values

--------------------------------------------------------------------------------
Why Action Encoding Matters:
    Actions are price levels [50, 100, 150, 200, 250].
    Q-Learning works with integer indices [0, 1, 2, 3, 4].
    This file provides clean conversion between them.

--------------------------------------------------------------------------------
Functions:
    encode_state()         → [inventory, days] → single integer
    decode_state()         → single integer → [inventory, days]
    get_action_index()     → price value → action index
    get_price_from_index() → action index → price value
    initialize_q_table()   → creates zero-initialized Q-table
    validate_state()       → checks state is within valid bounds
    validate_action()      → checks action index is valid
    get_state_space_info() → prints complete state space summary
    get_action_space_info()→ prints complete action space summary

--------------------------------------------------------------------------------
Q-Table Structure (for Member 2):
    Shape  : (3131, 5)
    Rows   : state indices (0 to 3130)
    Columns: action indices (0 to 4)
    Values : Q-values, initialized to 0.0

    Access pattern:
        q_value = Q[encode_state(inventory, days)][action_index]

--------------------------------------------------------------------------------
Handoff to Member 2 (Q-Learning Agent):
    from state_action_space import (
        encode_state, decode_state,
        get_price_from_index, initialize_q_table,
        TOTAL_STATES, N_ACTIONS
    )
================================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================
# Importing from Week 1 MDP design — these MUST stay consistent
# across all team members' files

# State space bounds
MAX_INVENTORY  = 100
MAX_DAYS       = 30
TOTAL_INVENTORY = 100
TOTAL_DAYS      = 30

# Action space — price levels
PRICE_LEVELS = [50, 100, 150, 200, 250]
N_ACTIONS    = len(PRICE_LEVELS)

# Derived constants (computed, not hardcoded)
TOTAL_STATES  = (MAX_INVENTORY + 1) * (MAX_DAYS + 1)  # 3131
Q_TABLE_SHAPE = (TOTAL_STATES, N_ACTIONS)              # (3131, 5)

print("State & Action Space constants loaded:")
print(f"  Total states  : {TOTAL_STATES}")
print(f"  Total actions : {N_ACTIONS}")
print(f"  Q-Table shape : {Q_TABLE_SHAPE}")
print(f"  Q-Table size  : {TOTAL_STATES * N_ACTIONS} values")