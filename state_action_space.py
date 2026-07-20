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

# ============================================================
# FUNCTION 1: encode_state
# ============================================================

def encode_state(inventory, days):
    """
    Convert 2D state [inventory, days] to single integer index.

    This is the CORE function of this file. Q-Learning needs a
    single integer to index into the Q-table row. Since our state
    has two dimensions, we flatten it using this formula:

        index = inventory × (MAX_DAYS + 1) + days

    Why this formula?
        Think of it like converting row/column to a flat array index.
        Each inventory level "owns" (MAX_DAYS + 1) = 31 indices —
        one for each possible day count.

        inventory=0, days=0  → 0  × 31 + 0  = 0
        inventory=0, days=30 → 0  × 31 + 30 = 30
        inventory=1, days=0  → 1  × 31 + 0  = 31
        inventory=100, days=30 → 100 × 31 + 30 = 3130

    Args:
        inventory (int): remaining inventory (0 to 100)
        days      (int): days until departure (0 to 30)

    Returns:
        int: unique state index (0 to 3130)
    """
    return int(inventory * (MAX_DAYS + 1) + days)


# ============================================================
# FUNCTION 2: decode_state
# ============================================================

def decode_state(index):
    """
    Convert single integer index back to [inventory, days].

    Reverse of encode_state() — used for debugging and logging
    to show human-readable state from Q-table index.

    Args:
        index (int): state index (0 to 3130)

    Returns:
        tuple: (inventory, days)
    """
    inventory = index // (MAX_DAYS + 1)
    days      = index  % (MAX_DAYS + 1)
    return int(inventory), int(days)


if __name__ == "__main__":
    print("=" * 55)
    print("STATE ENCODING VERIFICATION")
    print("=" * 55)
    test_states = [
        [100, 30],   # start state
        [50,  15],   # mid episode
        [0,   0],    # terminal state
        [1,   1],
        [99,  29]
    ]
    print(f"{'State':<15} {'Index':<8} {'Decoded':<15} {'Match'}")
    print("-" * 55)
    for inv, d in test_states:
        idx       = encode_state(inv, d)
        inv2, d2  = decode_state(idx)
        match     = "✓" if (inv == inv2 and d == d2) else "✗"
        print(f"[{inv}, {d}]"
              f"{'':>{13-len(str(inv))-len(str(d))}}"
              f"{idx:<8} [{inv2}, {d2}]"
              f"{'':>{12-len(str(inv2))-len(str(d2))}}"
              f"{match}")
    print("=" * 55)
    print(f"Start state    [100, 30] → index: {encode_state(100, 30)}")
    print(f"Terminal state [0, 0]    → index: {encode_state(0, 0)}")

# ============================================================
# FUNCTION 3: get_action_index
# ============================================================

def get_action_index(price):
    """
    Convert a price value to its action index.

    Q-Learning works with integer indices [0,1,2,3,4], not
    price values. This function translates cleanly between them.

    Args:
        price (int): price value from PRICE_LEVELS

    Returns:
        int: action index (0 to 4)

    Raises:
        ValueError: if price not in PRICE_LEVELS

    Example:
        get_action_index(50)  → 0
        get_action_index(150) → 2
        get_action_index(250) → 4
    """
    if price not in PRICE_LEVELS:
        raise ValueError(
            f"Price {price} not in PRICE_LEVELS {PRICE_LEVELS}"
        )
    return PRICE_LEVELS.index(price)


# ============================================================
# FUNCTION 4: get_price_from_index
# ============================================================

def get_price_from_index(action_index):
    """
    Convert action index to its price value.

    Reverse of get_action_index(). Used by Q-Learning agent
    when it needs to take a price action based on Q-table output.

    Args:
        action_index (int): index 0 to 4

    Returns:
        int: price value

    Raises:
        ValueError: if index out of range

    Example:
        get_price_from_index(0) → 50
        get_price_from_index(2) → 150
        get_price_from_index(4) → 250
    """
    if action_index < 0 or action_index >= N_ACTIONS:
        raise ValueError(
            f"action_index {action_index} out of range "
            f"[0, {N_ACTIONS-1}]"
        )
    return PRICE_LEVELS[action_index]


if __name__ == "__main__":
    print("=" * 45)
    print("ACTION SPACE ENCODING VERIFICATION")
    print("=" * 45)
    print(f"{'Index':<8} {'Price':<8} {'Round-trip'}")
    print("-" * 45)
    for i, price in enumerate(PRICE_LEVELS):
        idx        = get_action_index(price)
        price_back = get_price_from_index(idx)
        match      = "✓" if price == price_back else "✗"
        print(f"{i:<8} ₹{price:<7} {match}")
    print("=" * 45)

# ============================================================
# FUNCTION 5: validate_state
# ============================================================

def validate_state(inventory, days):
    """
    Check if a state [inventory, days] is within valid bounds.

    Used as a safety check in the Q-Learning training loop to
    catch invalid states before they cause IndexError on Q-table.

    Args:
        inventory (int): remaining inventory
        days      (int): days remaining

    Returns:
        bool: True if valid, False if out of bounds

    Example:
        validate_state(50, 15) → True
        validate_state(-1, 15) → False  (negative inventory)
        validate_state(50, 35) → False  (days > MAX_DAYS)
    """
    valid = (
        0 <= inventory <= MAX_INVENTORY and
        0 <= days      <= MAX_DAYS
    )

    if not valid:
        print(f"Invalid state: inventory={inventory}, days={days}")
        print(f"  inventory must be 0 to {MAX_INVENTORY}")
        print(f"  days must be 0 to {MAX_DAYS}")

    return valid


# ============================================================
# FUNCTION 6: validate_action
# ============================================================

def validate_action(action_index):
    """
    Check if an action index is within valid range.

    Args:
        action_index (int): index to validate

    Returns:
        bool: True if valid, False if out of range
    """
    valid = 0 <= action_index < N_ACTIONS

    if not valid:
        print(f"Invalid action: {action_index}")
        print(f"  Must be 0 to {N_ACTIONS - 1}")

    return valid


if __name__ == "__main__":
    print("=" * 45)
    print("VALIDATION TESTS")
    print("=" * 45)
    print("State validation:")
    tests = [(50, 15, True), (100, 30, True),
             (-1, 15, False), (50, 35, False), (101, 10, False)]
    for inv, d, expected in tests:
        result = validate_state(inv, d)
        match  = "✓" if result == expected else "✗"
        print(f"  [{inv}, {d}] → {result} {match}")

    print("\nAction validation:")
    for a, expected in [(0, True),(4, True),(5, False),(-1, False)]:
        result = validate_action(a)
        match  = "✓" if result == expected else "✗"
        print(f"  action={a} → {result} {match}")


# ============================================================
# FUNCTION 7: initialize_q_table
# ============================================================

def initialize_q_table(init_value=0.0):
    """
    Create and return a zero-initialized Q-table.

    The Q-table is the core data structure of Q-Learning:
        Q[state_index][action_index] = expected_cumulative_reward

    Shape: (3131, 5) — 3131 states × 5 price actions

    Why initialize to 0?
        Starting at 0 is the standard approach — it means the agent
        initially assumes every (state, action) pair gives 0 reward,
        then updates values as it explores and gains experience.

        Alternative: initialize to a small positive value (optimistic
        initialization) to encourage early exploration — but 0 is
        simpler and works well for this problem.

    Args:
        init_value (float): initial Q-value (default 0.0)

    Returns:
        np.ndarray: Q-table of shape (3131, 5)

    Usage (Member 2):
        Q = initialize_q_table()
        q_value = Q[encode_state(50, 15)][get_action_index(150)]
        Q[encode_state(50, 15)][2] = 750.0  # after update
    """
    Q = np.full(Q_TABLE_SHAPE, fill_value=init_value, dtype=np.float64)

    print("=" * 50)
    print("Q-TABLE INITIALIZED")
    print("=" * 50)
    print(f"Shape         : {Q.shape}")
    print(f"Total values  : {Q.size}")
    print(f"Initial value : {init_value}")
    print(f"Memory usage  : {Q.nbytes / 1024:.2f} KB")
    print(f"dtype         : {Q.dtype}")
    print("=" * 50)

    return Q


if __name__ == "__main__":
    Q = initialize_q_table()

    # Show a sample lookup
    inv, days = 100, 30
    state_idx  = encode_state(inv, days)
    action_idx = get_action_index(150)
    print(f"\nSample Q-table lookup:")
    print(f"  State  : [{inv}, {days}] → index {state_idx}")
    print(f"  Action : ₹150 → index {action_idx}")
    print(f"  Q-value: {Q[state_idx][action_idx]}")

# ============================================================
# FUNCTION 8: get_state_space_info
# ============================================================

def get_state_space_info():
    """
    Print a complete summary of the state space.

    Used for documentation and team review. Shows the full
    mathematical specification of the state space.

    Returns:
        dict: state space specification
    """
    info = {
        'dimensions'   : 2,
        'variables'    : ['remaining_inventory', 'days_until_departure'],
        'inventory'    : {'min': 0, 'max': MAX_INVENTORY},
        'days'         : {'min': 0, 'max': MAX_DAYS},
        'total_states' : TOTAL_STATES,
        'start_state'  : [TOTAL_INVENTORY, TOTAL_DAYS],
        'start_index'  : encode_state(TOTAL_INVENTORY, TOTAL_DAYS),
        'terminal_conditions': [
            'days_until_departure == 0',
            'remaining_inventory  == 0'
        ],
        'encoding_formula': 'index = inventory × 31 + days'
    }

    print("\n" + "=" * 55)
    print("STATE SPACE SPECIFICATION")
    print("=" * 55)
    print(f"Dimensions     : {info['dimensions']}D → encoded to 1D")
    print(f"Variables      : {info['variables']}")
    print(f"Inventory      : 0 to {info['inventory']['max']}")
    print(f"Days           : 0 to {info['days']['max']}")
    print(f"Total states   : {info['total_states']}")
    print(f"Start state    : {info['start_state']} "
          f"→ index {info['start_index']}")
    print(f"Terminal when  : {info['terminal_conditions']}")
    print(f"Encoding       : {info['encoding_formula']}")
    print("=" * 55)

    return info


# ============================================================
# FUNCTION 9: get_action_space_info
# ============================================================

def get_action_space_info():
    """
    Print a complete summary of the action space.

    Returns:
        dict: action space specification
    """
    info = {
        'type'         : 'Discrete',
        'n_actions'    : N_ACTIONS,
        'price_levels' : PRICE_LEVELS,
        'index_map'    : {i: p for i, p in enumerate(PRICE_LEVELS)},
        'min_price'    : min(PRICE_LEVELS),
        'max_price'    : max(PRICE_LEVELS)
    }

    print("\n" + "=" * 55)
    print("ACTION SPACE SPECIFICATION")
    print("=" * 55)
    print(f"Type           : {info['type']}")
    print(f"Total actions  : {info['n_actions']}")
    print(f"Price levels   : {info['price_levels']}")
    print(f"Index mapping  :")
    for idx, price in info['index_map'].items():
        print(f"    {idx} → ₹{price}")
    print("=" * 55)

    return info