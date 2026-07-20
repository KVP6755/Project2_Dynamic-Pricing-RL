# Project 2 Contributions — Subhashree Behera

## Week 1 — MDP Design & Documentation

### Role: Member 1 — \\MDP Design & Documentation

### Files Owned
- `mdp/mdp_design.py` — complete MDP implementation
- `mdp/mdp_documentation.md` — written spec for team

### MDP Summary
| Component | Specification |
|---|---|
| State | [remaining_inventory (0-100), days_until_departure (0-30)] |
| State Space | 3,131 total states |
| Actions | 5 price levels: ₹50, ₹100, ₹150, ₹200, ₹250 |
| Reward | price × bookings_made per step |
| Episode | 30 days, 100 rooms, start=[100,30] |
| Terminal | days=0 OR inventory=0 |

### Functions Built
| Function | Purpose |
|---|---|
| `define_state_space()` | Documents/validates 3131-state space |
| `define_action_space()` | Documents 5 price level actions |
| `define_reward_function()` | Calculates step revenue |
| `define_demand_function()` | Stochastic booking probability |
| `simulate_one_step()` | One MDP transition (for Gym step()) |
| `simulate_episode()` | Full 30-day episode simulation |
| `summarize_mdp()` | Complete spec report for team |

### Baseline Episode Results
- Random pricing: ~₹3,200 revenue
- Fixed ₹250 (high): ~₹1,500 revenue
- Fixed ₹50 (low): ~₹2,000 revenue


---

## Week 2 Contributions — State & Action Space Design

### File Owned
`state_action_space.py`

### Responsibility
Translate the Week 1 MDP design into exact mathematical
structures Q-Learning requires — state encoding, action
encoding, Q-table initialization, and validation utilities.

### Why State Encoding is Needed
Q-Learning uses a Q-table indexed by integers. Our state is
2D: [inventory (0-100), days (0-30)]. We encode it to a
single integer:

`index = inventory × 31 + days`

This maps all 3,131 unique states to indices 0-3130.

### Functions Built
| Function | Purpose |
|---|---|
| `encode_state()` | [inventory, days] → single Q-table index |
| `decode_state()` | Q-table index → [inventory, days] |
| `get_action_index()` | Price value → action index |
| `get_price_from_index()` | Action index → price value |
| `validate_state()` | Bounds check before Q-table access |
| `validate_action()` | Range check on action index |
| `initialize_q_table()` | Creates (3131, 5) zero Q-table |
| `get_state_space_info()` | Full state space summary |
| `get_action_space_info()` | Full action space summary |

### Q-Table Specification
| Property | Value |
|---|---|
| Shape | (3131, 5) |
| Rows | State indices (0 to 3130) |
| Columns | Action indices (0 to 4) |
| Values | Q-values, init = 0.0 |
| Memory | 122.30 KB |

### Encoding Verification
| State | Index | Decoded | Match |
|---|---|---|---|
| [100, 30] | 3130 | [100, 30] | ✓ |
| [50, 15] | 1565 | [50, 15] | ✓ |
| [0, 0] | 0 | [0, 0] | ✓ |

### Handoff to Member 2
```python
from state_action_space import (
    encode_state, decode_state,
    get_price_from_index, initialize_q_table,
    TOTAL_STATES, N_ACTIONS
)
```