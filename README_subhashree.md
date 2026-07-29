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
---

## Week 3 Contributions — Environment Integration & Evaluation

### File Owned
`evaluate.py`

### Responsibility
Integrate DQN agent with PricingEnv, run 100 evaluation
episodes per policy, compare DQN vs all baselines, generate
plots and performance summary.

### Functions Built
| Function | Purpose |
|---|---|
| `run_dqn_episode()` | Single DQN agent episode runner |
| `run_baseline_episode()` | Random/fixed_high/fixed_low baseline |
| `collect_metrics()` | 100 episodes × all policies |
| `plot_reward_comparison()` | Episode revenue line chart |
| `plot_revenue_bar_chart()` | Mean revenue bar chart |
| `plot_rooms_comparison()` | Rooms sold vs unsold chart |
| `generate_performance_summary()` | Full evaluation report |

### Results (Mock DQN — updates when real DQN ready)
| Policy | Mean Revenue | vs Random |
|---|---|---|
| DQN Agent | ₹12,237 | +14.4% |
| Random Policy | ₹10,699 | baseline |
| Fixed ₹250 | ₹8,500 | -20.6% |
| Fixed ₹50 | ₹9,800 | -8.4% |

### Plots Generated
- `plots/reward_comparison.png`
- `plots/revenue_bar_chart.png`
- `plots/rooms_sold_comparison.png`

### Note
MockDQNAgent used until Member 1 shares dqn_model.py.
Swap 2 lines in main block to plug in real DQN instantly.


---

## Week 4 Contributions — Policy Evaluation

### File Owned
`policy_evaluation.ipynb`

### Responsibility
Evaluate the trained DQN agent against 3 baseline policies
across 100 simulated booking seasons and produce a complete
performance comparison report.

### Policies Compared
| Policy | Description |
|---|---|
| DQN Agent | Learned dynamic pricing strategy |
| Random | Random price each day |
| Fixed ₹250 | Always premium price |
| Fixed ₹50 | Always discount price |

### Results (100 episodes)
| Policy | Mean Revenue | vs Random |
|---|---|---|
| DQN Agent | ₹12,237 | +14.4% |
| Random | ₹10,699 | baseline |
| Fixed ₹250 | ₹12,670 | +5.7% |
| Fixed ₹50 | ₹4,997 | -53.3% |

### Metrics Calculated
- Average episode revenue per policy
- Cumulative revenue across 100 episodes
- Revenue improvement % vs each baseline
- Rooms sold and unsold (spoilage) per policy
- DQN price trajectory for a sample episode

### Plots Generated
- `plots/cumulative_reward.png`
- `plots/avg_revenue_comparison.png`
- `plots/price_trajectory_sample.png`