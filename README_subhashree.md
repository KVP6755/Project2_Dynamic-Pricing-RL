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
