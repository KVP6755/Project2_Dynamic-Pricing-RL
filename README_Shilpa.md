# README - Shilpa S Nair

## Project 2: Travel & Hospitality – Dynamic Pricing using Reinforcement Learning (RL)


### Week 1

### Team Role
**Member 2 – Custom Gymnasium Environment**

## Author

**Shilpa S Nair**


---

## Task Description

Implemented a custom Gymnasium environment (`PricingEnv`) for the Dynamic Pricing Reinforcement Learning project.

The environment simulates a travel/hospitality pricing scenario where an RL agent can interact with the environment by selecting different pricing levels.

---

## Responsibilities Completed

- Created a custom `PricingEnv` class using Gymnasium.
- Defined the action space with five pricing levels.
- Defined the observation space consisting of:
  - Remaining Inventory
  - Days Left
- Implemented the `reset()` method.
- Implemented the `step()` method.
- Implemented the `render()` method.
- Implemented the `close()` method.
- Tested the environment using sample actions.

---

## Files Implemented

```
gym_env/
│
├── pricing_env.py
└── test_env.py
```

---

## Environment Details

### State

The environment state is represented as:

```
[Remaining Inventory, Days Left]
```

Example:

```
[100, 30]
```

---

### Action Space

Five pricing levels are available:

| Action | Price (₹) |
|---------|-----------|
| 0 | 4000 |
| 1 | 4500 |
| 2 | 5000 |
| 3 | 5500 |
| 4 | 6000 |

---

### Reward

The reward is calculated as:

```
Reward = Bookings × Selected Price
```

(Currently uses placeholder booking logic for Week 1.)

---

## Technologies Used

- Python
- Gymnasium
- NumPy

---

## Testing

The environment was tested by:

- Resetting the environment.
- Selecting sample pricing actions.
- Updating inventory and remaining days.
- Displaying environment information using `render()`.
- Closing the environment successfully.

---

## Week 1 Status

| Task | Status |
|------|--------|
| Create PricingEnv | ✅ Completed |
| Implement reset() | ✅ Completed |
| Implement step() | ✅ Completed |
| Implement render() | ✅ Completed |
| Implement close() | ✅ Completed |
| Environment Testing | ✅ Completed |

---

