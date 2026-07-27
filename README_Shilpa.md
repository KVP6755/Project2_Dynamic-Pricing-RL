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



----------------------------------------------------------------------------------------------------------


### Week 2

### Team Role
**Member 2 – Q-Learning Agent Development**

## Author

**Shilpa S Nair**

---

## Task Description

Implemented the Q-Learning Agent for the Dynamic Pricing Reinforcement Learning project.

The agent interacts with the custom Gymnasium environment, selects pricing actions using the epsilon-greedy strategy, updates the Q-table based on rewards, and learns an optimal pricing policy through multiple training episodes.

---

## Responsibilities Completed

- Created the `QLearningAgent` class.
- Initialized the Q-table for all state-action pairs.
- Implemented the epsilon-greedy action selection policy.
- Implemented the Q-Learning update equation.
- Developed the multi-episode training loop.
- Added epsilon decay for balancing exploration and exploitation.
- Stored rewards obtained in each training episode.
- Generated a training summary including reward statistics.
- Separated testing logic into a dedicated test file.

---

## Files Implemented

```
rl_agent/
│
├── q_learning.py
└── test_q_learning.py
```

---

## Agent Details

### State

The agent observes the environment state as:

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

### Learning Strategy

The Q-Learning agent uses:

- Epsilon-Greedy Action Selection
- Q-Learning Update Equation
- Learning Rate
- Discount Factor
- Epsilon Decay

to learn the optimal pricing policy over multiple training episodes.

---

### Reward

The reward received from the environment is calculated as:

```
Reward = Bookings × Selected Price
```

The agent updates its Q-table using the received reward and the maximum future Q-value.

---

## Technologies Used

- Python
- NumPy
- Gymnasium
- Reinforcement Learning (Q-Learning)

---

## Testing

The Q-Learning agent was tested by:

- Initializing the agent.
- Resetting the environment.
- Selecting actions using the epsilon-greedy policy.
- Updating the Q-table after each action.
- Running multiple training episodes.
- Recording episode rewards.
- Displaying the training summary.
- Verifying the agent using a separate test file.

---

## Week 2 Status

| Task | Status |
|------|--------|
| Create QLearningAgent | ✅ Completed |
| Initialize Q-Table | ✅ Completed |
| Implement Epsilon-Greedy Policy | ✅ Completed |
| Implement Q-Learning Update | ✅ Completed |
| Develop Training Loop | ✅ Completed |
| Add Epsilon Decay | ✅ Completed |
| Store Reward History | ✅ Completed |
| Generate Training Summary | ✅ Completed |
| Separate Testing Module | ✅ Completed |



----------------------------------------------------------------------------------------------------------

### Week 3

### Team Role
**Member 1 – DQN Model Development**


---

## Task Description

Implemented the Deep Q-Network (DQN) model for the Dynamic Pricing Reinforcement Learning project.

The DQN agent uses a deep neural network to estimate Q-values for pricing actions, follows an epsilon-greedy strategy for action selection, and utilizes a target network to improve training stability.

---

## Responsibilities Completed

- Created the `DQNModel` class.
- Designed the Deep Q-Network (DQN) architecture.
- Implemented the epsilon-greedy action selection policy.
- Configured exploration parameters (epsilon, epsilon decay, epsilon minimum).
- Created a separate target network.
- Implemented target network synchronization.
- Tested model initialization and action selection.
- Verified target network updates.

---

## Files Implemented

```
dqn/
│
└── dqn_model.py
```

---

## Model Details

### State

The DQN agent receives the environment state as:

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

### Neural Network Architecture

The Deep Q-Network consists of:

- Input Layer (2 Features)
- Hidden Layer (64 Neurons, ReLU)
- Hidden Layer (64 Neurons, ReLU)
- Output Layer (5 Actions, Linear Activation)

---

### Exploration Strategy

The DQN agent uses the Epsilon-Greedy policy:

- Random action selection during exploration.
- Best predicted action during exploitation.
- Epsilon decay to gradually reduce exploration over time.

---

### Target Network

A separate Target Network is maintained to improve learning stability by periodically copying the weights from the main DQN network.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Deep Reinforcement Learning (DQN)

---

## Testing

The DQN implementation was tested by:

- Initializing the DQN model.
- Verifying the neural network architecture.
- Testing epsilon-greedy action selection.
- Synchronizing the target network.
- Validating successful execution of all implemented components.

---

## Week 3 Status

| Task | Status |
|------|--------|
| Create DQN Model | ✅ Completed |
| Design Neural Network | ✅ Completed |
| Implement Epsilon-Greedy Policy | ✅ Completed |
| Configure Exploration Parameters | ✅ Completed |
| Create Target Network | ✅ Completed |
| Synchronize Target Network | ✅ Completed |
| Model Testing | ✅ Completed |

---