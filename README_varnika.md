# Project 2: Travel & Hospitality – Dynamic Pricing using Reinforcement Learning

## Author
**Varnika Valliammai V**

## Branch
`varnika`

---

# Overview

This repository contains my contributions to **Project 2 – Travel & Hospitality: Dynamic Pricing using Reinforcement Learning (RL)** completed as part of the Infotact Technical Internship.

The objective of this project is to develop an intelligent pricing agent that learns optimal room/seat pricing strategies over a 30-day booking horizon using Reinforcement Learning techniques.

---

# My Contributions

## Week 1 – Market Simulation 

Implemented the business simulation components required for the RL environment.

### Tasks Completed

- Inventory Management
- Customer Demand Simulation
- Revenue Calculation
- Booking Probability Logic

### Features

- Simulated hotel/airline inventory updates
- Implemented stochastic customer booking behavior
- Calculated revenue using:

```
Reward = Price × Bookings
```

- Added realistic booking probability based on:
  - Price sensitivity
  - Remaining booking days
  - Customer urgency

---

## Week 2 – Environment Integration & Training

Integrated the Q-Learning agent with the pricing environment.

### Tasks Completed

- Connected Q-Learning agent with Pricing Environment
- Executed multi-episode training
- Stored episode rewards
- Generated training history

### Features

- Multi-episode Q-Learning training
- Reward tracking
- Training history generation
- Performance data collection

---

## Week 3 – Experience Replay & DQN Training Pipeline 

Implemented the Deep Q-Network (DQN) training pipeline.

### Files

- `replay_buffer.py`
- `train_dqn.py`

### Tasks Completed

- Experience Replay Buffer
- Mini-batch Sampling
- DQN Bellman Update
- Training Pipeline
- Target Network Synchronization
- Model Saving
- Training History Saving
- Step Log Generation

### Features

- Replay buffer implementation
- Random mini-batch sampling
- Bellman Q-value updates
- Epsilon decay
- Target network updates
- Model checkpoint saving
- CSV training history generation
- Environment step logging

### Output Files

```
dqn_trained_model.keras
dqn_training_history.csv
step_logs.csv
```

---

## Week 4 – Business Dashboard 

Developed an interactive dashboard to visualize DQN training performance.

### File

```
business_dashboard.ipynb
```

### Dashboard Visualizations

- Revenue Trend
- Occupancy Rate
- Reward Trend
- Pricing Distribution

### Technologies Used

- Plotly
- Matplotlib
- Pandas
- NumPy

The dashboard loads:

- `dqn_training_history.csv`
- `step_logs.csv`

to generate business analytics and performance visualizations.

---

# Project Workflow

```
Customer Demand
        │
        ▼
Pricing Environment
        │
        ▼
Replay Buffer
        │
        ▼
DQN Training
        │
        ▼
Training History
        │
        ▼
Business Dashboard
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Plotly
- Reinforcement Learning (Q-Learning & DQN)

---

# Repository Outputs

After training, the following files are generated:

```
dqn_trained_model.keras
dqn_training_history.csv
step_logs.csv
```

These outputs are used for model evaluation and dashboard visualization.

---

# Learning Outcomes

Through this project I gained practical experience in:

- Reinforcement Learning
- Deep Q Networks (DQN)
- Experience Replay
- Target Networks
- Q-Learning Integration
- Reward Optimization
- Business KPI Analysis
- Interactive Dashboard Development
- Data Visualization using Plotly

---

# Acknowledgement

This work was completed as part of the **Infotact Technical Internship – Project 2: Travel & Hospitality Dynamic Pricing using Reinforcement Learning**.
