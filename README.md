# Project2_Dynamic-Pricing-RL
Travel &amp; Hospitality Dynamic Pricing using Reinforcement Learning (RL) - Infotact Solution Internship Project
# Project 2: Dynamic Pricing using Reinforcement Learning

This repository implements a Reinforcement Learning (RL) framework using Q-Learning to optimize dynamic pricing strategies, maximizing total revenue against a traditional static baseline.

## 🚀 Performance Summary

The agent was evaluated over extended training episodes. It successfully learned an optimal pricing policy that consistently outperforms the fixed rule-based strategy.

| Metric | Value |
| :--- | :--- |
| **Average Reward** | 57.26 |
| **Maximum Reward** | 99.00 |
| **Average RL Revenue** | **5353.47** |
| **Baseline Revenue** | 5000.00 |

> **Result:** The Q-Learning dynamic pricing agent achieved a **7.07% revenue increase** over the baseline pricing model.

---

## 📁 Repository Structure

```text
Project2_Dynamic-Pricing-RL/
│
└── src/
    ├── baseline_pricing.ipynb   # Rule-based static pricing logic & baseline setup
    └── training_analysis.ipynb   # Q-Learning pipeline, visualization, & metric evaluation
```

---

## 🛠️ Implementation Details

### 1. Baseline Pricing Strategy
* Located in `src/baseline_pricing.ipynb`.
* Establishes a constant or rule-based market pricing mechanism.
* Outputs a steady baseline revenue benchmark of **5000** units for evaluation comparison.

### 2. Q-Learning Training & Analysis
* Located in `src/training_analysis.ipynb`.
* Implements the RL environment interaction loop, exploration vs. exploitation trade-offs, and Q-table updates.
* **Visualizations included:**
  * **Episode vs. Reward:** Track agent learning convergence.
  * **Revenue Comparison:** Real-time line chart mapping Q-learning performance fluctuations directly against the red baseline marker.

---

## 💻 How to Run

1. Activate your virtual environment:
   ```bash
   # Windows
   .venv\Scripts\activate
   ```
2. Navigate to the codebase folder and open the notebooks via VS Code or Jupyter Labs:
   ```bash
   cd src
   ```
3. Execute `baseline_pricing.ipynb` first to register benchmarks, followed by `training_analysis.ipynb` to train the reinforcement learning model.
