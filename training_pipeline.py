"""
Project 2 Week 2: Environment Integration & Training
=====================================================
Member 3: Integrates Q-Learning agent with PricingEnv,
runs training for multiple episodes, and stores rewards
and training history.

Author  : Varnika Valliammai V
File    : training_pipeline.py
"""

import numpy as np
import pandas as pd

# Import MDP constants from Member 1 (Week 1)
from mdp_design import (
    PRICE_LEVELS,
    TOTAL_INVENTORY,
    TOTAL_DAYS,
    simulate_one_step
)

# ─────────────────────────────────────────
# 1. State Encoding
# ─────────────────────────────────────────
def encode_state(inventory, days):
    """
    Converts [inventory, days] into a single integer index
    for Q-table lookup.

    Q-table needs one row per state — we flatten the 2D
    state [inventory, days] into one unique integer.

    Parameters:
        inventory : rooms remaining (0-100)
        days      : days left (0-30)

    Returns:
        state_index : unique integer (0 to 3130)
    """
    return int(inventory * (TOTAL_DAYS + 1) + days)


# ─────────────────────────────────────────
# 2. Training Configuration
# ─────────────────────────────────────────
TOTAL_EPISODES = 500
RANDOM_STATE   = 42


# ─────────────────────────────────────────
# 3. Single Episode Runner
# ─────────────────────────────────────────
def run_episode(agent, episode_num, training=True):
    """
    Runs one complete 30-day booking episode.

    Parameters:
        agent       : QLearningAgent from Member 2
        episode_num : current episode number
        training    : if True agent learns, if False agent exploits

    Returns:
        episode_reward  : total revenue this episode
        episode_history : list of daily step logs
    """
    # Start state: full inventory, full days
    inventory = TOTAL_INVENTORY
    days      = TOTAL_DAYS
    state     = encode_state(inventory, days)

    episode_reward  = 0
    episode_history = []
    done            = False
    day             = 0

    while not done:

        # Agent picks action (price index 0-4)
        if training:
            action = agent.choose_action(state)
        else:
            action = agent.best_action(state)


        next_state_raw, reward, done, info = simulate_one_step(
            [inventory, days], action
        )

        next_inventory = next_state_raw[0]
        next_days      = next_state_raw[1]
        next_state     = encode_state(next_inventory, next_days)

        # Agent updates Q-table (training only)
        if training:
            agent.update(
                state      = state,
                action     = action,
                reward     = reward,
                next_state = next_state,
                done       = done
            )

        # Log this day
        episode_history.append({
            'episode'   : episode_num,
            'day'       : day + 1,
            'inventory' : inventory,
            'days_left' : days,
            'action'    : action,
            'price'     : PRICE_LEVELS[action],
            'bookings'  : info['bookings_made'],
            'reward'    : reward
        })

        episode_reward += reward
        inventory       = next_inventory
        days            = next_days
        state           = next_state
        day            += 1

    return episode_reward, episode_history

# 4. Full Training Loop
def train_agent(agent, total_episodes=TOTAL_EPISODES):
    """
    Runs the full training loop across multiple episodes.

    Parameters:
        agent          : QLearningAgent from Member 2
        total_episodes : number of training episodes

    Returns:
        training_history : DataFrame with per-episode results
        all_step_logs    : DataFrame with per-day step details
    """
    episode_rewards = []
    all_step_logs   = []

    print("=" * 50)
    print(f"STARTING TRAINING: {total_episodes} episodes")
    print("=" * 50)

    for episode in range(total_episodes):
        episode_reward, episode_history = run_episode(
            agent, episode_num=episode, training=True
        )

        episode_rewards.append(episode_reward)
        all_step_logs.extend(episode_history)

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1:4d}/{total_episodes} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(last 50): {avg_reward:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\n✓ Training complete.")

    # Build summary DataFrame
    training_history = pd.DataFrame({
        'episode'          : range(total_episodes),
        'total_reward'     : episode_rewards,
        'cumulative_reward': np.cumsum(episode_rewards),
        'avg_reward_50'    : pd.Series(episode_rewards)
                             .rolling(50, min_periods=1).mean()
    })

    all_step_logs_df = pd.DataFrame(all_step_logs)

    return training_history, all_step_logs_df



# 5. Save results 

def save_training_results(training_history, step_logs,
                          history_path='training_history.csv',
                          steps_path='step_logs.csv'):
  
    training_history.to_csv(history_path, index=False)
    step_logs.to_csv(steps_path, index=False)
    print(f"Training history saved : {history_path}")
    print(f"Step logs saved        : {steps_path}")


# 6. Training summary

def print_training_summary(training_history):
    """Prints a summary of training performance."""
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total episodes      : {len(training_history)}")
    print(f"Best episode reward : ₹{training_history['total_reward'].max():,.2f}")
    print(f"Worst episode reward: ₹{training_history['total_reward'].min():,.2f}")
    print(f"Average reward      : ₹{training_history['total_reward'].mean():,.2f}")
    print(f"Final avg (last 50) : ₹{training_history['avg_reward_50'].iloc[-1]:,.2f}")
    print(f"Total revenue earned: ₹{training_history['cumulative_reward'].iloc[-1]:,.2f}")



# 7. Run everything

if __name__ == "__main__":

    # Import Member 2's Q-Learning agent
    from q_learning_agent import QLearningAgent

    np.random.seed(RANDOM_STATE)

    # Total states = 101 inventory levels × 31 day levels
    n_states  = (TOTAL_INVENTORY + 1) * (TOTAL_DAYS + 1)
    n_actions = len(PRICE_LEVELS)

    # Initialize Q-Learning agent
    agent = QLearningAgent(
        n_states      = n_states,
        n_actions     = n_actions,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_decay = 0.995,
        epsilon_min   = 0.01
    )

    # Train
    training_history, step_logs = train_agent(
        agent, total_episodes=TOTAL_EPISODES
    )

    # Summary
    print_training_summary(training_history)

    # Save 
    save_training_results(training_history, step_logs)

    print("\n✓ training_pipeline.py complete.")
 
