"""
Project 2 Week 3: DQN Training Pipeline

Author  : Varnika Valliammai V
File    : train_dqn.py
"""

import numpy as np
import pandas as pd
import random
import os
import sys

# Add project root to path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(project_root)

# Import teammates' modules
from replay_buffer import ReplayBuffer
from dqn.dqn_model import DQNModel

from mdp.mdp_design import (
    PRICE_LEVELS,
    TOTAL_INVENTORY,
    TOTAL_DAYS,
    simulate_one_step
)

# 1. Training Configuration

CONFIG = {
    'total_episodes': 500,
    'batch_size': 16,
    'gamma': 0.95,
    'learning_rate': 0.001,
    'buffer_capacity': 10000,
    'target_update_freq': 10,
    'random_state': 42
}

# Member 1 uses raw [inventory, days] as state
# so state_size = 2 (not encoded integer)
STATE_SIZE  = 2
N_ACTIONS   = len(PRICE_LEVELS)



# 2. State preparation

def prepare_state(inventory, days):
    """
    Converts inventory and days into a numpy array
    that Member 1's Keras model expects.

    Member 1 uses state_size=2, so state = [inventory, days]
    as a raw array — NOT encoded to a single integer.

    Parameters:
        inventory : rooms remaining (0-100)
        days      : days left (0-30)

    Returns:
        state_array : numpy array of shape (2,)
    """
    return np.array([inventory, days], dtype=np.float32)


# ─────────────────────────────────────────
# 3. DQN Training Step
# ─────────────────────────────────────────
def train_step(dqn, buffer, batch_size=32, gamma=0.95):
    """
    One training step using a mini-batch from the buffer.

    DQN Bellman Update:
        Q_target = reward + gamma × max(Q_target(next_state))
        Loss     = MSE(Q_main(state, action), Q_target)

    Uses Member 1's target_model for stable Q targets
    and main model for current Q predictions.

    Parameters:
        dqn        : DQNModel from Member 1
        buffer     : ReplayBuffer (your file)
        batch_size : mini-batch size
        gamma      : discount factor

    Returns:
        loss : training loss for this step
    """
    if not buffer.is_ready(batch_size):
        return None

    # Sample random mini-batch
    states, actions, rewards, next_states, dones = \
        buffer.sample(batch_size)

    # Predict current Q values for all states
    current_q = dqn.model.predict(states, verbose=0)

    # Predict next Q values using TARGET network (stable targets)
    next_q = dqn.target_model.predict(next_states, verbose=0)

    # Update Q values using Bellman equation
    for i in range(batch_size):
        if dones[i]:
            # Terminal state — no future reward
            current_q[i][actions[i]] = rewards[i]
        else:
            # Non-terminal — add discounted future reward
            current_q[i][actions[i]] = (
                rewards[i] + gamma * np.max(next_q[i])
            )

    # Train main network on updated Q values
    history = dqn.model.fit(
        states, current_q,
        epochs=1, verbose=0
    )

    return history.history['loss'][0]


# 4. Full Training Loop

def train_dqn(config=CONFIG):
    """
    Full DQN training pipeline using Member 1's model.

    Each episode:
    1. Reset to [100 rooms, 30 days]
    2. Agent picks action using epsilon-greedy
    3. Environment returns reward and next state
    4. Experience stored in replay buffer
    5. Mini-batch sampled and DQN updated
    6. Target network synced every N episodes

    Parameters:
        config : training configuration dictionary

    Returns:
        dqn              : trained DQNModel
        training_history : DataFrame of episode results
    """
    np.random.seed(config['random_state'])
    random.seed(config['random_state'])

    # Initialize Member 1's DQN model
    dqn = DQNModel(
        state_size    = STATE_SIZE,
        action_size   = N_ACTIONS,
        learning_rate = config['learning_rate']
    )

    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=config['buffer_capacity'])

    episode_rewards = []
    episode_losses  = []
    step_logs = []

    print("=" * 55)
    print(f"STARTING DQN TRAINING: {config['total_episodes']} episodes")
    print("=" * 55)

    for episode in range(config['total_episodes']):

        # Reset to start state
        inventory = TOTAL_INVENTORY
        days      = TOTAL_DAYS
        state     = prepare_state(inventory, days)

        episode_reward = 0
        episode_loss   = []
        done           = False

        while not done:

            # Member 1's epsilon-greedy action selection
            action = dqn.choose_action(state)

            # Step environment using Subhashree's MDP
            next_raw, reward, done, info = simulate_one_step(
                [inventory, days], action
            )
            next_inventory = next_raw[0]
            next_days      = next_raw[1]
            next_state     = prepare_state(next_inventory, next_days)
                       
            # Log this environment step
            step_logs.append({
            "episode": episode,
            "day": TOTAL_DAYS - days + 1,
            "inventory": inventory,
            "days_left": days,
            "action": action,
            "price": PRICE_LEVELS[action],
            "bookings": info["bookings_made"],
            "reward": reward
   })
             

            # Store experience in replay buffer
            buffer.push(state, action, reward, next_state, done)

            # Train on mini-batch
            loss = train_step(
                    dqn,
                    buffer,
                    batch_size=config['batch_size'],
                    gamma=config['gamma']
            )

            if loss is not None:
                episode_loss.append(loss)

                # Decay epsilon after every successful training step
                if dqn.epsilon > dqn.epsilon_min:
                   dqn.epsilon = max(
                          dqn.epsilon_min,
                          dqn.epsilon * dqn.epsilon_decay
                    )

            episode_reward += reward
            inventory = next_inventory
            days = next_days
            state = next_state
        # Update target network every N episodes
        if (episode + 1) % config['target_update_freq'] == 0:
            dqn.update_target_network()
            print(f"  [Episode {episode+1}] Target network updated.")

        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1:4d}/{config['total_episodes']} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(50): {avg_reward:8.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Eps: {dqn.epsilon:.3f}")

    print("\n✓ DQN Training complete.")

    # Build history DataFrame
    training_history = pd.DataFrame({
        'episode'          : range(config['total_episodes']),
        'total_reward'     : episode_rewards,
        'avg_loss'         : episode_losses,
        'cumulative_reward': np.cumsum(episode_rewards),
        'avg_reward_50'    : pd.Series(episode_rewards)
                             .rolling(50, min_periods=1).mean()
    })
    step_logs_df = pd.DataFrame(step_logs)

    return dqn, training_history, step_logs_df



# 5. Save and Load Model

def save_model(
    dqn,
    training_history,
    step_logs,
    model_path="dqn_trained_model.keras",
    history_path="dqn_training_history.csv",
    step_logs_path="step_logs.csv"
):

    dqn.model.save(model_path)

    training_history.to_csv(history_path, index=False)

    step_logs.to_csv(step_logs_path, index=False)

    print(f"Model saved   : {model_path}")
    print(f"History saved : {history_path}")
    print(f"Step logs saved : {step_logs_path}")


def load_model(model_path='dqn_trained_model.keras'):
    """
    Loads saved DQN model for evaluation.
    """
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model


# 6. Training Summary

def print_training_summary(training_history):
    """Prints DQN training performance summary."""
    print("\n" + "=" * 50)
    print("DQN TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total episodes      : {len(training_history)}")
    print(f"Best episode reward : ₹{training_history['total_reward'].max():,.2f}")
    print(f"Average reward      : ₹{training_history['total_reward'].mean():,.2f}")
    print(f"Final avg (last 50) : ₹{training_history['avg_reward_50'].iloc[-1]:,.2f}")
    print(f"Final avg loss      : {training_history['avg_loss'].iloc[-1]:.4f}")



# 7. Run everything

if __name__ == "__main__":

    # Train DQN
    dqn, training_history, step_logs_df = train_dqn(CONFIG)

    # Print summary
    print_training_summary(training_history)

    # Save for teammates
    save_model(dqn, training_history, step_logs_df)

    print("\n✓ train_dqn.py complete.")
