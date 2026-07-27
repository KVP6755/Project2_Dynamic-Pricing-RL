"""
Implements the Experience Replay Buffer
for DQN training. Stores past experiences and samples
random mini-batches for stable training.

Author  : Varnika Valliammai V
File    : replay_buffer.py
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Fixed-size buffer that stores past (state, action,
    reward, next_state, done) experience tuples.

    New experiences overwrite oldest ones when full
    (circular buffer using deque with maxlen).
    """

    def __init__(self, capacity=10000):
        """
        Parameters:
            capacity : maximum number of experiences to store
                       (older ones get overwritten when full)
        """
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Adds one experience tuple to the buffer.

        Parameters:
            state      : current state (encoded integer)
            action     : action taken (price index 0-4)
            reward     : reward received (revenue)
            next_state : next state after action
            done       : True if episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size=32):
        """
        Randomly samples a mini-batch of experiences.

        Random sampling breaks the correlation between
        consecutive experiences — critical for stable DQN.

        Parameters:
            batch_size : number of experiences to sample

        Returns:
            states, actions, rewards, next_states, dones
            as separate numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)

        states      = np.array([e[0] for e in batch])
        actions     = np.array([e[1] for e in batch])
        rewards     = np.array([e[2] for e in batch],
                               dtype=np.float32)
        next_states = np.array([e[3] for e in batch])
        dones       = np.array([e[4] for e in batch],
                               dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns current number of stored experiences."""
        return len(self.buffer)

    def is_ready(self, batch_size=32):
        """
        Returns True if buffer has enough experiences
        to start training (need at least batch_size).
        """
        return len(self.buffer) >= batch_size

    def summary(self):
        """Prints buffer status."""
        print(f"Replay Buffer: {len(self.buffer)}"
              f"/{self.capacity} experiences stored")

# Quick test
if __name__ == "__main__":
    print("=" * 50)
    print("REPLAY BUFFER TEST")
    print("=" * 50)

    buffer = ReplayBuffer(capacity=1000)

    # Push fake experiences
    for i in range(100):
        state      = np.random.randint(0, 3131)
        action     = np.random.randint(0, 5)
        reward     = np.random.uniform(0, 500)
        next_state = np.random.randint(0, 3131)
        done       = bool(np.random.randint(0, 2))
        buffer.push(state, action, reward, next_state, done)

    buffer.summary()
    print(f"Ready to train: {buffer.is_ready(batch_size=32)}")

    # Sample a batch
    states, actions, rewards, next_states, dones = \
        buffer.sample(batch_size=32)

    print(f"\nSampled batch shapes:")
    print(f"  states      : {states.shape}")
    print(f"  actions     : {actions.shape}")
    print(f"  rewards     : {rewards.shape}")
    print(f"  next_states : {next_states.shape}")
    print(f"  dones       : {dones.shape}")