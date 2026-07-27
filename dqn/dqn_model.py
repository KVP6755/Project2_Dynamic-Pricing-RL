# ============================================================
# Day 1 - DQN Neural Network
# Member 1
# ============================================================

import os
import sys
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.layers import Input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(project_root)


class DQNModel:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Exploration Parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = self.build_model()

        # Create Target Network
        self.target_model = self.build_model()

        # Copy weights from main model
        self.update_target_network()

    # --------------------------------------------------------
    # Build Deep Neural Network
    # --------------------------------------------------------

    def build_model(self):

        # model = Sequential()

        # model.add(
        #     Dense(
        #         64,
        #         input_dim=self.state_size,
        #         activation="relu"
        #     )
        # )

        
        model = Sequential()

        model.add(Input(shape=(self.state_size,)))

        model.add(Dense(
        64,
        activation="relu"
    ))

        model.add(Dense(
        64,
        activation="relu"
    ))

        model.add(Dense(
        self.action_size,
        activation="linear"
    ))

        model.add(
            Dense(
                64,
                activation="relu"
            )
        )

        model.add(
            Dense(
                self.action_size,
                activation="linear"
            )
        )

        model.compile(
            loss="mse",
            optimizer=Adam(
                learning_rate=self.learning_rate
            )
        )

        return model
    # --------------------------------------------------------
# Choose Action using Epsilon-Greedy Policy
# --------------------------------------------------------
    def choose_action(self, state):

        # Generate a random number
        random_number = random.uniform(0, 1)

        # Exploration
        if random_number < self.epsilon:

            action = random.randrange(self.action_size)

            print("\n===== Exploration =====")
            print("Random Number :", round(random_number, 4))
            print("Random Action :", action)

        # Exploitation
        else:

            state = np.reshape(state, (1, self.state_size))

            q_values = self.model.predict(
                state,
                verbose=0
            )

            action = np.argmax(q_values[0])

            print("\n===== Exploitation =====")
            print("Random Number :", round(random_number, 4))
            print("Predicted Q-Values :", q_values[0])
            print("Best Action :", action)

        return action


    # --------------------------------------------------------
# Update Target Network
# --------------------------------------------------------
    def update_target_network(self):

    # Copy weights from main network
        self.target_model.set_weights(
         self.model.get_weights()
        )

        
    

# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("         DAY 3 - TARGET NETWORK TEST")
    print("=" * 60)

    # Create DQN Model
    state_size = 2
    action_size = 5

    dqn = DQNModel(
        state_size,
        action_size
    )

    print("\n1. DQN Model Initialization")
    print("-" * 40)
    print("Input Features :", state_size)
    print("Output Actions :", action_size)
    print("Learning Rate  :", dqn.learning_rate)
    print("Epsilon        :", dqn.epsilon)

    print("\nMain Network Created Successfully!")

    print("\n2. Target Network")
    print("-" * 40)
    print("Target Network Created Successfully!")

    # Sample State
    state = np.array([100, 30])

    print("\n3. Test Epsilon-Greedy Policy")
    print("-" * 40)
    print("Current State :", state)

    action = dqn.choose_action(state)

    print("Selected Action :", action)

    print("\n4. Synchronizing Target Network")
    print("-" * 40)

    dqn.update_target_network()

    print("Target Network Updated Successfully!")

    print("Weights Copied Successfully!")

    print("\n" + "=" * 60)
    print("DAY 3 IMPLEMENTATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("✓ Deep Neural Network Initialized")
    print("✓ Epsilon-Greedy Policy Tested")
    print("✓ Target Network Created")
    print("✓ Target Network Synchronized")
    print("\nThe DQN model is now ready for training.")
    print("=" * 60)