import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from rl_agent.q_learning import QLearningAgent


if __name__ == "__main__":

    # Create Q-Learning Agent
    agent = QLearningAgent()

    print("=" * 45)
    print("      Q-LEARNING AGENT TEST")
    print("=" * 45)

    # Agent Information
    print("\n1. Agent Initialization")
    print("-" * 30)
    print(f"State Size        : {agent.state_size}")
    print(f"Action Size       : {agent.action_size}")
    print(f"Learning Rate     : {agent.learning_rate}")
    print(f"Discount Factor   : {agent.discount_factor}")
    print(f"Epsilon           : {agent.epsilon}")
    print(f"Q-Table Shape     : {agent.q_table.shape}")

    # Reset Environment
    print("\n2. Environment Reset")
    print("-" * 30)

    state, info = agent.env.reset()

    print("Current State :", state)
    print("Inventory     :", state[0])
    print("Days Left     :", state[1])

    # Test Epsilon-Greedy Policy
    print("\n3. Epsilon-Greedy Action Selection")
    print("-" * 30)

    action = agent.choose_action(state)

    print("\nReturned Action :", action)
    print("Selected Price  :", agent.env.price_levels[action])

    # Test Q-Table Update
    print("\n4. Q-Table Update")
    print("-" * 30)

    next_state, reward, terminated, truncated, info = agent.env.step(action)

    updated_q = agent.update_q_table(
        state,
        action,
        reward,
        next_state,
        terminated
    )

    print("\nCurrent State :", state)
    print("Selected Action :", action)
    print("Reward :", reward)
    print("Next State :", next_state)

    print("\nUpdated Q-Value :", updated_q)

    inventory, days_left = state

    print("Stored Q-Value :",
          agent.q_table[inventory, days_left, action])

    # Training Loop
    print("\n5. Training Loop")
    print("-" * 30)

    agent.train(episodes=3)

    print("\n" + "=" * 45)
    print("Q-Learning Agent Tested Successfully")
    print("=" * 45)