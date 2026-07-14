from pricing_env import PricingEnv

# Create environment
env = PricingEnv()

print("=" * 40)
print("Testing Pricing Environment")
print("=" * 40)

# Reset Environment
state, info = env.reset()

print("\n1. Initial State")
print("State :", state)
print("Info :", info)

# Display Current State
print("\n2. Render Environment")
env.render()

# Take Sample Actions
actions = [0, 2, 4]

for action in actions:

    print("\n" + "=" * 40)
    print(f"Taking Action {action}")
    print("=" * 40)

    next_state, reward, terminated, truncated, info = env.step(action)

    print("Next State :", next_state)
    print("Reward :", reward)
    print("Terminated :", terminated)
    print("Truncated :", truncated)
    print("Info :", info)

    env.render()

    if terminated:
        print("\nEpisode Finished!")
        break

# Close Environment
env.close()

print("\nEnvironment Test Completed Successfully!")