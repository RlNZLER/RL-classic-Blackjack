import gymnasium as gym

# Create our training environment - a cart with a pole that needs balancing
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - card values, dealer's visible card, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [player_hand, dealer_hand, usable_ace]

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = stick, 1 = hit
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the agent stays in the game
    # terminated: True if the agent goes bust (over 21)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()