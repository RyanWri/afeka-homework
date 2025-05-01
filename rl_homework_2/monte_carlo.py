from collections import defaultdict
import random


# Helper functions for Monte Carlo
def generate_episode(env, policy, max_steps=100):
    episode = []
    state = env.reset()
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def monte_carlo_prediction(env, policy, num_episodes=5000, gamma=0.99):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = {}
    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = gamma * G + reward
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
    for s in returns_sum:
        V[s] = returns_sum[s] / returns_count[s]
    return V


def random_policy(state):
    return random.choice([0, 1, 2, 3])  # uniform random action
