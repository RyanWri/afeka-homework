from collections import defaultdict
import random


def td0_prediction(env, policy, num_episodes=5000, gamma=0.99, alpha=0.1):
    """
    Temporal Difference (TD(0)) prediction algorithm.
    Args:
        env: GridWorldEnv environment
        policy: function mapping state -> action
        num_episodes: number of episodes to run
        gamma: discount factor
        alpha: learning rate
    Returns:
        V: dict of state -> estimated value
    """
    V = defaultdict(float)

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # TD(0) update rule
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

    return V


def q_learning(env, num_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Q-Learning algorithm (off-policy TD control).
    Args:
        env: GridWorldEnv environment
        num_episodes: number of episodes to train
        gamma: discount factor
        alpha: learning rate
        epsilon: exploration rate for ε-greedy policy

    Returns:
        Q: dict mapping (state, action) to value
    """
    Q = defaultdict(float)

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                q_vals = [Q[(state, a)] for a in env.action_space]
                max_q = max(q_vals)
                best_actions = [a for a in env.action_space if Q[(state, a)] == max_q]
                action = random.choice(best_actions)

            next_state, reward, done, _ = env.step(action)

            # Q-Learning update
            next_q_vals = [Q[(next_state, a)] for a in env.action_space]
            max_next_q = max(next_q_vals)
            Q[(state, action)] += alpha * (
                reward + gamma * max_next_q - Q[(state, action)]
            )

            state = next_state

    return Q


def sarsa(env, num_episodes=5000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    SARSA algorithm (on-policy TD control).
    Args:
        env: GridWorldEnv environment
        num_episodes: number of episodes to train
        gamma: discount factor
        alpha: learning rate
        epsilon: exploration rate for ε-greedy policy

    Returns:
        Q: dict mapping (state, action) to value
    """
    Q = defaultdict(float)

    for _ in range(num_episodes):
        state = env.reset()

        # ε-greedy action selection for the starting state
        if random.random() < epsilon:
            action = random.choice(env.action_space)
        else:
            q_vals = [Q[(state, a)] for a in env.action_space]
            max_q = max(q_vals)
            best_actions = [a for a in env.action_space if Q[(state, a)] == max_q]
            action = random.choice(best_actions)

        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)

            # ε-greedy action selection for the next state
            if random.random() < epsilon:
                next_action = random.choice(env.action_space)
            else:
                q_vals = [Q[(next_state, a)] for a in env.action_space]
                max_q = max(q_vals)
                best_actions = [
                    a for a in env.action_space if Q[(next_state, a)] == max_q
                ]
                next_action = random.choice(best_actions)

            # SARSA update
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )

            state, action = next_state, next_action

    return Q


def td_lambda(env, num_episodes=5000, gamma=0.99, alpha=0.1, lambda_=0.8):
    """
    TD(λ) using backward view with eligibility traces.
    Args:
        env: GridWorldEnvRC environment
        num_episodes: number of episodes to train
        gamma: discount factor
        alpha: learning rate
        lambda_: trace decay parameter

    Returns:
        V: dict mapping state -> estimated value
    """
    V = defaultdict(float)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        eligibility = defaultdict(float)

        while not done:
            # Random policy for simplicity (can be replaced)
            action = random.choice(env.action_space)
            next_state, reward, done, _ = env.step(action)

            # TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Increase eligibility for current state
            eligibility[state] += 1

            # Update all states
            for s in list(eligibility.keys()):
                V[s] += alpha * delta * eligibility[s]
                eligibility[s] *= gamma * lambda_

            state = next_state

    return V
