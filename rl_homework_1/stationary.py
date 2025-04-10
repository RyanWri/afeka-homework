import numpy as np


def get_reward_and_cost(arm):
    reward_means = [0.8, 0.6, 0.9, 0.4, 0.7]
    reward_stds = [0.1, 0.1, 0.1, 0.1, 0.1]
    costs = [0.2, 0.1, 0.3, 0.05, 0.15]

    reward = np.random.normal(reward_means[arm], reward_stds[arm])
    return reward, costs[arm]


def epsilon_greedy_with_costs(num_arms, num_steps, epsilon):
    counts = np.zeros(num_arms)
    Q_values = np.zeros(num_arms)
    cumulative_net_rewards = []
    total_net_reward = 0
    chosen_arms = []
    for t in range(1, num_steps + 1):
        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(Q_values)

        chosen_arms.append(arm)
        reward, cost = get_reward_and_cost(arm)
        net_reward = reward - cost
        counts[arm] += 1
        Q_values[arm] += (net_reward - Q_values[arm]) / counts[arm]
        total_net_reward += net_reward
        cumulative_net_rewards.append(total_net_reward)

    return cumulative_net_rewards, chosen_arms


def ucb_with_costs(num_arms, num_steps, c=1.0):
    """
    UCB algorithm with costs.
    """
    counts = np.zeros(num_arms)
    Q_values = np.zeros(num_arms)
    cumulative_net_rewards = []
    total_net_reward = 0
    for arm in range(num_arms):
        reward, cost = get_reward_and_cost(arm)
        net_reward = reward - cost
        Q_values[arm] = net_reward
        counts[arm] = 1
        total_net_reward += net_reward
        cumulative_net_rewards.append(total_net_reward)

    for t in range(num_arms + 1, num_steps + 1):
        ucb_values = Q_values + c * np.sqrt(np.log(t) / counts)
        arm = np.argmax(ucb_values)
        reward, cost = get_reward_and_cost(arm)
        net_reward = reward - cost
        counts[arm] += 1
        Q_values[arm] += (net_reward - Q_values[arm]) / counts[arm]
        total_net_reward += net_reward
        cumulative_net_rewards.append(total_net_reward)

    return cumulative_net_rewards


def calculate_regret(cumulative_rewards):
    """
    Calculate the regret over time.
    """
    reward_means = [0.8, 0.6, 0.9, 0.4, 0.7]
    costs = [0.2, 0.1, 0.3, 0.05, 0.15]
    expected_net_rewards = [m - c for m, c in zip(reward_means, costs)]
    optimal_arm = np.argmax(expected_net_rewards)
    optimal_net_reward = expected_net_rewards[optimal_arm]
    regrets = [
        optimal_net_reward * (i + 1) - r for i, r in enumerate(cumulative_rewards)
    ]
    return regrets
