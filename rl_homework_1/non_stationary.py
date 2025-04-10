import numpy as np


def get_reward_and_cost_nonstationary(arm, step):
    """
    Non-stationary environment where costs drift every 100 steps.
    """
    base_reward_means = [0.8, 0.6, 0.9, 0.4, 0.7]
    reward_stds = [0.1, 0.1, 0.1, 0.1, 0.1]
    base_costs = [0.2, 0.1, 0.3, 0.05, 0.15]

    # Every 100 steps, add a random drift between -0.1 and 0.1 to each cost
    drift = (step // 100) * np.random.uniform(-0.1, 0.1, size=len(base_costs))
    current_costs = np.array(base_costs) + drift
    current_costs = np.maximum(current_costs, 0)  # Ensure non-negative costs

    reward = np.random.normal(base_reward_means[arm], reward_stds[arm])
    return reward, current_costs[arm]


def epsilon_greedy_nonstationary(num_arms, num_steps, epsilon):
    counts = np.zeros(num_arms)
    Q_values = np.zeros(num_arms)

    cumulative_net_rewards = []
    total_net_reward = 0

    for t in range(1, num_steps + 1):
        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(Q_values)

        reward, cost = get_reward_and_cost_nonstationary(arm, t)
        net_reward = reward - cost

        counts[arm] += 1
        Q_values[arm] += (net_reward - Q_values[arm]) / counts[arm]

        total_net_reward += net_reward
        cumulative_net_rewards.append(total_net_reward)

    return cumulative_net_rewards
