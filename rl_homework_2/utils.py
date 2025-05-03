import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


class RLConfig:
    """
    Configuration for RL experiments.
    Attributes:
        gamma: Discount factor.
        alpha: Learning rate.
        epsilon: Exploration rate for ε-greedy policies.
        episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
    """

    def __init__(self, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=5000, max_steps=100):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps = max_steps


@dataclass
class RLResult:
    name: str  # e.g. "Value Iteration"
    time_seconds: float  # total runtime
    iterations: int  # DP sweeps or episodes run
    value_function: dict  # V(s)
    q_function: dict = None  # optional Q(s,a)
    policy: dict = None  # derived policy π(s)
    rewards_per_episode: list = None  # for learning curves
    notes: str = ""  # any extra context


def record_episode_metrics(env, policy_fn, config):
    """
    Run episodes using policy_fn on env, record total reward and steps per episode.
    Returns:
        rewards: list of total reward per episode
        lengths: list of number of steps per episode
        runtimes: list of runtime per episode (optional)
    """
    rewards = []
    lengths = []
    runtimes = []
    for episode in range(config.episodes):
        start_time = time.time()
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < config.max_steps:
            action = policy_fn(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)
        runtimes.append(time.time() - start_time)
    return rewards, lengths, runtimes


def _plot_grid(grid, env, title, cmap="viridis", show=True):
    """
    Internal helper to render a 2D grid (value or Q for a single action).
    """
    plt.imshow(grid, origin="upper", cmap=cmap)
    for r in range(env.rows):
        for c in range(env.cols):
            coord = (r, c)
            if coord in env.obstacles:
                plt.text(
                    c, r, "■", ha="center", va="center", fontsize=16, color="black"
                )
            elif coord == env.goal_state:
                plt.text(
                    c, r, "G", ha="center", va="center", fontsize=12, color="white"
                )
            elif not np.isnan(grid[r, c]):
                plt.text(
                    c,
                    r,
                    f"{grid[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
    plt.title(title)
    plt.xticks(np.arange(env.cols))
    plt.yticks(np.arange(env.rows))
    plt.gca().invert_yaxis()
    plt.grid(False)
    if show:
        plt.show()


def plot_value_function(V, env, title="Value Function"):
    """
    Plot the state-value function V as a heatmap.
    """
    grid = np.full((env.rows, env.cols), np.nan)
    for (r, c), v in V.items():
        grid[r, c] = v
    _plot_grid(grid, env, title, cmap="viridis")


def plot_q_values(Q, env, title="Q-Values"):
    """
    Plot the action-value function Q as a 2x2 grid of heatmaps, one per action.
    Action ordering: 0=up, 1=right, 2=down, 3=left.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    actions = [0, 1, 2, 3]
    cmaps = ["Reds", "Blues", "Greens", "Purples"]
    for idx, action in enumerate(actions):
        grid = np.full((env.rows, env.cols), np.nan)
        for (r, c), _ in grid.items() if False else []:
            pass
        for (s, a), q in Q.items():
            if a == action:
                r, c = s
                grid[r, c] = q
        ax = axes[idx // 2, idx % 2]
        ax.imshow(grid, origin="upper", cmap=cmaps[idx])
        ax.set_title(f"{title} (action={action})")
        ax.set_xticks(np.arange(env.cols))
        ax.set_yticks(np.arange(env.rows))
        for r in range(env.rows):
            for c in range(env.cols):
                coord = (r, c)
                if coord in env.obstacles:
                    ax.text(
                        c, r, "■", ha="center", va="center", fontsize=16, color="black"
                    )
                elif coord == env.goal_state:
                    ax.text(
                        c, r, "G", ha="center", va="center", fontsize=12, color="white"
                    )
                elif not np.isnan(grid[r, c]):
                    ax.text(
                        c,
                        r,
                        f"{grid[r, c]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )
        ax.invert_yaxis()
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_policy_arrows_from_q(Q, env, title="Policy"):
    """
    Derive greedy policy from Q and plot arrows on the grid.
    """
    # build grid of arrows
    arrow_u = np.zeros((env.rows, env.cols))
    arrow_v = np.zeros((env.rows, env.cols))
    # mapping actions to (dr, dc)
    action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    V = {}
    for (s, a), q in Q.items():
        V.setdefault(s, {})[a] = q
    policy = {}
    for s, actions in V.items():
        best_a = max(actions, key=actions.get)
        policy[s] = best_a
        dr, dc = action_map[best_a]
        r, c = s
        arrow_u[r, c] = dc
        arrow_v[r, c] = -dr
    # background: value heatmap
    value_est = {s: max(actions.values()) for s, actions in V.items()}
    plot_value_function(value_est, env, title + " (value background)")
    # overlay arrows
    plt.quiver(
        np.arange(env.cols),
        np.arange(env.rows),
        arrow_u,
        arrow_v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
    )
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()


def plot_episode_rewards(rewards, title="Reward per Episode"):
    """
    Plot total reward collected in each episode over time.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.show()
