from copy import deepcopy


def extract_mdp(env):
    """
    Extract the full MDP model from a deterministic GridWorldEnv.
    Returns:
      - states: list of all non-obstacle states
      - actions: list of all possible actions
      - P: dict mapping state -> action -> list of (prob, next_state, reward)
    """
    states = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.obstacles
    ]
    actions = env.action_space
    P = {}
    for s in states:
        P[s] = {}
        for a in actions:
            env_copy = deepcopy(env)
            env_copy.state = s
            next_s, r, _, _ = env_copy.step(a)
            P[s][a] = [(1.0, next_s, r)]
    return states, actions, P
