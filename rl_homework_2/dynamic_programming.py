def value_iteration(states, actions, P, gamma=0.99, theta=1e-4, max_iterations=1000):
    V = {s: 0.0 for s in states}
    for i in range(1, max_iterations + 1):
        delta = 0
        for s in states:
            v_old = V[s]
            V[s] = max(
                sum(prob * (r + gamma * V[s_next]) for prob, s_next, r in P[s][a])
                for a in actions
            )
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    policy = {}
    for s in states:
        q_vals = {
            a: sum(prob * (r + gamma * V[s_next]) for prob, s_next, r in P[s][a])
            for a in actions
        }
        policy[s] = max(q_vals, key=q_vals.get)
    return V, policy, i
