import numpy as np
from grid import negative_grid, standard_grid
import matplotlib.pyplot as plt
from iterative_policy_evaluation import print_policy, print_values

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")


def play_game(grid, policy):
    # Start in a random valid position
    start_states = grid.actions.keys()
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)

    states_and_rewards = [(s, a, 0)]  # no reward for starting position
    while True:
        old_s = s
        r = grid.move(a)
        s = grid.current_state()

        if old_s == s:
            # If the move resulted in no movement, give crazy low reward.
            states_and_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            # no action at terminal states
            states_and_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_and_rewards.append((s, a, r))

    # calculate returns working backwards from terminal state
    G = 0
    states_and_returns = []
    first = True
    for s, a, r in reversed(states_and_rewards):
        if first:
            # Ignore terminal state
            first = False
        else:
            # map the current state with s' value
            states_and_returns.append((s, a, G))
        G = r + GAMMA * G

    states_and_returns.reverse()
    return states_and_returns


def max_dict(d):
    # does argmax (key) & max (val) from a dictionary
    max_key = None
    max_val = float("-inf")
    for k, v in d.iteritems():
        if v > max_val:
            max_key = k
            max_val = v

    return max_key, max_val


if __name__ == "__main__":
    grid = negative_grid()

    print("\nRewards:")
    print_values(grid.rewards, grid)

    # Generate random starter policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print("\nPolicy:")
    print_policy(policy, grid)

    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0  # needs init so argmax can happen
                returns[(s, a)] = []
        else:
            # Terminal state or impossible state
            pass

    deltas = []  # FOR DEBUGGING

    # Policy Iteration
    for t in range(2000):
        if t % 200:
            print(t)

        biggest_delta = 0

        # Policy Improvement
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, a, G in states_and_returns:
            sa = (s, a)
            if sa not in seen_states:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                seen_states.add(sa)
                biggest_delta = max(biggest_delta, np.abs(old_q - Q[s][a]))

        deltas.append(biggest_delta)

        # Update Policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("\nValues:")
    print_values(V, grid)

    print("\nOptimal Policy:")
    print_policy(policy, grid)
