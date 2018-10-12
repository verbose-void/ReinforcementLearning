import numpy as np
from grid import negative_grid, standard_grid
from iterative_policy_evaluation import print_policy, print_values

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")


def play_game(grid, policy):
    # Start in a random valid position
    start_states = grid.actions.keys()
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]  # no reward for starting position
    while not grid.game_over():
        # For all actions in the given policy, get all rewards by traversing the game world.
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    # calculate returns working backwards from terminal state
    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        if first:
            # Ignore terminal state
            first = False
        else:
            # map the current state with s' value
            states_and_returns.append((s, G))
        G = r + GAMMA * G

    states_and_returns.reverse()
    return states_and_returns


if __name__ == "__main__":
    grid = negative_grid()

    print("\nRewards:")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "R",
        (2, 1): "R",
        (2, 2): "R",
        (2, 3): "U",
    }

    print("\nPolicy:")
    print_policy(policy, grid)

    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = []
        else:
            # Terminal state or impossible state
            V[s] = 0

    for t in range(100):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            seen_states.add(s)
            returns[s] = G
            V[s] = np.mean(returns[s])
            seen_states.add(s)

    print("\nValues:")
    print_values(V, grid)

    print("\nPolicy:")
    print_policy(policy, grid)
