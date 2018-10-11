import numpy as np
from iterative_policy_evaluation import print_policy, print_values
from grid import standard_grid, negative_grid

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")

if __name__ == "__main__":
    grid = negative_grid()
    states = grid.all_states()

    # initialize all states in the Value function
    V = {}
    for s in states:
        V[s] = 0

    # initialize policy with random policies
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print("\nInitial Policy:")
    print_policy(policy, grid)

    # Find Optimal Value Function
    while True:
        delta = 0
        for s in states:
            old_v = V[s]
            max_v = float("-inf")
            if s in policy:
                # Value Iteration Equation Here TODO
                best_a = None
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > max_v:
                        max_v = v
                        best_a = a
                V[s] = max_v
                policy[s] = best_a
                delta = max(delta, np.abs(old_v - V[s]))

        if delta < SMALL_ENOUGH:
            break

    print("\nValues:")
    print_values(V, grid)

    print("\nPolicy:")
    print_policy(policy, grid)
